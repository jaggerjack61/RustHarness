use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use dialoguer::{Input as TextInput, Password};
use reqwest::blocking::Client;
use serde_json::Value;
use termimad::crossterm::terminal;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::agent::{AgentConfig, AgentHarness, HarnessError, split_multi, tls_insecure_enabled};
use crate::cancel::CancelToken;
use crate::constants::{
    DEFAULT_BASE_URL, DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TURNS, DEFAULT_REASONING_EFFORT,
    DISPLAY_TRUNCATION_LIMIT, FETCH_TIMEOUT_SECS, NONSTANDARD_REASONING_EFFORTS, REASONING_OPTIONS,
};
use crate::display::ResponseBuffer;
use crate::events::Event;
use crate::input::{Input, Prompt, read_line};
use crate::keys::KeyListener;
use crate::markdown::markdown_lines;
use crate::prompts::{prompt_selection, theme};
use crate::providers::{self, SavedProvider};
use crate::ui::{self, BULLET, CROSS, RESULT, SPINNER, THINKING, TICK, Tone, WARN, paint};

const LIVE_REFRESH_INTERVAL: Duration = Duration::from_millis(50);
const SPINNER_INTERVAL: Duration = Duration::from_millis(80);
const MAX_DIFF_LINES: usize = 6;
const MAX_EDITS_SHOWN: usize = 3;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ReasoningEffort {
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "harness",
    version,
    about = "Interactive Nasa Level Genius Agent - chat with an AI that can read, write, edit, and run commands."
)]
struct Cli {
    #[arg(short = 'm', long, env = "HARNESS_MODEL")]
    model: Option<String>,

    #[arg(short = 'k', long, env = "OPENAI_API_KEY")]
    api_key: Option<String>,

    #[arg(short = 'u', long, env = "HARNESS_BASE_URL", default_value = DEFAULT_BASE_URL)]
    base_url: String,

    #[arg(short = 'd', long = "dir")]
    working_dir: Option<PathBuf>,

    #[arg(
        long,
        env = "HARNESS_MAX_TURNS",
        default_value_t = DEFAULT_MAX_TURNS,
        value_parser = parse_positive_usize
    )]
    max_turns: usize,

    #[arg(long, env = "HARNESS_PROMPT")]
    system_prompt: Option<String>,

    #[arg(long, value_enum, default_value = DEFAULT_REASONING_EFFORT)]
    reasoning_effort: ReasoningEffort,

    #[arg(
        long,
        env = "HARNESS_CONTEXT_WINDOW",
        value_parser = parse_positive_i64
    )]
    context_window: Option<i64>,

    #[arg(long, action = ArgAction::SetTrue, overrides_with = "no_stream")]
    stream: bool,

    #[arg(long = "no-stream", action = ArgAction::SetTrue, overrides_with = "stream")]
    no_stream: bool,

    #[arg(long)]
    no_markdown: bool,
}

fn parse_positive_usize(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|value| *value >= 1)
        .ok_or_else(|| format!("expected a positive integer, got {value:?}"))
}

fn parse_positive_i64(value: &str) -> Result<i64, String> {
    value
        .parse::<i64>()
        .ok()
        .filter(|value| *value >= 1)
        .ok_or_else(|| format!("expected a positive integer, got {value:?}"))
}

pub fn run() -> Result<()> {
    load_executable_env()?;
    run_with(Cli::parse())
}

fn load_executable_env() -> Result<()> {
    let executable = std::env::current_exe().context("could not determine executable path")?;
    load_executable_env_from(&executable)
}

fn load_executable_env_from(executable: &Path) -> Result<()> {
    let env_path = executable
        .parent()
        .context("executable path has no parent directory")?
        .join(".env");
    match dotenvy::from_path_override(&env_path) {
        Ok(()) => Ok(()),
        Err(dotenvy::Error::Io(error)) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("could not load {}", env_path.display())),
    }
}

fn run_with(args: Cli) -> Result<()> {
    let working_dir = args
        .working_dir
        .clone()
        .unwrap_or(std::env::current_dir().context("could not determine current directory")?);
    let provider_path = providers::config_path()?;
    let mut saved_providers = providers::load(&provider_path)?;
    let model_path = provider_path.with_file_name("last-model.json");
    let model = match args.model.clone() {
        Some(model) => Some(model),
        None => providers::load_model(&model_path)?,
    };
    let mut config = AgentConfig::new(model.unwrap_or_default());
    config.api_key = args.api_key.clone();
    config.base_url = args.base_url.clone();
    config.working_dir = Some(working_dir.clone());
    config.max_turns = args.max_turns;
    config.reasoning_effort = Some(args.reasoning_effort.as_str().to_owned());
    config.context_window = args.context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW);
    if let Some(system_prompt) = args.system_prompt {
        config.system_prompt = system_prompt;
    }
    if args.api_key.is_none() && args.base_url == DEFAULT_BASE_URL && !saved_providers.is_empty() {
        config.base_url = saved_providers
            .iter()
            .map(|provider| provider.base_url.as_str())
            .collect::<Vec<_>>()
            .join(";;");
        config.api_key = Some(
            saved_providers
                .iter()
                .map(|provider| provider.api_key.as_str())
                .collect::<Vec<_>>()
                .join(";;"),
        );
    }
    if config
        .api_key
        .as_deref()
        .is_none_or(|key| key.trim().is_empty())
        && config.base_url == DEFAULT_BASE_URL
    {
        ui::hint("No provider configured. Add one to get started.");
        let Some((base_url, api_key)) = prompt_login()? else {
            return Ok(());
        };
        saved_providers.insert(
            0,
            SavedProvider {
                base_url: base_url.clone(),
                api_key: api_key.clone(),
            },
        );
        providers::save(&provider_path, &saved_providers)?;
        config.base_url = base_url;
        config.api_key = Some(api_key);
    }
    let mut agent = AgentHarness::new(config)?;
    let mut use_stream = args.stream || !args.no_stream;

    let (base_url, api_key) = agent.provider_config();
    let mut fetcher = ModelFetcher::new(base_url.clone(), api_key);
    let models = fetcher.get();
    if agent.model.is_empty() {
        fetcher.print_failures();
        anyhow::ensure!(
            !models.is_empty(),
            "No models available. Specify a model with --model or check your provider."
        );
        let Some(selected) = prompt_selection(&models, "", "Select a model", "Current model")
        else {
            return Ok(());
        };
        agent.select_model(selected, None);
    }
    apply_model_metadata(&mut agent, &fetcher, args.context_window);
    providers::save_model(&model_path, &agent.model)?;
    print_banner(&agent, &working_dir, use_stream);

    if let Some(warning) =
        check_reasoning_compatibility(&base_url, agent.reasoning_effort.as_deref())
    {
        ui::warning(&warning);
        println!();
    }

    let mut prompt = Prompt::new(Some(provider_path.with_file_name("history")));
    let mut model_checked = false;
    let mut interrupted = false;
    loop {
        prompt.set_effort(
            agent
                .reasoning_effort
                .as_deref()
                .unwrap_or(DEFAULT_REASONING_EFFORT),
        );
        let input = match prompt.read()? {
            Input::Line(line) => {
                interrupted = false;
                apply_prompt_effort(&mut agent, prompt.effort());
                line
            }
            Input::Interrupted if !interrupted => {
                interrupted = true;
                ui::hint("Press Ctrl+C again to quit.");
                continue;
            }
            Input::Interrupted | Input::Eof => break,
        };
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        let command = input.to_lowercase();
        match command.as_str() {
            "/exit" | "/quit" => break,
            "/help" => {
                println!(
                    "\n{}\n",
                    ui::help_text(terminal_width().saturating_sub(1), ui::styled())
                );
                continue;
            }
            "/clear" => {
                agent.clear_history();
                clear_terminal();
                print_banner(&agent, &working_dir, use_stream);
                ui::success("Conversation cleared.");
                println!();
                continue;
            }
            "/stream" => {
                use_stream = !use_stream;
                ui::success(&format!(
                    "Streaming {}.",
                    if use_stream { "on" } else { "off" }
                ));
                println!();
                continue;
            }
            "/login" => {
                if let Some((base_url, api_key)) = prompt_login()? {
                    let mut updated_providers = saved_providers.clone();
                    updated_providers.retain(|provider| provider.base_url != base_url);
                    updated_providers.insert(
                        0,
                        SavedProvider {
                            base_url: base_url.clone(),
                            api_key: api_key.clone(),
                        },
                    );
                    if let Err(error) = providers::save(&provider_path, &updated_providers) {
                        ui::error(&format!("Could not save provider: {error}"));
                        println!();
                        continue;
                    }
                    saved_providers = updated_providers;
                    agent.add_provider(&base_url, &api_key);
                    let (urls, keys) = agent.provider_config();
                    fetcher = ModelFetcher::new(urls, keys);
                    model_checked = false;
                    ui::success("Provider saved.");
                    ui::hint("Loading its models in the background · /models to choose one");
                    if let Some(warning) =
                        check_reasoning_compatibility(&base_url, agent.reasoning_effort.as_deref())
                    {
                        ui::warning(&warning);
                    }
                }
                println!();
                continue;
            }
            "/model" | "/models" => {
                if !fetcher.ready() {
                    ui::hint("Loading models…");
                }
                let mut models = fetcher.get();
                if models.is_empty() {
                    ui::hint("Fetching models…");
                    fetcher.refresh();
                    models = fetcher.get();
                }
                fetcher.print_failures();
                if models.is_empty() {
                    ui::warning("No models available from the configured providers.");
                } else if let Some(selected) =
                    prompt_selection(&models, &agent.model, "Select a model", "Current model")
                {
                    agent.select_model(selected, None);
                    apply_model_metadata(&mut agent, &fetcher, args.context_window);
                    model_checked = true;
                    if let Err(error) = providers::save_model(&model_path, &agent.model) {
                        ui::error(&format!("Could not save model: {error}"));
                    }
                    ui::success(&format!(
                        "Model set to {}.",
                        ui::style(&agent.model, Tone::Bold)
                    ));
                }
                println!();
                continue;
            }
            "/reasoning" => {
                let current = agent.reasoning_effort.as_deref().unwrap_or("high");
                if let Some(selected) = prompt_selection(
                    REASONING_OPTIONS,
                    current,
                    "Select reasoning effort",
                    "Current effort",
                ) {
                    ui::success(&format!(
                        "Reasoning effort set to {}.",
                        ui::style(&selected, Tone::Bold)
                    ));
                    agent.reasoning_effort = Some(selected);
                }
                println!();
                continue;
            }
            "/context" => {
                if let Some(context) = read_multiline_context()? {
                    let line_count = context.lines().count();
                    agent.set_custom_context(Some(context));
                    ui::success(&format!(
                        "Custom context set ({line_count} {}).",
                        plural(line_count, "line")
                    ));
                }
                println!();
                continue;
            }
            "/context clear" => {
                agent.clear_custom_context();
                ui::success("Custom context cleared.");
                println!();
                continue;
            }
            "/context show" => {
                if let Some(context) = agent.get_custom_context() {
                    let count = context.lines().count();
                    println!(
                        "\n{} {}",
                        ui::style("Custom context", Tone::Bold),
                        ui::style(&format!("· {count} {}", plural(count, "line")), Tone::Muted)
                    );
                    for line in context.lines() {
                        println!("{} {line}", ui::style("│", Tone::Muted));
                    }
                } else {
                    ui::hint("No custom context set. Use /context to add one.");
                }
                println!();
                continue;
            }
            _ if is_unknown_command(&command) => {
                ui::warning(&format!("Unknown command {input}."));
                ui::hint("Type /help to see all commands.");
                println!();
                continue;
            }
            _ => {}
        }

        if !model_checked && (fetcher.requires_routing() || fetcher.ready()) {
            let models = fetcher.get();
            apply_model_metadata(&mut agent, &fetcher, args.context_window);
            if let Some(warning) = check_model_available(&agent.model, &models) {
                ui::warning(&warning);
            }
            model_checked = true;
        }

        println!();
        let mut display = run_turn(&mut agent, input, use_stream, args.no_markdown);
        display.complete();
        println!();
    }
    ui::hint("Goodbye!");
    Ok(())
}

/// Hides the terminal cursor so it does not sit on the animated status line.
struct HiddenCursor;

impl HiddenCursor {
    fn new() -> Self {
        print!("\x1b[?25l");
        let _ = io::stdout().flush();
        Self
    }
}

impl Drop for HiddenCursor {
    fn drop(&mut self) {
        print!("\x1b[?25h");
        let _ = io::stdout().flush();
    }
}

/// Adopt a reasoning effort chosen with Tab at the prompt.
fn apply_prompt_effort(agent: &mut AgentHarness, effort: &str) {
    if agent.reasoning_effort.as_deref() == Some(effort) {
        return;
    }
    agent.reasoning_effort = Some(effort.to_owned());
    let (base_url, _) = agent.provider_config();
    if let Some(warning) = check_reasoning_compatibility(&base_url, Some(effort)) {
        ui::warning(&warning);
    }
}

/// Run one user turn, animating the live region from a ticker thread while the
/// agent blocks the main thread. Esc (or Ctrl+C) interrupts the turn.
fn run_turn(agent: &mut AgentHarness, input: &str, stream: bool, no_markdown: bool) -> CliDisplay {
    let mut display = CliDisplay::new(no_markdown);
    let animate = display.interactive;
    let cancel = agent.cancel_token();
    cancel.reset();
    let keys = if animate {
        KeyListener::start(cancel.clone())
    } else {
        None
    };
    if keys.is_some() {
        display.cancel = Some(cancel);
    }
    let _cursor = animate.then(HiddenCursor::new);
    let display = Mutex::new(display);
    let stop = AtomicBool::new(false);
    let response = thread::scope(|scope| {
        if animate {
            scope.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    thread::sleep(SPINNER_INTERVAL);
                    if !stop.load(Ordering::Relaxed) {
                        display
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .tick();
                    }
                }
            });
        }
        let response = agent.run_with_callback(input, stream, &mut |event| {
            display
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .on_event(event);
        });
        stop.store(true, Ordering::Relaxed);
        response
    });
    drop(keys);
    let mut display = display.into_inner().unwrap_or_else(PoisonError::into_inner);
    match response {
        Ok(_) => {}
        Err(HarnessError::Cancelled) => display.interrupt(),
        Err(error) => display.fail(&error.to_string()),
    }
    display
}

fn print_banner(agent: &AgentHarness, working_dir: &Path, streaming: bool) {
    let info = ui::BannerInfo {
        model: &agent.model,
        reasoning_effort: agent.reasoning_effort.as_deref().unwrap_or("high"),
        context_window: agent.context_window,
        working_dir,
        streaming,
    };
    println!("{}\n", ui::banner(&info, terminal_width(), ui::styled()));
}

/// A lone `/word` that is not a known command (paths like `/usr/bin` are not commands).
fn is_unknown_command(input: &str) -> bool {
    input.len() > 1
        && input.starts_with('/')
        && !input[1..].contains(['/', ' ', '\t', '\n'])
        && !ui::COMMANDS.iter().any(|(command, _)| *command == input)
}

fn plural(count: usize, noun: &str) -> String {
    if count == 1 {
        noun.to_owned()
    } else {
        format!("{noun}s")
    }
}

fn prompt_login() -> Result<Option<(String, String)>> {
    println!();
    ui::hint("Add a provider · leave a field empty to cancel");
    let interactive = io::stdin().is_terminal();
    let url = if interactive {
        TextInput::<String>::with_theme(&theme())
            .with_prompt("Base URL")
            .allow_empty(true)
            .interact_text()
            .unwrap_or_default()
    } else {
        print!("Base URL: ");
        io::stdout().flush()?;
        read_line()?.unwrap_or_default()
    };
    if url.trim().is_empty() {
        ui::hint("Cancelled.");
        return Ok(None);
    }
    let url = match providers::validate_url(&url) {
        Ok(url) => url,
        Err(error) => {
            ui::error(&error.to_string());
            return Ok(None);
        }
    };
    let key = if interactive {
        Password::with_theme(&theme())
            .with_prompt("API key")
            .allow_empty_password(true)
            .report(false)
            .interact()
            .unwrap_or_default()
    } else {
        print!("API key: ");
        io::stdout().flush()?;
        read_line()?.unwrap_or_default()
    };
    let key = key.trim();
    if key.is_empty() {
        ui::hint("Cancelled.");
        return Ok(None);
    }
    if key.contains(";;") {
        ui::error("Enter one API key per /login.");
        return Ok(None);
    }
    Ok(Some((url, key.to_owned())))
}

fn read_multiline_context() -> io::Result<Option<String>> {
    println!();
    ui::hint("Paste or type the context. Finish with a line containing only '.'; Ctrl+D cancels.");
    let gutter = ui::style("│ ", Tone::Muted);
    let interactive = io::stdin().is_terminal();
    let mut lines = Vec::new();
    loop {
        if interactive {
            print!("{gutter}");
            io::stdout().flush()?;
        }
        let Some(line) = read_line()? else {
            println!();
            ui::hint("Cancelled.");
            return Ok(None);
        };
        let line = line.trim_end_matches(['\r', '\n']);
        if line.trim() == "." {
            break;
        }
        lines.push(line.to_owned());
    }
    if lines.is_empty() {
        ui::hint("No text entered — context unchanged.");
        Ok(None)
    } else {
        Ok(Some(lines.join("\n")))
    }
}

fn clear_terminal() {
    if io::stdout().is_terminal() {
        print!("\x1b[2J\x1b[3J\x1b[H");
        let _ = io::stdout().flush();
    }
}

fn check_reasoning_compatibility(base_url: &str, effort: Option<&str>) -> Option<String> {
    let first_base_url = split_multi(Some(base_url)).into_iter().next();
    effort
        .filter(|effort| NONSTANDARD_REASONING_EFFORTS.contains(effort))
        .filter(|_| first_base_url.is_some_and(|url| url.contains("openai.com")))
        .map(|effort| {
            format!(
                "Reasoning effort '{effort}' is not supported by OpenAI and may cause errors. Use low/medium/high, or switch --base-url to a compatible provider."
            )
        })
}

fn check_model_available(model: &str, models: &[String]) -> Option<String> {
    (!models.is_empty() && !models.iter().any(|available| available == model)).then(|| {
        format!(
            "Model '{model}' was not found among the {} available models. Use /models to pick one.",
            models.len()
        )
    })
}

#[derive(Clone, Debug, Default)]
struct ModelCatalog {
    models: Vec<String>,
    sources: BTreeMap<String, Vec<usize>>,
    failures: Vec<String>,
    context_windows: BTreeMap<String, i64>,
}

fn model_context_window(model: &Value) -> Option<i64> {
    [
        "/context_window",
        "/context_length",
        "/max_context_length",
        "/max_model_len",
        "/architecture/context_length",
        "/top_provider/context_length",
    ]
    .into_iter()
    .filter_map(|path| model.pointer(path))
    .find_map(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
            .filter(|value| *value > 0)
    })
}

fn apply_model_metadata(
    agent: &mut AgentHarness,
    fetcher: &ModelFetcher,
    override_window: Option<i64>,
) {
    let endpoint_indices = fetcher.endpoint_indices_for(&agent.model);
    agent.context_window = override_window
        .or_else(|| fetcher.context_window_for(&agent.model))
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);
    agent.select_model(agent.model.clone(), endpoint_indices);
}

fn fetch_models(base_url: &str, api_key: Option<&str>) -> Result<ModelCatalog> {
    let client = Client::builder()
        .timeout(Duration::from_secs(FETCH_TIMEOUT_SECS))
        .danger_accept_invalid_certs(tls_insecure_enabled())
        .build()?;
    let urls = split_multi(Some(base_url));
    let keys = split_multi(api_key);
    let mut models = BTreeSet::new();
    let mut sources = BTreeMap::<String, Vec<usize>>::new();
    let mut failures = Vec::new();
    let mut context_windows = BTreeMap::new();

    for (index, configured_url) in urls.iter().enumerate() {
        let url = format!("{}/models", configured_url.trim_end_matches('/'));
        let mut request = client.get(url);
        if let Some(key) = keys.get(index).or_else(|| keys.last()) {
            request = request.bearer_auth(key);
        }
        let result = request
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(|response| response.json::<Value>());
        match result {
            Ok(value) => {
                let endpoint_models = value
                    .get("data")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten();
                for metadata in endpoint_models {
                    let Some(model) = metadata.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    models.insert(model.to_owned());
                    sources.entry(model.to_owned()).or_default().push(index);
                    if let Some(window) = model_context_window(metadata) {
                        context_windows
                            .entry(model.to_owned())
                            .and_modify(|existing: &mut i64| *existing = (*existing).min(window))
                            .or_insert(window);
                    }
                }
            }
            Err(error) => {
                let failure = format!(
                    "Failed to fetch models from {configured_url}: {}",
                    error_chain(&error)
                );
                failures.push(failure);
            }
        }
    }

    Ok(ModelCatalog {
        models: models.into_iter().collect(),
        sources,
        failures,
        context_windows,
    })
}

fn error_chain(error: &dyn std::error::Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(error) = source {
        message.push_str(": ");
        message.push_str(&error.to_string());
        source = error.source();
    }
    message
}

#[derive(Clone)]
struct ModelFetcher {
    base_url: String,
    api_key: Option<String>,
    requires_routing: bool,
    state: Arc<(Mutex<Option<ModelCatalog>>, Condvar)>,
}

impl ModelFetcher {
    fn new(base_url: String, api_key: Option<String>) -> Self {
        let requires_routing = split_multi(Some(&base_url)).len() > 1;
        let fetcher = Self {
            base_url,
            api_key,
            requires_routing,
            state: Arc::new((Mutex::new(None), Condvar::new())),
        };
        let worker = fetcher.clone();
        thread::spawn(move || {
            let catalog =
                fetch_models(&worker.base_url, worker.api_key.as_deref()).unwrap_or_default();
            worker.store(catalog);
        });
        fetcher
    }

    fn ready(&self) -> bool {
        self.state.0.lock().expect("model state poisoned").is_some()
    }

    fn get(&self) -> Vec<String> {
        let (lock, ready) = &*self.state;
        let models = ready
            .wait_while(lock.lock().expect("model state poisoned"), |models| {
                models.is_none()
            })
            .expect("model state poisoned");
        models
            .as_ref()
            .map(|catalog| catalog.models.clone())
            .unwrap_or_default()
    }

    fn requires_routing(&self) -> bool {
        self.requires_routing
    }

    fn endpoint_indices_for(&self, model: &str) -> Option<Vec<usize>> {
        self.state
            .0
            .lock()
            .expect("model state poisoned")
            .as_ref()
            .and_then(|catalog| catalog.sources.get(model).cloned())
    }

    fn context_window_for(&self, model: &str) -> Option<i64> {
        self.state
            .0
            .lock()
            .expect("model state poisoned")
            .as_ref()
            .and_then(|catalog| catalog.context_windows.get(model).copied())
    }

    fn print_failures(&self) {
        let state = self.state.0.lock().expect("model state poisoned");
        if let Some(catalog) = state.as_ref() {
            for failure in &catalog.failures {
                ui::error(failure);
            }
        }
    }

    fn refresh(&self) {
        match fetch_models(&self.base_url, self.api_key.as_deref()) {
            Ok(catalog) => self.store(catalog),
            Err(error) => {
                ui::error(&format!("Failed to fetch models: {error}"));
                self.store(ModelCatalog::default());
            }
        }
    }

    fn store(&self, catalog: ModelCatalog) {
        let (lock, ready) = &*self.state;
        *lock.lock().expect("model state poisoned") = Some(catalog);
        ready.notify_all();
    }
}

#[derive(Clone)]
struct TokenStatus {
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    turn_input: u64,
    context_window: i64,
    model: String,
    reasoning_effort: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Active,
    Done,
    Failed,
    Interrupted,
}

/// What the agent is doing right now, shown next to the spinner.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Phase {
    Waiting,
    Thinking,
    Writing,
    Tool(String),
}

impl Phase {
    fn label(&self) -> String {
        match self {
            Self::Waiting => "Working".to_owned(),
            Self::Thinking => "Thinking".to_owned(),
            Self::Writing => "Writing".to_owned(),
            Self::Tool(name) => match name.as_str() {
                "bash" => "Running command".to_owned(),
                "read" => "Reading".to_owned(),
                "write" => "Writing file".to_owned(),
                "edit" => "Editing".to_owned(),
                other => format!("Running {other}"),
            },
        }
    }
}

/// Everything shown on the status line.
struct StatusInfo<'a> {
    outcome: Outcome,
    /// Spinner animation frame.
    frame: usize,
    label: &'a str,
    elapsed_secs: f64,
    /// Show the "esc to interrupt" hint.
    interruptible: bool,
    tokens: Option<&'a TokenStatus>,
}

/// Render the one-line status: spinner (or outcome), elapsed time, and token stats.
/// Lower-priority segments are dropped until the line fits in `max_width`.
fn render_status(info: &StatusInfo<'_>, max_width: usize, styled: bool) -> String {
    let StatusInfo {
        outcome,
        frame,
        label,
        elapsed_secs,
        interruptible,
        tokens,
    } = *info;
    let lead: Vec<(String, Tone)> = match outcome {
        Outcome::Active => vec![
            (SPINNER[frame % SPINNER.len()].to_owned(), Tone::Accent),
            (format!(" {label}…"), Tone::Plain),
            (
                format!(" {}", ui::format_duration(elapsed_secs, false)),
                Tone::Muted,
            ),
        ],
        Outcome::Done => vec![
            (TICK.to_owned(), Tone::Success),
            (
                format!(" Done in {}", ui::format_duration(elapsed_secs, true)),
                Tone::Muted,
            ),
        ],
        Outcome::Failed => vec![
            (CROSS.to_owned(), Tone::Error),
            (
                format!(" Failed after {}", ui::format_duration(elapsed_secs, true)),
                Tone::Muted,
            ),
        ],
        Outcome::Interrupted => vec![
            ("■".to_owned(), Tone::Warning),
            (
                format!(
                    " Interrupted after {}",
                    ui::format_duration(elapsed_secs, true)
                ),
                Tone::Muted,
            ),
        ],
    };

    // (priority, text, tone): higher priorities are dropped first.
    let mut segments: Vec<(u8, String, Tone)> = Vec::new();
    if interruptible && outcome == Outcome::Active {
        segments.push((2, "esc to interrupt".to_owned(), Tone::Muted));
    }
    if let Some(tokens) = tokens {
        segments.push((
            1,
            format!(
                "↑ {} ↓ {}",
                ui::compact_number(tokens.input_tokens),
                ui::compact_number(tokens.output_tokens)
            ),
            Tone::Muted,
        ));
        if tokens.context_window > 0 {
            let percentage = tokens.turn_input as f64 / tokens.context_window as f64 * 100.0;
            let text = if percentage < 10.0 {
                format!("ctx {percentage:.1}%")
            } else {
                format!("ctx {percentage:.0}%")
            };
            let tone = if percentage < 50.0 {
                Tone::Muted
            } else if percentage < 80.0 {
                Tone::Warning
            } else {
                Tone::Error
            };
            segments.push((3, text, tone));
        }
        if tokens.cached_tokens > 0 && tokens.input_tokens > 0 {
            let rate = tokens.cached_tokens as f64 / tokens.input_tokens as f64 * 100.0;
            segments.push((5, format!("cache {rate:.0}%"), Tone::Muted));
        }
        segments.push((4, tokens.model.clone(), Tone::Muted));
        if let Some(effort) = tokens.reasoning_effort.as_deref() {
            segments.push((6, effort.to_owned(), Tone::Muted));
        }
    }

    const SEPARATOR: &str = " · ";
    let lead_width: usize = lead
        .iter()
        .map(|(text, _)| UnicodeWidthStr::width(text.as_str()))
        .sum();
    let width_of = |segments: &[(u8, String, Tone)]| {
        lead_width
            + segments
                .iter()
                .map(|(_, text, _)| {
                    UnicodeWidthStr::width(SEPARATOR) + UnicodeWidthStr::width(text.as_str())
                })
                .sum::<usize>()
    };
    while width_of(&segments) > max_width && !segments.is_empty() {
        let lowest = segments
            .iter()
            .enumerate()
            .max_by_key(|(_, (priority, _, _))| *priority)
            .map(|(index, _)| index)
            .expect("segments is not empty");
        segments.remove(lowest);
    }

    if lead_width > max_width {
        let plain: String = lead.iter().map(|(text, _)| text.as_str()).collect();
        return paint(&ui::truncate(&plain, max_width), Tone::Muted, styled);
    }
    let mut output: String = lead
        .iter()
        .map(|(text, tone)| paint(text, *tone, styled))
        .collect();
    for (_, text, tone) in &segments {
        output.push_str(&paint(SEPARATOR, Tone::Muted, styled));
        output.push_str(&paint(text, *tone, styled));
    }
    output
}

#[derive(Default)]
struct LiveRegion {
    line_count: usize,
}

impl LiveRegion {
    fn replace(&mut self, lines: &[String]) {
        let mut output = String::from("\x1b[?2026h");
        self.push_clear_sequence(&mut output);
        for (index, line) in lines.iter().enumerate() {
            if index > 0 {
                output.push_str("\r\n");
            }
            output.push_str(line);
        }
        output.push_str("\x1b[?2026l");
        print!("{output}");
        let _ = io::stdout().flush();
        self.line_count = lines.len();
    }

    fn clear(&mut self) {
        if self.line_count == 0 {
            return;
        }
        let mut output = String::from("\x1b[?2026h");
        self.push_clear_sequence(&mut output);
        output.push_str("\x1b[?2026l");
        print!("{output}");
        let _ = io::stdout().flush();
        self.line_count = 0;
    }

    fn push_clear_sequence(&self, output: &mut String) {
        if self.line_count == 0 {
            return;
        }
        output.push('\r');
        if self.line_count > 1 {
            output.push_str(&format!("\x1b[{}A", self.line_count - 1));
        }
        output.push_str("\x1b[J");
    }
}

struct CliDisplay {
    no_markdown: bool,
    interactive: bool,
    styled: bool,
    response_buffer: ResponseBuffer,
    thinking_line_buf: String,
    thinking_first_line: bool,
    /// Blank thinking lines held back until more thinking text follows.
    thinking_blank_lines: usize,
    live: LiveRegion,
    last_tokens: Option<TokenStatus>,
    last_live_refresh: Option<Instant>,
    finished: bool,
    started: Instant,
    phase: Phase,
    pending_tool: Option<(String, Value)>,
    /// Number of output blocks printed so far; used to separate blocks with a blank line.
    blocks: usize,
    /// Set while Esc can interrupt the turn.
    cancel: Option<CancelToken>,
}

impl CliDisplay {
    fn new(no_markdown: bool) -> Self {
        Self {
            no_markdown,
            interactive: io::stdout().is_terminal(),
            styled: ui::styled(),
            response_buffer: ResponseBuffer::new(),
            thinking_line_buf: String::new(),
            thinking_first_line: true,
            thinking_blank_lines: 0,
            live: LiveRegion::default(),
            last_tokens: None,
            last_live_refresh: None,
            finished: false,
            started: Instant::now(),
            phase: Phase::Waiting,
            pending_tool: None,
            blocks: 0,
            cancel: None,
        }
    }

    fn begin_block(&mut self) {
        if self.blocks > 0 {
            println!();
        }
        self.blocks += 1;
    }

    fn status_line(&self, outcome: Outcome, width: usize) -> String {
        let elapsed = self.started.elapsed();
        let cancelling = self.cancel.as_ref().is_some_and(CancelToken::is_cancelled);
        let label = if cancelling {
            "Interrupting".to_owned()
        } else {
            self.phase.label()
        };
        let info = StatusInfo {
            outcome,
            frame: (elapsed.as_millis() / SPINNER_INTERVAL.as_millis()) as usize,
            label: &label,
            elapsed_secs: elapsed.as_secs_f64(),
            interruptible: self.cancel.is_some() && !cancelling,
            tokens: self.last_tokens.as_ref(),
        };
        render_status(&info, width, self.styled)
    }

    fn live_lines(&self, width: usize, height: usize) -> Vec<String> {
        let width = width.saturating_sub(1).max(1);
        let mut lines = Vec::new();
        let gap = |lines: &mut Vec<String>, blocks: usize| {
            if blocks > 0 || !lines.is_empty() {
                lines.push(String::new());
            }
        };

        if !self.thinking_line_buf.trim().is_empty() {
            if self.thinking_first_line {
                gap(&mut lines, self.blocks);
            }
            lines.extend(self.thinking_lines(&self.thinking_line_buf, width));
        }

        let response = self.response_buffer.text();
        if !response.trim().is_empty() {
            gap(&mut lines, self.blocks);
            lines.extend(prefixed_lines(
                response.trim_start(),
                &format!("{BULLET} "),
                "  ",
                width,
            ));
        }

        if let Some((name, arguments)) = self.pending_tool.as_ref() {
            gap(&mut lines, self.blocks);
            lines.push(tool_header_line(
                name,
                arguments,
                Tone::Muted,
                width,
                self.styled,
            ));
        }

        gap(&mut lines, self.blocks);
        lines.push(self.status_line(Outcome::Active, width));

        let max_lines = height.saturating_sub(1).max(1);
        if lines.len() > max_lines {
            lines.drain(..lines.len() - max_lines);
        }
        lines
    }

    fn refresh_live(&mut self, force: bool) {
        if !self.interactive || self.finished {
            return;
        }
        if !force
            && self
                .last_live_refresh
                .is_some_and(|last| last.elapsed() < LIVE_REFRESH_INTERVAL)
        {
            return;
        }
        let (width, height) = terminal_dimensions();
        let lines = self.live_lines(width, height);
        self.live.replace(&lines);
        self.last_live_refresh = Some(Instant::now());
    }

    /// Advance the spinner animation.
    fn tick(&mut self) {
        self.refresh_live(false);
    }

    fn prepare_output(&mut self) {
        if self.interactive {
            self.live.clear();
            self.last_live_refresh = None;
        }
    }

    /// Print an error for the whole turn, dropping any half-streamed output.
    fn fail(&mut self, message: &str) {
        self.prepare_output();
        self.response_buffer.reset();
        self.thinking_line_buf.clear();
        self.pending_tool = None;
        self.begin_block();
        let width = terminal_width().saturating_sub(1).max(4);
        for (index, line) in prefixed_lines(message, "", "", width.saturating_sub(2))
            .into_iter()
            .enumerate()
        {
            let prefix = if index == 0 {
                format!("{} ", paint(CROSS, Tone::Error, self.styled))
            } else {
                "  ".to_owned()
            };
            println!("{prefix}{}", paint(&line, Tone::Error, self.styled));
        }
        self.finish(Outcome::Failed);
    }

    /// End a turn the user interrupted, dropping any half-streamed response.
    fn interrupt(&mut self) {
        self.prepare_output();
        self.response_buffer.reset();
        self.pending_tool = None;
        self.finish(Outcome::Interrupted);
    }

    /// Finish a successful turn (no-op if the turn already failed).
    fn complete(&mut self) {
        self.finish(Outcome::Done);
    }

    fn finish(&mut self, outcome: Outcome) {
        if self.finished {
            return;
        }
        self.prepare_output();
        if !self.thinking_line_buf.is_empty() {
            self.finish_thinking();
        }
        self.finished = true;
        if self.blocks > 0 {
            println!();
        }
        let width = terminal_width().saturating_sub(1).max(1);
        println!("{}", self.status_line(outcome, width));
    }

    fn print_final_response(&mut self, content: &str) {
        self.prepare_output();
        self.response_buffer.reset();
        self.phase = Phase::Waiting;
        let content = content.trim();
        if !content.is_empty() {
            self.begin_block();
            let width = terminal_width().saturating_sub(1).max(4);
            let bullet = format!("{BULLET} ");
            let lines = if self.no_markdown {
                prefixed_lines(content, &bullet, "  ", width)
            } else {
                markdown_lines(content, width - 2, self.styled)
                    .into_iter()
                    .enumerate()
                    .map(|(index, line)| {
                        format!("{}{line}", if index == 0 { bullet.as_str() } else { "  " })
                    })
                    .collect()
            };
            for line in lines {
                println!("{}", line.trim_end_matches(' '));
            }
        }
        self.refresh_live(true);
    }

    fn thinking_lines(&self, text: &str, width: usize) -> Vec<String> {
        let first_prefix = if self.thinking_first_line {
            format!("{THINKING} ")
        } else {
            "  ".to_owned()
        };
        prefixed_lines(&sanitize(text), &first_prefix, "  ", width)
            .into_iter()
            .map(|line| paint(&line, Tone::Thinking, self.styled))
            .collect()
    }

    fn print_thinking_line(&mut self, line: &str) {
        if line.trim().is_empty() {
            if !self.thinking_first_line {
                self.thinking_blank_lines += 1;
            }
            return;
        }
        if self.thinking_first_line {
            self.begin_block();
        } else if std::mem::take(&mut self.thinking_blank_lines) > 0 {
            println!();
        }
        let width = terminal_width().saturating_sub(1).max(4);
        for line in self.thinking_lines(line, width) {
            println!("{line}");
        }
        self.thinking_first_line = false;
    }

    fn flush_thinking_lines(&mut self) {
        let Some(newline) = self.thinking_line_buf.rfind('\n') else {
            return;
        };
        let remainder = self.thinking_line_buf[newline + 1..].to_owned();
        let completed = self.thinking_line_buf[..=newline].to_owned();
        self.prepare_output();
        for line in completed.split_terminator('\n') {
            self.print_thinking_line(line);
        }
        self.thinking_line_buf = remainder;
    }

    fn finish_thinking(&mut self) {
        self.prepare_output();
        let remainder = std::mem::take(&mut self.thinking_line_buf);
        if !remainder.is_empty() {
            self.print_thinking_line(&remainder);
        }
        self.thinking_first_line = true;
        self.thinking_blank_lines = 0;
    }

    fn print_notice(&mut self, message: &str) {
        self.prepare_output();
        self.begin_block();
        println!(
            "{} {}",
            paint(WARN, Tone::Warning, self.styled),
            paint(message, Tone::Warning, self.styled)
        );
        self.refresh_live(true);
    }

    fn print_tool_block(&mut self, name: &str, arguments: &Value, result: &str) {
        self.prepare_output();
        self.begin_block();
        let width = terminal_width().saturating_sub(1).max(12);
        let bullet = if tool_failed(name, result) {
            Tone::Error
        } else {
            Tone::Success
        };
        println!(
            "{}",
            tool_header_line(name, arguments, bullet, width, self.styled)
        );
        let body = tool_body(name, arguments, result, width - 5, self.styled);
        for (index, line) in body.iter().enumerate() {
            if index == 0 {
                println!("  {}  {line}", paint(RESULT, Tone::Muted, self.styled));
            } else {
                println!("     {line}");
            }
        }
    }

    fn on_event(&mut self, event: &Event) {
        match event {
            Event::TurnStart => {
                self.response_buffer.reset();
                self.phase = Phase::Waiting;
                self.refresh_live(true);
            }
            Event::Tokens {
                input_tokens,
                output_tokens,
                cached_tokens,
                turn_input,
                context_window,
                model,
                reasoning_effort,
                ..
            } => {
                self.last_tokens = Some(TokenStatus {
                    input_tokens: *input_tokens,
                    output_tokens: *output_tokens,
                    cached_tokens: *cached_tokens,
                    turn_input: *turn_input,
                    context_window: *context_window,
                    model: model.clone(),
                    reasoning_effort: reasoning_effort.clone(),
                });
                self.refresh_live(true);
            }
            Event::Thinking { content } => {
                self.prepare_output();
                for line in content.lines() {
                    self.print_thinking_line(line);
                }
                self.thinking_first_line = true;
                self.thinking_blank_lines = 0;
                self.refresh_live(true);
            }
            Event::ThinkingDelta { content } => {
                if !content.is_empty() {
                    self.phase = Phase::Thinking;
                    self.thinking_line_buf.push_str(content);
                    self.flush_thinking_lines();
                    self.refresh_live(false);
                }
            }
            Event::ThinkingEnd => {
                self.finish_thinking();
                self.phase = Phase::Waiting;
                self.refresh_live(true);
            }
            Event::TextDelta { content } => {
                if !content.is_empty() {
                    self.phase = Phase::Writing;
                    self.response_buffer.append(content);
                    self.refresh_live(false);
                }
            }
            Event::TextEnd { content } | Event::Text { content } => {
                self.print_final_response(content);
            }
            Event::ToolCall { name, arguments } => {
                self.phase = Phase::Tool(name.clone());
                self.pending_tool = Some((name.clone(), arguments.clone()));
                self.refresh_live(true);
            }
            Event::ToolResult { name, result } => {
                let arguments = self
                    .pending_tool
                    .take()
                    .filter(|(pending, _)| pending == name)
                    .map_or(Value::Null, |(_, arguments)| arguments);
                self.print_tool_block(name, &arguments, result);
                self.phase = Phase::Waiting;
                self.refresh_live(true);
            }
            Event::FinishReason { reason, .. } => {
                let warning = match reason.as_str() {
                    "length" => {
                        "The response was truncated at the model's output limit.".to_owned()
                    }
                    "content_filter" => {
                        "The response was stopped by the provider's content filter.".to_owned()
                    }
                    _ => format!("The response stopped with finish reason {reason:?}."),
                };
                self.print_notice(&warning);
            }
            Event::HistoryTrimmed { summarized } => {
                self.print_notice(&format!(
                    "Context window nearly full: summarized {summarized} earlier {}.",
                    plural(*summarized, "message")
                ));
            }
        }
    }
}

fn tool_failed(name: &str, result: &str) -> bool {
    result.starts_with("Error") || (name == "bash" && split_exit_code(result).1.is_some())
}

/// Title and short argument summary for a tool call, e.g. `("Bash", "cargo test")`.
fn tool_header(name: &str, arguments: &Value) -> (String, String) {
    let text = |key: &str| arguments.get(key).and_then(Value::as_str).unwrap_or("");
    match name {
        "bash" => {
            let command = text("command").trim();
            let mut lines = command.lines();
            let first = lines.next().unwrap_or("").to_owned();
            let detail = if lines.next().is_some() {
                format!("{first} …")
            } else {
                first
            };
            ("Bash".to_owned(), detail)
        }
        "read" => {
            let offset = arguments.get("offset").and_then(Value::as_u64);
            let limit = arguments.get("limit").and_then(Value::as_u64);
            let range = match (offset, limit) {
                (None, None) => String::new(),
                (offset, Some(limit)) => {
                    let start = offset.unwrap_or(1);
                    format!(":{start}-{}", start + limit.saturating_sub(1))
                }
                (Some(offset), None) => format!(":{offset}-"),
            };
            ("Read".to_owned(), format!("{}{range}", text("path")))
        }
        "write" => ("Write".to_owned(), text("path").to_owned()),
        "edit" => ("Edit".to_owned(), text("path").to_owned()),
        other => (other.to_owned(), format_arguments(arguments)),
    }
}

fn tool_header_line(
    name: &str,
    arguments: &Value,
    bullet: Tone,
    width: usize,
    styled: bool,
) -> String {
    let (title, detail) = tool_header(name, arguments);
    let title_width = UnicodeWidthStr::width(title.as_str());
    let room = width.saturating_sub(2 + title_width + 2);
    let detail = ui::truncate(&sanitize(&detail), room);
    let detail = if detail.is_empty() {
        String::new()
    } else {
        format!("({detail})")
    };
    format!(
        "{} {}{detail}",
        paint(BULLET, bullet, styled),
        paint(
            &ui::truncate(&title, width.saturating_sub(2)),
            Tone::Bold,
            styled
        )
    )
}

/// Summarize a tool result for display. Lines are at most `width` columns wide.
fn tool_body(
    name: &str,
    arguments: &Value,
    result: &str,
    width: usize,
    styled: bool,
) -> Vec<String> {
    let width = width.max(4);
    if result.starts_with("Error") {
        let lines: Vec<String> = result
            .lines()
            .flat_map(|line| wrap_line(&sanitize(line), width))
            .collect();
        return capped(lines, Tone::Error, styled);
    }
    match name {
        "read" => {
            let count = result.lines().count();
            vec![format!(
                "Read {} {}",
                paint(&count.to_string(), Tone::Bold, styled),
                plural(count, "line")
            )]
        }
        "write" => {
            let content = arguments
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("");
            let total = content.lines().count();
            let mut lines = vec![format!(
                "Wrote {} {}",
                paint(&total.to_string(), Tone::Bold, styled),
                plural(total, "line")
            )];
            let number_width = total.min(DISPLAY_TRUNCATION_LIMIT).to_string().len();
            for (index, line) in content.lines().take(DISPLAY_TRUNCATION_LIMIT).enumerate() {
                lines.push(format!(
                    "{} {}",
                    paint(
                        &format!("{:>number_width$}", index + 1),
                        Tone::Muted,
                        styled
                    ),
                    ui::truncate(&sanitize(line), width.saturating_sub(number_width + 1))
                ));
            }
            if total > DISPLAY_TRUNCATION_LIMIT {
                lines.push(more_lines(total - DISPLAY_TRUNCATION_LIMIT, styled));
            }
            lines
        }
        "edit" => edit_body(arguments, width, styled),
        "bash" => {
            let (output, exit_code) = split_exit_code(result);
            let output = output.trim_end();
            let mut lines = if output.trim().is_empty() || output == "(no output)" {
                vec![paint("(no output)", Tone::Muted, styled)]
            } else {
                preview_lines(output, width, Tone::Muted, styled)
            };
            if let Some(code) = exit_code {
                lines.push(paint(&format!("exit code {code}"), Tone::Error, styled));
            }
            lines
        }
        _ => preview_lines(result, width, Tone::Muted, styled),
    }
}

fn edit_body(arguments: &Value, width: usize, styled: bool) -> Vec<String> {
    let edits = arguments
        .get("edits")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut lines = vec![format!(
        "Applied {} {}",
        paint(&edits.len().to_string(), Tone::Bold, styled),
        plural(edits.len(), "edit")
    )];
    for (index, edit) in edits.iter().take(MAX_EDITS_SHOWN).enumerate() {
        if index > 0 {
            lines.push(paint("⋮", Tone::Muted, styled));
        }
        let text = |key: &str| edit.get(key).and_then(Value::as_str).unwrap_or("");
        let (removed, added) = changed_lines(text("oldText"), text("newText"));
        for (diff_lines, sign, tone) in [(removed, '-', Tone::Removed), (added, '+', Tone::Added)] {
            for line in diff_lines.iter().take(MAX_DIFF_LINES) {
                let line = ui::truncate(&sanitize(line), width.saturating_sub(2));
                lines.push(paint(&format!("{sign} {line}"), tone, styled));
            }
            if diff_lines.len() > MAX_DIFF_LINES {
                lines.push(more_lines(diff_lines.len() - MAX_DIFF_LINES, styled));
            }
        }
    }
    if edits.len() > MAX_EDITS_SHOWN {
        let hidden = edits.len() - MAX_EDITS_SHOWN;
        lines.push(paint(
            &format!("… +{hidden} more {}", plural(hidden, "edit")),
            Tone::Muted,
            styled,
        ));
    }
    lines
}

/// Lines that differ between `old` and `new`, ignoring their shared leading and trailing lines.
fn changed_lines<'a>(old: &'a str, new: &'a str) -> (Vec<&'a str>, Vec<&'a str>) {
    let old: Vec<&str> = old.lines().collect();
    let new: Vec<&str> = new.lines().collect();
    let prefix = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
    let max_suffix = old.len().min(new.len()) - prefix;
    let suffix = old
        .iter()
        .rev()
        .zip(new.iter().rev())
        .take(max_suffix)
        .take_while(|(a, b)| a == b)
        .count();
    let removed = old[prefix..old.len() - suffix].to_vec();
    let added = new[prefix..new.len() - suffix].to_vec();
    if removed.is_empty() && added.is_empty() {
        (old, new)
    } else {
        (removed, added)
    }
}

/// Split a trailing `[Exit code: N]` marker from bash output.
fn split_exit_code(result: &str) -> (&str, Option<i64>) {
    let trimmed = result.trim_end();
    if let Some(start) = trimmed.rfind("[Exit code: ")
        && let Some(code) = trimmed[start + 12..]
            .strip_suffix(']')
            .and_then(|code| code.trim().parse().ok())
    {
        return (&trimmed[..start], Some(code));
    }
    (result, None)
}

fn preview_lines(text: &str, width: usize, tone: Tone, styled: bool) -> Vec<String> {
    let lines: Vec<String> = text
        .lines()
        .map(|line| ui::truncate(&sanitize(line), width))
        .collect();
    capped(lines, tone, styled)
}

fn capped(lines: Vec<String>, tone: Tone, styled: bool) -> Vec<String> {
    let total = lines.len();
    let mut output: Vec<String> = lines
        .into_iter()
        .take(DISPLAY_TRUNCATION_LIMIT)
        .map(|line| paint(&line, tone, styled))
        .collect();
    if total > DISPLAY_TRUNCATION_LIMIT {
        output.push(more_lines(total - DISPLAY_TRUNCATION_LIMIT, styled));
    }
    output
}

fn more_lines(count: usize, styled: bool) -> String {
    paint(
        &format!("… +{count} {}", plural(count, "line")),
        Tone::Muted,
        styled,
    )
}

/// Make tool output safe to measure and print: expand tabs, drop ANSI escapes and
/// other control characters.
fn sanitize(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '\t' => output.push_str("    "),
            '\n' => output.push('\n'),
            '\x1b' => {
                if chars.peek() == Some(&'[') {
                    chars.next();
                    for next in chars.by_ref() {
                        if ('@'..='~').contains(&next) {
                            break;
                        }
                    }
                }
            }
            character if character.is_control() => {}
            character => output.push(character),
        }
    }
    output
}

fn terminal_dimensions() -> (usize, usize) {
    terminal::size()
        .map(|(width, height)| (usize::from(width).max(1), usize::from(height).max(2)))
        .unwrap_or((80, 24))
}

fn terminal_width() -> usize {
    terminal_dimensions().0
}

fn format_arguments(arguments: &Value) -> String {
    let Some(arguments) = arguments.as_object() else {
        return arguments.to_string();
    };
    arguments
        .iter()
        .map(|(key, value)| {
            let value = value
                .as_str()
                .map(|value| format!("{value:?}"))
                .unwrap_or_else(|| value.to_string());
            format!("{key}={value}")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn wrap_line(line: &str, width: usize) -> Vec<String> {
    if line.is_empty() || UnicodeWidthStr::width(line) <= width {
        return vec![line.to_owned()];
    }
    let mut remaining = line;
    let mut chunks = Vec::new();
    while UnicodeWidthStr::width(remaining) > width {
        let mut used = 0;
        let mut hard_end = 0;
        let mut soft_end = None;
        for (index, character) in remaining.char_indices() {
            let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
            if used + character_width > width {
                break;
            }
            used += character_width;
            hard_end = index + character.len_utf8();
            if character.is_whitespace() {
                soft_end = Some(hard_end);
            }
        }
        if hard_end == 0 {
            hard_end = remaining
                .char_indices()
                .next()
                .map_or(remaining.len(), |(_, character)| character.len_utf8());
        }
        let cut = soft_end.filter(|cut| *cut > 0).unwrap_or(hard_end);
        let chunk = remaining[..cut].trim_end();
        if !chunk.is_empty() {
            chunks.push(chunk.to_owned());
        }
        remaining = remaining[cut..].trim_start();
    }
    chunks.push(remaining.to_owned());
    chunks
}

fn prefixed_lines(text: &str, first_prefix: &str, rest_prefix: &str, width: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut output = Vec::new();
    let mut first = true;
    for line in text.split('\n') {
        let requested_prefix = if first { first_prefix } else { rest_prefix };
        let prefix = if UnicodeWidthStr::width(requested_prefix) < width {
            requested_prefix
        } else {
            ""
        };
        let available = width.saturating_sub(UnicodeWidthStr::width(prefix)).max(1);
        for (index, chunk) in wrap_line(line, available).into_iter().enumerate() {
            let prefix = if index == 0 {
                prefix
            } else if UnicodeWidthStr::width(rest_prefix) < width {
                rest_prefix
            } else {
                ""
            };
            output.push(format!("{prefix}{chunk}"));
            first = false;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    fn sample_token_status() -> TokenStatus {
        TokenStatus {
            input_tokens: 5_000,
            output_tokens: 200,
            cached_tokens: 1_000,
            turn_input: 5_000,
            context_window: 1_000_000,
            model: "a-very-long-model-name".to_owned(),
            reasoning_effort: Some("high".to_owned()),
        }
    }

    fn sample_token_event() -> Event {
        Event::Tokens {
            input_tokens: 5_000,
            output_tokens: 200,
            total_tokens: 5_200,
            cached_tokens: 1_000,
            turn_input: 5_000,
            turn_output: 200,
            turn_cached: 1_000,
            context_window: 1_000_000,
            model: "model".to_owned(),
            reasoning_effort: Some("high".to_owned()),
        }
    }

    fn quiet_display() -> CliDisplay {
        let mut display = CliDisplay::new(false);
        display.interactive = false;
        display.styled = false;
        display
    }

    #[test]
    fn cli_defaults_and_stream_flags_match_python() {
        let args = Cli::try_parse_from(["harness"]).unwrap();
        assert!(args.model.is_none());
        assert!(args.context_window.is_none());
        assert!(!args.no_stream);

        let args = Cli::try_parse_from(["harness", "--no-stream"]).unwrap();
        assert!(args.no_stream);
        let args = Cli::try_parse_from(["harness", "--stream"]).unwrap();
        assert!(args.stream);
        assert!(!args.no_stream);
    }

    #[test]
    fn model_metadata_rejects_invalid_windows_and_accepts_provider_formats() {
        for value in [
            json!(null),
            json!(0),
            json!(-1),
            json!("unknown"),
            json!(1.5),
        ] {
            assert_eq!(
                model_context_window(&json!({"context_window": value})),
                None
            );
        }
        assert_eq!(
            model_context_window(&json!({"context_length": "32768"})),
            Some(32768)
        );
        assert_eq!(
            model_context_window(&json!({"architecture": {"context_length": 65536}})),
            Some(65536)
        );
        assert_eq!(
            model_context_window(
                &json!({"context_window": 0, "top_provider": {"context_length": 128000}})
            ),
            Some(128000)
        );
    }

    #[test]
    fn switching_models_updates_context_and_preserves_explicit_overrides() {
        let fetcher = ModelFetcher {
            base_url: String::new(),
            api_key: None,
            requires_routing: false,
            state: Arc::new((
                Mutex::new(Some(ModelCatalog {
                    context_windows: BTreeMap::from([("small".to_owned(), 32768)]),
                    sources: BTreeMap::from([("small".to_owned(), vec![0])]),
                    ..ModelCatalog::default()
                })),
                Condvar::new(),
            )),
        };
        let mut agent = AgentHarness::new(AgentConfig::new("small")).unwrap();
        apply_model_metadata(&mut agent, &fetcher, None);
        assert_eq!(agent.context_window, 32768);
        apply_model_metadata(&mut agent, &fetcher, Some(8192));
        assert_eq!(agent.context_window, 8192);
        agent.select_model("missing", None);
        apply_model_metadata(&mut agent, &fetcher, None);
        assert_eq!(agent.context_window, DEFAULT_CONTEXT_WINDOW);
    }

    #[test]
    fn model_catalog_fetches_metadata_and_uses_the_smallest_provider_window() {
        use std::io::Read;
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            for window in [128000, 64000] {
                let (mut stream, _) = listener.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut request = Vec::new();
                let mut byte = [0];
                while !request.ends_with(b"\r\n\r\n") {
                    stream.read_exact(&mut byte).unwrap();
                    request.push(byte[0]);
                }
                assert!(
                    String::from_utf8(request)
                        .unwrap()
                        .starts_with("GET /models ")
                );
                let body = json!({"data": [{"id": "shared", "context_length": window}, {"id": "unknown"}]}).to_string();
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        let catalog = fetch_models(&format!("{base_url};;{base_url}"), Some("key")).unwrap();
        server.join().unwrap();
        assert_eq!(catalog.models, ["shared", "unknown"]);
        assert_eq!(catalog.sources["shared"], [0, 1]);
        assert_eq!(catalog.context_windows["shared"], 64000);
        assert!(!catalog.context_windows.contains_key("unknown"));
        assert!(catalog.failures.is_empty());
    }

    #[test]
    fn positive_values_are_validated() {
        assert!(Cli::try_parse_from(["harness", "--max-turns", "0"]).is_err());
        assert!(Cli::try_parse_from(["harness", "--context-window", "-1"]).is_err());
    }

    fn status(
        outcome: Outcome,
        label: &str,
        interruptible: bool,
        tokens: Option<&TokenStatus>,
        width: usize,
    ) -> String {
        let info = StatusInfo {
            outcome,
            frame: 2,
            label,
            elapsed_secs: 12.34,
            interruptible,
            tokens,
        };
        render_status(&info, width, false)
    }

    #[test]
    fn status_line_shows_all_details_when_space_allows() {
        let tokens = sample_token_status();
        let full = status(Outcome::Done, "Working", true, Some(&tokens), 200);
        assert!(full.starts_with("✓ Done in 12.3s"));
        assert!(full.contains("↑ 5k ↓ 200"));
        assert!(full.contains("ctx 0.5%"));
        assert!(full.contains("cache 20%"));
        assert!(full.contains("a-very-long-model-name"));
        assert!(full.contains("high"));
        assert!(
            !full.contains("esc"),
            "finished turns cannot be interrupted"
        );

        assert_eq!(
            status(Outcome::Active, "Thinking", false, None, 200),
            "⠹ Thinking… 12s"
        );
        assert_eq!(
            status(Outcome::Active, "Thinking", true, None, 200),
            "⠹ Thinking… 12s · esc to interrupt"
        );
        assert!(
            status(Outcome::Interrupted, "", false, None, 200).starts_with("■ Interrupted after")
        );
    }

    #[test]
    fn status_line_drops_low_priority_details_first() {
        let tokens = sample_token_status();
        let compact = status(Outcome::Done, "", false, Some(&tokens), 40);
        assert!(UnicodeWidthStr::width(compact.as_str()) <= 40);
        assert!(compact.contains("↑ 5k"));
        assert!(!compact.contains("cache"));
        assert!(!compact.contains("a-very-long-model-name"));

        let active = status(Outcome::Active, "Working", true, Some(&tokens), 60);
        assert!(active.contains("esc to interrupt"));
        assert!(!active.contains("a-very-long-model-name"));
    }

    #[test]
    fn status_line_never_wraps_on_narrow_terminals() {
        let tokens = sample_token_status();
        let outcomes = [
            Outcome::Active,
            Outcome::Done,
            Outcome::Failed,
            Outcome::Interrupted,
        ];
        for outcome in outcomes {
            for width in 1..=100 {
                let rendered = status(outcome, "Running command", true, Some(&tokens), width);
                assert!(
                    UnicodeWidthStr::width(rendered.as_str()) <= width,
                    "status exceeded terminal width {width}: {rendered:?}"
                );
            }
        }
    }

    #[test]
    fn live_frame_keeps_response_and_status_together() {
        let mut display = quiet_display();
        display.on_event(&Event::TextDelta {
            content: "## Summary\n\n**formatted** response".to_owned(),
        });
        display.on_event(&sample_token_event());

        let lines = display.live_lines(80, 24);
        assert!(lines.iter().any(|line| line.contains("● ## Summary")));
        let status = lines.last().unwrap();
        assert!(status.contains("Writing…"));
        assert!(status.contains("↑ 5k ↓ 200"));
    }

    #[test]
    fn live_frame_shows_pending_tool_call() {
        let mut display = quiet_display();
        display.on_event(&Event::ToolCall {
            name: "bash".to_owned(),
            arguments: json!({"command": "cargo test"}),
        });
        let lines = display.live_lines(80, 24);
        assert!(lines.iter().any(|line| line == "● Bash(cargo test)"));
        assert!(lines.last().unwrap().contains("Running command…"));
    }

    #[test]
    fn turn_start_preserves_previous_token_status() {
        let mut display = quiet_display();
        display.on_event(&sample_token_event());
        display.on_event(&Event::TurnStart);

        assert_eq!(
            display
                .last_tokens
                .as_ref()
                .map(|status| status.input_tokens),
            Some(5_000)
        );
    }

    #[test]
    fn tool_headers_summarize_arguments() {
        assert_eq!(
            tool_header("bash", &json!({"command": "ls\npwd"})),
            ("Bash".to_owned(), "ls …".to_owned())
        );
        assert_eq!(
            tool_header("read", &json!({"path": "a.rs", "offset": 10, "limit": 5})),
            ("Read".to_owned(), "a.rs:10-14".to_owned())
        );
        let line = tool_header_line(
            "bash",
            &json!({"command": "x".repeat(200)}),
            Tone::Success,
            40,
            false,
        );
        assert!(UnicodeWidthStr::width(line.as_str()) <= 40);
        assert!(line.ends_with("…)"));
    }

    #[test]
    fn bash_results_show_output_preview_and_exit_code() {
        let output = (1..=8)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let result = format!("{output}\n[Exit code: 2]");
        assert!(tool_failed("bash", &result));
        let body = tool_body("bash", &Value::Null, &result, 40, false);
        assert_eq!(body.first().map(String::as_str), Some("line 1"));
        assert!(body.contains(&"… +3 lines".to_owned()));
        assert_eq!(body.last().map(String::as_str), Some("exit code 2"));

        assert!(!tool_failed("bash", "ok"));
        assert_eq!(
            tool_body("bash", &Value::Null, "(no output)", 40, false),
            ["(no output)"]
        );
    }

    #[test]
    fn edit_results_show_a_compact_diff() {
        let arguments = json!({"path": "a.rs", "edits": [{
            "oldText": "fn a() {\n    old();\n}",
            "newText": "fn a() {\n    new();\n    more();\n}"
        }]});
        let body = tool_body(
            "edit",
            &arguments,
            "Successfully applied 1 edit(s) to a.rs.",
            60,
            false,
        );
        assert_eq!(
            body,
            [
                "Applied 1 edit",
                "-     old();",
                "+     new();",
                "+     more();"
            ]
        );
    }

    #[test]
    fn read_and_error_results_are_summarized() {
        assert_eq!(
            tool_body("read", &Value::Null, "a\nb\nc", 40, false),
            ["Read 3 lines"]
        );
        let body = tool_body("read", &Value::Null, "Error: File not found: x", 40, false);
        assert!(tool_failed("read", "Error: File not found: x"));
        assert_eq!(body, ["Error: File not found: x"]);
    }

    #[test]
    fn sanitize_strips_escapes_and_expands_tabs() {
        assert_eq!(sanitize("\x1b[31mred\x1b[0m\tok\r"), "red    ok");
    }

    #[test]
    fn unknown_commands_are_detected_without_catching_paths() {
        assert!(is_unknown_command("/foo"));
        assert!(!is_unknown_command("/help"));
        assert!(!is_unknown_command("/usr/bin is broken"));
        assert!(!is_unknown_command("/usr/bin"));
        assert!(!is_unknown_command("hello"));
    }

    #[test]
    fn prefixed_live_lines_are_bounded_and_preserve_trailing_line() {
        let lines = prefixed_lines("alpha beta gamma\n", "🤖 ", "", 10);

        assert!(lines.len() >= 3);
        assert!(
            lines
                .iter()
                .all(|line| UnicodeWidthStr::width(line.as_str()) <= 10)
        );
        assert_eq!(lines.last().map(String::as_str), Some(""));
    }

    #[test]
    fn wrapping_accounts_for_wide_terminal_characters() {
        let chunks = wrap_line("alpha 🤖 beta gamma delta", 10);
        assert!(chunks.len() > 1);
        assert!(
            chunks
                .iter()
                .all(|chunk| UnicodeWidthStr::width(chunk.as_str()) <= 10)
        );
    }

    #[test]
    fn provider_warnings_are_targeted() {
        assert!(check_reasoning_compatibility("https://api.openai.com/v1", Some("max")).is_some());
        assert!(check_reasoning_compatibility("https://api.openai.com/v1", Some("high")).is_none());
        assert!(check_model_available("missing", &["present".to_owned()]).is_some());
        assert!(check_model_available("missing", &[]).is_none());
        assert!(
            check_reasoning_compatibility(
                "https://other.example/v1;;https://api.openai.com/v1",
                Some("max")
            )
            .is_none()
        );
    }

    #[test]
    fn executable_env_is_loaded_from_executable_directory() {
        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("harness.exe");
        std::fs::write(
            directory.path().join(".env"),
            "HARNESS_RUST_DOTENV_TEST=from-sidecar\n",
        )
        .unwrap();

        load_executable_env_from(&executable).unwrap();

        assert_eq!(
            std::env::var("HARNESS_RUST_DOTENV_TEST").as_deref(),
            Ok("from-sidecar")
        );
    }

    #[test]
    fn missing_executable_env_is_optional() {
        let directory = tempfile::tempdir().unwrap();
        assert!(load_executable_env_from(&directory.path().join("harness.exe")).is_ok());
    }
}
