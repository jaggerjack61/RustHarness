use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::{ArgAction, Parser, ValueEnum};
use dialoguer::Password;
use reqwest::blocking::Client;
use serde_json::Value;
use termimad::crossterm::terminal;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::agent::{AgentConfig, AgentHarness, split_multi, tls_insecure_enabled};
use crate::constants::{
    DEFAULT_BASE_URL, DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TURNS, DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT, DISPLAY_TRUNCATION_LIMIT, FETCH_TIMEOUT_SECS,
    NONSTANDARD_REASONING_EFFORTS, REASONING_OPTIONS,
};
use crate::display::ResponseBuffer;
use crate::events::Event;
use crate::markdown::render_markdown;
use crate::prompts::prompt_selection;
use crate::providers::{self, SavedProvider};

const LIVE_REFRESH_INTERVAL: Duration = Duration::from_millis(50);

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
    #[arg(short = 'm', long, env = "HARNESS_MODEL", default_value = DEFAULT_MODEL)]
    model: String,

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
        default_value_t = DEFAULT_CONTEXT_WINDOW,
        value_parser = parse_positive_i64
    )]
    context_window: i64,

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
    let mut config = AgentConfig::new(&args.model);
    config.api_key = args.api_key.clone();
    config.base_url = args.base_url.clone();
    config.working_dir = Some(working_dir.clone());
    config.max_turns = args.max_turns;
    config.reasoning_effort = Some(args.reasoning_effort.as_str().to_owned());
    config.context_window = args.context_window;
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
    let mut agent = AgentHarness::new(config)?;
    let mut use_stream = args.stream || !args.no_stream;

    println!(
        "{}",
        build_welcome_box(
            &agent.model,
            agent.reasoning_effort.as_deref().unwrap_or("high"),
            agent.context_window,
            &working_dir,
            use_stream,
        )
    );
    println!();

    let (base_url, api_key) = agent.provider_config();
    let mut fetcher = ModelFetcher::new(base_url.clone(), api_key);
    println!("📦 Loading models in background… Use /models to browse.\n");
    if let Some(warning) =
        check_reasoning_compatibility(&base_url, agent.reasoning_effort.as_deref())
    {
        println!("{warning}\n");
    }

    let mut model_checked = false;
    loop {
        print!("▸ ");
        io::stdout().flush()?;
        let Some(input) = read_line()? else {
            println!("\nGoodbye!");
            break;
        };
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input.to_lowercase().as_str() {
            "/exit" => {
                println!("Goodbye!");
                break;
            }
            "/clear" => {
                agent.clear_history();
                clear_terminal();
                println!(
                    "{}\n",
                    build_welcome_box(
                        &agent.model,
                        agent.reasoning_effort.as_deref().unwrap_or("high"),
                        agent.context_window,
                        &working_dir,
                        use_stream,
                    )
                );
                if fetcher.ready() {
                    let models = fetcher.get();
                    if models.is_empty() {
                        println!("⚠️ Could not pre-fetch models — /models will retry on demand.\n");
                    } else {
                        println!(
                            "📦 {} models loaded. Use /models to switch.\n",
                            models.len()
                        );
                    }
                } else {
                    println!("📦 Loading models in background… Use /models to browse.\n");
                }
                println!("🔄 History cleared.\n");
                continue;
            }
            "/stream" => {
                use_stream = !use_stream;
                println!(
                    "✅ Streaming {}.",
                    if use_stream { "enabled" } else { "disabled" }
                );
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
                        println!("❌ Could not save provider: {error}");
                        continue;
                    }
                    saved_providers = updated_providers;
                    agent.add_provider(&base_url, &api_key);
                    let (urls, keys) = agent.provider_config();
                    fetcher = ModelFetcher::new(urls, keys);
                    model_checked = false;
                    println!("✅ Provider saved. Loading models… Use /models to select a model.");
                    if let Some(warning) =
                        check_reasoning_compatibility(&base_url, agent.reasoning_effort.as_deref())
                    {
                        println!("{warning}");
                    }
                }
                continue;
            }
            "/models" => {
                let mut models = fetcher.get();
                if models.is_empty() {
                    println!("\nFetching models…");
                    fetcher.refresh();
                    models = fetcher.get();
                }
                fetcher.print_failures();
                if let Some(selected) = prompt_selection(
                    &models,
                    &agent.model,
                    "📋 Available models",
                    "Current model",
                ) && selected != agent.model
                {
                    let endpoint_indices = fetcher.endpoint_indices_for(&selected);
                    agent.select_model(selected, endpoint_indices);
                    println!("✅ Model changed to: {}", agent.model);
                }
                continue;
            }
            "/reasoning" => {
                let current = agent.reasoning_effort.as_deref().unwrap_or("high");
                if let Some(selected) = prompt_selection(
                    REASONING_OPTIONS,
                    current,
                    "🧠 Reasoning effort",
                    "Current effort",
                ) && Some(selected.as_str()) != agent.reasoning_effort.as_deref()
                {
                    agent.reasoning_effort = Some(selected.clone());
                    println!("✅ Reasoning effort set to: {selected}");
                }
                continue;
            }
            "/context" => {
                if let Some(context) = read_multiline_context()? {
                    let line_count = context.lines().count();
                    agent.set_custom_context(Some(context));
                    println!("✅ Custom context set ({line_count} lines).");
                }
                continue;
            }
            "/context clear" => {
                agent.clear_custom_context();
                println!("✅ Custom context cleared.");
                continue;
            }
            "/context show" => {
                if let Some(context) = agent.get_custom_context() {
                    println!(
                        "\n📋 Current custom context ({} lines):\n{}\n{}\n{}",
                        context.lines().count(),
                        "─".repeat(40),
                        context,
                        "─".repeat(40)
                    );
                } else {
                    println!("No custom context set. Use /context to add one.");
                }
                continue;
            }
            _ => {}
        }

        if !model_checked && (fetcher.requires_routing() || fetcher.ready()) {
            let models = fetcher.get();
            let endpoint_indices = fetcher.endpoint_indices_for(&agent.model);
            agent.select_model(agent.model.clone(), endpoint_indices);
            if let Some(warning) = check_model_available(&agent.model, &models) {
                println!("{warning}\n");
            }
            model_checked = true;
        }

        println!();
        let mut display = CliDisplay::new(args.no_markdown);
        let response =
            agent.run_with_callback(input, use_stream, &mut |event| display.on_event(event));
        match response {
            Ok(response) if use_stream => {
                display.finish();
                if response.is_empty() && !display.streamed_any {
                    println!();
                }
                println!();
            }
            Ok(_) => {
                display.finish();
                println!();
            }
            Err(error) => {
                display.discard_preview();
                println!("\n❌ Error: {error}\n");
                display.finish();
            }
        }
    }
    Ok(())
}

fn read_line() -> io::Result<Option<String>> {
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(line)),
        Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(None),
        Err(error) => Err(error),
    }
}

fn prompt_login() -> Result<Option<(String, String)>> {
    println!("\nProvider login (press Enter at either prompt to cancel):");
    print!("Provider base URL: ");
    io::stdout().flush()?;
    let Some(url) = read_line()? else {
        println!("Cancelled.");
        return Ok(None);
    };
    if url.trim().is_empty() {
        println!("Cancelled.");
        return Ok(None);
    }
    let url = match providers::validate_url(&url) {
        Ok(url) => url,
        Err(error) => {
            println!("❌ {error}");
            return Ok(None);
        }
    };
    let key = if io::stdin().is_terminal() {
        Password::new()
            .with_prompt("API key")
            .allow_empty_password(true)
            .report(false)
            .interact()?
    } else {
        print!("API key: ");
        io::stdout().flush()?;
        let Some(key) = read_line()? else {
            println!("Cancelled.");
            return Ok(None);
        };
        key
    };
    let key = key.trim();
    if key.is_empty() {
        println!("Cancelled.");
        return Ok(None);
    }
    if key.contains(";;") {
        println!("❌ Enter one API key per /login.");
        return Ok(None);
    }
    Ok(Some((url, key.to_owned())))
}

fn read_multiline_context() -> io::Result<Option<String>> {
    println!("\n📝 Enter custom context (end with '.' on a line by itself):");
    let mut lines = Vec::new();
    loop {
        let Some(line) = read_line()? else {
            println!("\nCancelled.");
            return Ok(None);
        };
        let line = line.trim_end_matches(['\r', '\n']);
        if line.trim() == "." {
            break;
        }
        lines.push(line.to_owned());
    }
    if lines.is_empty() {
        println!("No text entered — context unchanged.");
        Ok(None)
    } else {
        Ok(Some(lines.join("\n")))
    }
}

fn clear_terminal() {
    if io::stdout().is_terminal() {
        print!("\x1b[2J\x1b[H");
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
                "⚠️  Reasoning effort '{effort}' is not supported by OpenAI and may cause errors. Use low/medium/high, or switch --base-url to a compatible provider."
            )
        })
}

fn check_model_available(model: &str, models: &[String]) -> Option<String> {
    (!models.is_empty() && !models.iter().any(|available| available == model)).then(|| {
        format!(
            "⚠️  Model '{model}' was not found in the available models list ({} models fetched). Use /models to select a valid model.",
            models.len()
        )
    })
}

fn build_welcome_box(
    model: &str,
    reasoning_effort: &str,
    context_window: i64,
    working_dir: &std::path::Path,
    streaming: bool,
) -> String {
    let lines = vec![
        "🤖 Nasa Level Genius Agent".to_owned(),
        format!("Model:           {model}"),
        format!("Reasoning:       {reasoning_effort}"),
        format!(
            "Context window:  {} tokens",
            format_number(context_window as u64)
        ),
        format!("Streaming:       {}", if streaming { "on" } else { "off" }),
        format!("CWD:             {}", working_dir.display()),
        "─".repeat(40),
        "Commands:  /exit  /clear  /login  /models  /reasoning".to_owned(),
        "           /stream  /context  /context show  /context clear".to_owned(),
    ];
    let inner_width = lines
        .iter()
        .map(|line| UnicodeWidthStr::width(line.as_str()))
        .max()
        .unwrap_or(0)
        + 4;
    let mut output = vec![format!("╔{}╗", "═".repeat(inner_width))];
    for line in lines {
        let padding = inner_width - 2 - UnicodeWidthStr::width(line.as_str());
        output.push(format!("║  {line}{}║", " ".repeat(padding)));
    }
    output.push(format!("╚{}╝", "═".repeat(inner_width)));
    output.join("\n")
}

fn format_number(value: u64) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

#[derive(Clone, Debug, Default)]
struct ModelCatalog {
    models: Vec<String>,
    sources: BTreeMap<String, Vec<usize>>,
    failures: Vec<String>,
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
                    .flatten()
                    .filter_map(|model| model.get("id").and_then(Value::as_str));
                for model in endpoint_models {
                    models.insert(model.to_owned());
                    sources.entry(model.to_owned()).or_default().push(index);
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

    fn print_failures(&self) {
        let state = self.state.0.lock().expect("model state poisoned");
        if let Some(catalog) = state.as_ref() {
            for failure in &catalog.failures {
                println!("❌ {failure}");
            }
        }
    }

    fn refresh(&self) {
        match fetch_models(&self.base_url, self.api_key.as_deref()) {
            Ok(catalog) => self.store(catalog),
            Err(error) => {
                println!("❌ Failed to fetch models: {error}");
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

#[derive(Clone, Copy)]
enum Tone {
    Plain,
    Blue,
    Green,
    Cyan,
    Yellow,
    Red,
    MagentaBold,
    Bold,
    Dim,
}

fn paint(text: &str, tone: Tone, enabled: bool) -> String {
    if !enabled || matches!(tone, Tone::Plain) {
        return text.to_owned();
    }
    let code = match tone {
        Tone::Plain => "",
        Tone::Blue => "34",
        Tone::Green => "32",
        Tone::Cyan => "1;36",
        Tone::Yellow => "33",
        Tone::Red => "1;31",
        Tone::MagentaBold => "1;35",
        Tone::Bold => "1",
        Tone::Dim => "2",
    };
    format!("\x1b[{code}m{text}\x1b[0m")
}

#[derive(Clone)]
struct StatusSegment {
    text: String,
    tone: Tone,
}

#[derive(Clone)]
struct TokenStatus {
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    cached_tokens: u64,
    turn_input: u64,
    turn_output: u64,
    context_window: i64,
    model: String,
    reasoning_effort: Option<String>,
}

impl TokenStatus {
    fn segments(
        &self,
        active: bool,
        include_model: bool,
        include_context: bool,
        include_cache: bool,
        cache_rate: bool,
        include_reasoning: bool,
    ) -> Vec<StatusSegment> {
        let mut parts = vec![StatusSegment {
            text: if active { "⠿" } else { "📊" }.to_owned(),
            tone: if active { Tone::Cyan } else { Tone::Bold },
        }];
        if include_model {
            parts.push(StatusSegment {
                text: self.model.clone(),
                tone: Tone::MagentaBold,
            });
        }
        parts.extend([
            StatusSegment {
                text: format!("In:{}", format_number(self.input_tokens)),
                tone: Tone::Cyan,
            },
            StatusSegment {
                text: format!("Out:{}", format_number(self.output_tokens)),
                tone: Tone::Green,
            },
            StatusSegment {
                text: format!("Tot:{}", format_number(self.total_tokens)),
                tone: Tone::Bold,
            },
        ]);
        if include_context && self.context_window > 0 {
            let percentage = self.turn_input as f64 / self.context_window as f64 * 100.0;
            parts.push(StatusSegment {
                text: format!("Ctx:{percentage:.1}%"),
                tone: if percentage < 50.0 {
                    Tone::Green
                } else if percentage < 80.0 {
                    Tone::Yellow
                } else {
                    Tone::Red
                },
            });
        }
        if include_cache && self.cached_tokens > 0 {
            let mut text = format!("Cache:{}", format_number(self.cached_tokens));
            if cache_rate {
                let rate = if self.input_tokens > 0 {
                    self.cached_tokens as f64 / self.input_tokens as f64 * 100.0
                } else {
                    0.0
                };
                text.push_str(&format!(" ({rate:.1}%)"));
            }
            parts.push(StatusSegment {
                text,
                tone: Tone::Yellow,
            });
        }
        if include_reasoning && let Some(effort) = self.reasoning_effort.as_deref() {
            parts.push(StatusSegment {
                text: format!("🧠 {effort}"),
                tone: Tone::Dim,
            });
        }
        parts.push(StatusSegment {
            text: format!(
                "[+{}/{}]",
                format_number(self.turn_input),
                format_number(self.turn_output)
            ),
            tone: Tone::Dim,
        });
        parts
    }

    fn render(&self, max_width: usize, styled: bool, active: bool) -> String {
        let layouts = [
            (true, true, true, true, true, "  "),
            (true, true, true, true, true, " "),
            (true, true, true, false, true, " "),
            (true, true, true, false, false, " "),
            (true, false, true, false, false, " "),
            (true, false, false, false, false, " "),
            (false, false, false, false, false, " "),
        ];
        let mut chosen = (Vec::new(), " ");
        for (
            include_model,
            include_context,
            include_cache,
            cache_rate,
            include_reasoning,
            spacing,
        ) in layouts
        {
            let parts = self.segments(
                active,
                include_model,
                include_context,
                include_cache,
                cache_rate,
                include_reasoning,
            );
            let width = parts
                .iter()
                .map(|part| UnicodeWidthStr::width(part.text.as_str()))
                .sum::<usize>()
                + spacing.len() * parts.len().saturating_sub(1);
            chosen = (parts, spacing);
            if width <= max_width {
                break;
            }
        }
        let chosen_width = chosen
            .0
            .iter()
            .map(|part| UnicodeWidthStr::width(part.text.as_str()))
            .sum::<usize>()
            + chosen.1.len() * chosen.0.len().saturating_sub(1);
        if chosen_width > max_width {
            let icon = StatusSegment {
                text: if active { "⠿" } else { "📊" }.to_owned(),
                tone: if active { Tone::Cyan } else { Tone::Bold },
            };
            let total = StatusSegment {
                text: format!("Tot:{}", format_number(self.total_tokens)),
                tone: Tone::Bold,
            };
            let compact_layouts = [vec![icon.clone(), total.clone()], vec![total], vec![icon]];
            chosen = compact_layouts
                .into_iter()
                .find(|parts| {
                    parts
                        .iter()
                        .map(|part| UnicodeWidthStr::width(part.text.as_str()))
                        .sum::<usize>()
                        + parts.len().saturating_sub(1)
                        <= max_width
                })
                .map_or_else(|| (Vec::new(), " "), |parts| (parts, " "));
        }
        chosen
            .0
            .iter()
            .map(|part| paint(&part.text, part.tone, styled))
            .collect::<Vec<_>>()
            .join(chosen.1)
    }
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

    fn commit(&mut self) {
        if self.line_count == 0 {
            return;
        }
        print!("\r\n");
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
    streamed_any: bool,
    interactive: bool,
    styled: bool,
    response_buffer: ResponseBuffer,
    thinking_line_buf: String,
    thinking_first_line: bool,
    live: LiveRegion,
    last_tokens: Option<TokenStatus>,
    last_live_refresh: Option<Instant>,
    finished: bool,
}

impl CliDisplay {
    fn new(no_markdown: bool) -> Self {
        let interactive = io::stdout().is_terminal();
        Self {
            no_markdown,
            streamed_any: false,
            interactive,
            styled: interactive && std::env::var_os("NO_COLOR").is_none(),
            response_buffer: ResponseBuffer::new(),
            thinking_line_buf: String::new(),
            thinking_first_line: true,
            live: LiveRegion::default(),
            last_tokens: None,
            last_live_refresh: None,
            finished: false,
        }
    }

    fn live_lines(&self, active: bool, width: usize, height: usize) -> Vec<String> {
        let width = width.saturating_sub(1).max(1);
        let mut lines = Vec::new();

        if !self.thinking_line_buf.is_empty() {
            let prefix = if self.thinking_first_line {
                "  🧠 "
            } else {
                "     "
            };
            let thinking_lines = prefixed_lines(&self.thinking_line_buf, prefix, "     ", width)
                .into_iter()
                .map(|line| paint(&line, Tone::Dim, self.styled));
            lines.extend(thinking_lines);
        }

        let response = self.response_buffer.text();
        if !response.is_empty() {
            lines.extend(prefixed_lines(&response, "🤖 ", "", width));
        }

        if let Some(status) = self.last_tokens.as_ref() {
            lines.push(status.render(width, self.styled, active));
        }

        let max_lines = height.saturating_sub(1).max(1);
        if lines.len() > max_lines {
            lines.drain(..lines.len() - max_lines);
        }
        lines
    }

    fn refresh_live(&mut self, active: bool, force: bool) {
        if !self.interactive {
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
        let lines = self.live_lines(active, width, height);
        self.live.replace(&lines);
        self.last_live_refresh = Some(Instant::now());
    }

    fn prepare_output(&mut self) {
        if self.interactive {
            self.live.clear();
            self.last_live_refresh = None;
        }
    }

    fn discard_preview(&mut self) {
        self.prepare_output();
        self.response_buffer.reset();
        self.thinking_line_buf.clear();
        self.thinking_first_line = true;
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if self.interactive {
            self.refresh_live(false, true);
            self.live.commit();
        } else if let Some(status) = self.last_tokens.as_ref() {
            println!("{}", status.render(usize::MAX, false, false));
        }
    }

    fn print_final_response(&mut self, content: &str, active: bool) {
        self.prepare_output();
        self.response_buffer.reset();
        if !content.is_empty() {
            if self.no_markdown {
                println!("🤖 {content}");
            } else {
                print!("🤖 ");
                let _ = io::stdout().flush();
                render_markdown(content);
                println!();
            }
        }
        self.refresh_live(active, true);
    }

    fn flush_thinking_lines(&mut self) {
        let Some(newline) = self.thinking_line_buf.rfind('\n') else {
            return;
        };
        let remainder = self.thinking_line_buf[newline + 1..].to_owned();
        let completed = self.thinking_line_buf[..=newline].to_owned();
        self.prepare_output();
        for line in completed.split_terminator('\n') {
            let prefix = if self.thinking_first_line {
                "  🧠 "
            } else {
                "     "
            };
            print_wrapped(
                line,
                prefix,
                "     ",
                Tone::Dim,
                self.styled,
                terminal_width(),
            );
            self.thinking_first_line = false;
        }
        self.thinking_line_buf = remainder;
    }

    fn finish_thinking(&mut self) {
        self.prepare_output();
        if !self.thinking_line_buf.is_empty() {
            let prefix = if self.thinking_first_line {
                "  🧠 "
            } else {
                "     "
            };
            print_wrapped(
                &self.thinking_line_buf,
                prefix,
                "     ",
                Tone::Dim,
                self.styled,
                terminal_width(),
            );
        }
        self.thinking_line_buf.clear();
        self.thinking_first_line = true;
    }

    fn on_event(&mut self, event: &Event) {
        match event {
            Event::TurnStart => {
                self.response_buffer.reset();
                self.streamed_any = false;
                self.refresh_live(true, true);
            }
            Event::Tokens {
                input_tokens,
                output_tokens,
                total_tokens,
                cached_tokens,
                turn_input,
                turn_output,
                context_window,
                model,
                reasoning_effort,
                ..
            } => {
                self.last_tokens = Some(TokenStatus {
                    input_tokens: *input_tokens,
                    output_tokens: *output_tokens,
                    total_tokens: *total_tokens,
                    cached_tokens: *cached_tokens,
                    turn_input: *turn_input,
                    turn_output: *turn_output,
                    context_window: *context_window,
                    model: model.clone(),
                    reasoning_effort: reasoning_effort.clone(),
                });
                if self.interactive {
                    self.refresh_live(true, true);
                }
            }
            Event::Thinking { content } => {
                self.prepare_output();
                print_wrapped(
                    content,
                    "  🧠 ",
                    "     ",
                    Tone::Dim,
                    self.styled,
                    terminal_width(),
                );
                self.refresh_live(true, true);
            }
            Event::ThinkingDelta { content } => {
                if !content.is_empty() {
                    self.thinking_line_buf.push_str(content);
                    self.flush_thinking_lines();
                    self.refresh_live(true, false);
                }
            }
            Event::ThinkingEnd => {
                self.finish_thinking();
                self.refresh_live(true, true);
            }
            Event::TextDelta { content } => {
                if !content.is_empty() {
                    self.response_buffer.append(content);
                    self.streamed_any = true;
                    self.refresh_live(true, false);
                }
            }
            Event::TextEnd { content } => {
                self.print_final_response(content, true);
            }
            Event::Text { content } => self.print_final_response(content, true),
            Event::ToolCall { name, arguments } => {
                self.prepare_output();
                self.print_tool_call(name, arguments);
                self.refresh_live(true, true);
            }
            Event::ToolResult { name, result } => {
                self.prepare_output();
                println!("     ├─ result:");
                print_truncated(
                    result,
                    "     │ ",
                    if matches!(name.as_str(), "write" | "edit") {
                        Tone::Green
                    } else {
                        Tone::Plain
                    },
                    self.styled,
                    terminal_width(),
                    "full output received by agent",
                );
                self.refresh_live(true, true);
            }
            Event::FinishReason { reason, .. } => {
                self.prepare_output();
                let warning = match reason.as_str() {
                    "length" => "Warning: The response was truncated because it reached the model's output limit.".to_owned(),
                    "content_filter" => "Warning: The response was stopped by the provider's content filter.".to_owned(),
                    _ => format!("Warning: The response stopped with finish reason {reason:?}."),
                };
                println!("{}", paint(&warning, Tone::Yellow, self.styled));
                self.refresh_live(true, true);
            }
            Event::HistoryTrimmed { summarized } => {
                self.prepare_output();
                let message =
                    format!("Context window reached: summarized {summarized} earlier message(s).");
                println!("{}", paint(&message, Tone::Yellow, self.styled));
                self.refresh_live(true, true);
            }
        }
    }

    fn print_tool_call(&self, name: &str, arguments: &Value) {
        if name == "bash" {
            let command = arguments
                .get("command")
                .and_then(Value::as_str)
                .unwrap_or("");
            println!();
            print_wrapped(
                command,
                "  🔧 ",
                "     ",
                Tone::Blue,
                self.styled,
                terminal_width(),
            );
        } else if name == "write" {
            let path = arguments.get("path").and_then(Value::as_str).unwrap_or("");
            println!(
                "\n  🔧 {}",
                paint(&format!("write(path={path:?})"), Tone::Green, self.styled)
            );
            print_truncated(
                arguments
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                "     │ ",
                Tone::Green,
                self.styled,
                terminal_width(),
                "full content sent to agent",
            );
        } else if name == "edit" {
            let path = arguments.get("path").and_then(Value::as_str).unwrap_or("");
            let edits = arguments.get("edits").and_then(Value::as_array);
            println!(
                "\n  🔧 {}",
                paint(
                    &format!(
                        "edit(path={path:?}) — {} edit(s)",
                        edits.map_or(0, |edits| edits.len())
                    ),
                    Tone::Green,
                    self.styled,
                )
            );
            for (index, edit) in edits.into_iter().flatten().enumerate() {
                println!("     ├─ Edit {}:", index + 1);
                if let Some(old_text) = edit.get("oldText").and_then(Value::as_str) {
                    println!(
                        "     │ {}",
                        paint(
                            &format!("oldText ({} lines):", old_text.lines().count()),
                            Tone::Green,
                            self.styled,
                        )
                    );
                    print_truncated(
                        old_text,
                        "     │ ",
                        Tone::Green,
                        self.styled,
                        terminal_width(),
                        "full content sent to agent",
                    );
                }
                if let Some(new_text) = edit.get("newText").and_then(Value::as_str) {
                    println!(
                        "     │ {}",
                        paint(
                            &format!("newText ({} lines):", new_text.lines().count()),
                            Tone::Green,
                            self.styled,
                        )
                    );
                    print_truncated(
                        new_text,
                        "     │ ",
                        Tone::Green,
                        self.styled,
                        terminal_width(),
                        "full content sent to agent",
                    );
                }
            }
        } else {
            println!("\n  🔧 {name}({})", format_arguments(arguments));
        }
    }
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

fn print_wrapped(
    text: &str,
    first_prefix: &str,
    rest_prefix: &str,
    tone: Tone,
    styled: bool,
    width: usize,
) {
    let mut first = true;
    for line in text.lines() {
        let prefix = if first { first_prefix } else { rest_prefix };
        let available = width.saturating_sub(UnicodeWidthStr::width(prefix)).max(1);
        for (index, chunk) in wrap_line(line, available).iter().enumerate() {
            let prefix = if index == 0 { prefix } else { rest_prefix };
            println!("{prefix}{}", paint(chunk, tone, styled));
        }
        first = false;
    }
}

fn print_truncated(
    text: &str,
    prefix: &str,
    tone: Tone,
    styled: bool,
    width: usize,
    full_output_note: &str,
) {
    let lines: Vec<_> = text.lines().collect();
    let available = width.saturating_sub(UnicodeWidthStr::width(prefix)).max(1);
    for line in lines.iter().take(DISPLAY_TRUNCATION_LIMIT) {
        for chunk in wrap_line(line, available) {
            println!("{prefix}{}", paint(&chunk, tone, styled));
        }
    }
    if lines.len() > DISPLAY_TRUNCATION_LIMIT {
        let notice = format!(
            "… truncated: {} of {} lines hidden ({full_output_note})",
            lines.len() - DISPLAY_TRUNCATION_LIMIT,
            lines.len()
        );
        println!("     └─ {}", paint(&notice, tone, styled));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn sample_token_status() -> TokenStatus {
        TokenStatus {
            input_tokens: 5_000,
            output_tokens: 200,
            total_tokens: 5_200,
            cached_tokens: 1_000,
            turn_input: 5_000,
            turn_output: 200,
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

    #[test]
    fn cli_defaults_and_stream_flags_match_python() {
        let args = Cli::try_parse_from(["harness"]).unwrap();
        assert_eq!(args.model, DEFAULT_MODEL);
        assert!(!args.no_stream);

        let args = Cli::try_parse_from(["harness", "--no-stream"]).unwrap();
        assert!(args.no_stream);
        let args = Cli::try_parse_from(["harness", "--stream"]).unwrap();
        assert!(args.stream);
        assert!(!args.no_stream);
    }

    #[test]
    fn positive_values_are_validated() {
        assert!(Cli::try_parse_from(["harness", "--max-turns", "0"]).is_err());
        assert!(Cli::try_parse_from(["harness", "--context-window", "-1"]).is_err());
    }

    #[test]
    fn welcome_box_rows_have_equal_display_width() {
        let box_text = build_welcome_box(
            "model",
            "high",
            1_000_000,
            std::path::Path::new("project"),
            true,
        );
        let widths: Vec<_> = box_text.lines().map(UnicodeWidthStr::width).collect();
        assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(box_text.contains("1,000,000 tokens"));
        assert!(box_text.contains("🤖 Nasa Level Genius Agent"));
        assert!(box_text.starts_with('╔'));
        assert!(box_text.ends_with('╝'));
    }

    #[test]
    fn status_bar_compacts_to_terminal_width() {
        let status = sample_token_status();

        let compact = status.render(48, false, false);
        assert!(UnicodeWidthStr::width(compact.as_str()) <= 48);
        assert!(!compact.contains("a-very-long-model-name"));
        assert!(!compact.contains("Cache:"));

        let full = status.render(200, false, false);
        assert!(full.contains("Ctx:0.5%"));
        assert!(full.contains("Cache:1,000 (20.0%)"));
        assert!(full.contains("🧠 high"));
    }

    #[test]
    fn status_bar_never_wraps_on_narrow_terminals() {
        let status = sample_token_status();

        for width in 1..=64 {
            let rendered = status.render(width, false, false);
            assert!(
                UnicodeWidthStr::width(rendered.as_str()) <= width,
                "status exceeded terminal width {width}: {rendered:?}"
            );
        }
    }

    #[test]
    fn live_frame_keeps_response_and_status_together() {
        let mut display = CliDisplay::new(false);
        display.interactive = false;
        display.styled = false;
        display.on_event(&Event::TextDelta {
            content: "## Summary\n\n**formatted** response".to_owned(),
        });
        display.on_event(&sample_token_event());

        let lines = display.live_lines(true, 80, 24);
        assert!(lines.iter().any(|line| line.contains("## Summary")));
        assert!(lines.last().is_some_and(|line| line.contains("Tot:5,200")));
    }

    #[test]
    fn turn_start_preserves_previous_token_status() {
        let mut display = CliDisplay::new(false);
        display.interactive = false;
        display.on_event(&sample_token_event());
        display.on_event(&Event::TurnStart);

        assert_eq!(
            display
                .last_tokens
                .as_ref()
                .map(|status| status.total_tokens),
            Some(5_200)
        );
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
