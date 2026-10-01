use std::collections::BTreeMap;
use std::env;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use reqwest::blocking::{Client, RequestBuilder, Response};
use serde_json::{Value, json};
use thiserror::Error;

use crate::cancel::CancelToken;
use crate::constants::{
    CONTEXT_WINDOW_TRIM_THRESHOLD, DEFAULT_BASE_URL, DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TURNS,
    MAX_RETRIES, RECENT_TURNS_TO_KEEP, RETRY_BASE_DELAY_SECS, SUMMARY_MAX_TOKENS,
    default_system_prompt,
};
use crate::events::{Callback, Event};
use crate::tools::ToolRegistry;

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("HTTP {status}: {message}")]
    Api { status: u16, message: String },
    #[error("HTTP request failed: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("invalid API response: {0}")]
    InvalidResponse(String),
    #[error("stream read failed: {0}")]
    Stream(#[from] std::io::Error),
    #[error("interrupted by the user")]
    Cancelled,
}

/// How often blocking waits check whether the user cancelled the turn.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Run blocking `work` on a helper thread so the caller can stop waiting when
/// `cancel` fires. An abandoned worker finishes in the background.
fn cancellable<T: Send + 'static>(
    cancel: &CancelToken,
    work: impl FnOnce() -> T + Send + 'static,
) -> Result<T, HarnessError> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(work());
    });
    recv_cancellable(&receiver, cancel)?
        .ok_or_else(|| HarnessError::InvalidResponse("request worker stopped unexpectedly".into()))
}

/// Receive the next value, or `None` once the sender is gone.
fn recv_cancellable<T>(
    receiver: &Receiver<T>,
    cancel: &CancelToken,
) -> Result<Option<T>, HarnessError> {
    loop {
        if cancel.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        match receiver.recv_timeout(CANCEL_POLL_INTERVAL) {
            Ok(value) => return Ok(Some(value)),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return Ok(None),
        }
    }
}

/// Sleep for `duration`, waking early with an error if the turn is cancelled.
fn sleep_cancellable(duration: Duration, cancel: &CancelToken) -> Result<(), HarnessError> {
    let deadline = Instant::now() + duration;
    loop {
        if cancel.is_cancelled() {
            return Err(HarnessError::Cancelled);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(());
        }
        thread::sleep(remaining.min(CANCEL_POLL_INTERVAL));
    }
}

impl HarnessError {
    fn retryable(&self) -> bool {
        match self {
            Self::Api { status, .. } => *status == 429 || (500..600).contains(status),
            Self::Transport(error) => error.is_timeout() || error.is_connect(),
            _ => false,
        }
    }
}

pub(crate) fn tls_insecure_enabled() -> bool {
    env::var("HARNESS_TLS_INSECURE").is_ok_and(|value| true_env_value(&value))
}

fn true_env_value(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

#[derive(Clone, Debug)]
pub struct AgentConfig {
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: String,
    pub system_prompt: String,
    pub working_dir: Option<PathBuf>,
    pub max_turns: usize,
    pub reasoning_effort: Option<String>,
    pub context_window: i64,
}

impl AgentConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: None,
            base_url: DEFAULT_BASE_URL.to_owned(),
            system_prompt: default_system_prompt(),
            working_dir: None,
            max_turns: DEFAULT_MAX_TURNS,
            reasoning_effort: None,
            context_window: DEFAULT_CONTEXT_WINDOW,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ToolCall {
    id: String,
    name: String,
    arguments: Value,
}

struct BackgroundTool {
    call: ToolCall,
    receiver: Receiver<String>,
    cancel: CancelToken,
}

#[derive(Default)]
struct BackgroundTools(Vec<BackgroundTool>);

impl Drop for BackgroundTools {
    fn drop(&mut self) {
        for tool in &self.0 {
            tool.cancel.cancel();
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_tokens: u64,
}

#[derive(Debug, Default)]
struct ProcessedResponse {
    text: Option<String>,
    reasoning: Option<String>,
    tool_calls: Vec<ToolCall>,
    usage: Usage,
    finish_reason: Option<String>,
}

pub struct AgentHarness {
    client: Client,
    session_id: String,
    endpoints: Vec<ApiEndpoint>,
    model_endpoint_indices: Option<Vec<usize>>,
    pub model: String,
    pub system_prompt: String,
    pub max_turns: usize,
    pub reasoning_effort: Option<String>,
    pub context_window: i64,
    pub tool_registry: ToolRegistry,
    pub messages: Vec<Value>,
    custom_context: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    last_prompt_tokens: u64,
    cancel: CancelToken,
}

#[derive(Clone, Debug, PartialEq)]
struct ApiEndpoint {
    api_key: String,
    base_url: String,
}

impl AgentHarness {
    pub fn new(config: AgentConfig) -> Result<Self, HarnessError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .danger_accept_invalid_certs(tls_insecure_enabled())
            .build()?;
        let api_key = config
            .api_key
            .filter(|value| !value.is_empty())
            .or_else(|| env::var("OPENAI_API_KEY").ok())
            .filter(|value| !value.is_empty());
        let endpoints = build_endpoints(api_key.as_deref(), &config.base_url);
        let tool_registry = ToolRegistry::new(config.working_dir);
        let cancel = tool_registry.cancel.clone();
        Ok(Self {
            client,
            session_id: new_session_id(),
            endpoints,
            model_endpoint_indices: None,
            model: config.model,
            system_prompt: config.system_prompt,
            max_turns: config.max_turns,
            reasoning_effort: config.reasoning_effort,
            context_window: config.context_window,
            tool_registry,
            messages: Vec::new(),
            custom_context: None,
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
            last_prompt_tokens: 0,
            cancel,
        })
    }

    /// Token that stops the running turn (and any running command) when cancelled.
    /// It is reset at the start of every turn.
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }

    pub fn add_provider(&mut self, base_url: &str, api_key: &str) {
        let base_url = base_url.trim().trim_end_matches('/').to_owned();
        self.endpoints
            .retain(|endpoint| endpoint.base_url != base_url);
        self.endpoints.insert(
            0,
            ApiEndpoint {
                base_url,
                api_key: api_key.trim().to_owned(),
            },
        );
        self.model_endpoint_indices = None;
    }

    pub(crate) fn provider_config(&self) -> (String, Option<String>) {
        let urls = self
            .endpoints
            .iter()
            .map(|endpoint| endpoint.base_url.as_str())
            .collect::<Vec<_>>()
            .join(";;");
        let keys = self
            .endpoints
            .iter()
            .map(|endpoint| endpoint.api_key.as_str())
            .collect::<Vec<_>>()
            .join(";;");
        (urls, Some(keys))
    }

    pub fn select_model(&mut self, model: impl Into<String>, endpoint_indices: Option<Vec<usize>>) {
        self.model = model.into();
        let valid_indices = endpoint_indices
            .unwrap_or_default()
            .into_iter()
            .filter(|index| *index < self.endpoints.len())
            .collect::<Vec<_>>();
        self.model_endpoint_indices = (!valid_indices.is_empty()).then_some(valid_indices);
    }

    pub fn run(&mut self, prompt: &str) -> Result<String, HarnessError> {
        self.run_inner(prompt, false, None)
    }

    pub fn run_with_callback(
        &mut self,
        prompt: &str,
        stream: bool,
        callback: &mut Callback<'_>,
    ) -> Result<String, HarnessError> {
        self.run_inner(prompt, stream, Some(callback))
    }

    fn run_inner(
        &mut self,
        prompt: &str,
        stream: bool,
        mut callback: Option<&mut Callback<'_>>,
    ) -> Result<String, HarnessError> {
        self.cancel.reset();
        let mut rollback_messages = self.messages.clone();
        if self.messages.is_empty() {
            self.messages = self.build_initial_messages(prompt);
        } else {
            self.messages
                .push(json!({"role": "user", "content": prompt}));
        }

        let mut background = BackgroundTools::default();
        let result = (|| {
            for turn_index in 0..self.max_turns {
                if turn_index > 0 {
                    emit(&mut callback, Event::TurnStart);
                }
                self.check_cancelled()?;
                if self.collect_background_results(&mut background, &mut callback) {
                    rollback_messages = self.messages.clone();
                }
                self.trim_history(&mut callback);
                self.check_cancelled()?;

                let mut body = self.build_create_body();
                if stream {
                    body["stream"] = Value::Bool(true);
                    body["stream_options"] = json!({"include_usage": true});
                }
                let response = self.send_with_retry(&body)?;
                let processed = if stream {
                    process_stream(response, &mut callback, &self.cancel)?
                } else {
                    let value: Value = cancellable(&self.cancel, move || response.json())??;
                    parse_response(&value)?
                };

                self.track_and_emit_tokens(processed.usage, &mut callback);
                if !stream {
                    if let Some(reason) = processed
                        .finish_reason
                        .as_deref()
                        .filter(|reason| !matches!(*reason, "stop" | "tool_calls"))
                    {
                        emit(
                            &mut callback,
                            Event::FinishReason {
                                reason: reason.to_owned(),
                                content: processed.text.clone().unwrap_or_default(),
                            },
                        );
                    }
                    if let Some(reasoning) = processed.reasoning.as_ref() {
                        emit(
                            &mut callback,
                            Event::Thinking {
                                content: reasoning.clone(),
                            },
                        );
                    }
                }

                if processed.tool_calls.is_empty() {
                    if !background.0.is_empty() {
                        self.finalize_response(processed.text, stream, &mut callback);
                        if !self.collect_background_results(&mut background, &mut callback) {
                            self.wait_for_background_result(&mut background, &mut callback)?;
                        }
                        rollback_messages = self.messages.clone();
                        continue;
                    }
                    return Ok(self.finalize_response(processed.text, stream, &mut callback));
                }

                self.handle_tool_calls(
                    processed.text,
                    processed.tool_calls,
                    &mut background,
                    &mut callback,
                );
                rollback_messages = self.messages.clone();
            }

            while !background.0.is_empty() {
                self.wait_for_background_result(&mut background, &mut callback)?;
                rollback_messages = self.messages.clone();
            }
            Ok(self.finalize_response(
                Some("Max turns reached without a final response.".to_owned()),
                stream,
                &mut callback,
            ))
        })();

        if result.is_err() {
            self.messages = rollback_messages;
            for tool in &background.0 {
                tool.cancel.cancel();
                self.messages.push(json!({
                    "role": "user",
                    "content": format!("Cancellation requested for background tool call {} ({}) because the agent turn stopped.", tool.call.id, tool.call.name),
                }));
            }
        }
        result
    }

    fn build_create_body(&self) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": self.messages,
            "tools": self.tool_registry.get_definitions(),
        });
        if let Some(reasoning_effort) = self.reasoning_effort.as_ref() {
            body["reasoning_effort"] = Value::String(reasoning_effort.clone());
        }
        body
    }

    fn check_cancelled(&self) -> Result<(), HarnessError> {
        if self.cancel.is_cancelled() {
            Err(HarnessError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn build_model_request(&self, provider: &ApiEndpoint, body: &Value) -> RequestBuilder {
        let mut request = self
            .client
            .post(format!("{}/chat/completions", provider.base_url))
            .bearer_auth(&provider.api_key)
            .json(body);
        if reqwest::Url::parse(&provider.base_url)
            .is_ok_and(|url| url.host_str() == Some("opencode.ai"))
        {
            request = request
                .header("x-opencode-session", &self.session_id)
                .header(
                    reqwest::header::USER_AGENT,
                    concat!("harness-rs/", env!("CARGO_PKG_VERSION")),
                );
        }
        request
    }

    fn send_with_retry(&self, body: &Value) -> Result<Response, HarnessError> {
        for attempt in 0..=MAX_RETRIES {
            let endpoint_index = self.endpoint_index_for_attempt(attempt);
            let provider = &self.endpoints[endpoint_index];
            let request = self.build_model_request(provider, body);
            let response = cancellable(&self.cancel, move || {
                request
                    .send()
                    .map_err(HarnessError::Transport)
                    .and_then(api_response)
            })
            .and_then(|response| response);

            match response {
                Ok(response) => return Ok(response),
                Err(error) if attempt < MAX_RETRIES && error.retryable() => {
                    let jitter = rand::thread_rng().gen_range(0.0..0.5);
                    let delay = RETRY_BASE_DELAY_SECS * 2_f64.powi(attempt as i32) + jitter;
                    sleep_cancellable(Duration::from_secs_f64(delay), &self.cancel)?;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("retry loop always returns")
    }

    fn endpoint_index_for_attempt(&self, attempt: usize) -> usize {
        self.model_endpoint_indices
            .as_ref()
            .map(|indices| indices[attempt % indices.len()])
            .unwrap_or(attempt % self.endpoints.len())
    }

    fn handle_tool_calls(
        &mut self,
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
        background: &mut BackgroundTools,
        callback: &mut Option<&mut Callback<'_>>,
    ) {
        let wire_calls: Vec<Value> = tool_calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_owned()),
                    }
                })
            })
            .collect();
        if let Some(text) = text.as_ref().filter(|text| !text.is_empty()) {
            emit(
                callback,
                Event::TextEnd {
                    content: text.clone(),
                },
            );
        }
        self.messages.push(json!({
            "role": "assistant",
            "content": text,
            "tool_calls": wire_calls,
        }));

        for call in tool_calls {
            if self.cancel.is_cancelled() {
                // Keep the history valid: every tool call needs a matching result.
                self.messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "Error: Cancelled by the user before this tool ran.",
                }));
                continue;
            }
            emit(
                callback,
                Event::ToolCall {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                },
            );
            let result = if call.arguments.get("background").and_then(Value::as_bool) == Some(true)
            {
                let mut registry = self.tool_registry.clone();
                registry.cancel = CancelToken::new();
                let worker_call = call.clone();
                let (sender, receiver) = mpsc::channel();
                let cancel = registry.cancel.clone();
                thread::spawn(move || {
                    let result = registry.execute(&worker_call.name, &worker_call.arguments);
                    let _ = sender.send(result);
                });
                background.0.push(BackgroundTool {
                    call: call.clone(),
                    receiver,
                    cancel,
                });
                format!(
                    "Background tool call {} started. Its result will be delivered automatically when finished. Continue with other work in the meantime.",
                    call.id
                )
            } else {
                self.tool_registry.execute(&call.name, &call.arguments)
            };
            emit(
                callback,
                Event::ToolResult {
                    name: call.name.clone(),
                    result: result.clone(),
                },
            );
            self.messages.push(json!({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            }));
        }
    }

    fn collect_background_results(
        &mut self,
        background: &mut BackgroundTools,
        callback: &mut Option<&mut Callback<'_>>,
    ) -> bool {
        let mut completed = false;
        let mut index = 0;
        while index < background.0.len() {
            let result = match background.0[index].receiver.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Empty) => {
                    index += 1;
                    continue;
                }
                Err(TryRecvError::Disconnected) => {
                    "Error: background tool worker stopped unexpectedly.".to_owned()
                }
            };
            let tool = background.0.remove(index);
            self.messages.push(json!({
                "role": "user",
                "content": format!("Background tool call {} ({}) finished:\n{}", tool.call.id, tool.call.name, result),
            }));
            emit(
                callback,
                Event::BackgroundToolResult {
                    tool_call_id: tool.call.id,
                    name: tool.call.name,
                    arguments: tool.call.arguments,
                    result,
                },
            );
            completed = true;
        }
        completed
    }

    fn wait_for_background_result(
        &mut self,
        background: &mut BackgroundTools,
        callback: &mut Option<&mut Callback<'_>>,
    ) -> Result<(), HarnessError> {
        loop {
            self.check_cancelled()?;
            if self.collect_background_results(background, callback) {
                return Ok(());
            }
            sleep_cancellable(CANCEL_POLL_INTERVAL, &self.cancel)?;
        }
    }

    fn finalize_response(
        &mut self,
        text: Option<String>,
        stream: bool,
        callback: &mut Option<&mut Callback<'_>>,
    ) -> String {
        self.messages
            .push(json!({"role": "assistant", "content": text}));
        let text = text.unwrap_or_default();
        let event = if stream {
            Event::TextEnd {
                content: text.clone(),
            }
        } else {
            Event::Text {
                content: text.clone(),
            }
        };
        emit(callback, event);
        text
    }

    pub fn clear_history(&mut self) {
        self.messages.clear();
        self.session_id = new_session_id();
        self.input_tokens = 0;
        self.output_tokens = 0;
        self.cached_tokens = 0;
        self.last_prompt_tokens = 0;
    }

    pub fn set_custom_context(&mut self, text: Option<impl Into<String>>) {
        self.custom_context = text
            .map(Into::into)
            .filter(|value: &String| !value.is_empty());
        let system_content = self.build_system_content();
        if self
            .messages
            .first()
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
            == Some("system")
        {
            self.messages[0]["content"] = Value::String(system_content);
        }
    }

    pub fn clear_custom_context(&mut self) {
        self.set_custom_context(None::<String>);
    }

    pub fn get_custom_context(&self) -> Option<&str> {
        self.custom_context.as_deref()
    }

    pub fn build_system_content(&self) -> String {
        let cwd = self
            .tool_registry
            .working_dir
            .clone()
            .or_else(|| env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        let cwd_note = format!(
            "\nWorking directory: {}. All commands run from this directory.",
            cwd.display()
        );
        match self.custom_context.as_ref() {
            Some(context) => format!(
                "{}\n\n--- Additional Context ---\n{}\n--- End Additional Context ---\n{}",
                self.system_prompt, context, cwd_note
            ),
            None => format!("{}{}", self.system_prompt, cwd_note),
        }
    }

    pub fn build_initial_messages(&self, prompt: &str) -> Vec<Value> {
        vec![
            json!({"role": "system", "content": self.build_system_content()}),
            json!({"role": "user", "content": prompt}),
        ]
    }

    fn track_and_emit_tokens(&mut self, usage: Usage, callback: &mut Option<&mut Callback<'_>>) {
        self.input_tokens += usage.prompt_tokens;
        self.output_tokens += usage.completion_tokens;
        self.cached_tokens += usage.cached_tokens;
        self.last_prompt_tokens = usage.prompt_tokens;
        emit(
            callback,
            Event::Tokens {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                total_tokens: self.input_tokens + self.output_tokens,
                cached_tokens: self.cached_tokens,
                turn_input: usage.prompt_tokens,
                turn_output: usage.completion_tokens,
                turn_cached: usage.cached_tokens,
                context_window: self.context_window,
                model: self.model.clone(),
                reasoning_effort: self.reasoning_effort.clone(),
            },
        );
    }

    fn trim_history(&mut self, callback: &mut Option<&mut Callback<'_>>) {
        if self.messages.is_empty() || self.context_window <= 0 {
            return;
        }
        let estimated = self.estimate_token_count().max(self.last_prompt_tokens);
        let threshold = (self.context_window as f64 * CONTEXT_WINDOW_TRIM_THRESHOLD) as u64;
        if estimated <= threshold {
            return;
        }

        let mut rest = self.messages.as_slice();
        let mut system_messages = Vec::new();
        if role(rest.first()) == Some("system") {
            system_messages.push(rest[0].clone());
            rest = &rest[1..];
        }

        let mut previous_summaries = Vec::new();
        while role(rest.first()) == Some("system")
            && content(rest.first())
                .is_some_and(|text| text.starts_with("--- Summary of earlier conversation ---"))
        {
            previous_summaries.push(rest[0].clone());
            rest = &rest[1..];
        }

        let (mut to_summarize, to_keep) = split_history_for_trim(rest);
        if to_summarize.is_empty() {
            return;
        }
        previous_summaries.append(&mut to_summarize);
        let summarized_count = previous_summaries.len();
        let Some(summary) = self.summarize_turns(&previous_summaries) else {
            return;
        };
        if summary.is_empty() {
            return;
        }

        emit(
            callback,
            Event::HistoryTrimmed {
                summarized: summarized_count,
            },
        );
        system_messages.push(json!({
            "role": "system",
            "content": format!(
                "--- Summary of earlier conversation ---\n{summary}\n--- End Summary ---"
            ),
        }));
        system_messages.extend(to_keep);
        self.messages = system_messages;
    }

    fn summarize_turns(&self, turns: &[Value]) -> Option<String> {
        let conversation = turns
            .iter()
            .map(|message| {
                let role = message
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let mut text = value_to_content(message.get("content"));
                if role == "tool" {
                    text = text.chars().take(300).collect();
                }
                format!("{role}: {text}")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let body = json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Summarize the following conversation concisely, preserving key facts, decisions, and file paths."
                },
                {"role": "user", "content": conversation}
            ],
            "max_tokens": SUMMARY_MAX_TOKENS,
        });
        let response = self.send_with_retry(&body).ok()?;
        let response: Value = response.json().ok()?;
        response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }

    pub fn estimate_token_count(&self) -> u64 {
        serde_json::to_vec(&self.build_create_body())
            .map(|payload| payload.len() as u64 / 4)
            .unwrap_or(0)
    }
}

pub(crate) fn split_multi(value: Option<&str>) -> Vec<String> {
    value
        .into_iter()
        .flat_map(|value| value.split(";;"))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn new_session_id() -> String {
    format!("harness-{:032x}", rand::random::<u128>())
}

fn build_endpoints(api_key: Option<&str>, base_url: &str) -> Vec<ApiEndpoint> {
    let mut keys = split_multi(api_key);
    if keys.is_empty() {
        keys.push("sk-placeholder".to_owned());
    }
    let urls = split_multi(Some(base_url));
    let count = keys.len().max(urls.len()).max(1);

    (0..count)
        .map(|index| ApiEndpoint {
            api_key: keys
                .get(index)
                .unwrap_or_else(|| keys.last().expect("at least one API key"))
                .clone(),
            base_url: urls
                .get(index)
                .or_else(|| urls.last())
                .map(|url| url.trim_end_matches('/').to_owned())
                .unwrap_or_default(),
        })
        .collect()
}

fn api_response(response: Response) -> Result<Response, HarnessError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let code = status.as_u16();
    let message = response.text().unwrap_or_else(|_| {
        status
            .canonical_reason()
            .unwrap_or("request failed")
            .to_owned()
    });
    Err(HarnessError::Api {
        status: code,
        message,
    })
}

fn emit(callback: &mut Option<&mut Callback<'_>>, event: Event) {
    if let Some(callback) = callback.as_deref_mut() {
        callback(&event);
    }
}

fn parse_response(response: &Value) -> Result<ProcessedResponse, HarnessError> {
    reject_error_envelope(response)?;
    let Some(choice) = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
    else {
        return Ok(ProcessedResponse {
            usage: parse_usage(response.get("usage")),
            ..ProcessedResponse::default()
        });
    };
    let message = choice.get("message").unwrap_or(&Value::Null);
    Ok(ProcessedResponse {
        text: message
            .get("content")
            .and_then(Value::as_str)
            .map(str::to_owned),
        reasoning: extract_reasoning(message),
        tool_calls: parse_tool_calls(message.get("tool_calls"))?,
        usage: parse_usage(response.get("usage")),
        finish_reason: choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
    })
}

fn parse_tool_calls(value: Option<&Value>) -> Result<Vec<ToolCall>, HarnessError> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|call| {
            let arguments = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .and_then(|arguments| serde_json::from_str(arguments).ok())
                .unwrap_or_else(|| json!({}));
            let id = call.get("id").and_then(non_empty_string).ok_or_else(|| {
                HarnessError::InvalidResponse("tool call is missing an id".to_owned())
            })?;
            let name = call
                .pointer("/function/name")
                .and_then(non_empty_string)
                .ok_or_else(|| {
                    HarnessError::InvalidResponse("tool call is missing a function name".to_owned())
                })?;
            Ok(ToolCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments,
            })
        })
        .collect()
}

fn extract_reasoning(value: &Value) -> Option<String> {
    for key in ["reasoning_content", "thinking", "thought", "reasoning"] {
        if let Some(reasoning) = value.get(key).and_then(non_empty_string) {
            return Some(reasoning.to_owned());
        }
    }
    if let Some(extra) = value.get("model_extra") {
        for key in ["reasoning_content", "reasoning", "thinking", "thought"] {
            if let Some(reasoning) = extra.get(key).and_then(non_empty_string) {
                return Some(reasoning.to_owned());
            }
        }
    }
    None
}

fn non_empty_string(value: &Value) -> Option<&str> {
    value.as_str().filter(|value| !value.is_empty())
}

fn parse_usage(value: Option<&Value>) -> Usage {
    let value = value.unwrap_or(&Value::Null);
    Usage {
        prompt_tokens: value
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        completion_tokens: value
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cached_tokens: value
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
    }
}

#[derive(Default)]
struct StreamToolCall {
    id: String,
    name: String,
    argument_chunks: String,
}

#[derive(Default)]
struct StreamState {
    text: String,
    reasoning: String,
    thinking_open: bool,
    tool_calls: BTreeMap<u64, StreamToolCall>,
    usage: Usage,
    finish_reason: Option<String>,
}

fn process_stream(
    response: Response,
    callback: &mut Option<&mut Callback<'_>>,
    cancel: &CancelToken,
) -> Result<ProcessedResponse, HarnessError> {
    // Read on a helper thread so a stalled stream can still be cancelled.
    let (sender, lines) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(response).lines() {
            let failed = line.is_err();
            if sender.send(line).is_err() || failed {
                break;
            }
        }
    });
    let mut state = StreamState::default();
    let mut event_data = Vec::new();
    while let Some(line) = recv_cancellable(&lines, cancel)? {
        let line = line?;
        if line.is_empty() {
            if !event_data.is_empty()
                && !process_sse_data(&event_data.join("\n"), &mut state, callback)?
            {
                break;
            }
            event_data.clear();
        } else if let Some(data) = line.strip_prefix("data:") {
            event_data.push(data.strip_prefix(' ').unwrap_or(data).to_owned());
        }
    }
    if !event_data.is_empty() {
        let _ = process_sse_data(&event_data.join("\n"), &mut state, callback)?;
    }
    close_thinking(&mut state, callback);

    if let Some(reason) = state
        .finish_reason
        .as_deref()
        .filter(|reason| !matches!(*reason, "stop" | "tool_calls"))
    {
        emit(
            callback,
            Event::FinishReason {
                reason: reason.to_owned(),
                content: state.text.clone(),
            },
        );
    }

    let tool_calls = state
        .tool_calls
        .into_values()
        .map(|call| ToolCall {
            id: call.id,
            name: call.name,
            arguments: serde_json::from_str(&call.argument_chunks).unwrap_or_else(|_| json!({})),
        })
        .collect();
    Ok(ProcessedResponse {
        text: (!state.text.is_empty()).then_some(state.text),
        reasoning: (!state.reasoning.is_empty()).then_some(state.reasoning),
        tool_calls,
        usage: state.usage,
        finish_reason: state.finish_reason,
    })
}

fn process_sse_data(
    data: &str,
    state: &mut StreamState,
    callback: &mut Option<&mut Callback<'_>>,
) -> Result<bool, HarnessError> {
    if data.trim() == "[DONE]" {
        return Ok(false);
    }
    let chunk: Value = serde_json::from_str(data)
        .map_err(|error| HarnessError::InvalidResponse(error.to_string()))?;
    reject_error_envelope(&chunk)?;
    if chunk.get("usage").is_some_and(|usage| !usage.is_null()) {
        state.usage = parse_usage(chunk.get("usage"));
    }
    let Some(choice) = chunk
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
    else {
        return Ok(true);
    };
    if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
        state.finish_reason = Some(reason.to_owned());
    }
    let delta = choice.get("delta").unwrap_or(&Value::Null);

    let reasoning = extract_reasoning(delta);
    if let Some(reasoning) = reasoning {
        state.reasoning.push_str(&reasoning);
        state.thinking_open = true;
        emit(callback, Event::ThinkingDelta { content: reasoning });
    }
    if let Some(content) = delta.get("content").and_then(non_empty_string) {
        close_thinking(state, callback);
        state.text.push_str(content);
        emit(
            callback,
            Event::TextDelta {
                content: content.to_owned(),
            },
        );
    }
    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
        if !tool_calls.is_empty() {
            close_thinking(state, callback);
        }
        for call in tool_calls {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let accumulated = state.tool_calls.entry(index).or_default();
            if let Some(id) = call.get("id").and_then(non_empty_string) {
                accumulated.id = id.to_owned();
            }
            if let Some(name) = call.pointer("/function/name").and_then(non_empty_string) {
                accumulated.name = name.to_owned();
            }
            if let Some(arguments) = call
                .pointer("/function/arguments")
                .and_then(non_empty_string)
            {
                accumulated.argument_chunks.push_str(arguments);
            }
        }
    }
    Ok(true)
}

fn reject_error_envelope(value: &Value) -> Result<(), HarnessError> {
    let Some(error) = value.get("error").filter(|error| !error.is_null()) else {
        return Ok(());
    };
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| error.to_string());
    let status = error
        .get("status")
        .or_else(|| error.get("code"))
        .and_then(Value::as_u64)
        .and_then(|status| u16::try_from(status).ok())
        .unwrap_or(500);
    Err(HarnessError::Api { status, message })
}

fn close_thinking(state: &mut StreamState, callback: &mut Option<&mut Callback<'_>>) {
    if state.thinking_open {
        emit(callback, Event::ThinkingEnd);
        state.thinking_open = false;
    }
}

fn split_history_for_trim(messages: &[Value]) -> (Vec<Value>, Vec<Value>) {
    let mut prefix = Vec::new();
    let mut exchanges: Vec<Vec<Value>> = Vec::new();
    for message in messages {
        if role(Some(message)) == Some("user") {
            exchanges.push(vec![message.clone()]);
        } else if let Some(exchange) = exchanges.last_mut() {
            exchange.push(message.clone());
        } else {
            prefix.push(message.clone());
        }
    }
    if exchanges.is_empty() {
        return (Vec::new(), messages.to_vec());
    }
    let keep_at = exchanges.len().saturating_sub(RECENT_TURNS_TO_KEEP);
    let mut summarize = prefix;
    for exchange in &exchanges[..keep_at] {
        summarize.extend(exchange.iter().cloned());
    }
    let keep = exchanges[keep_at..].iter().flatten().cloned().collect();
    (summarize, keep)
}

fn role(message: Option<&Value>) -> Option<&str> {
    message?.get("role")?.as_str()
}

fn content(message: Option<&Value>) -> Option<&str> {
    message?.get("content")?.as_str()
}

fn value_to_content(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn background_calls_continue_and_deliver_results_for_both_response_modes() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        for stream_mode in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let mut config = AgentConfig::new("test-model");
            config.base_url = format!("http://{}/v1", listener.local_addr().unwrap());
            config.working_dir = Some(directory.path().to_owned());
            config.max_turns = 3;
            let release = directory.path().join("release");
            let foreground = directory.path().join("foreground");
            #[cfg(windows)]
            let command = "while (!(Test-Path release)) { Start-Sleep -Milliseconds 10 }; Write-Output background-output";
            #[cfg(not(windows))]
            let command = "while [ ! -f release ]; do sleep 0.01; done; printf background-output";
            let server = thread::spawn(move || {
                for index in 0..3 {
                    let deadline = Instant::now() + Duration::from_secs(5);
                    let mut socket = loop {
                        match listener.accept() {
                            Ok((socket, _)) => break socket,
                            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                                assert!(Instant::now() < deadline, "model request timed out");
                                thread::sleep(Duration::from_millis(10));
                            }
                            Err(error) => panic!("{error}"),
                        }
                    };
                    socket
                        .set_read_timeout(Some(Duration::from_secs(5)))
                        .unwrap();
                    let mut headers = Vec::new();
                    let mut byte = [0];
                    while !headers.ends_with(b"\r\n\r\n") {
                        socket.read_exact(&mut byte).unwrap();
                        headers.push(byte[0]);
                    }
                    let headers = String::from_utf8(headers).unwrap();
                    let length: usize = headers
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length: ")
                                .and_then(|value| value.parse().ok())
                        })
                        .unwrap();
                    let mut body = vec![0; length];
                    socket.read_exact(&mut body).unwrap();
                    let body: Value = serde_json::from_slice(&body).unwrap();
                    let message = match index {
                        0 => json!({"content": null, "tool_calls": [
                            {"id": "bg-1", "type": "function", "function": {
                                "name": "bash", "arguments": json!({"command": command, "background": true}).to_string()
                            }},
                            {"id": "fg-1", "type": "function", "function": {
                                "name": "write", "arguments": json!({"path": "foreground", "content": "done"}).to_string()
                            }}
                        ]}),
                        1 => {
                            assert!(foreground.exists());
                            let messages = body["messages"].as_array().unwrap();
                            assert!(
                                messages
                                    .iter()
                                    .any(|message| message["tool_call_id"] == "bg-1"
                                        && message["content"]
                                            .as_str()
                                            .unwrap()
                                            .contains("started"))
                            );
                            assert!(!messages.iter().any(|message| {
                                message["content"]
                                    .as_str()
                                    .is_some_and(|content| content.contains("background-output"))
                                    && message["role"] == "user"
                            }));
                            std::fs::write(&release, "done").unwrap();
                            json!({"content": "Other work finished."})
                        }
                        _ => {
                            assert!(body["messages"].as_array().unwrap().iter().any(|message| {
                                message["role"] == "user"
                                    && message["content"].as_str().is_some_and(|content| {
                                        content.contains("bg-1 (bash) finished:")
                                            && content.contains("background-output")
                                    })
                            }));
                            json!({"content": "All work finished."})
                        }
                    };
                    let finish = if index == 0 { "tool_calls" } else { "stop" };
                    let payload = if stream_mode {
                        let mut delta = message.clone();
                        if let Some(calls) = delta["tool_calls"].as_array_mut() {
                            for (index, call) in calls.iter_mut().enumerate() {
                                call["index"] = json!(index);
                            }
                        }
                        format!(
                            "data: {}\n\ndata: [DONE]\n\n",
                            json!({"choices": [{"delta": delta, "finish_reason": finish}]})
                        )
                    } else {
                        json!({"choices": [{"message": message, "finish_reason": finish}]})
                            .to_string()
                    };
                    write!(socket, "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        if stream_mode { "text/event-stream" } else { "application/json" }, payload.len(), payload).unwrap();
                }
            });
            let mut agent = AgentHarness::new(config).unwrap();
            let mut events = Vec::new();
            let response = agent
                .run_with_callback("Do the work", stream_mode, &mut |event| {
                    events.push(event.clone())
                })
                .unwrap();
            server.join().unwrap();
            assert_eq!(response, "All work finished.");
            assert!(events.iter().any(|event| matches!(event, Event::ToolCall { arguments, .. } if arguments["background"] == true)));
            assert!(events.iter().any(
                |event| matches!(event, Event::BackgroundToolResult { tool_call_id, result, .. }
                if tool_call_id == "bg-1" && result.contains("background-output"))
            ));
        }
    }

    #[test]
    fn background_completions_keep_call_ids_and_arguments_even_out_of_order() {
        let mut agent = AgentHarness::new(AgentConfig::new("test-model")).unwrap();
        let mut background = BackgroundTools::default();
        let mut senders = Vec::new();
        for id in ["first", "second"] {
            let (sender, receiver) = mpsc::channel();
            senders.push(sender);
            background.0.push(BackgroundTool {
                call: ToolCall {
                    id: id.to_owned(),
                    name: "read".to_owned(),
                    arguments: json!({"path": id, "background": true}),
                },
                receiver,
                cancel: CancelToken::new(),
            });
        }
        senders[1].send("Error: missing file".to_owned()).unwrap();
        let mut events = Vec::new();
        let mut callback = |event: &Event| events.push(event.clone());
        assert!(agent.collect_background_results(&mut background, &mut Some(&mut callback)));
        assert_eq!(background.0[0].call.id, "first");
        senders[0].send("first-output".to_owned()).unwrap();
        assert!(agent.collect_background_results(&mut background, &mut Some(&mut callback)));
        assert!(background.0.is_empty());
        assert!(
            matches!(&events[0], Event::BackgroundToolResult { tool_call_id, arguments, result, .. }
            if tool_call_id == "second" && arguments["path"] == "second" && result == "Error: missing file")
        );
        assert!(
            matches!(&events[1], Event::BackgroundToolResult { tool_call_id, result, .. }
            if tool_call_id == "first" && result == "first-output")
        );
    }

    #[test]
    fn cancelling_a_background_wait_stops_workers_without_reusing_the_turn_token() {
        let mut agent = AgentHarness::new(AgentConfig::new("test-model")).unwrap();
        let (sender, receiver) = mpsc::channel();
        let worker_cancel = CancelToken::new();
        let mut background = BackgroundTools(vec![BackgroundTool {
            call: ToolCall {
                id: "bg".into(),
                name: "bash".into(),
                arguments: json!({}),
            },
            receiver,
            cancel: worker_cancel.clone(),
        }]);
        agent.cancel.cancel();
        assert!(matches!(
            agent.wait_for_background_result(&mut background, &mut None),
            Err(HarnessError::Cancelled)
        ));
        drop(background);
        agent.cancel.reset();
        assert!(worker_cancel.is_cancelled());
        drop(sender);
    }

    #[test]
    fn foreground_calls_complete_synchronously_by_default_and_when_false() {
        let directory = tempfile::tempdir().unwrap();
        let mut config = AgentConfig::new("test-model");
        config.working_dir = Some(directory.path().to_owned());
        let mut agent = AgentHarness::new(config).unwrap();
        for arguments in [
            json!({"path": "file", "content": "default"}),
            json!({"path": "file", "content": "false", "background": false}),
        ] {
            let mut background = BackgroundTools::default();
            agent.handle_tool_calls(
                None,
                vec![ToolCall {
                    id: "fg".into(),
                    name: "write".into(),
                    arguments: arguments.clone(),
                }],
                &mut background,
                &mut None,
            );
            assert!(background.0.is_empty());
            assert_eq!(
                std::fs::read_to_string(directory.path().join("file")).unwrap(),
                arguments["content"].as_str().unwrap()
            );
            assert!(
                agent.messages.last().unwrap()["content"]
                    .as_str()
                    .unwrap()
                    .contains("Successfully wrote")
            );
        }
    }

    #[test]
    fn parses_text_reasoning_tools_and_usage() {
        let response = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": "Checking",
                    "reasoning_content": "Think",
                    "thinking": "lower priority",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "read", "arguments": "{\"path\":\"a.txt\"}"}
                    }]
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 4}
            }
        });
        let parsed = parse_response(&response).unwrap();
        assert_eq!(parsed.text.as_deref(), Some("Checking"));
        assert_eq!(parsed.reasoning.as_deref(), Some("Think"));
        assert_eq!(parsed.tool_calls[0].name, "read");
        assert_eq!(parsed.tool_calls[0].arguments, json!({"path": "a.txt"}));
        assert_eq!(parsed.usage.cached_tokens, 4);
    }

    #[test]
    fn malformed_tool_arguments_become_empty_object() {
        let response = json!({"choices": [{"message": {
            "content": null,
            "tool_calls": [{"id": "1", "function": {"name": "read", "arguments": "bad"}}]
        }}]});
        assert_eq!(
            parse_response(&response).unwrap().tool_calls[0].arguments,
            json!({})
        );
    }

    #[test]
    fn empty_choices_are_valid_empty_response() {
        let parsed = parse_response(&json!({"choices": []})).unwrap();
        assert!(parsed.text.is_none());
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn streamed_tool_fragments_are_combined_by_index() {
        let mut state = StreamState::default();
        let mut callback = None;
        process_sse_data(
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\"pa"}}]}}]}"#,
            &mut state,
            &mut callback,
        )
        .unwrap();
        process_sse_data(
            r#"{"choices":[{"finish_reason":"tool_calls","delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"x\"}"}}]}}]}"#,
            &mut state,
            &mut callback,
        )
        .unwrap();
        let call = state.tool_calls.get(&0).unwrap();
        assert_eq!(call.id, "c1");
        assert_eq!(call.name, "read");
        assert_eq!(
            serde_json::from_str::<Value>(&call.argument_chunks).unwrap(),
            json!({"path": "x"})
        );
    }

    #[test]
    fn history_split_keeps_complete_recent_exchanges() {
        let mut messages = Vec::new();
        for index in 0..8 {
            messages.push(json!({"role": "user", "content": format!("u{index}")}));
            messages.push(json!({"role": "assistant", "content": null, "tool_calls": []}));
            messages.push(json!({"role": "tool", "tool_call_id": index, "content": "ok"}));
        }
        let (summarize, keep) = split_history_for_trim(&messages);
        assert_eq!(summarize.len(), 6);
        assert_eq!(keep.len(), 18);
        assert_eq!(role(keep.first()), Some("user"));
    }

    #[test]
    fn config_defaults_match_python_library() {
        let config = AgentConfig::new("model");
        assert_eq!(config.max_turns, 1_000);
        assert_eq!(config.context_window, 1_000_000);
        assert!(config.reasoning_effort.is_none());
    }

    #[test]
    fn tls_insecure_values_match_python() {
        for value in ["1", "true", "TRUE", " yes ", "on"] {
            assert!(true_env_value(value));
        }
        for value in ["", "0", "false", "no", "off"] {
            assert!(!true_env_value(value));
        }
    }

    #[test]
    fn multi_keys_and_urls_are_paired_and_reuse_last_value() {
        let endpoints = build_endpoints(
            Some(" key-one ;; key-two ;; key-three "),
            "https://a.example/v1/;; https://b.example/v1 ;; ",
        );

        assert_eq!(
            endpoints,
            vec![
                ApiEndpoint {
                    api_key: "key-one".to_owned(),
                    base_url: "https://a.example/v1".to_owned(),
                },
                ApiEndpoint {
                    api_key: "key-two".to_owned(),
                    base_url: "https://b.example/v1".to_owned(),
                },
                ApiEndpoint {
                    api_key: "key-three".to_owned(),
                    base_url: "https://b.example/v1".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn opencode_requests_share_a_session_until_history_is_cleared() {
        let mut config = AgentConfig::new("test-model");
        config.base_url = "https://opencode.ai/zen/v1;;https://opencode.ai/zen/go/v1".to_owned();
        let mut agent = AgentHarness::new(config).unwrap();
        let session = agent.session_id.clone();
        for endpoint in &agent.endpoints {
            for body in [
                json!({"stream": false}),
                json!({"stream": true}),
                json!({"messages": [{"role": "user", "content": "Summarize"}]}),
            ] {
                let request = agent.build_model_request(endpoint, &body).build().unwrap();
                assert_eq!(request.headers()["x-opencode-session"], session);
                assert_eq!(
                    request.headers()[reqwest::header::USER_AGENT],
                    concat!("harness-rs/", env!("CARGO_PKG_VERSION"))
                );
            }
        }
        agent.select_model("another-model", Some(vec![1]));
        agent.add_provider("https://opencode.ai/zen/go/v1", "updated-key");
        assert_eq!(agent.session_id, session);
        let other_agent = AgentHarness::new(AgentConfig::new("test-model")).unwrap();
        assert_ne!(other_agent.session_id, session);
        agent.clear_history();
        assert_ne!(agent.session_id, session);
        let request = agent
            .build_model_request(&agent.endpoints[0], &json!({}))
            .build()
            .unwrap();
        assert_eq!(request.headers()["x-opencode-session"], agent.session_id);
    }

    #[test]
    fn unrelated_providers_do_not_receive_opencode_headers() {
        let agent = AgentHarness::new(AgentConfig::new("test-model")).unwrap();
        for base_url in [
            "https://api.openai.com/v1",
            "https://opencode.ai.example/v1",
            "https://example.com/opencode.ai",
        ] {
            let endpoint = ApiEndpoint {
                base_url: base_url.to_owned(),
                api_key: "key".to_owned(),
            };
            let request = agent
                .build_model_request(&endpoint, &json!({}))
                .build()
                .unwrap();
            assert!(!request.headers().contains_key("x-opencode-session"));
            assert!(!request.headers().contains_key(reqwest::header::USER_AGENT));
        }
    }

    #[test]
    fn adding_provider_updates_keys_and_resets_model_routing() {
        let mut config = AgentConfig::new("test-model");
        config.api_key = Some("original-key".to_owned());
        config.base_url = "https://original.example/v1".to_owned();
        let mut agent = AgentHarness::new(config).unwrap();
        agent.select_model("test-model", Some(vec![0]));
        agent
            .messages
            .push(json!({"role": "user", "content": "keep history"}));
        agent.set_custom_context(Some("keep context"));
        agent.add_provider("https://new.example/v1/", "new-key");
        assert_eq!(agent.endpoint_index_for_attempt(0), 0);
        assert_eq!(agent.endpoints[0].base_url, "https://new.example/v1");
        assert_eq!(agent.endpoints[1].api_key, "original-key");
        assert!(agent.model_endpoint_indices.is_none());
        agent.add_provider("https://new.example/v1", "updated-key");
        assert_eq!(agent.endpoints.len(), 2);
        assert_eq!(agent.endpoints[0].api_key, "updated-key");
        assert_eq!(agent.messages.len(), 1);
        assert_eq!(agent.get_custom_context(), Some("keep context"));
        let (urls, keys) = agent.provider_config();
        assert_eq!(build_endpoints(keys.as_deref(), &urls), agent.endpoints);
    }

    #[test]
    fn model_selection_keeps_only_valid_provider_indices() {
        let mut config = AgentConfig::new("old-model");
        config.api_key = Some("key-one;;key-two".to_owned());
        config.base_url = "https://a.example/v1;;https://b.example/v1".to_owned();
        let mut agent = AgentHarness::new(config).unwrap();

        agent.select_model("new-model", Some(vec![1, 99]));

        assert_eq!(agent.model, "new-model");
        assert_eq!(agent.model_endpoint_indices, Some(vec![1]));
        assert_eq!(agent.endpoint_index_for_attempt(0), 1);
        assert_eq!(agent.endpoint_index_for_attempt(3), 1);

        agent.select_model("unmapped-model", None);
        assert_eq!(agent.endpoint_index_for_attempt(0), 0);
        assert_eq!(agent.endpoint_index_for_attempt(1), 1);
        assert_eq!(agent.endpoint_index_for_attempt(2), 0);
    }
}
