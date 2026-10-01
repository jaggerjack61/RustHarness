# Harness Rust Port

This directory contains a standalone Rust port of the 
https://github.com/jaggerjack61/Harness repository's Python
agent. It keeps the OpenAI-compatible chat-completions protocol, streaming
SSE responses, reasoning fields, tool calling, retries, context
summarization, token events, markdown rendering, and the unrestricted
`read`, `write`, `edit`, and `bash` tools, plus a dynamic `find` tool when
ripgrep (`rg`) or `grep` is installed.

## Features

- **OpenAI-compatible** — works with OpenAI, DeepSeek, local models, or any
  chat-completions + tool-calling API.
- **Four built-in tools** — `read`, `write`, `edit`, and `bash` give the model
  full access to the filesystem and shell.
- **Dynamic content search** — `find` is exposed to the model when `rg` or
  `grep` is on PATH, preferring `rg`. It searches files recursively and returns
  filenames, line numbers, and matching text. Supports a search path,
  case-insensitive matching, and literal text or regex patterns; uses the
  same timeout, cancellation, and output limits as shell commands.
- **Background tools** — all tools accept `"background": true` (default `false`).
  Calls appear in chat immediately while the agent continues working. Completed
  output is delivered automatically with the original call ID. If other work
  finishes first, the agent waits for the output and responds to it. Cancellation
  also stops background commands.
- **Reasoning / chain-of-thought** — displays `reasoning_content` and similar
  fields from reasoning models as dimmed terminal output.
- **Markdown rendering** — responses are rendered as rich Markdown with
  syntax-highlighted code blocks, headings, lists, tables, etc.
- **Streaming** — responses stream token-by-token; the final answer is then
  rendered as formatted Markdown.
- **Configurable reasoning effort** — pass
  `--reasoning-effort low|medium|high|xhigh|max` for models that support it.
- **Polished terminal UI** — an animated status line shows what the agent is
  doing, elapsed time, and token/context usage. Tool calls are rendered as
  compact `● Bash(cargo test)` blocks with output previews, edits as
  red/green diffs, and each turn ends with a one-line summary. Colors follow
  `NO_COLOR` and are omitted when output is piped.
- **Line editing** — input history (persisted across sessions), Tab
  completion and inline hints for slash commands, and `\` + Enter for
  multi-line messages.
- **Quick reasoning switch** — press Tab at the prompt to cycle the reasoning
  effort; the `▰▰▰▱▱` meter before the `›` shows the current level.
- **Interruptible** — press Esc (or Ctrl+C) while the agent works to stop
  it, including any running shell command. Completed tool steps stay in
  the conversation.
- **Conversation history** — context is remembered across turns; use `/clear`
  to reset.
- **Automatic context management** — history is trimmed/summarized when it
  approaches the configured context window limit.
- **Retry with exponential backoff** — transient API errors (429, 5xx,
  timeouts) are retried automatically up to 3 times.
- **Real-time token tracking** — cumulative input/output/cache token counts
  and context-window usage are shown in the status bar.
- **Model switcher** — `/models` lets you choose a provider, then select
  one of its models at runtime.
- **Custom context** — `/context` lets you paste a block of text that is kept
  in the system message for the current session.
- **Cross-platform shell** — PowerShell 7 / Windows PowerShell 5.1 on
  Windows, `/bin/sh` on Unix/macOS; `HARNESS_SHELL` overrides the Windows
  shell choice.
- **Output safety** — tool outputs above 1,000 lines or 100 KB are dropped
  and the agent is asked to retry with narrower commands.

## Requirements

- A stable Rust toolchain (1.85+ for the `2024` edition)
- An API key for an OpenAI-compatible service

## Build

```bash
cd rust
cargo build --release
```

The executable is `target/release/harness` (`target\release\harness.exe` on
Windows).

At startup, Harness loads an optional `.env` from the executable's directory.
For a release build, place it beside `harness` or `harness.exe`, not in the
shell's current working directory. Values in this file override inherited
environment variables, matching the Python launcher's behavior.

```dotenv
OPENAI_API_KEY=key-for-provider-a;;key-for-provider-b
HARNESS_BASE_URL=https://provider-a.example/v1;;https://provider-b.example/v1
```

Multiple keys and URLs are separated by `;;` and paired in order. If one list
is shorter, its last value is reused. Retryable API failures rotate through
the configured providers.

## Run

```bash
export OPENAI_API_KEY="sk-..."
cargo run --release -- \
  --model deepseek-v4-pro \
  --base-url https://api.deepseek.com/v1 \
  --dir ..
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "sk-..."
cargo run --release -- --model deepseek-v4-pro --base-url https://api.deepseek.com/v1 --dir ..
```

Or pass the credentials directly on the command line, for example with
OpenCode's `kimi-k2.7-code` model:

```bash
harness.exe --model "kimi-k2.7-code" \
  --base-url "https://opencode.ai/zen/go/v1/" \
  --api-key "sk-your-api-key"
```

OpenCode Zen and Go providers automatically receive an `x-opencode-session`
header and a `harness-rs` user agent on every model completion request,
including streaming, retries, tool follow-ups, and context summarization.
The session ID is generated for each conversation, stays stable across turns
and provider/model switches, and renews on `/clear` or a new Harness instance.
This applies to OpenCode URLs supplied through `/login`, saved providers,
CLI options, environment variables, or the library; no extra configuration
is needed.

The CLI options and environment variables mirror the Python implementation.

| CLI option | Environment variable | Description |
|------------|----------------------|-------------|
| `-m`, `--model` | `HARNESS_MODEL` | Model name (defaults to the last selected model; prompts on first use) |
| `-k`, `--api-key` | `OPENAI_API_KEY` | API key, or multiple keys separated by `;;` |
| `-u`, `--base-url` | `HARNESS_BASE_URL` | Provider base URL(s), separated by `;;` (default `https://api.openai.com/v1`) |
| `-d`, `--dir` | — | Working directory |
| `--max-turns` | `HARNESS_MAX_TURNS` | Maximum tool/response turns (default `1000`) |
| `--system-prompt` | `HARNESS_PROMPT` | Override the system prompt |
| `--reasoning-effort` | — | `low`, `medium`, `high`, `xhigh`, or `max` (default `high`) |
| `--context-window` | `HARNESS_CONTEXT_WINDOW` | Override token context window (otherwise uses model metadata, falling back to `1_000_000`) |
| `--stream` / `--no-stream` | — | Force streaming on or off |
| `--no-markdown` | — | Disable markdown rendering |
| — | `HARNESS_TLS_INSECURE` | Set to `true` to disable TLS certificate verification for API and model requests |
| — | `NO_PROXY` | Comma-separated provider hosts that must bypass `HTTP_PROXY`/`HTTPS_PROXY` |
| — | `HARNESS_SHELL` | Windows shell override (default: `pwsh` if available, otherwise `powershell`) |

### Interactive commands

At the `›` prompt, you can type:

| Command | Description |
|---------|-------------|
| `/help` | Show commands and keyboard shortcuts |
| `/exit` | Quit the session |
| `/clear` | Clear conversation history and redraw the welcome box |
| `/stream` | Toggle streaming mode |
| `/login` | Save a provider name, URL, and API key for future sessions |
| `/models` | Choose a provider, then a model (type to filter long lists) |
| `/reasoning` | Switch reasoning effort |
| `/context` | Paste a custom context block |
| `/context clear` | Remove the custom context |
| `/context show` | Show the current custom context |

Keyboard shortcuts: **Tab** / **Shift+Tab** cycle the reasoning effort
(or complete a command when the line starts with `/`), **Esc** interrupts
the agent while it works (a second Ctrl+C during an interrupt quits),
**↑/↓** browse history,
ending a line with **`\`** continues the message on the next line,
**Ctrl+C** clears the current input (press twice to quit), and **Ctrl+D**
quits. Input history is stored as `history` next to `providers.json`.

Type `/login` to enter a provider name, an OpenAI-compatible provider base URL (including `/v1` when required) and API key. The key is hidden during interactive entry. Press Enter at any prompt to cancel. New providers are added first to the configured list, with existing providers retained; logging in again to the same URL updates its name and key. Credentials persist in a local JSON file: `~/Library/Application Support/Harness/providers.json` on macOS, `$XDG_CONFIG_HOME/harness/providers.json` (or `~/.config/harness/providers.json`) on Linux, and `%APPDATA%\Harness\providers.json` on Windows. Keys are stored as plain text; on macOS/Linux the directory and file are restricted to your user (700/600). Saved providers load automatically when no API key or custom base URL is supplied through CLI/environment options. Explicit credentials take precedence for that launch. Startup prompts for a provider when none is configured, and for a model when no previous selection exists. The last selected model and provider URL are saved in `last-model.json` beside the provider configuration and restored on later launches; `--model` or `HARNESS_MODEL` takes precedence. Use `/models` (or `/model`) to choose a provider, then select a model from its list. Requests and retries use the selected provider, even when another provider offers the same model ID. Providers saved without a name display their URL until renamed through `/login`. Context-window metadata from the provider’s `/models` response is applied on startup and model changes, with `1_000_000` as the fallback when metadata is absent. `--context-window` or `HARNESS_CONTEXT_WINDOW` overrides that metadata.

The default system prompt identifies the current platform and actual tool shell: PowerShell on Windows and `/bin/sh` on macOS/Linux. On macOS it also directs the model to use macOS/BSD command options. A custom `--system-prompt` still overrides the default.

## Library

```rust,no_run
use harness_rs::{AgentConfig, AgentHarness};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = AgentConfig::new("deepseek-v4-pro");
    config.base_url = "https://api.deepseek.com/v1".into();
    config.reasoning_effort = Some("high".into());

    let mut agent = AgentHarness::new(config)?;
    let response = agent.run("List the Rust files in this project.")?;
    println!("{response}");
    Ok(())
}
```

Use `AgentHarness::run_with_callback` to receive typed `Event` values and to
enable streaming. Background calls emit the usual `ToolCall` and an immediate
`ToolResult` acknowledgement. Completion emits `BackgroundToolResult`, including
`tool_call_id`, `name`, `arguments`, and `result`. Completed output is added to
conversation history as a notification before the next model request; an
in-flight model response continues normally.

```rust,no_run
use harness_rs::{AgentConfig, AgentHarness};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AgentConfig::new("deepseek-v4-pro");
    let mut agent = AgentHarness::new(config)?;
    let response = agent.run_with_callback(
        "Refactor the cli module.",
        true,
        &mut |event| println!("{event:?}"),
    )?;
    println!("{response}");
    Ok(())
}
```

## Test

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

## Security

Like the Python implementation, this port is intentionally not sandboxed. The
model can read and overwrite any file available to the current user and execute
arbitrary shell commands. Run it only in a controlled environment.
