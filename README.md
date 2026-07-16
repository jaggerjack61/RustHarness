# Harness Rust Port

This directory contains a standalone Rust port of the 
https://github.com/jaggerjack61/Harness repository's Python
agent. It keeps the OpenAI-compatible chat-completions protocol, streaming
SSE responses, reasoning fields, tool calling, retries, context
summarization, token events, markdown rendering, and the unrestricted
`read`, `write`, `edit`, and `bash` tools.

## Features

- **OpenAI-compatible** — works with OpenAI, DeepSeek, local models, or any
  chat-completions + tool-calling API.
- **Four built-in tools** — `read`, `write`, `edit`, and `bash` give the model
  full access to the filesystem and shell.
- **Reasoning / chain-of-thought** — displays `reasoning_content` and similar
  fields from reasoning models as dimmed terminal output.
- **Markdown rendering** — responses are rendered as rich Markdown with
  syntax-highlighted code blocks, headings, lists, tables, etc.
- **Streaming** — responses stream token-by-token; the final answer is then
  rendered as formatted Markdown.
- **Configurable reasoning effort** — pass
  `--reasoning-effort low|medium|high|xhigh|max` for models that support it.
- **Live events** — tool calls, tool results, thinking blocks, and a live
  token/context status bar are shown in real time.
- **Conversation history** — context is remembered across turns; use `/clear`
  to reset.
- **Automatic context management** — history is trimmed/summarized when it
  approaches the configured context window limit.
- **Retry with exponential backoff** — transient API errors (429, 5xx,
  timeouts) are retried automatically up to 3 times.
- **Real-time token tracking** — cumulative input/output/cache token counts
  and context-window usage are shown in the status bar.
- **Model switcher** — `/models` fetches the provider's model list and lets
  you switch at runtime.
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
  --api-key "sk-fPBvz8XkDa5mVa74TQyuxb1CwcrI3sRIweb0UR1SwPYK8lMcFgupIpEFxuBbgdI7"
```

The CLI options and environment variables mirror the Python implementation.

| CLI option | Environment variable | Description |
|------------|----------------------|-------------|
| `-m`, `--model` | `HARNESS_MODEL` | Model name (default `deepseek-v4-pro`) |
| `-k`, `--api-key` | `OPENAI_API_KEY` | API key |
| `-u`, `--base-url` | `HARNESS_BASE_URL` | Provider base URL (default `https://api.openai.com/v1`) |
| `-d`, `--dir` | — | Working directory |
| `--max-turns` | `HARNESS_MAX_TURNS` | Maximum tool/response turns (default `1000`) |
| `--system-prompt` | `HARNESS_PROMPT` | Override the system prompt |
| `--reasoning-effort` | — | `low`, `medium`, `high`, `xhigh`, or `max` (default `high`) |
| `--context-window` | `HARNESS_CONTEXT_WINDOW` | Token context window (default `1_000_000`) |
| `--stream` / `--no-stream` | — | Force streaming on or off |
| `--no-markdown` | — | Disable markdown rendering |
| — | `HARNESS_SHELL` | Windows shell override (default: `pwsh` if available, otherwise `powershell`) |

### Interactive commands

While the agent is running, you can type:

| Command | Description |
|---------|-------------|
| `/exit` | Quit the session |
| `/clear` | Clear conversation history and redraw the welcome box |
| `/stream` | Toggle streaming mode |
| `/models` | Fetch and switch models from the provider |
| `/reasoning` | Switch reasoning effort |
| `/context` | Paste a custom context block |
| `/context clear` | Remove the custom context |
| `/context show` | Show the current custom context |

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
enable streaming.

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
