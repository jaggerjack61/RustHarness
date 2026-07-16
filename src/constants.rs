pub const DEFAULT_MAX_TURNS: usize = 1_000;
pub const DEFAULT_CONTEXT_WINDOW: i64 = 1_000_000;
pub const DEFAULT_MODEL: &str = "deepseek-v4-pro";
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_REASONING_EFFORT: &str = "high";

pub const REASONING_OPTIONS: &[&str] = &["low", "medium", "high", "xhigh", "max"];
pub const NONSTANDARD_REASONING_EFFORTS: &[&str] = &["xhigh", "max"];

pub const MAX_RETRIES: usize = 3;
pub const RETRY_BASE_DELAY_SECS: f64 = 1.0;

pub const CONTEXT_WINDOW_TRIM_THRESHOLD: f64 = 0.8;
pub const RECENT_TURNS_TO_KEEP: usize = 6;
pub const SUMMARY_MAX_TOKENS: u64 = 500;

pub const MAX_OUTPUT_LINES: usize = 1_000;
pub const MAX_OUTPUT_BYTES: usize = 100_000;
pub const BASH_TIMEOUT_SECS: u64 = 60;

pub const FETCH_TIMEOUT_SECS: u64 = 10;
pub const DISPLAY_TRUNCATION_LIMIT: usize = 5;

pub const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an expert coding assistant. You have access to the following tools:

- **read** — Read file contents. Use for examining files, with optional offset/limit for large files.
- **write** — Create or overwrite a file. Automatically creates parent directories.
- **edit** — Make precise text replacements in files. Each edit specifies oldText and newText.
- **bash** — Execute shell commands (PowerShell on Windows, bash on Unix). Use for listing files, running tests, installing packages, etc.

On Windows, all bash commands run in PowerShell. Use PowerShell commands (ls, Get-Content, Select-String, Get-ChildItem, etc.). Powershell uses ; instead of &&.
Always use these tools when you need to interact with the file system or execute commands.
Your tool results may be truncated for display — keep your commands and file reads concise.
When searching or listing files, limit output with Select-Object -First, | head, or similar.
Be concise and helpful."#;
