pub const DEFAULT_MAX_TURNS: usize = 1_000;
pub const DEFAULT_CONTEXT_WINDOW: i64 = 1_000_000;
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
- **bash** — Execute commands using the platform shell described below. Use for listing files, running tests, installing packages, etc.

All tools accept an optional background boolean, defaulting to false. Use background: true for independent work that can run while you continue. Background calls return a started acknowledgement; their results arrive automatically in a later message with the tool call ID.

Always use these tools when you need to interact with the file system or execute commands.
Your tool results may be truncated for display — keep your commands and file reads concise.
When searching or listing files, limit output using commands supported by the platform shell.
Be concise and helpful."#;

pub fn default_system_prompt() -> String {
    let mut prompt = system_prompt_for_platform(std::env::consts::OS);
    if crate::tools::tool_definitions()
        .iter()
        .any(|definition| definition["function"]["name"] == "find")
    {
        prompt.push_str("\n\nUse the find tool to search file contents. It returns matching lines with filenames and line numbers; narrow the path or pattern to keep results manageable.");
    }
    prompt
}

fn system_prompt_for_platform(platform: &str) -> String {
    let shell_guidance = match platform {
        "windows" => format!(
            "Platform: Windows. The bash tool runs commands through {} with -NoProfile -Command. Use PowerShell syntax such as Get-Content, Select-String, Get-ChildItem, and Select-Object -First. Use ; to separate commands; do not assume && is available in Windows PowerShell 5.1.",
            crate::tools::get_windows_shell()
        ),
        "macos" => "Platform: macOS. The bash tool executes /bin/sh -c, regardless of the user's interactive shell. Use POSIX shell syntax and macOS/BSD command options. Limit output with head. Do not assume Bash-specific syntax or GNU utilities are available.".to_owned(),
        "linux" => "Platform: Linux. The bash tool executes /bin/sh -c, regardless of the user's interactive shell. Use POSIX shell syntax and limit output with head. Do not assume Bash-specific syntax is supported.".to_owned(),
        other => format!(
            "Platform: {other}. The bash tool executes /bin/sh -c. Use POSIX shell syntax and limit output with head. Do not assume Bash-specific syntax is supported."
        ),
    };
    format!("{DEFAULT_SYSTEM_PROMPT}\n\n{shell_guidance}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompts_describe_each_platform_shell() {
        for platform in ["macos", "linux", "freebsd"] {
            let prompt = system_prompt_for_platform(platform);
            assert!(prompt.contains("/bin/sh -c"));
            assert!(!prompt.contains("PowerShell"));
            assert!(!prompt.contains("Select-Object"));
        }
        assert!(system_prompt_for_platform("macos").contains("macOS/BSD"));
        let windows = system_prompt_for_platform("windows");
        assert!(windows.contains("PowerShell"));
        assert!(windows.contains("-NoProfile -Command"));
        assert!(!windows.contains("/bin/sh"));
        let prompt = default_system_prompt();
        assert!(prompt.starts_with(&system_prompt_for_platform(std::env::consts::OS)));
        let find_available = crate::tools::tool_definitions()
            .iter()
            .any(|definition| definition["function"]["name"] == "find");
        assert_eq!(prompt.contains("Use the find tool"), find_available);
    }
}
