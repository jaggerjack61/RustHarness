use std::borrow::Borrow;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use path_clean::PathClean;
use serde_json::{Map, Value, json};
use wait_timeout::ChildExt;

use crate::cancel::CancelToken;
use crate::constants::{BASH_TIMEOUT_SECS, MAX_OUTPUT_BYTES, MAX_OUTPUT_LINES};

/// How often a running command checks for cancellation.
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(50);

static WINDOWS_SHELL: OnceLock<String> = OnceLock::new();

/// Return the OpenAI function-calling schemas for all supported tools.
pub fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read the contents of a file. Supports reading portions of large files with offset and limit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read (relative or absolute)."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Line number to start reading from (1-indexed)."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read."
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "write",
                "description": "Create or overwrite a file with the given content. Automatically creates parent directories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write (relative or absolute)."
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Make precise, targeted edits to a file. Each edit specifies oldText (exact text to find) and newText (replacement). Multiple edits can be applied in one call.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to edit (relative or absolute)."
                        },
                        "edits": {
                            "type": "array",
                            "description": "List of edits to apply. Each edit has oldText (exact text to replace) and newText (replacement text).",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "oldText": {
                                        "type": "string",
                                        "description": "Exact text to find and replace."
                                    },
                                    "newText": {
                                        "type": "string",
                                        "description": "Replacement text."
                                    }
                                },
                                "required": ["oldText", "newText"]
                            }
                        }
                    },
                    "required": ["path", "edits"]
                }
            }
        }),
        json!({
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a shell command and return its output (stdout and stderr combined).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute."
                        }
                    },
                    "required": ["command"]
                }
            }
        }),
    ]
}

/// Discard a response which is too large to safely return to the model.
pub fn enforce_output_limits(result: impl Into<String>, tool_name: &str) -> String {
    let result = result.into();
    if result.is_empty() {
        return result;
    }

    let byte_count = result.len();
    if byte_count > MAX_OUTPUT_BYTES {
        return format!(
            "Error: The '{tool_name}' tool response exceeded the {}-byte limit ({} UTF-8 bytes returned). The output has been discarded. Please try again with a narrower command or file slice.",
            format_number(MAX_OUTPUT_BYTES),
            format_number(byte_count),
        );
    }

    let line_count = result.bytes().filter(|byte| *byte == b'\n').count()
        + if result.ends_with('\n') { 0 } else { 1 };
    if line_count <= MAX_OUTPUT_LINES {
        return result;
    }

    format!(
        "Error: The '{tool_name}' tool response exceeded the {}-line limit ({} lines returned). The output has been discarded. Please try again using line-limiting commands such as `head -n <N>`, `tail -n <N>`, or `Select-Object -First <N>`.",
        format_number(MAX_OUTPUT_LINES),
        format_number(line_count),
    )
}

/// Compatibility alias for callers using the Python helper's former name.
pub fn enforce_line_limit(result: impl Into<String>, tool_name: &str) -> String {
    enforce_output_limits(result, tool_name)
}

fn format_number(value: usize) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, byte) in digits.bytes().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            formatted.push(',');
        }
        formatted.push(char::from(byte));
    }
    formatted
}

/// Resolve a path relative to `cwd`, cleaning `.` and `..` and following any
/// symlinks in the existing part of the path.
pub fn resolve_path(path: &str, cwd: Option<&Path>) -> io::Result<PathBuf> {
    let path = Path::new(path);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        let current_dir = std::env::current_dir()?;
        let base = cwd
            .map(Path::to_path_buf)
            .unwrap_or_else(|| current_dir.clone());
        let base = if base.is_absolute() {
            base
        } else {
            current_dir.join(base)
        };
        base.join(path)
    }
    .clean();

    if let Ok(canonical) = fs::canonicalize(&absolute) {
        return Ok(canonical);
    }

    // `canonicalize` requires the leaf to exist. Resolve the deepest existing
    // ancestor so writes through a symlinked directory still match Path.resolve.
    let mut ancestor = absolute.as_path();
    let mut missing = Vec::<OsString>::new();
    while !ancestor.exists() {
        let Some(name) = ancestor.file_name() else {
            return Ok(absolute);
        };
        missing.push(name.to_os_string());
        let Some(parent) = ancestor.parent() else {
            return Ok(absolute);
        };
        ancestor = parent;
    }

    let mut resolved = fs::canonicalize(ancestor)?;
    for component in missing.iter().rev() {
        resolved.push(component);
    }
    Ok(resolved.clean())
}

/// Read UTF-8 text, replacing invalid sequences and normalizing all line
/// endings to `\n`, as Python's text mode does.
pub fn read_file(
    path: &str,
    offset: Option<i64>,
    limit: Option<i64>,
    cwd: Option<&Path>,
) -> String {
    if let Some(offset) = offset
        && offset < 1
    {
        return format!("Error: offset must be an integer >= 1 (got {offset})");
    }
    if let Some(limit) = limit
        && limit < 1
    {
        return format!("Error: limit must be an integer >= 1 (got {limit})");
    }

    let resolved = match resolve_path(path, cwd) {
        Ok(path) => path,
        Err(error) => return format!("Error reading file: {error}"),
    };
    if !resolved.exists() {
        return format!("Error: File not found: {path}");
    }

    let file = match File::open(&resolved) {
        Ok(file) => file,
        Err(error) => return format!("Error reading file: {error}"),
    };
    let mut reader = BufReader::new(file);
    let start_line = offset.unwrap_or(1) as usize;
    let limit = limit.map(|value| value as usize);
    let mut result = String::new();
    let mut raw_chunk = Vec::new();
    let mut byte_count = 0usize;
    let mut selected_lines = 0usize;
    let mut current_line = 1usize;

    loop {
        let bytes_read = match read_universal_line_chunk(&mut reader, &mut raw_chunk) {
            Ok(bytes_read) => bytes_read,
            Err(error) => return format!("Error reading file: {error}"),
        };
        if bytes_read == 0 {
            break;
        }

        if current_line >= start_line {
            if selected_lines >= MAX_OUTPUT_LINES {
                return format!(
                    "Error: The 'read' tool response exceeded the {}-line limit. The output has been discarded. Please try again with a narrower file slice.",
                    format_number(MAX_OUTPUT_LINES),
                );
            }

            let chunk = String::from_utf8_lossy(&raw_chunk);
            let chunk_bytes = chunk.len();
            if byte_count + chunk_bytes > MAX_OUTPUT_BYTES {
                return format!(
                    "Error: The 'read' tool response exceeded the {}-byte limit. The output has been discarded. Please try again with a narrower file slice.",
                    format_number(MAX_OUTPUT_BYTES),
                );
            }
            result.push_str(&chunk);
            byte_count += chunk_bytes;
        }

        if raw_chunk.ends_with(b"\n") {
            if current_line >= start_line {
                selected_lines += 1;
                if limit.is_some_and(|limit| selected_lines >= limit) {
                    break;
                }
            }
            current_line += 1;
        }
    }

    result
}

// Read at most one universal-newline text line, with a hard chunk size so a
// single enormous line cannot be allocated before the output guard runs.
fn read_universal_line_chunk<R: BufRead>(
    reader: &mut R,
    output: &mut Vec<u8>,
) -> io::Result<usize> {
    output.clear();
    let chunk_limit = MAX_OUTPUT_BYTES + 1;

    while output.len() < chunk_limit {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            break;
        }

        if let Some(index) = available
            .iter()
            .position(|byte| matches!(*byte, b'\n' | b'\r'))
        {
            let room = chunk_limit - output.len();
            if index >= room {
                output.extend_from_slice(&available[..room]);
                reader.consume(room);
                break;
            }

            let newline = available[index];
            output.extend_from_slice(&available[..index]);
            reader.consume(index + 1);
            if newline == b'\r' {
                let after_cr = reader.fill_buf()?;
                if after_cr.first() == Some(&b'\n') {
                    reader.consume(1);
                }
            }
            output.push(b'\n');
            break;
        }

        let take = available.len().min(chunk_limit - output.len());
        output.extend_from_slice(&available[..take]);
        reader.consume(take);
    }

    Ok(output.len())
}

/// Create or overwrite a UTF-8 file while preserving supplied line endings.
pub fn write_file(path: &str, content: &str, cwd: Option<&Path>) -> String {
    let resolved = match resolve_path(path, cwd) {
        Ok(path) => path,
        Err(error) => return format!("Error writing file: {error}"),
    };

    if let Some(parent) = resolved.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        return format!("Error writing file: {error}");
    }
    match fs::write(&resolved, content.as_bytes()) {
        Ok(()) => format!("Successfully wrote to {path}."),
        Err(error) => format!("Error writing file: {error}"),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextEdit {
    pub old_text: String,
    pub new_text: String,
}

impl TextEdit {
    pub fn new(old_text: impl Into<String>, new_text: impl Into<String>) -> Self {
        Self {
            old_text: old_text.into(),
            new_text: new_text.into(),
        }
    }
}

#[derive(Debug)]
struct PositionedEdit {
    start: usize,
    end: usize,
    index: usize,
    replacement: String,
}

/// Apply exact, unique replacements without modifying the file if any edit is
/// invalid, missing, ambiguous, or overlaps another edit.
pub fn edit_file(path: &str, edits: &[TextEdit], cwd: Option<&Path>) -> String {
    if edits.is_empty() {
        return "Error: edits must be a non-empty list.".to_owned();
    }
    for (index, edit) in edits.iter().enumerate() {
        if edit.old_text.is_empty() {
            return format!("Error: Edit {index}: oldText must be a non-empty string.");
        }
    }

    let resolved = match resolve_path(path, cwd) {
        Ok(path) => path,
        Err(error) => return format!("Error reading file: {error}"),
    };
    if !resolved.exists() {
        return format!("Error: File not found: {path}");
    }

    let bytes = match fs::read(&resolved) {
        Ok(bytes) => bytes,
        Err(error) => return format!("Error reading file: {error}"),
    };
    let text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => return format!("Error reading file: {error}"),
    };

    let file_newline = dominant_newline(&text).unwrap_or("\n");
    let mut normalized = None::<(String, Vec<usize>)>;
    let mut positioned = Vec::with_capacity(edits.len());

    for (index, edit) in edits.iter().enumerate() {
        let exact_matches: Vec<_> = text.match_indices(&edit.old_text).collect();
        if exact_matches.len() > 1 {
            return format!(
                "Error: Edit {index}: oldText found {} times - provide more context to uniquely identify the location.",
                exact_matches.len(),
            );
        }
        if let Some((start, _)) = exact_matches.first() {
            positioned.push(PositionedEdit {
                start: *start,
                end: *start + edit.old_text.len(),
                index,
                replacement: edit.new_text.clone(),
            });
            continue;
        }

        let (normalized_text, raw_offsets) =
            normalized.get_or_insert_with(|| normalize_with_raw_offsets(&text));
        let normalized_old = normalize_newlines(&edit.old_text);
        let normalized_matches: Vec<_> = normalized_text.match_indices(&normalized_old).collect();
        if normalized_matches.is_empty() {
            return format!("Error: Edit {index}: oldText not found in file.");
        }
        if normalized_matches.len() > 1 {
            return format!(
                "Error: Edit {index}: oldText found {} times - provide more context to uniquely identify the location.",
                normalized_matches.len(),
            );
        }

        let normalized_start = normalized_matches[0].0;
        let normalized_end = normalized_start + normalized_old.len();
        let start = raw_offsets[normalized_start];
        let end = raw_offsets[normalized_end];
        let replacement_newline = dominant_newline(&text[start..end]).unwrap_or(file_newline);
        let replacement = if edit.new_text.contains('\r') {
            edit.new_text.clone()
        } else {
            edit.new_text.replace('\n', replacement_newline)
        };
        positioned.push(PositionedEdit {
            start,
            end,
            index,
            replacement,
        });
    }

    positioned.sort_by_key(|edit| edit.start);
    for pair in positioned.windows(2) {
        if pair[1].start < pair[0].end {
            return format!(
                "Error: Edit {} overlaps with edit {}.",
                pair[1].index, pair[0].index,
            );
        }
    }

    let replacement_bytes: usize = positioned.iter().map(|edit| edit.replacement.len()).sum();
    let removed_bytes: usize = positioned.iter().map(|edit| edit.end - edit.start).sum();
    let mut updated = String::with_capacity(text.len() - removed_bytes + replacement_bytes);
    let mut cursor = 0usize;
    for edit in positioned {
        updated.push_str(&text[cursor..edit.start]);
        updated.push_str(&edit.replacement);
        cursor = edit.end;
    }
    updated.push_str(&text[cursor..]);

    match fs::write(&resolved, updated.as_bytes()) {
        Ok(()) => format!("Successfully applied {} edit(s) to {path}.", edits.len()),
        Err(error) => format!("Error writing file: {error}"),
    }
}

fn normalize_newlines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

// The offset map has one raw byte offset for every normalized byte boundary.
// Match indices are UTF-8 boundaries, so only those entries are consumed.
fn normalize_with_raw_offsets(text: &str) -> (String, Vec<usize>) {
    let bytes = text.as_bytes();
    let mut normalized = String::with_capacity(text.len());
    let mut raw_offsets = Vec::with_capacity(text.len() + 1);
    raw_offsets.push(0);
    let mut raw = 0usize;

    while raw < bytes.len() {
        if bytes[raw] == b'\r' {
            let consumed = if bytes.get(raw + 1) == Some(&b'\n') {
                2
            } else {
                1
            };
            raw += consumed;
            normalized.push('\n');
            raw_offsets.push(raw);
            continue;
        }

        let character = text[raw..].chars().next().expect("raw is in bounds");
        let width = character.len_utf8();
        normalized.push(character);
        for byte in 1..=width {
            raw_offsets.push(raw + byte);
        }
        raw += width;
    }

    (normalized, raw_offsets)
}

fn dominant_newline(text: &str) -> Option<&'static str> {
    let crlf = text.match_indices("\r\n").count();
    let lf = text.bytes().filter(|byte| *byte == b'\n').count() - crlf;
    let cr = text.bytes().filter(|byte| *byte == b'\r').count() - crlf;
    if crlf == 0 && lf == 0 && cr == 0 {
        None
    } else if crlf >= lf && crlf >= cr {
        Some("\r\n")
    } else if lf >= cr {
        Some("\n")
    } else {
        Some("\r")
    }
}

/// Execute a command using PowerShell on Windows and `/bin/sh` elsewhere.
pub fn run_bash(command: &str, cwd: Option<&Path>) -> String {
    if command.is_empty() {
        return "Error: command must be a non-empty string.".to_owned();
    }
    run_bash_with_timeout(command, cwd, Duration::from_secs(BASH_TIMEOUT_SECS), None)
}

/// Wait for `child` until it exits, `timeout` elapses (`Ok(None)`), or `cancel` fires.
fn wait_cancellable(
    child: &mut std::process::Child,
    timeout: Duration,
    cancel: Option<&CancelToken>,
) -> io::Result<WaitOutcome> {
    let deadline = Instant::now() + timeout;
    loop {
        if cancel.is_some_and(CancelToken::is_cancelled) {
            return Ok(WaitOutcome::Cancelled);
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(WaitOutcome::TimedOut);
        }
        if let Some(status) = child.wait_timeout(remaining.min(CANCEL_POLL_INTERVAL))? {
            return Ok(WaitOutcome::Exited(status));
        }
    }
}

enum WaitOutcome {
    Exited(ExitStatus),
    TimedOut,
    Cancelled,
}

/// Kill the shell and, on Unix, every process in its process group.
fn kill_tree(child: &mut std::process::Child) {
    #[cfg(unix)]
    if let Ok(pid) = i32::try_from(child.id()) {
        // SAFETY: kill(2) has no memory-safety preconditions; a negative pid targets
        // the process group created for this child.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn run_bash_with_timeout(
    command: &str,
    cwd: Option<&Path>,
    timeout: Duration,
    cancel: Option<&CancelToken>,
) -> String {
    let mut process = shell_command(command);
    process
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut process, 0);
    if let Some(cwd) = cwd {
        process.current_dir(cwd);
    }

    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(error) => return format!("Error executing command: {error}"),
    };
    let stdout = child.stdout.take().expect("stdout was configured as piped");
    let stderr = child.stderr.take().expect("stderr was configured as piped");
    let stdout_reader = thread::spawn(move || read_all_bounded(stdout));
    let stderr_reader = thread::spawn(move || read_all_bounded(stderr));

    let status = match wait_cancellable(&mut child, timeout, cancel) {
        Ok(WaitOutcome::Exited(status)) => status,
        Ok(WaitOutcome::Cancelled) => {
            kill_tree(&mut child);
            drop(stdout_reader);
            drop(stderr_reader);
            return "Error: Command interrupted by the user.".to_owned();
        }
        Ok(WaitOutcome::TimedOut) => {
            kill_tree(&mut child);
            // Readers are detached deliberately: a grandchild may still hold a
            // copied pipe handle after the shell itself has been terminated.
            drop(stdout_reader);
            drop(stderr_reader);
            return format!(
                "Error: Command timed out after {} seconds.",
                timeout.as_secs()
            );
        }
        Err(error) => {
            kill_tree(&mut child);
            return format!("Error executing command: {error}");
        }
    };

    let (mut output, stdout_count) = match join_reader(stdout_reader) {
        Ok(output) => output,
        Err(error) => return format!("Error executing command: {error}"),
    };
    let (stderr, stderr_count) = match join_reader(stderr_reader) {
        Ok(output) => output,
        Err(error) => return format!("Error executing command: {error}"),
    };
    let total_bytes = stdout_count.saturating_add(stderr_count);
    if total_bytes > MAX_OUTPUT_BYTES {
        return format!(
            "Error: The 'bash' tool response exceeded the {}-byte limit ({} UTF-8 bytes returned). The output has been discarded. Please try again with a narrower command or file slice.",
            format_number(MAX_OUTPUT_BYTES),
            format_number(total_bytes),
        );
    }
    output.extend_from_slice(&stderr);

    let mut output = String::from_utf8_lossy(&output).into_owned();
    if !status.success() {
        output.push_str(&format!("\n[Exit code: {}]", exit_code(status)));
    }
    let output = output.trim();
    if output.is_empty() {
        "(no output)".to_owned()
    } else {
        output.to_owned()
    }
}

fn read_all_bounded(mut reader: impl Read) -> io::Result<(Vec<u8>, usize)> {
    let mut output = Vec::with_capacity(MAX_OUTPUT_BYTES.min(8 * 1024));
    let mut total = 0usize;
    let mut buffer = [0u8; 8 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read);
        if output.len() <= MAX_OUTPUT_BYTES {
            let remaining = MAX_OUTPUT_BYTES + 1 - output.len();
            output.extend_from_slice(&buffer[..read.min(remaining)]);
        }
    }
    Ok((output, total))
}

fn join_reader(
    handle: thread::JoinHandle<io::Result<(Vec<u8>, usize)>>,
) -> io::Result<(Vec<u8>, usize)> {
    match handle.join() {
        Ok(result) => result,
        Err(_) => Err(io::Error::other("shell output reader panicked")),
    }
}

fn exit_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        -status.signal().unwrap_or(1)
    }
    #[cfg(not(unix))]
    {
        1
    }
}

#[cfg(windows)]
fn shell_command(command: &str) -> Command {
    let mut process = Command::new(get_windows_shell());
    process.args(["-NoProfile", "-Command", command]);
    process
}

#[cfg(not(windows))]
fn shell_command(command: &str) -> Command {
    let mut process = Command::new("/bin/sh");
    process.args(["-c", command]);
    process
}

/// Select PowerShell 7 when available, then Windows PowerShell 5.1.
pub fn get_windows_shell() -> &'static str {
    WINDOWS_SHELL
        .get_or_init(|| choose_windows_shell(std::env::var("HARNESS_SHELL").ok(), command_exists))
        .as_str()
}

fn choose_windows_shell(
    override_shell: Option<String>,
    mut available: impl FnMut(&str) -> bool,
) -> String {
    if let Some(shell) = override_shell.filter(|shell| !shell.is_empty()) {
        return shell;
    }
    ["pwsh", "powershell"]
        .into_iter()
        .find(|candidate| available(candidate))
        .unwrap_or("powershell")
        .to_owned()
}

fn command_exists(command: &str) -> bool {
    let command_path = Path::new(command);
    if command_path.components().count() > 1 {
        return command_path.is_file();
    }

    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    let extensions = std::env::var_os("PATHEXT")
        .map(|value| {
            value
                .to_string_lossy()
                .split(';')
                .filter(|extension| !extension.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![".COM".into(), ".EXE".into(), ".BAT".into(), ".CMD".into()]);

    std::env::split_paths(&path).any(|directory| {
        let plain = directory.join(command);
        plain.is_file()
            || (command_path.extension().is_none()
                && extensions
                    .iter()
                    .any(|extension| directory.join(format!("{command}{extension}")).is_file()))
    })
}

#[derive(Clone, Debug, Default)]
pub struct ToolRegistry {
    pub working_dir: Option<PathBuf>,
    /// Interrupts long-running commands when the user cancels a turn.
    pub cancel: CancelToken,
}

impl ToolRegistry {
    pub fn new(working_dir: Option<PathBuf>) -> Self {
        Self {
            working_dir,
            cancel: CancelToken::new(),
        }
    }

    pub fn with_working_dir(working_dir: impl Into<PathBuf>) -> Self {
        Self::new(Some(working_dir.into()))
    }

    pub fn get_definitions(&self) -> Vec<Value> {
        tool_definitions()
    }

    /// Execute a named tool. Invalid model-provided arguments are returned as
    /// tool errors rather than panicking or aborting the agent loop.
    pub fn execute<A: Borrow<Value>>(&self, name: &str, arguments: A) -> String {
        if !matches!(name, "read" | "write" | "edit" | "bash") {
            return format!("Error: Unknown tool: {name}");
        }
        let arguments = arguments.borrow();
        let Some(arguments) = arguments.as_object() else {
            return "Error: tool arguments must be a dict.".to_owned();
        };

        let result = match name {
            "read" => self.execute_read(arguments),
            "write" => self.execute_write(arguments),
            "edit" => self.execute_edit(arguments),
            "bash" => self.execute_bash(arguments),
            _ => unreachable!("tool name was validated above"),
        };
        enforce_output_limits(result, name)
    }

    fn execute_read(&self, arguments: &Map<String, Value>) -> String {
        let Some(path) = arguments.get("path").and_then(Value::as_str) else {
            return "Error: 'path' is required and must be a string for read tool.".to_owned();
        };
        let offset = match positive_integer_argument(arguments, "offset", "read") {
            Ok(value) => value,
            Err(error) => return error,
        };
        let limit = match positive_integer_argument(arguments, "limit", "read") {
            Ok(value) => value,
            Err(error) => return error,
        };
        read_file(path, offset, limit, self.working_dir.as_deref())
    }

    fn execute_write(&self, arguments: &Map<String, Value>) -> String {
        let Some(path) = arguments.get("path").and_then(Value::as_str) else {
            return "Error: 'path' is required and must be a string for write tool.".to_owned();
        };
        let Some(content) = arguments.get("content").and_then(Value::as_str) else {
            return "Error: 'content' is required and must be a string for write tool.".to_owned();
        };
        write_file(path, content, self.working_dir.as_deref())
    }

    fn execute_edit(&self, arguments: &Map<String, Value>) -> String {
        let Some(path) = arguments.get("path").and_then(Value::as_str) else {
            return "Error: 'path' is required and must be a string for edit tool.".to_owned();
        };
        let Some(raw_edits) = arguments.get("edits").and_then(Value::as_array) else {
            return "Error: 'edits' is required and must be a non-empty list for edit tool."
                .to_owned();
        };
        if raw_edits.is_empty() {
            return "Error: 'edits' is required and must be a non-empty list for edit tool."
                .to_owned();
        }

        let mut edits = Vec::with_capacity(raw_edits.len());
        for (index, edit) in raw_edits.iter().enumerate() {
            let Some(edit) = edit.as_object() else {
                return format!("Error: edit {index} must be a dict.");
            };
            let Some(old_text) = edit.get("oldText").and_then(Value::as_str) else {
                return format!("Error: edit {index} oldText must be a non-empty string.");
            };
            if old_text.is_empty() {
                return format!("Error: edit {index} oldText must be a non-empty string.");
            }
            let Some(new_text) = edit.get("newText").and_then(Value::as_str) else {
                return format!("Error: edit {index} newText must be a string.");
            };
            edits.push(TextEdit::new(old_text, new_text));
        }
        edit_file(path, &edits, self.working_dir.as_deref())
    }

    fn execute_bash(&self, arguments: &Map<String, Value>) -> String {
        let Some(command) = arguments.get("command").and_then(Value::as_str) else {
            return "Error: 'command' is required and must be a non-empty string for bash tool."
                .to_owned();
        };
        if command.is_empty() {
            return "Error: 'command' is required and must be a non-empty string for bash tool."
                .to_owned();
        }
        run_bash_with_timeout(
            command,
            self.working_dir.as_deref(),
            Duration::from_secs(BASH_TIMEOUT_SECS),
            Some(&self.cancel),
        )
    }
}

fn positive_integer_argument(
    arguments: &Map<String, Value>,
    key: &str,
    tool: &str,
) -> Result<Option<i64>, String> {
    let Some(value) = arguments.get(key) else {
        return Ok(None);
    };
    match value.as_i64() {
        Some(value) if value >= 1 => Ok(Some(value)),
        _ => Err(format!(
            "Error: '{key}' must be an integer >= 1 for {tool} tool."
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn definitions_match_openai_schema() {
        let definitions = tool_definitions();
        let names: Vec<_> = definitions
            .iter()
            .map(|definition| definition["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["read", "write", "edit", "bash"]);
        for definition in definitions {
            assert_eq!(definition["type"], "function");
            assert!(definition["function"]["description"].is_string());
            assert_eq!(definition["function"]["parameters"]["type"], "object");
        }
    }

    #[test]
    fn output_guards_count_utf8_bytes_and_lines() {
        assert_eq!(enforce_output_limits(String::new(), "read"), "");
        assert_eq!(
            enforce_output_limits("x".repeat(MAX_OUTPUT_BYTES), "bash").len(),
            MAX_OUTPUT_BYTES
        );
        let oversized = enforce_output_limits("\u{e9}".repeat(MAX_OUTPUT_BYTES / 2 + 1), "read");
        assert!(oversized.starts_with("Error:"));
        assert!(oversized.contains("UTF-8 bytes"));

        let exact_lines = "x\n".repeat(MAX_OUTPUT_LINES);
        assert_eq!(
            enforce_output_limits(exact_lines.clone(), "bash"),
            exact_lines
        );
        let too_many = enforce_output_limits("x\n".repeat(MAX_OUTPUT_LINES + 1), "bash");
        assert!(too_many.contains("1,000-line limit"));
        assert!(!too_many.contains("x\nx"));
    }

    #[test]
    fn resolves_relative_clean_paths() {
        let directory = tempdir().unwrap();
        fs::create_dir(directory.path().join("nested")).unwrap();
        let resolved = resolve_path("nested/../new.txt", Some(directory.path())).unwrap();
        let expected = fs::canonicalize(directory.path()).unwrap().join("new.txt");
        assert_eq!(resolved, expected);
    }

    #[test]
    fn reads_slices_and_normalizes_all_newlines() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("lines.txt"),
            b"line1\r\nline2\rline3\nline4",
        )
        .unwrap();

        assert_eq!(
            read_file("lines.txt", Some(2), Some(2), Some(directory.path())),
            "line2\nline3\n"
        );
        assert_eq!(
            read_file("lines.txt", Some(3), None, Some(directory.path())),
            "line3\nline4"
        );
    }

    #[test]
    fn read_validates_ranges_and_missing_files() {
        let directory = tempdir().unwrap();
        assert!(read_file("x", Some(0), None, Some(directory.path())).contains("offset"));
        assert!(read_file("x", None, Some(-1), Some(directory.path())).contains("limit"));
        assert!(
            read_file("missing", None, None, Some(directory.path())).contains("File not found")
        );
    }

    #[test]
    fn read_replaces_invalid_utf8() {
        let directory = tempdir().unwrap();
        fs::write(directory.path().join("binary.txt"), b"a\xffb").unwrap();
        assert_eq!(
            read_file("binary.txt", None, None, Some(directory.path())),
            "a\u{fffd}b"
        );
    }

    #[test]
    fn streamed_read_enforces_byte_and_line_limits() {
        let directory = tempdir().unwrap();
        fs::write(
            directory.path().join("wide.txt"),
            "x".repeat(MAX_OUTPUT_BYTES + 1),
        )
        .unwrap();
        assert!(read_file("wide.txt", None, None, Some(directory.path())).contains("byte limit"));

        fs::write(
            directory.path().join("long.txt"),
            "x\n".repeat(MAX_OUTPUT_LINES + 1),
        )
        .unwrap();
        assert!(read_file("long.txt", None, None, Some(directory.path())).contains("line limit"));
    }

    #[test]
    fn writes_nested_files_and_preserves_line_endings() {
        let directory = tempdir().unwrap();
        let result = write_file(
            "nested/file.txt",
            "first\r\nsecond\n",
            Some(directory.path()),
        );
        assert!(result.contains("Successfully"));
        assert_eq!(
            fs::read(directory.path().join("nested/file.txt")).unwrap(),
            b"first\r\nsecond\n"
        );
    }

    #[test]
    fn applies_multiple_non_overlapping_edits() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("edit.txt");
        fs::write(&path, "alpha beta gamma").unwrap();
        let edits = [
            TextEdit::new("alpha", "ALPHA"),
            TextEdit::new("gamma", "GAMMA"),
        ];
        assert!(edit_file("edit.txt", &edits, Some(directory.path())).contains("Successfully"));
        assert_eq!(fs::read_to_string(path).unwrap(), "ALPHA beta GAMMA");
    }

    #[test]
    fn normalized_edit_preserves_crlf_in_match_and_replacement() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("windows.txt");
        fs::write(&path, b"first\r\nsecond\r\nthird\r\n").unwrap();
        let old = read_file("windows.txt", Some(1), Some(2), Some(directory.path()));
        let edits = [TextEdit::new(old, "FIRST\nSECOND\n")];
        assert!(edit_file("windows.txt", &edits, Some(directory.path())).contains("Successfully"));
        assert_eq!(fs::read(path).unwrap(), b"FIRST\r\nSECOND\r\nthird\r\n");
    }

    #[test]
    fn exact_edit_preserves_explicit_mixed_line_endings() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("mixed.txt");
        fs::write(&path, b"one\r\ntwo\n---\none\ntwo\n").unwrap();
        let edits = [TextEdit::new("one\r\ntwo", "ONE\r\nTWO\nTHREE")];
        assert!(edit_file("mixed.txt", &edits, Some(directory.path())).contains("Successfully"));
        assert_eq!(
            fs::read(path).unwrap(),
            b"ONE\r\nTWO\nTHREE\n---\none\ntwo\n"
        );
    }

    #[test]
    fn edit_rejects_ambiguous_missing_and_overlapping_matches() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("edit.txt");

        fs::write(&path, "foo bar foo").unwrap();
        let ambiguous = edit_file(
            "edit.txt",
            &[TextEdit::new("foo", "x")],
            Some(directory.path()),
        );
        assert!(ambiguous.contains("found 2 times"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "foo bar foo");

        fs::write(&path, "foo bar").unwrap();
        let overlapping = edit_file(
            "edit.txt",
            &[
                TextEdit::new("foo", "first"),
                TextEdit::new("foo", "second"),
            ],
            Some(directory.path()),
        );
        assert!(overlapping.contains("overlaps"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "foo bar");

        let missing = edit_file(
            "edit.txt",
            &[TextEdit::new("absent", "x")],
            Some(directory.path()),
        );
        assert!(missing.contains("not found"));
    }

    #[test]
    fn edit_rejects_non_utf8_without_modifying_it() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("binary.txt");
        let original: &[u8] = b"hello \xff world";
        fs::write(&path, original).unwrap();
        let result = edit_file(
            "binary.txt",
            &[TextEdit::new("hello", "hi")],
            Some(directory.path()),
        );
        assert!(result.starts_with("Error reading file:"));
        assert_eq!(fs::read(path).unwrap(), original);
    }

    #[test]
    fn windows_shell_selection_prefers_override_then_pwsh() {
        assert_eq!(
            choose_windows_shell(Some("custom-shell".into()), |_| true),
            "custom-shell"
        );
        assert_eq!(
            choose_windows_shell(None, |candidate| candidate == "pwsh"),
            "pwsh"
        );
        assert_eq!(
            choose_windows_shell(None, |candidate| candidate == "powershell"),
            "powershell"
        );
        assert_eq!(choose_windows_shell(None, |_| false), "powershell");
    }

    #[test]
    fn shell_runs_in_working_directory_and_reports_failures() {
        let directory = tempdir().unwrap();
        #[cfg(windows)]
        let create_marker = "Set-Content -NoNewline -Path marker.txt -Value ok";
        #[cfg(not(windows))]
        let create_marker = "printf ok > marker.txt";
        let result = run_bash(create_marker, Some(directory.path()));
        assert_eq!(result, "(no output)");
        assert_eq!(
            fs::read_to_string(directory.path().join("marker.txt")).unwrap(),
            "ok"
        );

        #[cfg(windows)]
        let fail = "Write-Error boom; exit 7";
        #[cfg(not(windows))]
        let fail = "printf boom >&2; exit 7";
        let result = run_bash(fail, None);
        assert!(result.contains("boom"));
        assert!(result.contains("[Exit code: 7]"));
    }

    #[test]
    fn shell_timeout_is_enforced() {
        #[cfg(windows)]
        let sleep = "Start-Sleep -Seconds 2";
        #[cfg(not(windows))]
        let sleep = "sleep 2";
        let result = run_bash_with_timeout(sleep, None, Duration::from_millis(20), None);
        assert!(result.contains("timed out"));
    }

    #[test]
    fn cancellation_interrupts_running_commands() {
        #[cfg(windows)]
        let sleep = "Start-Sleep -Seconds 10";
        #[cfg(not(windows))]
        let sleep = "sleep 10 | cat";
        let cancel = CancelToken::new();
        let trigger = cancel.clone();
        let canceller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            trigger.cancel();
        });
        let started = Instant::now();
        let result = run_bash_with_timeout(sleep, None, Duration::from_secs(10), Some(&cancel));
        canceller.join().unwrap();
        assert!(result.contains("interrupted by the user"), "{result}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn registry_validates_arguments_and_uses_working_directory() {
        let directory = tempdir().unwrap();
        let registry = ToolRegistry::with_working_dir(directory.path());
        assert!(registry.execute("read", json!({})).starts_with("Error:"));
        assert!(
            registry
                .execute("read", json!({"path": "x", "offset": false}))
                .starts_with("Error:")
        );
        assert!(
            registry
                .execute("edit", json!({"path": "x", "edits": []}))
                .starts_with("Error:")
        );
        assert!(registry.execute("bash", json!([])).starts_with("Error:"));
        assert!(
            registry
                .execute("missing", json!({}))
                .contains("Unknown tool")
        );

        assert!(
            registry
                .execute("write", json!({"path": "file.txt", "content": "data"}))
                .contains("Successfully")
        );
        assert_eq!(
            registry.execute("read", json!({"path": "file.txt"})),
            "data"
        );
        assert_eq!(registry.get_definitions().len(), 4);
    }
}
