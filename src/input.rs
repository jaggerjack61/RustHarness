//! Interactive prompt with history, slash-command completion, inline hints, and
//! Tab to cycle the reasoning effort.

use std::borrow::Cow;
use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::{Hint, Hinter};
use rustyline::history::DefaultHistory;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{
    Cmd, CompletionType, ConditionalEventHandler, Config, Context, Editor, EventContext,
    EventHandler, Helper, KeyCode, KeyEvent, Modifiers, RepeatCount,
};

use crate::constants::REASONING_OPTIONS;
use crate::ui::{self, COMMANDS, PROMPT, Tone};

/// Index into [`REASONING_OPTIONS`], shared between the editor and its key handlers.
type EffortLevel = Arc<AtomicUsize>;

fn effort_index(effort: &str) -> usize {
    REASONING_OPTIONS
        .iter()
        .position(|option| *option == effort)
        .unwrap_or(REASONING_OPTIONS.len() / 2)
}

/// Fixed-width meter such as `▰▰▰▱▱`, one cell per reasoning level.
fn effort_meter(index: usize, styled: bool) -> String {
    let filled = "▰".repeat(index + 1);
    let empty = "▱".repeat(REASONING_OPTIONS.len().saturating_sub(index + 1));
    format!(
        "{}{}",
        ui::paint(&filled, Tone::Accent, styled),
        ui::paint(&empty, Tone::Muted, styled)
    )
}

fn prompt_text(index: usize, styled: bool) -> String {
    format!(
        "{} {}",
        effort_meter(index, styled),
        ui::paint(&format!("{PROMPT} "), Tone::AccentBold, styled)
    )
}

/// Tab / Shift+Tab: complete slash commands, otherwise cycle the reasoning effort.
struct CycleEffort {
    level: EffortLevel,
    step: isize,
}

impl ConditionalEventHandler for CycleEffort {
    fn handle(
        &self,
        _event: &rustyline::Event,
        _count: RepeatCount,
        _positive: bool,
        ctx: &EventContext<'_>,
    ) -> Option<Cmd> {
        if ctx.line().starts_with('/') {
            return None;
        }
        let count = REASONING_OPTIONS.len() as isize;
        let current = self.level.load(Ordering::Relaxed) as isize;
        let next = (current + self.step).rem_euclid(count) as usize;
        self.level.store(next, Ordering::Relaxed);
        Some(Cmd::Repaint)
    }
}

pub enum Input {
    Line(String),
    Interrupted,
    Eof,
}

pub struct CommandHint {
    display: String,
    completion: Option<String>,
}

impl Hint for CommandHint {
    fn display(&self) -> &str {
        &self.display
    }

    fn completion(&self) -> Option<&str> {
        self.completion.as_deref()
    }
}

struct CommandHelper {
    styled: bool,
    level: EffortLevel,
}

fn matching_commands(prefix: &str) -> impl Iterator<Item = &'static (&'static str, &'static str)> {
    COMMANDS
        .iter()
        .filter(move |(command, _)| prefix.starts_with('/') && command.starts_with(prefix))
}

impl Completer for CommandHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        let candidates = matching_commands(&line[..pos])
            .map(|(command, _)| Pair {
                display: (*command).to_owned(),
                replacement: (*command).to_owned(),
            })
            .collect();
        Ok((0, candidates))
    }
}

impl Hinter for CommandHelper {
    type Hint = CommandHint;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<CommandHint> {
        if line.is_empty() {
            let effort = REASONING_OPTIONS[self.level.load(Ordering::Relaxed)];
            return Some(CommandHint {
                // Leading spaces keep the block cursor from covering the hint's first letter.
                display: format!("  {effort} reasoning · Tab to change · /help for commands"),
                completion: None,
            });
        }
        if pos < line.len() || line.len() < 2 {
            return None;
        }
        let (command, description) = matching_commands(line).next()?;
        let rest = &command[line.len()..];
        Some(CommandHint {
            display: format!("{rest}  {description}"),
            completion: Some(rest.to_owned()),
        })
    }
}

impl Highlighter for CommandHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        prompt: &'p str,
        default: bool,
    ) -> Cow<'b, str> {
        if !default {
            // A search prompt (e.g. Ctrl+R) rather than ours.
            return Cow::Borrowed(prompt);
        }
        // Rebuilt from the live level so Tab updates it; the meter's width never changes.
        Cow::Owned(prompt_text(self.level.load(Ordering::Relaxed), self.styled))
    }

    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        if self.styled {
            Cow::Owned(ui::paint(hint, Tone::Muted, true))
        } else {
            Cow::Borrowed(hint)
        }
    }

    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        if self.styled && COMMANDS.iter().any(|(command, _)| *command == line) {
            Cow::Owned(ui::paint(line, Tone::Accent, true))
        } else {
            Cow::Borrowed(line)
        }
    }

    fn highlight_char(&self, _line: &str, _pos: usize, kind: CmdKind) -> bool {
        self.styled && kind != CmdKind::MoveCursor
    }
}

impl Validator for CommandHelper {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        Ok(if ctx.input().ends_with('\\') {
            ValidationResult::Incomplete
        } else {
            ValidationResult::Valid(None)
        })
    }
}

impl Helper for CommandHelper {}

/// Join lines continued with a trailing backslash.
fn join_continuations(input: &str) -> String {
    input.replace("\\\n", "\n").replace("\\\r\n", "\n")
}

pub struct Prompt {
    editor: Option<Editor<CommandHelper, DefaultHistory>>,
    history_path: Option<PathBuf>,
    level: EffortLevel,
}

impl Prompt {
    pub fn new(history_path: Option<PathBuf>) -> Self {
        let level = EffortLevel::default();
        let interactive = io::stdin().is_terminal() && io::stdout().is_terminal();
        let editor = interactive
            .then(|| build_editor(history_path.as_ref(), &level))
            .flatten();
        Self {
            editor,
            history_path,
            level,
        }
    }

    pub fn set_effort(&self, effort: &str) {
        self.level.store(effort_index(effort), Ordering::Relaxed);
    }

    /// The reasoning effort currently shown in the prompt (changed with Tab).
    pub fn effort(&self) -> &'static str {
        REASONING_OPTIONS[self.level.load(Ordering::Relaxed)]
    }

    pub fn read(&mut self) -> io::Result<Input> {
        let Some(editor) = self.editor.as_mut() else {
            let prompt = format!("{PROMPT} ");
            print!("{}", ui::style(&prompt, Tone::AccentBold));
            io::stdout().flush()?;
            return Ok(match read_line()? {
                Some(line) => Input::Line(line),
                None => {
                    println!();
                    Input::Eof
                }
            });
        };
        let prompt = prompt_text(self.level.load(Ordering::Relaxed), false);
        match editor.readline(&prompt) {
            Ok(line) => {
                if !line.trim().is_empty() {
                    let _ = editor.add_history_entry(line.as_str());
                    if let Some(path) = self.history_path.as_ref() {
                        if let Some(parent) = path.parent() {
                            let _ = std::fs::create_dir_all(parent);
                        }
                        let _ = editor.save_history(path);
                    }
                }
                Ok(Input::Line(join_continuations(&line)))
            }
            Err(ReadlineError::Interrupted) => Ok(Input::Interrupted),
            Err(ReadlineError::Eof) => Ok(Input::Eof),
            Err(ReadlineError::Io(error)) => Err(error),
            Err(error) => Err(io::Error::other(error)),
        }
    }
}

fn build_editor(
    history_path: Option<&PathBuf>,
    level: &EffortLevel,
) -> Option<Editor<CommandHelper, DefaultHistory>> {
    let config = Config::builder()
        .completion_type(CompletionType::List)
        .history_ignore_dups(true)
        .ok()?
        .max_history_size(1_000)
        .ok()?
        .auto_add_history(false)
        .build();
    let mut editor = Editor::with_config(config).ok()?;
    editor.set_helper(Some(CommandHelper {
        styled: ui::styled(),
        level: Arc::clone(level),
    }));
    for (key, step) in [(KeyCode::Tab, 1), (KeyCode::BackTab, -1)] {
        editor.bind_sequence(
            KeyEvent(key, Modifiers::NONE),
            EventHandler::Conditional(Box::new(CycleEffort {
                level: Arc::clone(level),
                step,
            })),
        );
    }
    if let Some(path) = history_path {
        let _ = editor.load_history(path);
    }
    Some(editor)
}

/// Read one raw line from stdin; `None` on end of input.
pub fn read_line() -> io::Result<Option<String>> {
    let mut line = String::new();
    match io::stdin().read_line(&mut line) {
        Ok(0) => Ok(None),
        Ok(_) => Ok(Some(line)),
        Err(error) if error.kind() == io::ErrorKind::Interrupted => Ok(None),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustyline::history::DefaultHistory;

    #[test]
    fn hints_complete_unique_command_prefixes() {
        let helper = CommandHelper {
            styled: false,
            level: EffortLevel::default(),
        };
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let hint = helper.hint("/mod", 4, &ctx).unwrap();
        assert_eq!(hint.completion(), Some("els"));
        assert!(hint.display().contains("Choose a provider and model"));
        assert!(helper.hint("hello", 5, &ctx).is_none());
        assert!(helper.hint("/", 1, &ctx).is_none());
    }

    #[test]
    fn completion_lists_matching_commands() {
        let helper = CommandHelper {
            styled: false,
            level: EffortLevel::default(),
        };
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let (start, candidates) = helper.complete("/context ", 9, &ctx).unwrap();
        assert_eq!(start, 0);
        let names: Vec<_> = candidates
            .iter()
            .map(|pair| pair.replacement.as_str())
            .collect();
        assert_eq!(names, ["/context show", "/context clear"]);
        assert!(helper.complete("text", 4, &ctx).unwrap().1.is_empty());
    }

    #[test]
    fn prompt_width_is_the_same_for_every_effort() {
        let widths: Vec<_> = (0..REASONING_OPTIONS.len())
            .map(|index| ui::visible_width(&prompt_text(index, true)))
            .collect();
        assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));
        assert_eq!(prompt_text(effort_index("high"), false), "▰▰▰▱▱ › ");
        assert_eq!(prompt_text(effort_index("max"), false), "▰▰▰▰▰ › ");
        assert_eq!(effort_index("unknown"), effort_index("high"));
    }

    #[test]
    fn empty_line_hint_names_the_current_effort() {
        let helper = CommandHelper {
            styled: false,
            level: Arc::new(AtomicUsize::new(effort_index("low"))),
        };
        let history = DefaultHistory::new();
        let ctx = Context::new(&history);
        let hint = helper.hint("", 0, &ctx).unwrap();
        assert!(hint.display().starts_with("  low reasoning"));
        assert_eq!(hint.completion(), None);
    }

    #[test]
    fn backslash_continuations_become_newlines() {
        assert_eq!(join_continuations("first\\\nsecond"), "first\nsecond");
        assert_eq!(join_continuations("plain"), "plain");
    }
}
