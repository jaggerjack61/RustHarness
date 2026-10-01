//! Shared terminal styling: palette, glyphs, status messages, and the banner.

use std::io::{self, IsTerminal};
use std::path::Path;
use std::sync::OnceLock;

use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

pub const APP_NAME: &str = "Nasa Level Genius Agent";

pub const BULLET: &str = "●";
pub const RESULT: &str = "⎿";
pub const PROMPT: &str = "›";
pub const TICK: &str = "✓";
pub const CROSS: &str = "✗";
pub const WARN: &str = "⚠";
pub const THINKING: &str = "✻";
pub const SPARK: &str = "✦";
pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Slash commands with their help text, in display order.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/help", "Show commands and shortcuts"),
    ("/models", "Choose a provider and model"),
    ("/model", "Browse and switch models (alias)"),
    ("/reasoning", "Set the reasoning effort"),
    ("/login", "Add a named OpenAI-compatible provider"),
    ("/context", "Attach custom context to the system prompt"),
    ("/context show", "Show the custom context"),
    ("/context clear", "Remove the custom context"),
    ("/stream", "Toggle streaming output"),
    ("/clear", "Clear the conversation and screen"),
    ("/exit", "Quit"),
];

const SHORTCUTS: &[(&str, &str)] = &[
    (
        "Tab / Shift+Tab",
        "Cycle reasoning effort (completes /commands)",
    ),
    ("Esc", "Interrupt the agent while it works"),
    ("↑ / ↓", "Browse input history"),
    ("\\ then Enter", "Continue on a new line"),
    ("Ctrl+C", "Clear the input (twice to quit)"),
    ("Ctrl+D", "Quit"),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Plain,
    Accent,
    AccentBold,
    Bold,
    Muted,
    Thinking,
    Success,
    Warning,
    Error,
    Added,
    Removed,
}

impl Tone {
    fn code(self) -> &'static str {
        match self {
            Self::Plain => "",
            Self::Accent => "38;5;75",
            Self::AccentBold => "1;38;5;75",
            Self::Bold => "1",
            Self::Muted => "38;5;245",
            Self::Thinking => "3;38;5;245",
            Self::Success => "38;5;78",
            Self::Warning => "38;5;214",
            Self::Error => "38;5;203",
            Self::Added => "38;5;114",
            Self::Removed => "38;5;210",
        }
    }
}

/// Whether stdout should receive ANSI styling.
pub fn styled() -> bool {
    static STYLED: OnceLock<bool> = OnceLock::new();
    *STYLED.get_or_init(|| io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none())
}

pub fn paint(text: &str, tone: Tone, enabled: bool) -> String {
    if !enabled || tone == Tone::Plain || text.is_empty() {
        return text.to_owned();
    }
    format!("\x1b[{}m{text}\x1b[0m", tone.code())
}

pub fn style(text: &str, tone: Tone) -> String {
    paint(text, tone, styled())
}

pub fn success(message: &str) {
    println!("{} {message}", style(TICK, Tone::Success));
}

pub fn warning(message: &str) {
    println!(
        "{} {}",
        style(WARN, Tone::Warning),
        style(message, Tone::Warning)
    );
}

pub fn error(message: &str) {
    println!(
        "{} {}",
        style(CROSS, Tone::Error),
        style(message, Tone::Error)
    );
}

pub fn hint(message: &str) {
    println!("{}", style(message, Tone::Muted));
}

/// Display width of `text`, ignoring ANSI escape sequences.
pub fn visible_width(text: &str) -> usize {
    let mut width = 0;
    let mut chars = text.chars();
    while let Some(character) = chars.next() {
        if character == '\x1b' {
            if chars.next() == Some('[') {
                for next in chars.by_ref() {
                    if ('@'..='~').contains(&next) {
                        break;
                    }
                }
            }
            continue;
        }
        width += UnicodeWidthChar::width(character).unwrap_or(0);
    }
    width
}

/// Cut plain `text` to at most `width` columns, ending with an ellipsis when shortened.
pub fn truncate(text: &str, width: usize) -> String {
    if UnicodeWidthStr::width(text) <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    let mut output = String::new();
    let mut used = 0;
    for character in text.chars() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if used + character_width + 1 > width {
            break;
        }
        used += character_width;
        output.push(character);
    }
    output.push('…');
    output
}

/// Like [`truncate`], but keeps the end of the text (useful for paths).
pub fn truncate_start(text: &str, width: usize) -> String {
    if UnicodeWidthStr::width(text) <= width {
        return text.to_owned();
    }
    if width == 0 {
        return String::new();
    }
    let mut kept = Vec::new();
    let mut used = 0;
    for character in text.chars().rev() {
        let character_width = UnicodeWidthChar::width(character).unwrap_or(0);
        if used + character_width + 1 > width {
            break;
        }
        used += character_width;
        kept.push(character);
    }
    std::iter::once('…').chain(kept.into_iter().rev()).collect()
}

/// Short human form of a count: `950`, `5.2k`, `123k`, `1.5M`.
pub fn compact_number(value: u64) -> String {
    fn scaled(value: f64, suffix: &str) -> String {
        let text = if value < 100.0 {
            format!("{value:.1}")
        } else {
            format!("{value:.0}")
        };
        format!("{}{suffix}", text.strip_suffix(".0").unwrap_or(&text))
    }
    match value {
        0..1_000 => value.to_string(),
        1_000..999_950 => scaled(value as f64 / 1_000.0, "k"),
        _ => scaled(value as f64 / 1_000_000.0, "M"),
    }
}

pub fn format_duration(seconds: f64, precise: bool) -> String {
    if seconds < 60.0 {
        if precise {
            format!("{seconds:.1}s")
        } else {
            format!("{}s", seconds as u64)
        }
    } else {
        let total = seconds as u64;
        format!("{}m {:02}s", total / 60, total % 60)
    }
}

/// Replace the home directory prefix with `~`.
pub fn display_path(path: &Path) -> String {
    let text = path.display().to_string();
    if let Some(home) = std::env::var_os("HOME").filter(|home| !home.is_empty()) {
        let home = home.to_string_lossy();
        if let Some(rest) = text.strip_prefix(home.as_ref())
            && (rest.is_empty() || rest.starts_with(std::path::MAIN_SEPARATOR))
        {
            return format!("~{rest}");
        }
    }
    text
}

pub struct BannerInfo<'a> {
    pub model: &'a str,
    pub reasoning_effort: &'a str,
    pub context_window: i64,
    pub working_dir: &'a Path,
    pub streaming: bool,
}

pub fn banner(info: &BannerInfo<'_>, terminal_width: usize, styled: bool) -> String {
    const LABEL_WIDTH: usize = 11;
    let rows = [
        ("model", info.model.to_owned()),
        ("reasoning", info.reasoning_effort.to_owned()),
        (
            "context",
            format!(
                "{} tokens",
                compact_number(info.context_window.max(0) as u64)
            ),
        ),
        (
            "streaming",
            if info.streaming { "on" } else { "off" }.to_owned(),
        ),
        ("directory", display_path(info.working_dir)),
    ];
    let version = format!("v{}", env!("CARGO_PKG_VERSION"));
    let title_width = UnicodeWidthStr::width(SPARK) + 1 + APP_NAME.len() + 2 + version.len();
    let content_width = rows
        .iter()
        .map(|(_, value)| 2 + LABEL_WIDTH + UnicodeWidthStr::width(value.as_str()))
        .max()
        .unwrap_or(0)
        .max(title_width);
    // Inner width excludes the two border columns and one space of padding on each side.
    let inner = content_width
        .max(44)
        .min(terminal_width.saturating_sub(4))
        .max(20);

    let border = |text: &str| paint(text, Tone::Muted, styled);
    let line = |content: String| {
        let padding = inner.saturating_sub(visible_width(&content));
        format!(
            "{} {content}{} {}",
            border("│"),
            " ".repeat(padding),
            border("│")
        )
    };

    let mut output = vec![border(&format!("╭{}╮", "─".repeat(inner + 2)))];
    let title = if title_width <= inner {
        format!(
            "{} {}  {}",
            paint(SPARK, Tone::Accent, styled),
            paint(APP_NAME, Tone::Bold, styled),
            paint(&version, Tone::Muted, styled)
        )
    } else {
        paint(&truncate(APP_NAME, inner), Tone::Bold, styled)
    };
    output.push(line(title));
    output.push(line(String::new()));
    for (label, value) in rows {
        let available = inner.saturating_sub(2 + LABEL_WIDTH);
        let value = if label == "directory" {
            truncate_start(&value, available)
        } else {
            truncate(&value, available)
        };
        output.push(line(format!(
            "  {}{value}",
            paint(&format!("{label:<LABEL_WIDTH$}"), Tone::Muted, styled)
        )));
    }
    output.push(border(&format!("╰{}╯", "─".repeat(inner + 2))));
    output.push(String::new());
    let tips = "Type a message to start · /help for commands · Ctrl+D to quit";
    output.push(paint(
        &truncate(tips, terminal_width.saturating_sub(2)),
        Tone::Muted,
        styled,
    ));
    output.join("\n")
}

pub fn help_text(width: usize, styled: bool) -> String {
    const KEY_WIDTH: usize = 18;
    let description_width = width.saturating_sub(KEY_WIDTH + 3);
    let row = |key: &str, description: &str| {
        let padding = KEY_WIDTH.saturating_sub(UnicodeWidthStr::width(key));
        format!(
            "  {}{}{}",
            paint(key, Tone::Accent, styled),
            " ".repeat(padding),
            truncate(description, description_width)
        )
    };
    let mut output = vec![paint("Commands", Tone::Bold, styled)];
    output.extend(
        COMMANDS
            .iter()
            .map(|(command, description)| row(command, description)),
    );
    output.push(String::new());
    output.push(paint("Shortcuts", Tone::Bold, styled));
    output.extend(
        SHORTCUTS
            .iter()
            .map(|(keys, description)| row(keys, description)),
    );
    output.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_banner(width: usize, directory: &str) -> String {
        banner(
            &BannerInfo {
                model: "model",
                reasoning_effort: "high",
                context_window: 1_000_000,
                working_dir: Path::new(directory),
                streaming: true,
            },
            width,
            false,
        )
    }

    #[test]
    fn banner_box_rows_have_equal_display_width() {
        let text = sample_banner(120, "project");
        let box_rows: Vec<_> = text.lines().take_while(|line| !line.is_empty()).collect();
        let widths: Vec<_> = box_rows.iter().map(|line| visible_width(line)).collect();
        assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));
        assert!(text.contains("1M tokens"));
        assert!(text.contains(APP_NAME));
        assert!(text.starts_with('╭'));
    }

    #[test]
    fn banner_fits_narrow_terminals() {
        let directory = "/a/very/long/path/that/would/never/fit/inside/a/narrow/terminal/window";
        let text = sample_banner(40, directory);
        assert!(text.lines().all(|line| visible_width(line) <= 40));
        assert!(text.contains("…"));
        assert!(text.contains("window"));
    }

    #[test]
    fn styled_banner_keeps_rows_aligned() {
        let text = banner(
            &BannerInfo {
                model: "model",
                reasoning_effort: "high",
                context_window: 128_000,
                working_dir: Path::new("project"),
                streaming: false,
            },
            100,
            true,
        );
        let widths: Vec<_> = text
            .lines()
            .take_while(|line| !line.is_empty())
            .map(visible_width)
            .collect();
        assert!(widths.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn compact_numbers_are_short() {
        assert_eq!(compact_number(950), "950");
        assert_eq!(compact_number(5_000), "5k");
        assert_eq!(compact_number(5_200), "5.2k");
        assert_eq!(compact_number(123_456), "123k");
        assert_eq!(compact_number(999_999), "1M");
        assert_eq!(compact_number(1_500_000), "1.5M");
    }

    #[test]
    fn visible_width_ignores_ansi_sequences() {
        assert_eq!(visible_width(&paint("hello", Tone::Accent, true)), 5);
        assert_eq!(visible_width("● ok"), 4);
    }

    #[test]
    fn truncation_respects_width() {
        assert_eq!(truncate("abcdef", 4), "abc…");
        assert_eq!(truncate("abc", 4), "abc");
        assert_eq!(truncate_start("abcdef", 4), "…def");
    }

    #[test]
    fn durations_switch_to_minutes() {
        assert_eq!(format_duration(4.27, false), "4s");
        assert_eq!(format_duration(4.27, true), "4.3s");
        assert_eq!(format_duration(65.0, true), "1m 05s");
    }

    #[test]
    fn help_lists_every_command() {
        let help = help_text(100, false);
        for (command, _) in COMMANDS {
            assert!(help.contains(command));
        }
        assert!(
            help_text(40, false)
                .lines()
                .all(|line| visible_width(line) <= 40)
        );
    }
}
