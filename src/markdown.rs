use std::io::{self, IsTerminal};
use std::sync::OnceLock;

use termimad::crossterm::style::{Attribute, Color};
use termimad::{CompoundStyle, MadSkin, gray};

const MIN_RENDER_WIDTH: usize = 3;

fn markdown_skin() -> &'static MadSkin {
    static SKIN: OnceLock<MadSkin> = OnceLock::new();

    SKIN.get_or_init(|| {
        let mut skin = MadSkin::default_dark();
        let accent = Color::AnsiValue(75);
        let muted = Color::AnsiValue(245);
        let code = Color::AnsiValue(116);

        skin.inline_code = CompoundStyle::with_fg(code);
        skin.code_block.set_fgbg(code, gray(2));
        skin.bold.set_fg(Color::Reset);
        skin.bold.add_attr(Attribute::Bold);
        skin.italic.add_attr(Attribute::Italic);
        skin.bullet.set_fg(accent);
        skin.quote_mark.set_fg(muted);
        skin.horizontal_rule.set_fg(muted);
        skin.table.set_fg(muted);

        for header in &mut skin.headers {
            header.compound_style.remove_attr(Attribute::Underlined);
            header.add_attr(Attribute::Bold);
            header.set_fg(Color::Reset);
            header.align = termimad::Alignment::Left;
        }
        skin.headers[0].set_fg(accent);
        skin.headers[1].set_fg(accent);

        skin
    })
}

fn plain_skin() -> &'static MadSkin {
    static SKIN: OnceLock<MadSkin> = OnceLock::new();
    SKIN.get_or_init(MadSkin::no_style)
}

/// Render Markdown to stdout, using terminal styling only when stdout is a TTY.
pub fn render_markdown(text: &str) {
    let skin = if io::stdout().is_terminal() {
        markdown_skin()
    } else {
        plain_skin()
    };
    skin.print_text(text);
}

/// Render Markdown into terminal lines no wider than `width`, without leading or
/// trailing blank lines. ANSI styling is applied only when `styled` is true.
pub fn markdown_lines(text: &str, width: usize, styled: bool) -> Vec<String> {
    let skin = if styled {
        markdown_skin()
    } else {
        plain_skin()
    };
    let rendered = skin
        .text(text, Some(width.max(MIN_RENDER_WIDTH)))
        .to_string();
    let lines: Vec<String> = rendered
        .lines()
        .map(|line| line.trim_end().to_owned())
        .collect();
    let is_blank = |line: &String| crate::ui::visible_width(line) == 0 && !line.contains('\x1b');
    let start = lines
        .iter()
        .position(|line| !is_blank(line))
        .unwrap_or(lines.len());
    let end = lines
        .iter()
        .rposition(|line| !is_blank(line))
        .map_or(start, |end| end + 1);
    lines[start..end].to_vec()
}

/// Convert Markdown to terminal-shaped plain text without ANSI escape sequences.
pub fn markdown_to_plain(text: &str, width: usize) -> String {
    plain_skin()
        .text(text, Some(width.max(MIN_RENDER_WIDTH)))
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_conversion_removes_inline_markup_and_ansi_sequences() {
        let rendered = markdown_to_plain("# Summary\n\nUse **bold** and `code`.", 80);

        assert!(rendered.contains("Summary"));
        assert!(rendered.contains("Use bold and code."));
        assert!(!rendered.contains("**"));
        assert!(!rendered.contains('\u{1b}'));
    }

    #[test]
    fn plain_conversion_preserves_structured_content() {
        let markdown = "- first\n- second\n\n```rust\nlet answer = 42;\n```";
        let rendered = markdown_to_plain(markdown, 80);

        assert!(rendered.contains("first"));
        assert!(rendered.contains("second"));
        assert!(rendered.contains("let answer = 42;"));
        assert!(!rendered.contains("```"));
    }

    #[test]
    fn plain_conversion_wraps_to_the_requested_width() {
        let rendered = markdown_to_plain(
            "This sentence is deliberately long enough to wrap across lines.",
            24,
        );

        assert!(rendered.lines().count() > 1);
        assert!(rendered.lines().all(|line| line.chars().count() <= 24));
    }

    #[test]
    fn empty_markdown_renders_as_empty_text() {
        assert!(markdown_to_plain("", 100).trim().is_empty());
    }

    #[test]
    fn terminal_skin_avoids_magenta_for_primary_elements() {
        let skin = markdown_skin();

        assert_ne!(skin.inline_code.get_fg(), Some(Color::Magenta));
        assert_ne!(
            skin.headers[0].compound_style.get_fg(),
            Some(Color::Magenta)
        );
        assert_ne!(skin.bullet.get_fg(), Some(Color::Magenta));
    }

    #[test]
    fn markdown_lines_trim_surrounding_blank_lines_and_fit_width() {
        let lines = markdown_lines("\n\n# Title\n\nSome body text that wraps.\n\n", 12, false);
        assert!(lines.first().is_some_and(|line| line.contains("Title")));
        assert!(lines.last().is_some_and(|line| !line.trim().is_empty()));
        assert!(lines.iter().all(|line| line.chars().count() <= 12));
    }
}
