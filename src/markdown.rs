use std::io::{self, IsTerminal};
use std::sync::OnceLock;

use termimad::crossterm::style::{Attribute, Color};
use termimad::{CompoundStyle, MadSkin, gray};

const MIN_RENDER_WIDTH: usize = 3;

fn markdown_skin() -> &'static MadSkin {
    static SKIN: OnceLock<MadSkin> = OnceLock::new();

    SKIN.get_or_init(|| {
        let mut skin = MadSkin::default_dark();
        let code_background = gray(3);

        skin.inline_code = CompoundStyle::with_fgbg(Color::Cyan, code_background);
        skin.code_block.set_fgbg(Color::Cyan, code_background);
        skin.bold.set_fg(Color::White);
        skin.bullet.set_fg(Color::Cyan);
        skin.quote_mark.set_fg(Color::DarkGrey);
        skin.horizontal_rule.set_fg(Color::DarkGrey);
        skin.table.set_fg(Color::White);

        for header in &mut skin.headers {
            header.compound_style.remove_attr(Attribute::Underlined);
            header.add_attr(Attribute::Bold);
            header.set_fg(Color::White);
        }
        for header in &mut skin.headers[2..] {
            header.set_fg(Color::Grey);
        }

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

        assert_eq!(skin.inline_code.get_fg(), Some(Color::Cyan));
        assert_eq!(skin.headers[0].compound_style.get_fg(), Some(Color::White));
        assert_ne!(skin.bullet.get_fg(), Some(Color::Magenta));
    }
}
