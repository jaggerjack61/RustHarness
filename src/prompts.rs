use std::io::{self, IsTerminal, Write};

use console::{Style, style};
use dialoguer::theme::ColorfulTheme;
use dialoguer::{FuzzySelect, Input, Select};

use crate::ui;

/// Lists longer than this get type-to-filter selection.
const FUZZY_THRESHOLD: usize = 12;
const VISIBLE_ROWS: usize = 12;

const SEPARATOR: &str = "──────────────────────────────────────────────────";

#[derive(Debug, PartialEq, Eq)]
enum NumericSelectionError {
    NotANumber,
    OutOfRange,
}

fn parse_numeric_selection(
    input: &str,
    option_count: usize,
) -> Result<Option<usize>, NumericSelectionError> {
    let input = input.trim();
    if input.is_empty() {
        return Ok(None);
    }

    let number = input
        .parse::<i128>()
        .map_err(|_| NumericSelectionError::NotANumber)?;
    if number < 1 || number > option_count as i128 {
        return Err(NumericSelectionError::OutOfRange);
    }

    Ok(Some(number as usize - 1))
}

fn numeric_menu<S: AsRef<str>>(
    options: &[S],
    current: &str,
    title: &str,
    current_label: &str,
) -> String {
    let mut menu = format!("\n{title}:\n{SEPARATOR}\n");
    for (index, option) in options.iter().enumerate() {
        let option = option.as_ref();
        let marker = if option == current { " → " } else { "   " };
        menu.push_str(&format!("{marker}{:2}. {option}\n", index + 1));
    }
    menu.push_str(&format!("{SEPARATOR}\n{current_label}: {current}\n\n"));
    menu
}

/// Print numbered options and return the selected value.
///
/// Empty, non-numeric, and out-of-range input all cancel the selection.
pub fn prompt_selection_numeric<S: AsRef<str>>(
    options: &[S],
    current: &str,
    title: &str,
    current_label: &str,
) -> Option<String> {
    if options.is_empty() {
        return None;
    }

    print!("{}", numeric_menu(options, current, title, current_label));
    let _ = io::stdout().flush();

    let choice = if io::stdin().is_terminal() {
        match Input::<String>::new()
            .with_prompt("Select number (or press Enter to cancel)")
            .allow_empty(true)
            .report(false)
            .interact_text()
        {
            Ok(choice) => choice,
            Err(_) => {
                println!("\nCancelled.");
                return None;
            }
        }
    } else {
        print!("Select number (or press Enter to cancel): ");
        let _ = io::stdout().flush();
        match crate::input::read_line() {
            Ok(Some(choice)) => choice,
            _ => return None,
        }
    };

    match parse_numeric_selection(&choice, options.len()) {
        Ok(Some(index)) => Some(options[index].as_ref().to_owned()),
        Ok(None) => None,
        Err(NumericSelectionError::NotANumber) => {
            ui::error("Invalid input. Please enter a number.");
            None
        }
        Err(NumericSelectionError::OutOfRange) => {
            ui::error(&format!(
                "Invalid selection. Please enter 1-{}.",
                options.len()
            ));
            None
        }
    }
}

/// Dialoguer theme matching the CLI palette.
pub fn theme() -> ColorfulTheme {
    if !ui::styled() {
        console::set_colors_enabled_stderr(false);
    }
    let accent = Style::new().for_stderr().color256(75);
    let muted = Style::new().for_stderr().color256(245);
    ColorfulTheme {
        prompt_prefix: style("?".to_owned()).for_stderr().color256(75).bold(),
        prompt_suffix: style("›".to_owned()).for_stderr().color256(245),
        success_prefix: style("✓".to_owned()).for_stderr().color256(78),
        success_suffix: style("·".to_owned()).for_stderr().color256(245),
        error_prefix: style("✗".to_owned()).for_stderr().color256(203),
        hint_style: muted.clone(),
        values_style: accent.clone(),
        active_item_style: accent.clone().bold(),
        active_item_prefix: style("›".to_owned()).for_stderr().color256(75).bold(),
        inactive_item_prefix: style(" ".to_owned()).for_stderr(),
        picked_item_prefix: style("›".to_owned()).for_stderr().color256(75),
        fuzzy_match_highlight_style: Style::new().for_stderr().color256(75).bold().underlined(),
        ..ColorfulTheme::default()
    }
}

/// Let the user choose with arrow keys (and type-to-filter for long lists),
/// falling back to a numeric prompt when stdin is not a terminal.
pub fn prompt_selection<S: AsRef<str>>(
    options: &[S],
    current: &str,
    title: &str,
    current_label: &str,
) -> Option<String> {
    if options.is_empty() {
        ui::hint("Nothing to choose from.");
        return None;
    }

    if !io::stdin().is_terminal() {
        return prompt_selection_numeric(options, current, title, current_label);
    }

    let items: Vec<&str> = options.iter().map(|option| option.as_ref()).collect();
    let default = items
        .iter()
        .position(|option| *option == current)
        .unwrap_or(0);
    let fuzzy = items.len() > FUZZY_THRESHOLD;
    println!();
    let keys = if fuzzy {
        "↑/↓ move · type to filter · Enter select · Esc cancel"
    } else {
        "↑/↓ move · Enter select · Esc cancel"
    };
    let width = termimad::crossterm::terminal::size()
        .map_or(80, |(width, _)| usize::from(width))
        .saturating_sub(1);
    ui::hint(&ui::truncate(&format!("{current_label}: {current}"), width));
    ui::hint(&ui::truncate(keys, width));
    let theme = theme();
    let result = if fuzzy {
        FuzzySelect::with_theme(&theme)
            .with_prompt(title)
            .items(&items)
            .default(default)
            .max_length(VISIBLE_ROWS)
            .report(false)
            .interact_opt()
    } else {
        Select::with_theme(&theme)
            .with_prompt(title)
            .items(&items)
            .default(default)
            .max_length(VISIBLE_ROWS)
            .report(false)
            .interact_opt()
    };
    match result {
        Ok(Some(index)) => Some(items[index].to_owned()),
        Ok(None) => {
            ui::hint("Cancelled.");
            None
        }
        Err(error) => {
            ui::error(&format!(
                "Interactive selection failed ({error}); falling back to numeric input."
            ));
            prompt_selection_numeric(options, current, title, current_label)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_parser_accepts_one_based_choices() {
        assert_eq!(parse_numeric_selection(" 2 ", 3), Ok(Some(1)));
    }

    #[test]
    fn numeric_parser_treats_empty_input_as_cancellation() {
        assert_eq!(parse_numeric_selection("  ", 3), Ok(None));
    }

    #[test]
    fn numeric_parser_rejects_non_numbers_and_out_of_range_values() {
        assert_eq!(
            parse_numeric_selection("two", 3),
            Err(NumericSelectionError::NotANumber)
        );
        assert_eq!(
            parse_numeric_selection("0", 3),
            Err(NumericSelectionError::OutOfRange)
        );
        assert_eq!(
            parse_numeric_selection("4", 3),
            Err(NumericSelectionError::OutOfRange)
        );
    }

    #[test]
    fn numeric_menu_marks_the_current_option() {
        let menu = numeric_menu(&["alpha", "beta"], "beta", "Options", "Current");

        assert!(menu.contains("    1. alpha"));
        assert!(menu.contains(" →  2. beta"));
        assert!(menu.contains("Current: beta"));
    }
}
