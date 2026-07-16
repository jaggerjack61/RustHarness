use std::io::{self, IsTerminal, Write};

use dialoguer::{Input, Select};

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

    let choice = match Input::<String>::new()
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
    };

    match parse_numeric_selection(&choice, options.len()) {
        Ok(Some(index)) => Some(options[index].as_ref().to_owned()),
        Ok(None) => None,
        Err(NumericSelectionError::NotANumber) => {
            println!("❌ Invalid input. Please enter a number.");
            None
        }
        Err(NumericSelectionError::OutOfRange) => {
            println!("❌ Invalid selection. Please enter 1-{}.", options.len());
            None
        }
    }
}

/// Let the user choose with arrow keys, falling back to a numeric prompt.
pub fn prompt_selection<S: AsRef<str>>(
    options: &[S],
    current: &str,
    title: &str,
    current_label: &str,
) -> Option<String> {
    if options.is_empty() {
        println!("No options available.");
        return None;
    }

    if !io::stdin().is_terminal() {
        return prompt_selection_numeric(options, current, title, current_label);
    }

    println!("\n{title}:");
    println!("{SEPARATOR}");
    println!("{current_label}: {current}");
    println!("Use ↑/↓ to navigate, Enter to select, Esc to cancel.\n");

    let items: Vec<&str> = options.iter().map(|option| option.as_ref()).collect();
    let default = items
        .iter()
        .position(|option| *option == current)
        .unwrap_or(0);
    match Select::new()
        .items(&items)
        .default(default)
        .clear(false)
        .report(false)
        .interact_opt()
    {
        Ok(Some(index)) => Some(items[index].to_owned()),
        Ok(None) => None,
        Err(error) => {
            println!("❌ Interactive selection failed ({error}); falling back to numeric input.");
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
