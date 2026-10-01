fn main() {
    if let Err(error) = harness_rs::cli::run() {
        use std::io::IsTerminal;
        if std::io::stderr().is_terminal() && std::env::var_os("NO_COLOR").is_none() {
            eprintln!("\x1b[38;5;203m✗ Error: {error:#}\x1b[0m");
        } else {
            eprintln!("Error: {error:#}");
        }
        std::process::exit(1);
    }
}
