fn main() {
    if let Err(error) = harness_rs::cli::run() {
        eprintln!("Error: {error:#}");
        std::process::exit(1);
    }
}
