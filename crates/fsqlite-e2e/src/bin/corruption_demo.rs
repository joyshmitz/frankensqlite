//! Guided corruption + recovery walkthrough binary (bd-1w6k.7.5).
//!
//! A single command that produces a compelling demo for a human to watch:
//!
//! ```text
//! cargo run --bin corruption-demo
//! cargo run --bin corruption-demo -- --json
//! ```

use std::io::{self, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_mode = args.iter().any(|a| a == "--json");
    let help = args.iter().any(|a| a == "-h" || a == "--help");

    if help {
        print_help();
        return;
    }

    let report = fsqlite_e2e::corruption_walkthrough::run_walkthrough();

    if json_mode {
        match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("error: failed to serialize walkthrough: {e}");
                std::process::exit(1);
            }
        }
    } else {
        let text = fsqlite_e2e::corruption_walkthrough::format_walkthrough(&report);
        let _ = io::stdout().write_all(text.as_bytes());
    }

    if !report.all_passed {
        std::process::exit(1);
    }
}

fn print_help() {
    let text = "\
corruption-demo — FrankenSQLite corruption + recovery walkthrough

Demonstrates how FrankenSQLite's WAL-FEC recovery compares to C SQLite
when databases are corrupted.  Runs 4 representative scenarios:

  1. WAL corruption within FEC tolerance  (FrankenSQLite recovers)
  2. Single-bit WAL corruption (bitrot)   (FrankenSQLite recovers)
  3. WAL corruption beyond FEC capacity   (both engines lose data)
  4. Database header zeroed               (catastrophic, no recovery)

USAGE:
    corruption-demo [OPTIONS]

OPTIONS:
    --json    Output results as JSON instead of narrative text
    -h, --help    Show this help message
";
    let _ = io::stdout().write_all(text.as_bytes());
}

#[cfg(test)]
mod tests {
    fn run_with(args: &[&str]) -> (i32, String) {
        // Capture behavior via the library directly.
        let json_mode = args.contains(&"--json");
        let help = args.contains(&"-h") || args.contains(&"--help");

        if help {
            return (0, "corruption-demo".to_owned());
        }

        let report = fsqlite_e2e::corruption_walkthrough::run_walkthrough();
        let output = if json_mode {
            serde_json::to_string_pretty(&report).unwrap_or_default()
        } else {
            fsqlite_e2e::corruption_walkthrough::format_walkthrough(&report)
        };

        let code = i32::from(!report.all_passed);
        (code, output)
    }

    #[test]
    fn test_help_output() {
        let (code, output) = run_with(&["corruption-demo", "-h"]);
        assert_eq!(code, 0);
        assert!(output.contains("corruption-demo"));
    }

    #[test]
    fn test_narrative_mode() {
        let (code, output) = run_with(&["corruption-demo"]);
        assert_eq!(code, 0, "walkthrough should pass");
        assert!(output.contains("Walkthrough"));
        assert!(output.contains("Scenario 1"));
    }

    #[test]
    fn test_json_mode() {
        let (code, output) = run_with(&["corruption-demo", "--json"]);
        assert_eq!(code, 0, "walkthrough should pass");
        let parsed: serde_json::Value = serde_json::from_str(&output).expect("valid JSON");
        assert!(parsed["sections"].is_array());
        assert_eq!(parsed["sections"].as_array().unwrap().len(), 4);
    }
}
