use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use fsqlite_harness::durability_matrix::{
    BEAD_ID, DEFAULT_ROOT_SEED, build_validated_durability_matrix, render_operator_workflow,
    write_matrix_json,
};

#[derive(Debug)]
struct CliConfig {
    root_seed: u64,
    output: Option<PathBuf>,
    workflow: bool,
}

fn print_help() {
    let help = "\
durability_matrix_manifest — deterministic durability matrix generator (bd-mblr.7.4)

USAGE:
    cargo run -p fsqlite-harness --bin durability_matrix_manifest -- [OPTIONS]

OPTIONS:
    --root-seed <u64>     Root seed for deterministic matrix generation
                          (default: 0xB740_0000_0000_0001)
    --output <PATH>       Write output to file (stdout when omitted)
    --workflow            Emit operator workflow text instead of JSON
    -h, --help            Show this help
";
    println!("{help}");
}

fn parse_u64(value: &str) -> Result<u64, String> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).map_err(|_| format!("invalid hex u64 value: {value}"))
    } else {
        value
            .parse::<u64>()
            .map_err(|_| format!("invalid u64 value: {value}"))
    }
}

fn parse_args(args: &[String]) -> Result<CliConfig, String> {
    let mut config = CliConfig {
        root_seed: DEFAULT_ROOT_SEED,
        output: None,
        workflow: false,
    };

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--root-seed" => {
                index += 1;
                if index >= args.len() {
                    return Err("--root-seed requires a value".to_owned());
                }
                config.root_seed = parse_u64(&args[index])?;
            }
            "--output" => {
                index += 1;
                if index >= args.len() {
                    return Err("--output requires a value".to_owned());
                }
                config.output = Some(PathBuf::from(&args[index]));
            }
            "--workflow" => config.workflow = true,
            "-h" | "--help" => {
                print_help();
                return Err(String::new());
            }
            unknown => return Err(format!("unknown option: {unknown}")),
        }
        index += 1;
    }

    Ok(config)
}

fn run(args: &[String]) -> Result<(), String> {
    let config = parse_args(args)?;
    let matrix = build_validated_durability_matrix(config.root_seed)?;

    if config.workflow {
        let workflow = render_operator_workflow(&matrix);
        if let Some(output) = &config.output {
            std::fs::write(output, workflow.as_bytes()).map_err(|error| {
                format!(
                    "durability_matrix_workflow_write_failed path={} error={error}",
                    output.display()
                )
            })?;
        } else {
            println!("{workflow}");
        }
        return Ok(());
    }

    if let Some(output) = &config.output {
        write_matrix_json(output, &matrix)?;
    } else {
        let payload = serde_json::to_string_pretty(&matrix)
            .map_err(|error| format!("durability_matrix_json_serialize_failed: {error}"))?;
        println!("{payload}");
    }

    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) if error.is_empty() => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR bead_id={BEAD_ID} durability_matrix_manifest failed: {error}");
            ExitCode::from(2)
        }
    }
}
