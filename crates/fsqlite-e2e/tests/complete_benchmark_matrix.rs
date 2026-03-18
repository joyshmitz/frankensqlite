use std::error::Error;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root should exist")
        .to_path_buf()
}

fn artifact_dir() -> PathBuf {
    let repo_root = repo_root();
    std::env::var_os("FSQLITE_MATRIX_ARTIFACT_DIR").map_or_else(
        || {
            repo_root.join(
                "artifacts/perf/bd-db300.1.2/run_20260317T083212Z__rev_6b071b159606__beads_60f9bb3f1ace",
            )
        },
        PathBuf::from,
    )
}

fn bench_binary() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_realdb-e2e").map_or_else(
        || panic!("CARGO_BIN_EXE_realdb-e2e should be available for integration tests"),
        PathBuf::from,
    )
}

fn print_progress(label: &str, line: &str) {
    if let Ok(summary) = serde_json::from_str::<serde_json::Value>(line) {
        let benchmark_id = summary["benchmark_id"].as_str().unwrap_or("unknown");
        let median_ops = summary["throughput"]["median_ops_per_sec"]
            .as_f64()
            .unwrap_or(0.0);
        let p95_ms = summary["latency"]["p95_ms"].as_f64().unwrap_or(0.0);
        let measurement_count = summary["measurement_count"].as_u64().unwrap_or(0);
        println!(
            "[bench:{label}] {benchmark_id} median_ops/s={median_ops:.1} p95_ms={p95_ms:.1} n={measurement_count}"
        );
    } else if !line.trim().is_empty() {
        println!("[bench:{label}] {line}");
    }
}

fn run_bench(label: &str, args: &[String]) -> Result<ExitStatus, Box<dyn Error>> {
    let mut child = Command::new(bench_binary())
        .current_dir(repo_root())
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdout = child
        .stdout
        .take()
        .ok_or("bench child stdout should be piped")?;
    for line in BufReader::new(stdout).lines() {
        print_progress(label, &line?);
    }

    let status = child.wait()?;
    Ok(status)
}

fn run_command_capture_stdout(args: &[String]) -> Result<(ExitStatus, String), Box<dyn Error>> {
    let output = Command::new(bench_binary())
        .current_dir(repo_root())
        .args(args)
        .stderr(Stdio::inherit())
        .output()?;
    Ok((output.status, String::from_utf8(output.stdout)?))
}

fn assert_success(status: ExitStatus, label: &str) -> Result<(), Box<dyn Error>> {
    if status.success() {
        return Ok(());
    }
    Err(format!("{label} failed with status {:?}", status.code()).into())
}

fn apply_common_filters(args: &mut Vec<String>) {
    if let Ok(fixture_filter) = std::env::var("FSQLITE_MATRIX_DB") {
        args.push("--db".to_owned());
        args.push(fixture_filter);
    }
    if let Ok(workload_filter) = std::env::var("FSQLITE_MATRIX_WORKLOAD") {
        args.push("--preset".to_owned());
        args.push(workload_filter);
    }
    if let Ok(concurrency_filter) = std::env::var("FSQLITE_MATRIX_CONCURRENCY") {
        args.push("--concurrency".to_owned());
        args.push(concurrency_filter);
    }
    if let Ok(warmup) = std::env::var("FSQLITE_MATRIX_WARMUP") {
        args.push("--warmup".to_owned());
        args.push(warmup);
    }
    if let Ok(repeat) = std::env::var("FSQLITE_MATRIX_REPEAT") {
        args.push("--repeat".to_owned());
        args.push(repeat);
    }
    if let Ok(min_iters) = std::env::var("FSQLITE_MATRIX_MIN_ITERS") {
        args.push("--min-iters".to_owned());
        args.push(min_iters);
    }
    if let Ok(time_secs) = std::env::var("FSQLITE_MATRIX_TIME_SECS") {
        args.push("--time-secs".to_owned());
        args.push(time_secs);
    }
}

#[test]
#[ignore = "Runs the complete canonical benchmark matrix and writes artifact bundles."]
fn complete_benchmark_matrix() -> Result<(), Box<dyn Error>> {
    let artifact_dir = artifact_dir();
    fs::create_dir_all(&artifact_dir)?;

    let fixture_filter = std::env::var("FSQLITE_MATRIX_DB").ok();
    let workload_filter = std::env::var("FSQLITE_MATRIX_WORKLOAD").ok();
    let concurrency_filter = std::env::var("FSQLITE_MATRIX_CONCURRENCY").ok();
    let mode = std::env::var("FSQLITE_MATRIX_MODE").unwrap_or_else(|_| "full".to_owned());
    let stem = std::env::var("FSQLITE_MATRIX_OUTPUT_STEM").unwrap_or_else(|_| {
        fixture_filter
            .clone()
            .unwrap_or_else(|| "matrix".to_owned())
            .replace(',', "_")
    });

    let both_jsonl = artifact_dir.join(format!("{stem}_both.jsonl"));
    let both_md = artifact_dir.join(format!("{stem}_both.md"));
    let single_jsonl = artifact_dir.join(format!("{stem}_single_writer.jsonl"));
    let single_md = artifact_dir.join(format!("{stem}_single_writer.md"));
    let full_jsonl = artifact_dir.join(format!("{stem}_full.jsonl"));
    let manifest_json = artifact_dir.join(format!("{stem}.manifest.json"));

    let mut ran_both = false;
    let mut ran_single = false;
    let mut ran_sqlite = false;
    let mut ran_mvcc = false;

    if mode == "sqlite3" {
        let mut args = vec![
            "bench".to_owned(),
            "--engine".to_owned(),
            "sqlite3".to_owned(),
            "--output-jsonl".to_owned(),
            both_jsonl.to_string_lossy().into_owned(),
            "--output-md".to_owned(),
            both_md.to_string_lossy().into_owned(),
        ];
        apply_common_filters(&mut args);
        let sqlite = run_bench("sqlite3", &args)?;
        assert_success(sqlite, "sqlite3 bench")?;
        ran_sqlite = true;
    }

    if mode == "mvcc" {
        let mut args = vec![
            "bench".to_owned(),
            "--engine".to_owned(),
            "fsqlite".to_owned(),
            "--mvcc".to_owned(),
            "--output-jsonl".to_owned(),
            both_jsonl.to_string_lossy().into_owned(),
            "--output-md".to_owned(),
            both_md.to_string_lossy().into_owned(),
        ];
        apply_common_filters(&mut args);
        let mvcc = run_bench("mvcc", &args)?;
        assert_success(mvcc, "fsqlite mvcc bench")?;
        ran_mvcc = true;
    }

    if mode == "full" || mode == "both" {
        let mut args = vec![
            "bench".to_owned(),
            "--output-jsonl".to_owned(),
            both_jsonl.to_string_lossy().into_owned(),
            "--output-md".to_owned(),
            both_md.to_string_lossy().into_owned(),
        ];
        apply_common_filters(&mut args);
        let both = run_bench("both", &args)?;
        assert_success(both, "canonical both-mode bench")?;
        ran_both = true;
    }

    if mode == "full" || mode == "single_writer" {
        let mut args = vec![
            "bench".to_owned(),
            "--engine".to_owned(),
            "fsqlite".to_owned(),
            "--no-mvcc".to_owned(),
            "--output-jsonl".to_owned(),
            single_jsonl.to_string_lossy().into_owned(),
            "--output-md".to_owned(),
            single_md.to_string_lossy().into_owned(),
        ];
        apply_common_filters(&mut args);
        let single = run_bench("single_writer", &args)?;
        assert_success(single, "canonical single-writer bench")?;
        ran_single = true;
    }

    let mut combined = String::new();
    if ran_both || ran_sqlite || ran_mvcc {
        combined.push_str(&fs::read_to_string(&both_jsonl)?);
    }
    if ran_single {
        combined.push_str(&fs::read_to_string(&single_jsonl)?);
    }
    fs::write(&full_jsonl, combined)?;

    let manifest = serde_json::json!({
        "artifact_dir": artifact_dir,
        "fixture_filter": fixture_filter,
        "workload_filter": workload_filter,
        "concurrency_filter": concurrency_filter,
        "mode": mode,
        "matrix_both_jsonl": (ran_both || ran_sqlite || ran_mvcc).then_some(both_jsonl),
        "matrix_both_md": (ran_both || ran_sqlite || ran_mvcc).then_some(both_md),
        "matrix_single_writer_jsonl": ran_single.then_some(single_jsonl),
        "matrix_single_writer_md": ran_single.then_some(single_md),
        "matrix_full_jsonl": full_jsonl,
    });
    fs::write(&manifest_json, serde_json::to_vec_pretty(&manifest)?)?;

    Ok(())
}

#[test]
#[ignore = "Runs a targeted hot-path diagnosis for a single FrankenSQLite workload cell."]
fn hot_profile_diagnosis() -> Result<(), Box<dyn Error>> {
    let artifact_root = artifact_dir();
    fs::create_dir_all(&artifact_root)?;

    let db = std::env::var("FSQLITE_DIAG_DB")
        .or_else(|_| std::env::var("FSQLITE_MATRIX_DB"))
        .unwrap_or_else(|_| "frankensqlite".to_owned());
    let workload = std::env::var("FSQLITE_DIAG_WORKLOAD")
        .or_else(|_| std::env::var("FSQLITE_MATRIX_WORKLOAD"))
        .unwrap_or_else(|_| "commutative_inserts_disjoint_keys".to_owned());
    let concurrency = std::env::var("FSQLITE_DIAG_CONCURRENCY")
        .or_else(|_| std::env::var("FSQLITE_MATRIX_CONCURRENCY"))
        .unwrap_or_else(|_| "8".to_owned());
    let scale = std::env::var("FSQLITE_DIAG_SCALE").unwrap_or_else(|_| "100".to_owned());
    let output_dir = std::env::var_os("FSQLITE_DIAG_OUTPUT_DIR").map_or_else(
        || artifact_root.join(format!("{db}_{workload}_c{concurrency}_hot_profile")),
        PathBuf::from,
    );
    fs::create_dir_all(&output_dir)?;

    let pretty_json = output_dir.join("report.pretty.json");
    let mut args = vec![
        "hot-profile".to_owned(),
        "--db".to_owned(),
        db.clone(),
        "--workload".to_owned(),
        workload.clone(),
        "--concurrency".to_owned(),
        concurrency.clone(),
        "--scale".to_owned(),
        scale,
        "--output-dir".to_owned(),
        output_dir.to_string_lossy().into_owned(),
        "--pretty".to_owned(),
        "--mvcc".to_owned(),
    ];
    if std::env::var("FSQLITE_DIAG_INTEGRITY")
        .ok()
        .as_deref()
        .is_some_and(|value| value == "1")
    {
        args.push("--integrity-check".to_owned());
    }

    let (status, stdout) = run_command_capture_stdout(&args)?;
    if !stdout.trim().is_empty() {
        fs::write(&pretty_json, stdout.as_bytes())?;
        if let Ok(report) = serde_json::from_str::<serde_json::Value>(&stdout) {
            let engine_report = &report["engine_report"];
            let notes = engine_report["correctness"]["notes"].as_str().unwrap_or("");
            let error = engine_report["error"].as_str().unwrap_or("");
            println!(
                "[hot-profile] fixture={db} workload={workload} c={concurrency} ops/s={:.1} retries={} aborts={} error={} notes={}",
                engine_report["ops_per_sec"].as_f64().unwrap_or(0.0),
                engine_report["retries"].as_u64().unwrap_or(0),
                engine_report["aborts"].as_u64().unwrap_or(0),
                error,
                notes,
            );
        }
    }

    assert_success(status, "hot-profile diagnosis")?;
    Ok(())
}
