use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use fsqlite_harness::performance_regression_detector::{
    BEAD_ID, BenchmarkSample, HostContext, RegressionSeverity, RegressionTolerance,
    evaluate_candidate_against_baseline, load_baseline_samples, write_baseline_samples,
};
use tempfile::TempDir;

fn host() -> HostContext {
    HostContext {
        os: "linux".to_owned(),
        arch: "x86_64".to_owned(),
        cpu_model: "test-cpu".to_owned(),
        rustc_version: "nightly-2026-02-10".to_owned(),
    }
}

fn params() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("connections".to_owned(), "32".to_owned()),
        ("dataset_rows".to_owned(), "25000".to_owned()),
    ])
}

fn sample(run_id: &str, p95: f64, throughput: f64) -> BenchmarkSample {
    BenchmarkSample {
        scenario_id: "bm-write-contention-macro".to_owned(),
        run_id: run_id.to_owned(),
        git_sha: "abc1234".to_owned(),
        seed: 42,
        p50_micros: p95 * 0.70,
        p95_micros: p95,
        p99_micros: p95 * 1.20,
        throughput_ops_per_sec: throughput,
        host: host(),
        benchmark_params: params(),
    }
}

fn baseline_samples() -> Vec<BenchmarkSample> {
    vec![
        sample("base-1", 1_000.0, 5_000.0),
        sample("base-2", 1_020.0, 4_980.0),
        sample("base-3", 995.0, 5_050.0),
        sample("base-4", 1_010.0, 4_970.0),
        sample("base-5", 1_005.0, 5_010.0),
    ]
}

#[test]
fn test_noisy_candidate_is_not_flagged_as_regression() {
    let baseline = baseline_samples();
    let candidate = sample("candidate-noise", 1_025.0, 4_970.0);

    let report =
        evaluate_candidate_against_baseline(&baseline, &candidate, &RegressionTolerance::default())
            .expect("evaluate candidate");

    assert_eq!(report.bead_id, BEAD_ID);
    assert_eq!(report.assessment.severity, RegressionSeverity::None);
    assert!(
        report.assessment.confidence > 0.0,
        "confidence metadata should be populated"
    );
}

#[test]
fn test_material_regression_is_classified_critical() {
    let baseline = baseline_samples();
    let candidate = sample("candidate-regression", 1_400.0, 3_600.0);

    let report =
        evaluate_candidate_against_baseline(&baseline, &candidate, &RegressionTolerance::default())
            .expect("evaluate candidate");

    assert_eq!(report.assessment.severity, RegressionSeverity::Critical);
    assert!(
        report
            .assessment
            .reasons
            .iter()
            .any(|reason| reason.contains("critical threshold")),
        "critical assessment should include actionable reasons"
    );
}

#[test]
fn test_report_retains_host_context_and_benchmark_params() {
    let baseline = baseline_samples();
    let mut candidate = sample("candidate-context", 1_010.0, 5_000.0);
    candidate
        .benchmark_params
        .insert("checkpoint_interval".to_owned(), "2048".to_owned());

    let report =
        evaluate_candidate_against_baseline(&baseline, &candidate, &RegressionTolerance::default())
            .expect("evaluate candidate");

    assert_eq!(report.candidate.host.cpu_model, "test-cpu");
    assert_eq!(
        report
            .candidate
            .benchmark_params
            .get("checkpoint_interval")
            .map(String::as_str),
        Some("2048")
    );
}

fn detector_binary_path() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_performance_regression_detector"))
}

fn write_candidate_file(dir: &TempDir, name: &str, sample: &BenchmarkSample) -> PathBuf {
    let path = dir.path().join(name);
    let payload = serde_json::to_vec_pretty(sample).expect("serialize sample");
    fs::write(&path, payload).expect("write candidate");
    path
}

#[test]
fn test_cli_fail_on_warning_returns_non_zero() {
    let temp_dir = tempfile::tempdir().expect("create tempdir");
    let baseline_path = temp_dir.path().join("baseline.json");
    write_baseline_samples(&baseline_path, &baseline_samples()).expect("write baseline file");
    let baseline_loaded = load_baseline_samples(&baseline_path).expect("load baseline file");
    assert_eq!(baseline_loaded.len(), 5);

    let candidate_path = write_candidate_file(
        &temp_dir,
        "candidate.json",
        &sample("candidate-warn", 1_150.0, 4_900.0),
    );

    let output = Command::new(detector_binary_path())
        .arg("--baseline")
        .arg(&baseline_path)
        .arg("--candidate")
        .arg(&candidate_path)
        .arg("--fail-on")
        .arg("warning")
        .output()
        .expect("run performance_regression_detector");

    assert_eq!(
        output.status.code(),
        Some(1),
        "expected non-zero exit when severity >= warning"
    );
}
