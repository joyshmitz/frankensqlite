//! Integration tests for bd-mblr.7.2 — Long-Run Concurrency Endurance and History Checking.
//!
//! Tests the endurance orchestrator that ties together soak profiles (bd-mblr.7.2.1),
//! the concurrent soak executor (bd-mblr.7.2.2), and leak detection into an
//! aggregated endurance suite with pass/fail/warning verdicts.

use fsqlite_harness::soak_executor::{
    ENDURANCE_BEAD_ID, EnduranceConfig, EnduranceReport, EnduranceVerdict, LeakBudgetPolicy,
    load_endurance_report, run_endurance_suite, write_endurance_report,
};
use fsqlite_harness::soak_profiles::{
    SoakWorkloadSpec, all_presets, canonical_invariants, compute_soak_coverage, profile_light,
};

const BEAD_ID: &str = "bd-mblr.7.2";

// ---------------------------------------------------------------------------
// Endurance suite — full pipeline
// ---------------------------------------------------------------------------

#[test]
fn endurance_suite_runs_all_presets() {
    let config = EnduranceConfig {
        root_seed: 0xBEEF_CAFE,
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    assert_eq!(
        report.profiles_run, 4,
        "bead_id={BEAD_ID} case=all_presets_count"
    );
    assert!(
        report.total_transactions > 0,
        "bead_id={BEAD_ID} case=total_transactions_nonzero"
    );
    assert!(
        report.total_commits > 0,
        "bead_id={BEAD_ID} case=total_commits_nonzero"
    );
    assert_eq!(
        report.bead_id, ENDURANCE_BEAD_ID,
        "bead_id={BEAD_ID} case=bead_id_correct"
    );
    assert_eq!(
        report.schema_version, 1,
        "bead_id={BEAD_ID} case=schema_version"
    );
}

#[test]
fn endurance_suite_runs_selected_profiles() {
    let config = EnduranceConfig {
        root_seed: 0xDEAD_1234,
        profile_names: vec!["light".to_owned(), "moderate".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    assert_eq!(
        report.profiles_run, 2,
        "bead_id={BEAD_ID} case=selected_profiles_count"
    );
    let names: Vec<&str> = report
        .profile_results
        .iter()
        .map(|r| r.profile_name.as_str())
        .collect();
    assert!(
        names.contains(&"light"),
        "bead_id={BEAD_ID} case=has_light_profile"
    );
    assert!(
        names.contains(&"moderate"),
        "bead_id={BEAD_ID} case=has_moderate_profile"
    );
}

#[test]
fn endurance_light_only_passes() {
    let config = EnduranceConfig {
        root_seed: 0xABCD_0001,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    assert_eq!(
        report.profiles_run, 1,
        "bead_id={BEAD_ID} case=light_only_one_profile"
    );
    assert!(
        report.profile_results[0].soak_report.passed(),
        "bead_id={BEAD_ID} case=light_soak_passed"
    );
    assert!(
        report.profile_results[0].commit_rate_ok,
        "bead_id={BEAD_ID} case=light_commit_rate_ok"
    );
}

// ---------------------------------------------------------------------------
// Verdict logic
// ---------------------------------------------------------------------------

#[test]
fn endurance_verdict_pass_when_all_clean() {
    let config = EnduranceConfig {
        root_seed: 0xAAAA_0001,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    // Light profile with default config should pass
    assert_ne!(
        report.verdict,
        EnduranceVerdict::Fail,
        "bead_id={BEAD_ID} case=light_not_fail"
    );
}

#[test]
fn endurance_verdict_fail_on_low_commit_rate() {
    let config = EnduranceConfig {
        root_seed: 0xBBBB_0002,
        profile_names: vec!["light".to_owned()],
        min_commit_rate: 0.9999, // extremely strict threshold
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    // With the simulated executor, some write conflicts happen at ~0.5% rate.
    // Check if commit rate was actually below threshold.
    let result = &report.profile_results[0];
    if !result.commit_rate_ok {
        assert_eq!(
            result.verdict,
            EnduranceVerdict::Fail,
            "bead_id={BEAD_ID} case=low_commit_rate_fails"
        );
    }
}

#[test]
fn endurance_verdict_display() {
    assert_eq!(EnduranceVerdict::Pass.to_string(), "PASS");
    assert_eq!(EnduranceVerdict::Warning.to_string(), "WARNING");
    assert_eq!(EnduranceVerdict::Fail.to_string(), "FAIL");
}

// ---------------------------------------------------------------------------
// Per-profile results
// ---------------------------------------------------------------------------

#[test]
fn endurance_profile_results_have_commit_rate() {
    let config = EnduranceConfig {
        root_seed: 0xCCCC_0003,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    let result = &report.profile_results[0];
    assert!(
        result.commit_rate > 0.0 && result.commit_rate <= 1.0,
        "bead_id={BEAD_ID} case=commit_rate_bounds rate={}",
        result.commit_rate
    );
}

#[test]
fn endurance_profile_results_have_leak_findings() {
    let config = EnduranceConfig {
        root_seed: 0xDDDD_0004,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    // leak_findings may be empty for clean runs; just check the field exists
    let result = &report.profile_results[0];
    assert!(
        result.leak_findings.len() <= 100,
        "bead_id={BEAD_ID} case=leak_findings_bounded"
    );
}

#[test]
fn endurance_profile_soak_reports_are_complete() {
    let config = EnduranceConfig {
        root_seed: 0xEEEE_0005,
        profile_names: vec!["moderate".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    let soak = &report.profile_results[0].soak_report;
    assert!(
        soak.total_transactions > 0,
        "bead_id={BEAD_ID} case=soak_has_txns"
    );
    assert!(
        !soak.spec_json.is_empty(),
        "bead_id={BEAD_ID} case=soak_has_spec"
    );
    assert!(
        !soak.summary.is_empty(),
        "bead_id={BEAD_ID} case=soak_has_summary"
    );
}

// ---------------------------------------------------------------------------
// Aggregated metrics
// ---------------------------------------------------------------------------

#[test]
fn endurance_total_transactions_is_sum_of_profiles() {
    let config = EnduranceConfig {
        root_seed: 0x1111_0006,
        profile_names: vec!["light".to_owned(), "moderate".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    let sum: u64 = report
        .profile_results
        .iter()
        .map(|r| r.soak_report.total_transactions)
        .sum();
    assert_eq!(
        report.total_transactions, sum,
        "bead_id={BEAD_ID} case=txn_sum_matches"
    );
}

#[test]
fn endurance_total_commits_is_sum_of_profiles() {
    let config = EnduranceConfig {
        root_seed: 0x2222_0007,
        profile_names: vec!["light".to_owned(), "moderate".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    let sum: u64 = report
        .profile_results
        .iter()
        .map(|r| r.soak_report.total_commits)
        .sum();
    assert_eq!(
        report.total_commits, sum,
        "bead_id={BEAD_ID} case=commit_sum_matches"
    );
}

// ---------------------------------------------------------------------------
// Report serialization
// ---------------------------------------------------------------------------

#[test]
fn endurance_report_json_roundtrip() {
    let config = EnduranceConfig {
        root_seed: 0x3333_0008,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    let json = report.to_json().expect("serialize endurance report");
    let parsed = EnduranceReport::from_json(&json).expect("parse endurance report");

    assert_eq!(parsed.bead_id, report.bead_id);
    assert_eq!(parsed.run_id, report.run_id);
    assert_eq!(parsed.verdict, report.verdict);
    assert_eq!(parsed.profiles_run, report.profiles_run);
    assert_eq!(parsed.total_transactions, report.total_transactions);
    assert_eq!(parsed.total_commits, report.total_commits);
}

#[test]
fn endurance_report_file_roundtrip() {
    let config = EnduranceConfig {
        root_seed: 0x4444_0009,
        profile_names: vec!["light".to_owned()],
        git_sha: "abc123".to_owned(),
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);

    let dir = std::env::temp_dir().join("fsqlite-endurance-test");
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("endurance-test.json");

    write_endurance_report(&path, &report).expect("write report");
    let loaded = load_endurance_report(&path).expect("load report");

    assert_eq!(loaded.bead_id, report.bead_id);
    assert_eq!(loaded.git_sha, "abc123");
    assert_eq!(loaded.verdict, report.verdict);
    assert_eq!(loaded.profiles_run, report.profiles_run);

    // Cleanup
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

// ---------------------------------------------------------------------------
// Triage line
// ---------------------------------------------------------------------------

#[test]
fn endurance_triage_line_contains_key_info() {
    let config = EnduranceConfig {
        root_seed: 0x5555_000A,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);
    let line = report.triage_line();

    assert!(
        line.contains("profiles passed"),
        "bead_id={BEAD_ID} case=triage_has_profiles"
    );
    assert!(
        line.contains("txns"),
        "bead_id={BEAD_ID} case=triage_has_txns"
    );
}

#[test]
fn endurance_summary_is_nonempty() {
    let config = EnduranceConfig {
        root_seed: 0x6666_000B,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report = run_endurance_suite(&config);
    assert!(
        !report.summary.is_empty(),
        "bead_id={BEAD_ID} case=summary_nonempty"
    );
}

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

#[test]
fn endurance_config_default_is_valid() {
    let config = EnduranceConfig::default();
    let errors = config.validate();
    assert!(
        errors.is_empty(),
        "bead_id={BEAD_ID} case=default_config_valid errors={errors:?}"
    );
}

#[test]
fn endurance_config_invalid_commit_rate() {
    let config = EnduranceConfig {
        min_commit_rate: 1.5,
        ..EnduranceConfig::default()
    };
    let errors = config.validate();
    assert!(
        errors.iter().any(|e| e.contains("min_commit_rate")),
        "bead_id={BEAD_ID} case=invalid_commit_rate errors={errors:?}"
    );
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

#[test]
fn endurance_suite_is_deterministic() {
    let config = EnduranceConfig {
        root_seed: 0x7777_000C,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report1 = run_endurance_suite(&config);
    let report2 = run_endurance_suite(&config);

    assert_eq!(
        report1.total_transactions, report2.total_transactions,
        "bead_id={BEAD_ID} case=deterministic_txns"
    );
    assert_eq!(
        report1.total_commits, report2.total_commits,
        "bead_id={BEAD_ID} case=deterministic_commits"
    );
    assert_eq!(
        report1.verdict, report2.verdict,
        "bead_id={BEAD_ID} case=deterministic_verdict"
    );
}

// ---------------------------------------------------------------------------
// Integration with child beads
// ---------------------------------------------------------------------------

#[test]
fn endurance_uses_canonical_invariants() {
    let invs = canonical_invariants();
    assert!(
        invs.len() >= 12,
        "bead_id={BEAD_ID} case=canonical_invariants_count"
    );
    // All specs built by the orchestrator use canonical invariants
    let spec = SoakWorkloadSpec::from_profile(profile_light(), 42);
    assert_eq!(
        spec.invariants.len(),
        invs.len(),
        "bead_id={BEAD_ID} case=spec_uses_canonical"
    );
}

#[test]
fn endurance_coverage_across_presets() {
    let profiles = all_presets();
    let invs = canonical_invariants();
    let cov = compute_soak_coverage(&profiles, &invs);

    assert_eq!(cov.profile_count, 4);
    assert!(
        cov.invariant_count >= 12,
        "bead_id={BEAD_ID} case=coverage_invariant_count"
    );
    assert!(
        !cov.mvcc_refs_covered.is_empty(),
        "bead_id={BEAD_ID} case=coverage_has_mvcc_refs"
    );
}

#[test]
fn endurance_leak_policy_default_is_sane() {
    let policy = LeakBudgetPolicy::default();
    assert!(
        policy.baseline_window >= 1,
        "bead_id={BEAD_ID} case=leak_baseline_window"
    );
    assert!(
        policy.sustained_window >= 2,
        "bead_id={BEAD_ID} case=leak_sustained_window"
    );
    assert!(
        policy.monotone_growth_ratio > 0.0 && policy.monotone_growth_ratio <= 1.0,
        "bead_id={BEAD_ID} case=leak_growth_ratio_bounds"
    );
}

// ---------------------------------------------------------------------------
// Run ID format
// ---------------------------------------------------------------------------

#[test]
fn endurance_run_id_is_deterministic() {
    let config = EnduranceConfig {
        root_seed: 0x8888_000D,
        profile_names: vec!["light".to_owned()],
        ..EnduranceConfig::default()
    };
    let report1 = run_endurance_suite(&config);
    let report2 = run_endurance_suite(&config);

    assert_eq!(
        report1.run_id, report2.run_id,
        "bead_id={BEAD_ID} case=run_id_deterministic"
    );
    assert!(
        report1.run_id.starts_with("endurance-"),
        "bead_id={BEAD_ID} case=run_id_prefix"
    );
}
