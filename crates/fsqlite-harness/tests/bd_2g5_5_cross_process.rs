use std::path::{Path, PathBuf};

use fsqlite_harness::cross_process_crash_harness::{
    CRASH_POINT_ALL, CROSS_PROCESS_CRASH_BEAD_ID, CROSS_PROCESS_CRASH_SCHEMA_VERSION,
    CrossProcessCrashConfig, CrossProcessCrashReport, PROCESS_ROLE_ALL,
    load_cross_process_crash_report, run_cross_process_crash_harness,
    write_cross_process_crash_report, write_cross_process_event_log,
};

const BEAD_ID: &str = "bd-2g5.5.1";

fn cycles_from_env() -> usize {
    std::env::var("BD_2G5_5_CYCLES")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(100)
}

fn seed_from_env() -> u64 {
    std::env::var("BD_2G5_5_SEED")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(270_550_001)
}

fn run_tag_from_env() -> String {
    std::env::var("BD_2G5_5_RUN_TAG").unwrap_or_else(|_| "default".to_owned())
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("workspace_root_canonicalize_failed error={error}"))
}

fn write_artifacts(report: &CrossProcessCrashReport) -> Result<(PathBuf, PathBuf), String> {
    let root = workspace_root()?;
    let output_dir = root.join("test-results").join("bd_2g5_5");
    std::fs::create_dir_all(&output_dir).map_err(|error| {
        format!(
            "artifact_dir_create_failed path={} error={error}",
            output_dir.display()
        )
    })?;

    let report_path = output_dir.join(format!("{}.json", report.run_id));
    let events_path = output_dir.join(format!("{}.events.jsonl", report.run_id));

    write_cross_process_crash_report(&report_path, report)?;
    write_cross_process_event_log(&events_path, &report.events)?;
    Ok((report_path, events_path))
}

#[test]
fn cross_process_crash_matrix_is_complete_and_invariants_hold() {
    let minimum_cycles = PROCESS_ROLE_ALL.len() * CRASH_POINT_ALL.len();
    let cycles = cycles_from_env().max(minimum_cycles);
    let seed = seed_from_env();
    let run_tag = run_tag_from_env();

    let config = CrossProcessCrashConfig {
        seed,
        cycles,
        process_count: 8,
        run_id: format!("bd-2g5-5-cross-{run_tag}-seed-{seed:016x}-cycles-{cycles}"),
        trace_id: format!("trace-bd-2g5-5-1-{run_tag}-{seed:016x}"),
    };

    let report = run_cross_process_crash_harness(&config);

    assert_eq!(
        report.schema_version, CROSS_PROCESS_CRASH_SCHEMA_VERSION,
        "bead_id={BEAD_ID} case=schema_version"
    );
    assert_eq!(
        report.bead_id, CROSS_PROCESS_CRASH_BEAD_ID,
        "bead_id={BEAD_ID} case=bead_id"
    );
    assert_eq!(
        report.cycles_executed, cycles,
        "bead_id={BEAD_ID} case=cycles_executed"
    );
    assert_eq!(
        report.process_count, config.process_count,
        "bead_id={BEAD_ID} case=process_count"
    );
    assert!(
        report.scenario_matrix_complete,
        "bead_id={BEAD_ID} case=matrix_complete expected={} actual={}",
        report.scenario_matrix_expected, report.scenario_matrix_covered
    );
    assert_eq!(
        report.scenario_matrix_expected, minimum_cycles,
        "bead_id={BEAD_ID} case=matrix_expected"
    );
    assert!(
        report.slot_reclamation_pass,
        "bead_id={BEAD_ID} case=slot_reclamation"
    );
    assert!(
        report.seqlock_no_torn_reads,
        "bead_id={BEAD_ID} case=seqlock_no_torn_reads"
    );
    assert!(
        report.left_right_linearizable,
        "bead_id={BEAD_ID} case=left_right_linearizable"
    );
    assert_eq!(
        report.orphan_slots_after_run, 0,
        "bead_id={BEAD_ID} case=orphan_slots"
    );
    assert!(
        report.schema_conformance_errors.is_empty(),
        "bead_id={BEAD_ID} case=schema_conformance errors={:?}",
        report.schema_conformance_errors
    );
    assert!(
        report
            .events
            .iter()
            .all(|event| event.trace_id == report.trace_id
                && event.run_id == report.run_id
                && !event.scenario_id.is_empty()
                && !event.process_role.is_empty()
                && !event.crash_point.is_empty()
                && event.duration_micros > 0
                && !event.diagnostic.is_empty()),
        "bead_id={BEAD_ID} case=required_log_fields"
    );

    let (report_path, events_path) = write_artifacts(&report).expect("write artifacts");
    let loaded = load_cross_process_crash_report(&report_path).expect("load report");
    assert_eq!(
        loaded.triage_line(),
        report.triage_line(),
        "bead_id={BEAD_ID} case=triage_roundtrip"
    );

    println!("bead_id={BEAD_ID} path={}", report_path.display());
    println!("bead_id={BEAD_ID} events_path={}", events_path.display());
    println!("bead_id={BEAD_ID} replay_command={}", report.replay_command);
}

#[test]
fn report_is_deterministic_for_fixed_seed_and_config() {
    let config = CrossProcessCrashConfig {
        seed: 44,
        cycles: 40,
        process_count: 8,
        run_id: "bd-2g5-5-cross-deterministic".to_owned(),
        trace_id: "trace-bd-2g5-5-1-deterministic".to_owned(),
    };

    let first = run_cross_process_crash_harness(&config);
    let second = run_cross_process_crash_harness(&config);

    let first_json = first.to_json().expect("serialize first");
    let second_json = second.to_json().expect("serialize second");
    assert_eq!(
        first_json, second_json,
        "bead_id={BEAD_ID} case=determinism"
    );

    let parsed = CrossProcessCrashReport::from_json(&first_json).expect("parse report");
    assert_eq!(parsed.replay_command, config.replay_command());
    assert!(
        parsed
            .replay_command
            .contains("cargo test -p fsqlite-harness --test bd_2g5_5_cross_process"),
        "bead_id={BEAD_ID} case=replay_command_shape"
    );
}
