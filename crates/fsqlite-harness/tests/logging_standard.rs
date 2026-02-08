use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tempfile::tempdir;

use fsqlite_harness::log::{
    ConformanceDiff, LOG_SCHEMA_VERSION, LifecycleEventKind, REQUIRED_BUNDLE_FILES, RunStatus,
    init_repro_bundle, validate_bundle, validate_bundle_meta, validate_events_jsonl,
    validate_required_files,
};

const BEAD_ID: &str = "bd-1fpm";

#[test]
fn test_log_bundle_meta_json_schema_valid() {
    let temp = tempdir().expect("tempdir should be created");
    let mut bundle = init_repro_bundle(temp.path(), "harness", "meta_schema", 4242)
        .expect("bundle initialization should succeed");

    bundle
        .emit_event(LifecycleEventKind::Setup, "setup", BTreeMap::new())
        .expect("setup event should be emitted");

    let bundle_root = bundle
        .finish(RunStatus::Passed)
        .expect("bundle finalization should succeed");

    let meta = validate_bundle_meta(&bundle_root).expect("meta schema should validate");
    assert_eq!(
        meta.schema_version, LOG_SCHEMA_VERSION,
        "bead_id={BEAD_ID} case=meta_schema_version"
    );
    assert_eq!(meta.suite, "harness", "bead_id={BEAD_ID} case=meta_suite");
    assert_eq!(
        meta.case_id, "meta_schema",
        "bead_id={BEAD_ID} case=meta_case_id"
    );
    assert_eq!(meta.seed, 4242, "bead_id={BEAD_ID} case=meta_seed");
}

#[test]
fn test_events_jsonl_is_valid_jsonl() {
    let temp = tempdir().expect("tempdir should be created");
    let mut bundle = init_repro_bundle(temp.path(), "harness", "jsonl_validation", 7)
        .expect("bundle initialization should succeed");

    let mut payload = BTreeMap::new();
    payload.insert("phase".to_string(), Value::String("core".to_string()));
    payload.insert("step_idx".to_string(), json!(1));

    bundle
        .emit_event(LifecycleEventKind::Step, "do_work", payload)
        .expect("step event should be emitted");

    let bundle_root = bundle
        .finish(RunStatus::Passed)
        .expect("bundle finalization should succeed");

    let events = validate_events_jsonl(&bundle_root).expect("events.jsonl should parse");
    assert!(
        !events.is_empty(),
        "bead_id={BEAD_ID} case=events_non_empty"
    );
    assert_eq!(
        events.first().map(|event| event.kind),
        Some(LifecycleEventKind::RunStart),
        "bead_id={BEAD_ID} case=events_start"
    );
    assert_eq!(
        events.last().map(|event| event.kind),
        Some(LifecycleEventKind::RunEnd),
        "bead_id={BEAD_ID} case=events_end"
    );
}

#[test]
fn test_bundle_contains_required_files() {
    let temp = tempdir().expect("tempdir should be created");
    let incomplete_root = temp.path().join("incomplete_bundle");
    std::fs::create_dir_all(&incomplete_root).expect("incomplete root should be created");
    std::fs::write(incomplete_root.join("meta.json"), "{}").expect("meta stub should be written");

    let error =
        validate_required_files(&incomplete_root).expect_err("missing files must fail validation");
    let rendered = error.to_string();
    for required in REQUIRED_BUNDLE_FILES {
        if required != "meta.json" {
            assert!(
                rendered.contains(required),
                "bead_id={BEAD_ID} case=required_file_missing required={required} err={rendered}"
            );
        }
    }
}

#[test]
fn test_e2e_harness_emits_repro_bundle_on_failure() {
    let temp = tempdir().expect("tempdir should be created");
    let bundle_root = run_known_failing_harness_case(temp.path())
        .expect("known failing harness case should still emit bundle");

    validate_bundle(&bundle_root).expect("bundle should satisfy required validation checks");

    assert!(
        bundle_root.join("db_snapshot.json").is_file(),
        "bead_id={BEAD_ID} case=e2e_db_snapshot_present"
    );
    assert!(
        bundle_root.join("db-wal").is_file(),
        "bead_id={BEAD_ID} case=e2e_wal_present"
    );
    assert!(
        bundle_root.join("oracle_diff.json").is_file(),
        "bead_id={BEAD_ID} case=e2e_oracle_diff_present"
    );
}

fn run_known_failing_harness_case(base_dir: &Path) -> fsqlite_error::Result<PathBuf> {
    let mut bundle = init_repro_bundle(base_dir, "harness_e2e", "known_failure", 1337)?;

    bundle.emit_event(
        LifecycleEventKind::Setup,
        "setup",
        BTreeMap::from([("stage".to_string(), Value::String("begin".to_string()))]),
    )?;

    bundle.append_stdout("running known failing case")?;
    bundle.append_stderr("assertion failed: expected 1 got 2")?;

    bundle.write_artifact_json("db_snapshot.json", &json!({ "tables": [] }))?;
    bundle.write_artifact_json("db-wal", &json!({ "frames": [] }))?;

    bundle.record_conformance_diff(&ConformanceDiff {
        case_id: "known_failure".to_string(),
        sql: "SELECT 1".to_string(),
        params: "[]".to_string(),
        oracle_result: "[[1]]".to_string(),
        franken_result: "[[2]]".to_string(),
        diff: "[{\"index\":0,\"expected\":1,\"actual\":2}]".to_string(),
    })?;

    bundle.emit_event(
        LifecycleEventKind::Assertion,
        "assertion_failed",
        BTreeMap::from([("reason".to_string(), Value::String("mismatch".to_string()))]),
    )?;

    bundle.finish(RunStatus::Failed)
}
