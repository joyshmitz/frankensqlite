use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use fsqlite_harness::fault_vfs::{
    FaultSpec, FaultState, ReadDecision, SyncDecision, TEST_VFS_FAULT_COUNTER_NAME, WriteDecision,
};
use serde::Serialize;

const BEAD_ID: &str = "bd-3u7.3";
const SCENARIO_ID: &str = "TEST-VFS-FAULT-REPLAY";
const DEFAULT_SEED: u64 = 0x3A7D_3A7D_u64;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct ReplayRun {
    replay_seed: u64,
    steps: Vec<String>,
    trigger_log: Vec<String>,
    metric_name: String,
    metrics_by_fault_type: BTreeMap<String, u64>,
    metric_total: u64,
}

#[derive(Debug, Serialize)]
struct ReplaySuiteArtifact {
    schema_version: u32,
    bead_id: String,
    run_id: String,
    trace_id: String,
    scenario_id: String,
    seed: u64,
    alternate_seed: u64,
    duration_ms: u128,
    deterministic_match: bool,
    replay_commands: Vec<String>,
    primary: ReplayRun,
}

fn scenario_seed() -> u64 {
    std::env::var("BD_3U7_3_SEED")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(DEFAULT_SEED)
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("workspace_root_canonicalize_failed: {error}"))
}

fn write_suite_artifact(artifact: &ReplaySuiteArtifact) -> Result<PathBuf, String> {
    let root = workspace_root()?;
    let output_dir = root.join("test-results").join("bd_3u7_3");
    fs::create_dir_all(&output_dir).map_err(|error| {
        format!(
            "artifact_dir_create_failed path={} error={error}",
            output_dir.display()
        )
    })?;

    let output_path = output_dir.join(format!("{}.json", artifact.run_id));
    let payload = serde_json::to_string_pretty(artifact)
        .map_err(|error| format!("artifact_serialize_failed error={error}"))?;
    fs::write(&output_path, payload).map_err(|error| {
        format!(
            "artifact_write_failed path={} error={error}",
            output_path.display()
        )
    })?;

    Ok(output_path)
}

fn write_decision_label(decision: &WriteDecision) -> String {
    match decision {
        WriteDecision::Allow => "allow".to_owned(),
        WriteDecision::TornWrite { valid_bytes } => format!("torn_write:{valid_bytes}"),
        WriteDecision::PartialWrite { valid_bytes } => format!("partial_write:{valid_bytes}"),
        WriteDecision::IoError => "io_error".to_owned(),
        WriteDecision::DiskFull => "disk_full".to_owned(),
        WriteDecision::PoweredOff => "powered_off".to_owned(),
    }
}

fn read_decision_label(decision: &ReadDecision) -> &'static str {
    match decision {
        ReadDecision::Allow => "allow",
        ReadDecision::IoError => "io_error",
        ReadDecision::PoweredOff => "powered_off",
    }
}

fn sync_decision_label(decision: &SyncDecision) -> &'static str {
    match decision {
        SyncDecision::Allow => "allow",
        SyncDecision::PowerCut => "power_cut",
        SyncDecision::IoError => "io_error",
        SyncDecision::PoweredOff => "powered_off",
    }
}

fn run_fault_replay(seed: u64) -> ReplayRun {
    let state = FaultState::new_with_seed(seed);
    state.inject_fault(
        FaultSpec::write_failure("*.db")
            .after_count(1)
            .trigger_count(2)
            .build(),
    );
    state.inject_fault(
        FaultSpec::read_failure("*.db")
            .after_count(2)
            .trigger_count(1)
            .build(),
    );
    state.inject_fault(
        FaultSpec::latency("*.db")
            .latency_millis(0)
            .jitter_millis(2)
            .trigger_count(16)
            .build(),
    );
    state.inject_fault(
        FaultSpec::partial_write("*.wal")
            .at_offset_bytes(64)
            .bytes_written(7)
            .build(),
    );
    state.inject_fault(FaultSpec::disk_full("*.wal").trigger_count(1).build());
    state.inject_fault(FaultSpec::power_cut("*.db").after_nth_sync(1).build());

    let db = Path::new("scenario.db");
    let wal = Path::new("scenario.wal");
    let mut steps = Vec::new();

    let write_db_first = state.check_write(db, 0, 16);
    assert_eq!(write_db_first, WriteDecision::Allow);
    steps.push(format!(
        "write_db_0={}",
        write_decision_label(&write_db_first)
    ));

    let write_db_second = state.check_write(db, 16, 16);
    assert_eq!(write_db_second, WriteDecision::IoError);
    steps.push(format!(
        "write_db_1={}",
        write_decision_label(&write_db_second)
    ));

    let write_wal_partial = state.check_write(wal, 64, 16);
    assert_eq!(
        write_wal_partial,
        WriteDecision::PartialWrite { valid_bytes: 7 }
    );
    steps.push(format!(
        "write_wal_0={}",
        write_decision_label(&write_wal_partial)
    ));

    let write_wal_disk_full = state.check_write(wal, 80, 16);
    assert_eq!(write_wal_disk_full, WriteDecision::DiskFull);
    steps.push(format!(
        "write_wal_1={}",
        write_decision_label(&write_wal_disk_full)
    ));

    let read_db_first = state.check_read(db, 0, 8);
    assert_eq!(read_db_first, ReadDecision::Allow);
    steps.push(format!("read_db_0={}", read_decision_label(&read_db_first)));

    let read_db_second = state.check_read(db, 8, 8);
    assert_eq!(read_db_second, ReadDecision::Allow);
    steps.push(format!(
        "read_db_1={}",
        read_decision_label(&read_db_second)
    ));

    let read_db_third = state.check_read(db, 16, 8);
    assert_eq!(read_db_third, ReadDecision::IoError);
    steps.push(format!("read_db_2={}", read_decision_label(&read_db_third)));

    let sync_first = state.check_sync(db);
    assert_eq!(sync_first, SyncDecision::Allow);
    steps.push(format!("sync_db_0={}", sync_decision_label(&sync_first)));

    let sync_second = state.check_sync(db);
    assert_eq!(sync_second, SyncDecision::PowerCut);
    steps.push(format!("sync_db_1={}", sync_decision_label(&sync_second)));

    let write_after_power_cut = state.check_write(db, 32, 16);
    assert_eq!(write_after_power_cut, WriteDecision::PoweredOff);
    steps.push(format!(
        "write_db_after_power_cut={}",
        write_decision_label(&write_after_power_cut)
    ));

    let metrics = state.metrics_snapshot();
    let trigger_log = state
        .triggered_faults()
        .into_iter()
        .map(|record| format!("{}|{:?}|{}", record.spec_index, record.kind, record.detail))
        .collect();

    ReplayRun {
        replay_seed: state.replay_seed(),
        steps,
        trigger_log,
        metric_name: metrics.metric_name.to_owned(),
        metrics_by_fault_type: metrics.by_fault_type,
        metric_total: metrics.total,
    }
}

#[test]
fn test_e2e_bd_3u7_3_fault_vfs_deterministic_replay() {
    let seed = scenario_seed();
    let alternate_seed = seed ^ 0x9E37_79B9_7F4A_7C15_u64;
    let started = Instant::now();

    let first = run_fault_replay(seed);
    let second = run_fault_replay(seed);
    assert_eq!(
        first, second,
        "bead_id={BEAD_ID} deterministic replay failed for identical seed"
    );

    let alternate = run_fault_replay(alternate_seed);
    assert_ne!(
        first.trigger_log, alternate.trigger_log,
        "bead_id={BEAD_ID} alternate seed should change deterministic latency trace"
    );

    assert_eq!(first.metric_name, TEST_VFS_FAULT_COUNTER_NAME);
    for fault_type in [
        "write_failure",
        "read_failure",
        "latency",
        "partial_write",
        "disk_full",
        "power_cut",
    ] {
        assert!(
            first.metrics_by_fault_type.contains_key(fault_type),
            "bead_id={BEAD_ID} missing fault counter for {fault_type}",
        );
    }

    let run_id = format!("{BEAD_ID}-seed-{seed:016x}");
    let trace_id = format!("trace-{seed:016x}");
    let replay_commands = vec![
        format!(
            "BD_3U7_3_SEED={seed} cargo test -p fsqlite-harness --test bd_3u7_3_fault_vfs_deterministic_replay -- --nocapture"
        ),
        format!("BD_3U7_3_SEED={seed} scripts/verify_bd_3u7_3_fault_vfs_deterministic_replay.sh"),
    ];

    let artifact = ReplaySuiteArtifact {
        schema_version: 1,
        bead_id: BEAD_ID.to_owned(),
        run_id: run_id.clone(),
        trace_id: trace_id.clone(),
        scenario_id: SCENARIO_ID.to_owned(),
        seed,
        alternate_seed,
        duration_ms: started.elapsed().as_millis(),
        deterministic_match: true,
        replay_commands,
        primary: first,
    };

    let artifact_path =
        write_suite_artifact(&artifact).expect("bd-3u7.3 deterministic replay artifact write");
    println!(
        "INFO bead_id={BEAD_ID} case=suite_artifact path={} run_id={} trace_id={} scenario_id={SCENARIO_ID}",
        artifact_path.display(),
        run_id,
        trace_id,
    );
}
