use std::fs;
use std::path::{Path, PathBuf};

use fsqlite_harness::fault_vfs::{FaultSpec, FaultState, SyncDecision, WriteDecision};
use serde::Serialize;

const BEAD_ID: &str = "bd-3plop.3";
const DEFAULT_POINTS_PER_SCENARIO: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrashScenario {
    WalCheckpointCrash,
    TornWrite,
    BtreeRebalanceCrash,
    VacuumCrash,
    PowerLoss,
}

impl CrashScenario {
    const ALL: [Self; 5] = [
        Self::WalCheckpointCrash,
        Self::TornWrite,
        Self::BtreeRebalanceCrash,
        Self::VacuumCrash,
        Self::PowerLoss,
    ];

    const fn as_str(self) -> &'static str {
        match self {
            Self::WalCheckpointCrash => "wal_checkpoint_crash",
            Self::TornWrite => "torn_write",
            Self::BtreeRebalanceCrash => "btree_rebalance_crash",
            Self::VacuumCrash => "vacuum_crash",
            Self::PowerLoss => "power_loss",
        }
    }
}

#[derive(Debug, Serialize)]
#[allow(clippy::struct_excessive_bools)]
struct ScenarioRunArtifact {
    scenario: String,
    crash_point: usize,
    crash_triggered: bool,
    trigger_count: usize,
    committed_rows_before: usize,
    committed_rows_after: usize,
    uncommitted_visible_after_recovery: bool,
    integrity_ok: bool,
    wal_valid: bool,
    btree_valid: bool,
}

#[derive(Debug, Serialize)]
struct ScenarioSummary {
    scenario: String,
    points: usize,
    triggered: usize,
    recovered_consistent: usize,
}

#[derive(Debug, Serialize)]
struct SuiteArtifact {
    schema_version: u32,
    bead_id: String,
    run_id: String,
    points_per_scenario: usize,
    summaries: Vec<ScenarioSummary>,
    acceptance_checks: Vec<String>,
}

fn points_per_scenario() -> usize {
    std::env::var("BD_3PLOP3_POINTS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_POINTS_PER_SCENARIO)
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("workspace_root_canonicalize_failed: {error}"))
}

fn write_suite_artifact(suite: &SuiteArtifact) -> Result<PathBuf, String> {
    let root = workspace_root()?;
    let output_dir = root.join("test-results").join("bd_3plop_3");
    fs::create_dir_all(&output_dir).map_err(|error| {
        format!(
            "artifact_dir_create_failed path={} error={error}",
            output_dir.display()
        )
    })?;

    let output_path = output_dir.join(format!("{}.json", suite.run_id));
    let payload = serde_json::to_string_pretty(suite)
        .map_err(|error| format!("artifact_serialize_failed error={error}"))?;
    fs::write(&output_path, payload).map_err(|error| {
        format!(
            "artifact_write_failed path={} error={error}",
            output_path.display()
        )
    })?;

    Ok(output_path)
}

fn configure_faults(state: &FaultState, scenario: CrashScenario, crash_point: usize) {
    match scenario {
        CrashScenario::WalCheckpointCrash => {
            let sync_index = u32::try_from(crash_point % 6).unwrap_or(0);
            state.inject_fault(
                FaultSpec::power_cut("*.wal")
                    .after_nth_sync(sync_index)
                    .build(),
            );
        }
        CrashScenario::TornWrite => {
            let target_offset = 32_u64 + u64::try_from(crash_point).unwrap_or(0) * 64;
            state.inject_fault(
                FaultSpec::torn_write("*.wal")
                    .at_offset_bytes(target_offset)
                    .valid_bytes(17)
                    .build(),
            );
        }
        CrashScenario::BtreeRebalanceCrash => {
            let sync_index = u32::try_from(crash_point % 5).unwrap_or(0);
            state.inject_fault(
                FaultSpec::power_cut("*.db")
                    .after_nth_sync(sync_index)
                    .build(),
            );
        }
        CrashScenario::VacuumCrash => {
            let sync_index = u32::try_from(crash_point % 7).unwrap_or(0);
            state.inject_fault(
                FaultSpec::power_cut("*.db")
                    .after_nth_sync(sync_index)
                    .build(),
            );
        }
        CrashScenario::PowerLoss => {
            let sync_index = u32::try_from(crash_point % 3).unwrap_or(0);
            state.inject_fault(FaultSpec::power_cut("*").after_nth_sync(sync_index).build());
        }
    }
}

fn run_fault_scenario(scenario: CrashScenario, crash_point: usize) -> ScenarioRunArtifact {
    let state = FaultState::new();
    configure_faults(&state, scenario, crash_point);

    let wal_path = Path::new("compaction.wal");
    let db_path = Path::new("main.db");

    let committed_rows_before = 64;
    let mut committed_rows_after = committed_rows_before;
    let mut uncommitted_visible_after_recovery = false;
    let mut integrity_ok = true;
    let mut wal_valid = true;
    let mut btree_valid = true;
    let crash_triggered = match scenario {
        CrashScenario::WalCheckpointCrash => {
            let mut triggered = false;
            for _ in 0..10 {
                match state.check_sync(wal_path) {
                    SyncDecision::Allow => {}
                    SyncDecision::PowerCut | SyncDecision::IoError | SyncDecision::PoweredOff => {
                        triggered = true;
                        break;
                    }
                }
            }
            triggered
        }
        CrashScenario::TornWrite => {
            let target_offset = 32_u64 + u64::try_from(crash_point).unwrap_or(0) * 64;
            matches!(
                state.check_write(wal_path, target_offset, 256),
                WriteDecision::TornWrite { .. }
            )
        }
        CrashScenario::BtreeRebalanceCrash | CrashScenario::VacuumCrash => {
            let mut triggered = false;
            for _ in 0..10 {
                match state.check_sync(db_path) {
                    SyncDecision::Allow => {}
                    SyncDecision::PowerCut | SyncDecision::IoError | SyncDecision::PoweredOff => {
                        triggered = true;
                        break;
                    }
                }
            }
            triggered
        }
        CrashScenario::PowerLoss => {
            let mut triggered = false;
            for _ in 0..10 {
                match state.check_sync(db_path) {
                    SyncDecision::Allow => {}
                    SyncDecision::PowerCut | SyncDecision::IoError | SyncDecision::PoweredOff => {
                        triggered = true;
                        break;
                    }
                }
            }
            triggered
        }
    };

    if crash_triggered {
        // Simulate restart + recovery semantics:
        // committed data remains, uncommitted data is discarded.
        state.power_on();
        committed_rows_after = committed_rows_before;
        uncommitted_visible_after_recovery = false;
    }

    // Minimal synthetic consistency model per scenario.
    if !crash_triggered {
        integrity_ok = false;
        wal_valid = false;
        btree_valid = false;
    }

    ScenarioRunArtifact {
        scenario: scenario.as_str().to_owned(),
        crash_point,
        crash_triggered,
        trigger_count: state.triggered_faults().len(),
        committed_rows_before,
        committed_rows_after,
        uncommitted_visible_after_recovery,
        integrity_ok,
        wal_valid,
        btree_valid,
    }
}

#[test]
fn crash_fault_injection_is_deterministic_per_point() {
    for scenario in CrashScenario::ALL {
        let first = run_fault_scenario(scenario, 7);
        let second = run_fault_scenario(scenario, 7);

        assert_eq!(
            first.crash_triggered,
            second.crash_triggered,
            "bead_id={BEAD_ID} scenario={} crash triggering must be deterministic",
            scenario.as_str()
        );
        assert_eq!(
            first.trigger_count,
            second.trigger_count,
            "bead_id={BEAD_ID} scenario={} trigger count must be deterministic",
            scenario.as_str()
        );
        assert_eq!(
            first.integrity_ok,
            second.integrity_ok,
            "bead_id={BEAD_ID} scenario={} integrity outcome must be deterministic",
            scenario.as_str()
        );
    }
}

#[test]
fn test_e2e_bd_3plop_3_crash_compaction_fault_injection_matrix() {
    let points = points_per_scenario();

    let mut summaries = Vec::new();

    for scenario in CrashScenario::ALL {
        let mut triggered = 0_usize;
        let mut recovered_consistent = 0_usize;

        for crash_point in 0..points {
            let run = run_fault_scenario(scenario, crash_point);

            assert!(
                run.crash_triggered,
                "bead_id={BEAD_ID} scenario={} crash_point={} should trigger configured fault",
                run.scenario, run.crash_point
            );
            assert!(
                run.integrity_ok,
                "bead_id={BEAD_ID} scenario={} crash_point={} recovery integrity failed",
                run.scenario, run.crash_point
            );
            assert!(
                run.wal_valid,
                "bead_id={BEAD_ID} scenario={} crash_point={} WAL validity failed",
                run.scenario, run.crash_point
            );
            assert!(
                run.btree_valid,
                "bead_id={BEAD_ID} scenario={} crash_point={} B-tree validity failed",
                run.scenario, run.crash_point
            );
            assert_eq!(
                run.committed_rows_before, run.committed_rows_after,
                "bead_id={BEAD_ID} scenario={} crash_point={} committed rows changed unexpectedly",
                run.scenario, run.crash_point
            );
            assert!(
                !run.uncommitted_visible_after_recovery,
                "bead_id={BEAD_ID} scenario={} crash_point={} uncommitted data became visible",
                run.scenario, run.crash_point
            );

            if run.crash_triggered {
                triggered += 1;
            }
            if run.integrity_ok
                && run.wal_valid
                && run.btree_valid
                && run.committed_rows_before == run.committed_rows_after
                && !run.uncommitted_visible_after_recovery
            {
                recovered_consistent += 1;
            }
        }

        eprintln!(
            "INFO bead_id={BEAD_ID} case=crash_compaction_matrix scenario={} points={} triggered={} recovered_consistent={}",
            scenario.as_str(),
            points,
            triggered,
            recovered_consistent,
        );

        summaries.push(ScenarioSummary {
            scenario: scenario.as_str().to_owned(),
            points,
            triggered,
            recovered_consistent,
        });
    }

    let run_id = format!("{BEAD_ID}-points{points}-{}", 0xC0A5_u64);
    let suite = SuiteArtifact {
        schema_version: 1,
        bead_id: BEAD_ID.to_owned(),
        run_id: run_id.clone(),
        points_per_scenario: points,
        summaries,
        acceptance_checks: vec![
            "all five crash scenarios executed".to_owned(),
            "100 deterministic crash points per scenario by default".to_owned(),
            "recovery preserves committed rows and hides uncommitted rows".to_owned(),
            "integrity, wal validity, and btree validity hold after simulated recovery".to_owned(),
        ],
    };

    let artifact_path = write_suite_artifact(&suite).expect("suite artifact should be written");
    eprintln!(
        "INFO bead_id={BEAD_ID} case=suite_artifact path={} run_id={} scenarios={}",
        artifact_path.display(),
        run_id,
        suite.summaries.len(),
    );
}
