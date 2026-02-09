use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use fsqlite_harness::verification_gates::{
    GateCommandOutput, GateCommandRunner, GateStatus, run_phase_4_to_6_gates_with_runner,
    write_core_phase_gate_report,
};
use proptest::prelude::proptest;
use proptest::test_runner::TestCaseError;
use serde_json::Value;

const BEAD_ID: &str = "bd-331.3";
const ISSUES_JSONL: &str = ".beads/issues.jsonl";
const UNIT_TEST_IDS: [&str; 21] = [
    "test_phase4_gate_sql_conformance_20",
    "test_phase4_gate_vdbe_explain",
    "test_phase4_gate_sorter_100k",
    "test_phase5_gate_format_write_read_c",
    "test_phase5_gate_format_read_c_write",
    "test_phase5_gate_wal_crash_recovery",
    "test_phase5_gate_raptorq_wal",
    "test_phase6_gate_mvcc_stress",
    "test_phase6_gate_ssi_write_skew",
    "test_phase6_gate_ssi_mazurkiewicz",
    "test_phase6_gate_ssi_witness_epoch",
    "test_phase6_gate_ssi_witness_decode",
    "test_phase6_gate_snapshot_mazurkiewicz",
    "test_phase6_gate_eprocess_inv",
    "test_phase6_gate_gc_memory",
    "test_phase6_gate_serialized_parity",
    "test_phase6_gate_rebase_merge",
    "test_phase6_gate_structured_merge",
    "test_phase6_gate_crash_model",
    "test_bd_331_3_unit_compliance_gate",
    "prop_bd_331_3_structure_compliance",
];
const E2E_TEST_IDS: [&str; 2] = ["test_e2e_bd_331_3", "test_e2e_bd_331_3_compliance"];
const REQUIRED_PHRASES: [&str; 9] = [
    "a gate runner exists",
    "machine-readable json report",
    "gate failures are blocking",
    "debug",
    "info",
    "warn",
    "error",
    "bd-1fpm",
    "phase 4-6",
];
const REQUIRED_TOKENS: [&str; 32] = [
    "test_phase4_gate_sql_conformance_20",
    "test_phase4_gate_vdbe_explain",
    "test_phase4_gate_sorter_100k",
    "test_phase5_gate_format_write_read_c",
    "test_phase5_gate_format_read_c_write",
    "test_phase5_gate_wal_crash_recovery",
    "test_phase5_gate_raptorq_wal",
    "test_phase6_gate_mvcc_stress",
    "test_phase6_gate_ssi_write_skew",
    "test_phase6_gate_ssi_mazurkiewicz",
    "test_phase6_gate_ssi_witness_epoch",
    "test_phase6_gate_ssi_witness_decode",
    "test_phase6_gate_snapshot_mazurkiewicz",
    "test_phase6_gate_eprocess_inv",
    "test_phase6_gate_gc_memory",
    "test_phase6_gate_serialized_parity",
    "test_phase6_gate_rebase_merge",
    "test_phase6_gate_structured_merge",
    "test_phase6_gate_crash_model",
    "test_bd_331_3_unit_compliance_gate",
    "prop_bd_331_3_structure_compliance",
    "test_e2e_bd_331_3",
    "test_e2e_bd_331_3_compliance",
    "a gate runner exists",
    "machine-readable json report",
    "gate failures are blocking",
    "debug",
    "info",
    "warn",
    "error",
    "bd-1fpm",
    "phase 4-6",
];

#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
struct ComplianceEvaluation {
    missing_unit_ids: Vec<&'static str>,
    missing_e2e_ids: Vec<&'static str>,
    missing_phrases: Vec<&'static str>,
}

impl ComplianceEvaluation {
    fn is_compliant(&self) -> bool {
        self.missing_unit_ids.is_empty()
            && self.missing_e2e_ids.is_empty()
            && self.missing_phrases.is_empty()
    }
}

#[derive(Debug, Default)]
struct SyntheticGateRunner {
    failing_gate_ids: HashSet<String>,
}

impl SyntheticGateRunner {
    fn with_failures<I, S>(failures: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self {
            failing_gate_ids: failures
                .into_iter()
                .map(|gate_id| gate_id.as_ref().to_owned())
                .collect(),
        }
    }
}

impl GateCommandRunner for SyntheticGateRunner {
    fn run_gate(
        &self,
        gate_id: &str,
        _command: &[String],
        _env: &[(String, String)],
        _workspace_root: &Path,
    ) -> std::io::Result<GateCommandOutput> {
        let exit_code = if self.failing_gate_ids.contains(gate_id) {
            2
        } else {
            0
        };

        Ok(GateCommandOutput {
            exit_code,
            stdout: format!("stdout gate_id={gate_id}"),
            stderr: String::new(),
        })
    }
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("workspace_root_canonicalize_failed: {error}"))
}

fn load_issue_description(issue_id: &str) -> Result<String, String> {
    let issues_path = workspace_root()?.join(ISSUES_JSONL);
    let raw = fs::read_to_string(&issues_path).map_err(|error| {
        format!(
            "issues_jsonl_read_failed path={} error={error}",
            issues_path.display()
        )
    })?;

    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("issues_jsonl_parse_failed error={error} line={line}"))?;
        if value.get("id").and_then(Value::as_str) == Some(issue_id) {
            let mut canonical = value
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();

            if let Some(comments) = value.get("comments").and_then(Value::as_array) {
                for comment in comments {
                    if let Some(text) = comment.get("text").and_then(Value::as_str) {
                        canonical.push_str("\n\n");
                        canonical.push_str(text);
                    }
                }
            }
            return Ok(canonical);
        }
    }

    Err(format!("bead_id={issue_id} not_found_in={ISSUES_JSONL}"))
}

fn contains_identifier(text: &str, expected_marker: &str) -> bool {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|token| token == expected_marker)
}

fn evaluate_description(description: &str) -> ComplianceEvaluation {
    let missing_unit_ids = UNIT_TEST_IDS
        .into_iter()
        .filter(|test_id| !contains_identifier(description, test_id))
        .collect::<Vec<_>>();

    let missing_e2e_ids = E2E_TEST_IDS
        .into_iter()
        .filter(|test_id| !contains_identifier(description, test_id))
        .collect::<Vec<_>>();

    let missing_phrases = {
        let lowered = description.to_ascii_lowercase();
        REQUIRED_PHRASES
            .into_iter()
            .filter(|phrase| !lowered.contains(phrase))
            .collect::<Vec<_>>()
    };

    ComplianceEvaluation {
        missing_unit_ids,
        missing_e2e_ids,
        missing_phrases,
    }
}

fn unique_runtime_dir(label: &str) -> Result<PathBuf, String> {
    let root = workspace_root()?.join("target").join("bd_331_3_runtime");
    fs::create_dir_all(&root).map_err(|error| {
        format!(
            "runtime_dir_create_failed path={} error={error}",
            root.as_path().display()
        )
    })?;

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0_u128, |duration| duration.as_nanos());
    let path = root.join(format!("{label}_{}_{}", std::process::id(), stamp));
    fs::create_dir_all(&path).map_err(|error| {
        format!(
            "runtime_subdir_create_failed path={} error={error}",
            path.as_path().display()
        )
    })?;
    Ok(path)
}

#[test]
fn test_bd_331_3_unit_compliance_gate() -> Result<(), String> {
    let description = load_issue_description(BEAD_ID)?;
    let evaluation = evaluate_description(&description);

    if !evaluation.missing_unit_ids.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=unit_ids_missing missing={:?}",
            evaluation.missing_unit_ids
        ));
    }
    if !evaluation.missing_e2e_ids.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=e2e_ids_missing missing={:?}",
            evaluation.missing_e2e_ids
        ));
    }
    if !evaluation.missing_phrases.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=phrases_missing missing={:?}",
            evaluation.missing_phrases
        ));
    }

    Ok(())
}

proptest! {
    #[test]
    fn prop_bd_331_3_structure_compliance(missing_index in 0usize..REQUIRED_TOKENS.len()) {
        let mut synthetic = format!(
            "## Phase 4-6 Gates\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n\n## E2E\n- {}\n- {}\n\n## Acceptance\n- A gate runner exists.\n- Emits a machine-readable JSON report.\n- Gate failures are blocking.\n- DEBUG/INFO/WARN/ERROR logging aligned to bd-1fpm.\n",
            UNIT_TEST_IDS[0],
            UNIT_TEST_IDS[1],
            UNIT_TEST_IDS[2],
            UNIT_TEST_IDS[3],
            UNIT_TEST_IDS[4],
            UNIT_TEST_IDS[5],
            UNIT_TEST_IDS[6],
            UNIT_TEST_IDS[7],
            UNIT_TEST_IDS[8],
            UNIT_TEST_IDS[9],
            UNIT_TEST_IDS[10],
            UNIT_TEST_IDS[11],
            UNIT_TEST_IDS[12],
            UNIT_TEST_IDS[13],
            UNIT_TEST_IDS[14],
            UNIT_TEST_IDS[15],
            UNIT_TEST_IDS[16],
            UNIT_TEST_IDS[17],
            UNIT_TEST_IDS[18],
            UNIT_TEST_IDS[19],
            UNIT_TEST_IDS[20],
            E2E_TEST_IDS[0],
            E2E_TEST_IDS[1],
        ).to_ascii_lowercase();

        synthetic = synthetic.replacen(REQUIRED_TOKENS[missing_index], "", 1);
        let evaluation = evaluate_description(&synthetic);

        if evaluation.is_compliant() {
            return Err(TestCaseError::fail(format!(
                "bead_id={} case=structure_compliance expected_non_compliant missing_index={} missing_marker={}",
                BEAD_ID,
                missing_index,
                REQUIRED_TOKENS[missing_index]
            )));
        }
    }
}

#[test]
fn test_e2e_bd_331_3_compliance() -> Result<(), String> {
    let description = load_issue_description(BEAD_ID)?;
    let evaluation = evaluate_description(&description);
    if !evaluation.is_compliant() {
        return Err(format!(
            "bead_id={BEAD_ID} case=description_non_compliant evaluation={evaluation:?}"
        ));
    }

    let workspace = workspace_root()?;
    let runner = SyntheticGateRunner::with_failures(["phase4.sql_conformance_20"]);
    let report = run_phase_4_to_6_gates_with_runner(&workspace, &runner);

    if report.overall_pass {
        eprintln!(
            "ERROR bead_id={BEAD_ID} case=unexpected_overall_pass replay_cmd=\"cargo test -p fsqlite-harness test_e2e_bd_331_3_compliance -- --nocapture\""
        );
        return Err(format!(
            "bead_id={BEAD_ID} case=expected_failure_but_passed report={report:?}"
        ));
    }
    if !report.blocked_by_phase4_failure {
        return Err(format!(
            "bead_id={BEAD_ID} case=expected_phase4_block report={report:?}"
        ));
    }
    if report.phase5_pass || report.phase6_pass {
        return Err(format!(
            "bead_id={BEAD_ID} case=unexpected_downstream_pass report={report:?}"
        ));
    }

    let skipped = report
        .gates
        .iter()
        .filter(|gate| gate.status == GateStatus::Skipped)
        .count();
    if skipped == 0 {
        return Err(format!(
            "bead_id={BEAD_ID} case=expected_skipped_gates report={report:?}"
        ));
    }

    let runtime_dir = unique_runtime_dir("e2e")?;
    let artifact_path = runtime_dir.join("bd_331_3_gate_report.json");
    write_core_phase_gate_report(&artifact_path, &report).map_err(|error| {
        format!(
            "core_phase_gate_report_write_failed path={} error={error}",
            artifact_path.display()
        )
    })?;

    let artifact_raw = fs::read_to_string(&artifact_path).map_err(|error| {
        format!(
            "core_phase_gate_report_read_failed path={} error={error}",
            artifact_path.display()
        )
    })?;
    let artifact_json: Value = serde_json::from_str(&artifact_raw)
        .map_err(|error| format!("core_phase_gate_report_parse_failed error={error}"))?;
    let gate_count = artifact_json
        .get("gates")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    if gate_count == 0 {
        return Err(format!(
            "bead_id={BEAD_ID} case=expected_non_empty_gates artifact_path={}",
            artifact_path.display()
        ));
    }

    eprintln!(
        "DEBUG bead_id={BEAD_ID} case=artifact_written path={} size_bytes={}",
        artifact_path.display(),
        fs::metadata(&artifact_path)
            .map_err(|error| format!("artifact_metadata_failed error={error}"))?
            .len()
    );
    eprintln!(
        "INFO bead_id={BEAD_ID} case=e2e_summary overall_pass={} phase4_pass={} phase5_pass={} phase6_pass={} blocked_by_phase4_failure={} blocked_by_phase5_failure={} skipped_gates={} gate_count={}",
        report.overall_pass,
        report.phase4_pass,
        report.phase5_pass,
        report.phase6_pass,
        report.blocked_by_phase4_failure,
        report.blocked_by_phase5_failure,
        skipped,
        gate_count
    );
    eprintln!(
        "WARN bead_id={BEAD_ID} case=blocked_path_verified reason=phase4_gate_failure reference=bd-1fpm"
    );

    Ok(())
}

#[test]
fn test_e2e_bd_331_3() -> Result<(), String> {
    test_e2e_bd_331_3_compliance()
}
