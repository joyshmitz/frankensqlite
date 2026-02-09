use std::fs;
use std::path::{Path, PathBuf};

use fsqlite_core::repair_symbols::policy_controller::{
    AutoTunePragmaConfig, CandidateAction, DecisionReason, PolicyController, PolicyKnob,
    PolicySignals,
};
use proptest::prelude::proptest;
use proptest::test_runner::TestCaseError;
use serde_json::Value;

const BEAD_ID: &str = "bd-3e5r";
const ISSUES_JSONL: &str = ".beads/issues.jsonl";
const LOG_STANDARD_REF: &str = "bd-1fpm";
const UNIT_TEST_IDS: [&str; 2] = [
    "test_bd_3e5r_unit_compliance_gate",
    "prop_bd_3e5r_structure_compliance",
];
const E2E_TEST_IDS: [&str; 1] = ["test_e2e_bd_3e5r_compliance"];
const LOG_LEVEL_MARKERS: [&str; 4] = ["DEBUG", "INFO", "WARN", "ERROR"];
const SPEC_MARKERS: [&str; 9] = [
    "PolicyController",
    "argmin",
    "fsqlite.auto_tune",
    "fsqlite.profile",
    "fsqlite.bg_cpu_max",
    "remote_max_in_flight",
    "commit_encode_max",
    "hysteresis",
    "BOCPD",
];
const REQUIRED_TOKENS: [&str; 17] = [
    "test_bd_3e5r_unit_compliance_gate",
    "prop_bd_3e5r_structure_compliance",
    "test_e2e_bd_3e5r_compliance",
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR",
    "bd-1fpm",
    "PolicyController",
    "argmin",
    "fsqlite.auto_tune",
    "fsqlite.profile",
    "fsqlite.bg_cpu_max",
    "remote_max_in_flight",
    "commit_encode_max",
    "hysteresis",
    "BOCPD",
];

#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
struct ComplianceEvaluation {
    missing_unit_ids: Vec<&'static str>,
    missing_e2e_ids: Vec<&'static str>,
    missing_log_levels: Vec<&'static str>,
    missing_log_standard_ref: bool,
    missing_spec_markers: Vec<&'static str>,
}

impl ComplianceEvaluation {
    fn is_compliant(&self) -> bool {
        self.missing_unit_ids.is_empty()
            && self.missing_e2e_ids.is_empty()
            && self.missing_log_levels.is_empty()
            && !self.missing_log_standard_ref
            && self.missing_spec_markers.is_empty()
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

fn is_identifier_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn contains_identifier(text: &str, needle: &str) -> bool {
    text.match_indices(needle).any(|(start, _)| {
        let end = start + needle.len();
        let bytes = text.as_bytes();
        let before_ok = start == 0 || !is_identifier_char(bytes[start - 1]);
        let after_ok = end == bytes.len() || !is_identifier_char(bytes[end]);
        before_ok && after_ok
    })
}

fn evaluate_description(description: &str) -> ComplianceEvaluation {
    let missing_unit_ids = UNIT_TEST_IDS
        .into_iter()
        .filter(|id| !contains_identifier(description, id))
        .collect::<Vec<_>>();

    let missing_e2e_ids = E2E_TEST_IDS
        .into_iter()
        .filter(|id| !contains_identifier(description, id))
        .collect::<Vec<_>>();

    let missing_log_levels = LOG_LEVEL_MARKERS
        .into_iter()
        .filter(|level| !description.contains(level))
        .collect::<Vec<_>>();

    let missing_spec_markers = SPEC_MARKERS
        .into_iter()
        .filter(|marker| !description.contains(marker))
        .collect::<Vec<_>>();

    ComplianceEvaluation {
        missing_unit_ids,
        missing_e2e_ids,
        missing_log_levels,
        missing_log_standard_ref: !description.contains(LOG_STANDARD_REF),
        missing_spec_markers,
    }
}

fn synthetic_compliant_description() -> String {
    let mut text = String::from("## Unit Test Requirements\n");
    for id in UNIT_TEST_IDS {
        text.push_str("- ");
        text.push_str(id);
        text.push('\n');
    }
    text.push_str("\n## E2E Test\n");
    text.push_str("- test_e2e_bd_3e5r_compliance\n");
    text.push_str("\n## Logging Requirements\n");
    text.push_str("- DEBUG: stage-level progress\n");
    text.push_str("- INFO: summary counters\n");
    text.push_str("- WARN: degraded-mode/retry conditions\n");
    text.push_str("- ERROR: terminal failure diagnostics\n");
    text.push_str("- Reference: bd-1fpm\n");
    text.push_str("\n## Spec Markers\n");
    for marker in SPEC_MARKERS {
        text.push_str("- ");
        text.push_str(marker);
        text.push('\n');
    }
    text
}

#[test]
fn test_bd_3e5r_unit_compliance_gate() -> Result<(), String> {
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
    if !evaluation.missing_log_levels.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=logging_levels_missing missing={:?}",
            evaluation.missing_log_levels
        ));
    }
    if evaluation.missing_log_standard_ref {
        return Err(format!(
            "bead_id={BEAD_ID} case=logging_standard_missing expected_ref={LOG_STANDARD_REF}"
        ));
    }
    if !evaluation.missing_spec_markers.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=spec_markers_missing missing={:?}",
            evaluation.missing_spec_markers
        ));
    }

    Ok(())
}

proptest! {
    #[test]
    fn prop_bd_3e5r_structure_compliance(missing_index in 0usize..REQUIRED_TOKENS.len()) {
        let mut synthetic = synthetic_compliant_description();
        synthetic = synthetic.replacen(REQUIRED_TOKENS[missing_index], "", 1);

        let evaluation = evaluate_description(&synthetic);
        if evaluation.is_compliant() {
            return Err(TestCaseError::fail(format!(
                "bead_id={BEAD_ID} case=structure_compliance expected_non_compliant missing_index={} marker={}",
                missing_index,
                REQUIRED_TOKENS[missing_index]
            )));
        }
    }
}

#[test]
fn test_policy_controller_smoke() -> Result<(), String> {
    let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 8);
    let candidates = vec![
        CandidateAction::new(1, 2, 5.0, "baseline"),
        CandidateAction::new(2, 3, 3.0, "argmin expected loss"),
    ];
    let outcome = controller.evaluate_knob(
        PolicyKnob::BgCpuMax,
        2,
        &candidates,
        PolicySignals::default(),
        true,
        10,
    );

    if outcome.reason != DecisionReason::Applied(2) {
        return Err(format!(
            "bead_id={BEAD_ID} case=argmin_mismatch outcome={outcome:?}"
        ));
    }
    if controller.ledger().is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=missing_evidence_ledger_entry"
        ));
    }
    Ok(())
}

#[test]
fn test_e2e_bd_3e5r_compliance() -> Result<(), String> {
    let description = load_issue_description(BEAD_ID)?;
    let evaluation = evaluate_description(&description);

    let mut first = PolicyController::new(AutoTunePragmaConfig::default(), 32, 3, 16);
    let mut second = PolicyController::new(AutoTunePragmaConfig::default(), 32, 3, 16);

    let candidates = vec![
        CandidateAction::new(7, 4, 0.4, "increase remote permits"),
        CandidateAction::new(9, 3, 0.6, "keep"),
    ];
    let signals = PolicySignals {
        symbol_loss_rejects_h0: false,
        bocpd_regime_shift: true,
        regime_id: 42,
    };

    let out1 = first.evaluate_knob(
        PolicyKnob::RemoteMaxInFlight,
        2,
        &candidates,
        signals,
        true,
        100,
    );
    let out2 = second.evaluate_knob(
        PolicyKnob::RemoteMaxInFlight,
        2,
        &candidates,
        signals,
        true,
        100,
    );

    eprintln!(
        "DEBUG bead_id={BEAD_ID} case=e2e_start ledger_entries_first={} ledger_entries_second={}",
        first.ledger().len(),
        second.ledger().len()
    );
    eprintln!(
        "INFO bead_id={BEAD_ID} case=e2e_summary compliant={} reason={:?}",
        evaluation.is_compliant(),
        out1.reason
    );
    eprintln!(
        "WARN bead_id={BEAD_ID} case=e2e_bocpd regime_shift={} regime_id={}",
        signals.bocpd_regime_shift, signals.regime_id
    );
    eprintln!(
        "ERROR bead_id={BEAD_ID} case=e2e_guard replay_cmd=\"cargo test -p fsqlite-harness test_e2e_bd_3e5r_compliance -- --nocapture\""
    );

    if !evaluation.is_compliant() {
        return Err(format!(
            "bead_id={BEAD_ID} case=e2e_compliance_failure evaluation={evaluation:?}"
        ));
    }
    if out1 != out2 {
        return Err(format!(
            "bead_id={BEAD_ID} case=e2e_nondeterministic_outcome out1={out1:?} out2={out2:?}"
        ));
    }
    if first.ledger() != second.ledger() {
        return Err(format!(
            "bead_id={BEAD_ID} case=e2e_nondeterministic_ledger first={:?} second={:?}",
            first.ledger(),
            second.ledger()
        ));
    }

    Ok(())
}

#[test]
fn test_e2e_bd_3e5r() -> Result<(), String> {
    test_e2e_bd_3e5r_compliance()
}
