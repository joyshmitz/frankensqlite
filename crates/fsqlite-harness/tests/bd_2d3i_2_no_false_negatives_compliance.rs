use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;
use serde_json::Value;

const BEAD_ID: &str = "bd-2d3i.2";
const ISSUES_JSONL: &str = ".beads/issues.jsonl";
const LOG_STANDARD_REF: &str = "bd-1fpm";

const UNIT_TEST_IDS: [&str; 8] = [
    "prop_no_false_negatives_basic",
    "prop_no_false_negatives_under_symbol_loss",
    "prop_no_false_negatives_under_crash",
    "prop_no_false_negatives_multi_level",
    "prop_no_false_negatives_epoch_boundary",
    "test_false_negative_would_cause_serializability_violation",
    "test_bd_2d3i_2_unit_compliance_gate",
    "prop_bd_2d3i_2_structure_compliance",
];
const E2E_TEST_IDS: [&str; 4] = [
    "test_e2e_bd_2d3i_2_compliance",
    "e2e_prop_no_false_negatives_ci_smoke",
    "e2e_prop_no_false_negatives_nightly",
    "e2e_prop_shrinking_replay",
];
const LOG_LEVEL_MARKERS: [&str; 4] = ["DEBUG", "INFO", "WARN", "ERROR"];
const REQUIRED_TOKENS: [&str; 18] = [
    "prop_no_false_negatives_basic",
    "prop_no_false_negatives_under_symbol_loss",
    "prop_no_false_negatives_under_crash",
    "prop_no_false_negatives_multi_level",
    "prop_no_false_negatives_epoch_boundary",
    "test_false_negative_would_cause_serializability_violation",
    "test_bd_2d3i_2_unit_compliance_gate",
    "prop_bd_2d3i_2_structure_compliance",
    "test_e2e_bd_2d3i_2_compliance",
    "e2e_prop_no_false_negatives_ci_smoke",
    "e2e_prop_no_false_negatives_nightly",
    "e2e_prop_shrinking_replay",
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR",
    "bd-1fpm",
    "schedule_fingerprint",
];

#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
struct ComplianceEvaluation {
    missing_unit_ids: Vec<&'static str>,
    missing_e2e_ids: Vec<&'static str>,
    missing_log_levels: Vec<&'static str>,
    missing_log_standard_ref: bool,
    missing_required_tokens: Vec<&'static str>,
}

impl ComplianceEvaluation {
    fn is_compliant(&self) -> bool {
        self.missing_unit_ids.is_empty()
            && self.missing_e2e_ids.is_empty()
            && self.missing_log_levels.is_empty()
            && !self.missing_log_standard_ref
            && self.missing_required_tokens.is_empty()
    }
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("workspace_root_canonicalize_failed error={error}"))
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
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'))
        .any(|candidate| candidate == expected_marker)
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

    let missing_required_tokens = REQUIRED_TOKENS
        .into_iter()
        .filter(|token| !contains_identifier(description, token) && !description.contains(token))
        .collect::<Vec<_>>();

    ComplianceEvaluation {
        missing_unit_ids,
        missing_e2e_ids,
        missing_log_levels,
        missing_log_standard_ref: !description.contains(LOG_STANDARD_REF),
        missing_required_tokens,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct WitnessKey {
    level: u8,
    key: u16,
    epoch: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetectionOutcome {
    NoConflict,
    Discovered { count: usize },
    DurabilityContractViolated { missing: usize },
}

fn keys_overlap(read: WitnessKey, write: WitnessKey) -> bool {
    read.level == write.level && read.key == write.key
}

fn detect_conflicts(
    reads: &BTreeSet<WitnessKey>,
    writes: &BTreeSet<WitnessKey>,
    lost_read: &BTreeSet<WitnessKey>,
    lost_write: &BTreeSet<WitnessKey>,
    crash_read_plane: bool,
    crash_write_plane: bool,
) -> DetectionOutcome {
    let mut discovered = 0_usize;
    let mut missing = 0_usize;

    for read in reads {
        for write in writes {
            if !keys_overlap(*read, *write) {
                continue;
            }

            let read_visible = !crash_read_plane && !lost_read.contains(read);
            let write_visible = !crash_write_plane && !lost_write.contains(write);

            if read_visible || write_visible {
                discovered = discovered.saturating_add(1);
            } else {
                missing = missing.saturating_add(1);
            }
        }
    }

    if discovered > 0 {
        DetectionOutcome::Discovered { count: discovered }
    } else if missing > 0 {
        DetectionOutcome::DurabilityContractViolated { missing }
    } else {
        DetectionOutcome::NoConflict
    }
}

fn witness_key_strategy(level_max: u8, epoch_max: u8) -> impl Strategy<Value = WitnessKey> {
    (0_u8..=level_max, 1_u16..=512_u16, 0_u8..=epoch_max)
        .prop_map(|(level, key, epoch)| WitnessKey { level, key, epoch })
}

#[allow(clippy::cast_possible_truncation)]
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn sample_set(seed: &mut u64, count: usize, level_max: u8, epoch: u8) -> BTreeSet<WitnessKey> {
    let mut out = BTreeSet::new();
    for _ in 0..count {
        let next = xorshift64(seed);
        let level = u8::try_from(next % u64::from(level_max + 1)).expect("level fits u8");
        let key = u16::try_from((next >> 8) % 512 + 1).expect("key fits u16");
        out.insert(WitnessKey { level, key, epoch });
    }
    out
}

fn overlap_exists(reads: &BTreeSet<WitnessKey>, writes: &BTreeSet<WitnessKey>) -> bool {
    reads
        .iter()
        .any(|read| writes.iter().any(|write| keys_overlap(*read, *write)))
}

#[test]
fn test_bd_2d3i_2_unit_compliance_gate() -> Result<(), String> {
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
    if !evaluation.missing_required_tokens.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=required_tokens_missing missing={:?}",
            evaluation.missing_required_tokens
        ));
    }

    Ok(())
}

proptest! {
    #[test]
    fn prop_bd_2d3i_2_structure_compliance(missing_index in 0usize..REQUIRED_TOKENS.len()) {
        let mut synthetic = format!(
            "## Unit Test Requirements\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n- {}\n\n## E2E Test\n- {}\n- {}\n- {}\n- {}\n\n## Logging Requirements\n- DEBUG: stage-level progress and schedule_fingerprint\n- INFO: summary counters and completion status\n- WARN: degraded mode and high shrink count\n- ERROR: terminal failure diagnostics\n- Reference: {}\n",
            UNIT_TEST_IDS[0],
            UNIT_TEST_IDS[1],
            UNIT_TEST_IDS[2],
            UNIT_TEST_IDS[3],
            UNIT_TEST_IDS[4],
            UNIT_TEST_IDS[5],
            UNIT_TEST_IDS[6],
            UNIT_TEST_IDS[7],
            E2E_TEST_IDS[0],
            E2E_TEST_IDS[1],
            E2E_TEST_IDS[2],
            E2E_TEST_IDS[3],
            LOG_STANDARD_REF,
        );
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

proptest! {
    #[test]
    fn prop_no_false_negatives_basic(
        reads in prop::collection::btree_set(witness_key_strategy(0, 0), 1..32),
        writes in prop::collection::btree_set(witness_key_strategy(0, 0), 1..32),
    ) {
        let outcome = detect_conflicts(
            &reads,
            &writes,
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
            false,
        );

        if overlap_exists(&reads, &writes) {
            prop_assert_ne!(outcome, DetectionOutcome::NoConflict);
        }
    }
}

proptest! {
    #[test]
    fn prop_no_false_negatives_under_symbol_loss(
        reads in prop::collection::btree_set(witness_key_strategy(2, 1), 1..40),
        writes in prop::collection::btree_set(witness_key_strategy(2, 1), 1..40),
        loss_mod_read in 2_u8..8_u8,
        loss_mod_write in 2_u8..8_u8,
    ) {
        let lost_read = reads
            .iter()
            .filter(|key| key.key % u16::from(loss_mod_read) == 0)
            .copied()
            .collect::<BTreeSet<_>>();
        let lost_write = writes
            .iter()
            .filter(|key| key.key % u16::from(loss_mod_write) == 0)
            .copied()
            .collect::<BTreeSet<_>>();

        let outcome = detect_conflicts(&reads, &writes, &lost_read, &lost_write, false, false);
        if overlap_exists(&reads, &writes) {
            prop_assert_ne!(outcome, DetectionOutcome::NoConflict);
        }
    }
}

proptest! {
    #[test]
    fn prop_no_false_negatives_under_crash(
        reads in prop::collection::btree_set(witness_key_strategy(1, 1), 1..32),
        writes in prop::collection::btree_set(witness_key_strategy(1, 1), 1..32),
        crash_read_plane in any::<bool>(),
        crash_write_plane in any::<bool>(),
    ) {
        let outcome = detect_conflicts(
            &reads,
            &writes,
            &BTreeSet::new(),
            &BTreeSet::new(),
            crash_read_plane,
            crash_write_plane,
        );

        if overlap_exists(&reads, &writes) {
            prop_assert_ne!(outcome, DetectionOutcome::NoConflict);
        }
    }
}

proptest! {
    #[test]
    fn prop_no_false_negatives_multi_level(
        reads in prop::collection::btree_set(witness_key_strategy(4, 1), 1..48),
        writes in prop::collection::btree_set(witness_key_strategy(4, 1), 1..48),
    ) {
        let outcome = detect_conflicts(
            &reads,
            &writes,
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
            false,
        );

        if overlap_exists(&reads, &writes) {
            prop_assert_ne!(outcome, DetectionOutcome::NoConflict);
        }
    }
}

proptest! {
    #[test]
    fn prop_no_false_negatives_epoch_boundary(
        base_reads in prop::collection::btree_set(witness_key_strategy(2, 0), 1..32),
        base_writes in prop::collection::btree_set(witness_key_strategy(2, 0), 1..32),
    ) {
        let reads = base_reads
            .into_iter()
            .map(|key| WitnessKey { epoch: 0, ..key })
            .collect::<BTreeSet<_>>();
        let writes = base_writes
            .into_iter()
            .map(|key| WitnessKey { epoch: 1, ..key })
            .collect::<BTreeSet<_>>();

        let outcome = detect_conflicts(
            &reads,
            &writes,
            &BTreeSet::new(),
            &BTreeSet::new(),
            false,
            false,
        );

        if overlap_exists(&reads, &writes) {
            prop_assert_ne!(outcome, DetectionOutcome::NoConflict);
        }
    }
}

#[test]
fn test_false_negative_would_cause_serializability_violation() {
    let reads = BTreeSet::from([WitnessKey {
        level: 0,
        key: 101,
        epoch: 0,
    }]);
    let writes = BTreeSet::from([WitnessKey {
        level: 0,
        key: 101,
        epoch: 1,
    }]);
    let lost_read = reads.clone();
    let lost_write = writes.clone();

    let outcome = detect_conflicts(&reads, &writes, &lost_read, &lost_write, true, true);
    assert!(matches!(
        outcome,
        DetectionOutcome::DurabilityContractViolated { .. }
    ));
}

#[test]
fn e2e_prop_no_false_negatives_ci_smoke() {
    let mut seed = 0x5EED_CAFE_u64;
    for case_index in 0_u32..1_000_u32 {
        let reads = sample_set(&mut seed, 8, 2, 0);
        let writes = sample_set(&mut seed, 8, 2, 1);
        let crash_read = (xorshift64(&mut seed) & 1) == 0;
        let crash_write = (xorshift64(&mut seed) & 1) == 0;
        let outcome = detect_conflicts(
            &reads,
            &writes,
            &BTreeSet::new(),
            &BTreeSet::new(),
            crash_read,
            crash_write,
        );

        if overlap_exists(&reads, &writes) {
            assert!(
                !matches!(outcome, DetectionOutcome::NoConflict),
                "ERROR bead_id={BEAD_ID} case_index={case_index} seed={seed} schedule_fingerprint=ci-smoke false_negative_detected"
            );
        }
    }
}

#[test]
fn e2e_prop_no_false_negatives_nightly() {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|raw| raw.parse::<u32>().ok())
        .unwrap_or(2_000);
    let mut seed = 0xDEAD_BEEF_u64;

    for case_index in 0..cases {
        let reads = sample_set(&mut seed, 12, 4, 0);
        let writes = sample_set(&mut seed, 12, 4, 1);
        let loss_read = sample_set(&mut seed, 3, 4, 0);
        let loss_write = sample_set(&mut seed, 3, 4, 1);
        let outcome = detect_conflicts(&reads, &writes, &loss_read, &loss_write, false, false);

        if overlap_exists(&reads, &writes) {
            assert!(
                !matches!(outcome, DetectionOutcome::NoConflict),
                "ERROR bead_id={BEAD_ID} case_index={case_index} seed={seed} schedule_fingerprint=nightly false_negative_detected"
            );
        }
    }
}

#[test]
fn e2e_prop_shrinking_replay() {
    let reads = BTreeSet::from([
        WitnessKey {
            level: 2,
            key: 77,
            epoch: 0,
        },
        WitnessKey {
            level: 1,
            key: 12,
            epoch: 0,
        },
    ]);
    let writes = BTreeSet::from([WitnessKey {
        level: 2,
        key: 77,
        epoch: 1,
    }]);

    let minimized_loss = BTreeSet::from([WitnessKey {
        level: 1,
        key: 12,
        epoch: 0,
    }]);
    let outcome = detect_conflicts(
        &reads,
        &writes,
        &minimized_loss,
        &BTreeSet::new(),
        false,
        false,
    );
    assert!(
        matches!(outcome, DetectionOutcome::Discovered { .. }),
        "ERROR bead_id={BEAD_ID} case=replay seed=minimized schedule_fingerprint=replay-minimized outcome={outcome:?}"
    );
}

#[test]
fn test_e2e_bd_2d3i_2_compliance() -> Result<(), String> {
    let description = load_issue_description(BEAD_ID)?;
    let evaluation = evaluate_description(&description);

    eprintln!(
        "INFO bead_id={BEAD_ID} case=e2e_summary missing_unit_ids={} missing_e2e_ids={} missing_log_levels={} missing_required_tokens={} missing_log_standard_ref={}",
        evaluation.missing_unit_ids.len(),
        evaluation.missing_e2e_ids.len(),
        evaluation.missing_log_levels.len(),
        evaluation.missing_required_tokens.len(),
        evaluation.missing_log_standard_ref
    );
    eprintln!("DEBUG bead_id={BEAD_ID} case=e2e_trace schedule_fingerprint=bd_2d3i_2_compliance");

    for id in &evaluation.missing_unit_ids {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_unit_id id={id}");
    }
    for id in &evaluation.missing_e2e_ids {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_e2e_id id={id}");
    }
    for level in &evaluation.missing_log_levels {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_log_level level={level}");
    }
    for marker in &evaluation.missing_required_tokens {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_required_marker marker={marker}");
    }
    if evaluation.missing_log_standard_ref {
        eprintln!(
            "ERROR bead_id={BEAD_ID} case=missing_log_standard_ref expected={LOG_STANDARD_REF}"
        );
    }

    if !evaluation.is_compliant() {
        return Err(format!(
            "bead_id={BEAD_ID} case=e2e_compliance_failure evaluation={evaluation:?}"
        ));
    }

    Ok(())
}
