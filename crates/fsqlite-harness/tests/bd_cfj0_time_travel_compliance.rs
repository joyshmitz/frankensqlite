use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use fsqlite_core::tiered_storage::{
    DurabilityMode, FetchSymbolsRequest, RemoteTier, TieredStorage, UploadSegmentReceipt,
    UploadSegmentRequest,
};
use fsqlite_error::FrankenError;
use fsqlite_mvcc::{CommitLog, CommitRecord, TimeTravelTarget, resolve_timestamp_via_commit_log};
use fsqlite_types::cx::{Cx, cap};
use fsqlite_types::{CommitSeq, ObjectId, Oti, RemoteCap, SymbolRecord, SymbolRecordFlags, TxnId};
use proptest::prelude::proptest;
use proptest::test_runner::TestCaseError;
use serde_json::Value;

const BEAD_ID: &str = "bd-cfj0";
const ISSUES_JSONL: &str = ".beads/issues.jsonl";
const LOG_STANDARD_REF: &str = "bd-1fpm";
const UNIT_TEST_IDS: [&str; 2] = [
    "test_bd_cfj0_unit_compliance_gate",
    "prop_bd_cfj0_structure_compliance",
];
const E2E_TEST_IDS: [&str; 1] = ["test_e2e_bd_cfj0_compliance"];
const LOG_LEVEL_MARKERS: [&str; 4] = ["DEBUG", "INFO", "WARN", "ERROR"];
const REQUIRED_TOKENS: [&str; 8] = [
    "test_bd_cfj0_unit_compliance_gate",
    "prop_bd_cfj0_structure_compliance",
    "test_e2e_bd_cfj0_compliance",
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR",
    "bd-1fpm",
];

#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
struct ComplianceEvaluation {
    missing_unit_ids: Vec<&'static str>,
    missing_e2e_ids: Vec<&'static str>,
    missing_log_levels: Vec<&'static str>,
    missing_log_standard_ref: bool,
}

impl ComplianceEvaluation {
    fn is_compliant(&self) -> bool {
        self.missing_unit_ids.is_empty()
            && self.missing_e2e_ids.is_empty()
            && self.missing_log_levels.is_empty()
            && !self.missing_log_standard_ref
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

    ComplianceEvaluation {
        missing_unit_ids,
        missing_e2e_ids,
        missing_log_levels,
        missing_log_standard_ref: !description.contains(LOG_STANDARD_REF),
    }
}

fn commit_record(txn_id: u64, commit_seq: u64, timestamp_unix_ns: u64) -> CommitRecord {
    CommitRecord {
        txn_id: TxnId::new(txn_id).expect("test txn id must be non-zero"),
        commit_seq: CommitSeq::new(commit_seq),
        pages: Vec::new().into(),
        timestamp_unix_ns,
    }
}

#[derive(Debug, Default)]
struct MockRemoteTier {
    object_symbols: HashMap<ObjectId, Vec<SymbolRecord>>,
    fetch_calls: usize,
}

impl MockRemoteTier {
    fn set_object_symbols(&mut self, object_id: ObjectId, records: Vec<SymbolRecord>) {
        self.object_symbols.insert(object_id, records);
    }

    fn fetch_calls(&self) -> usize {
        self.fetch_calls
    }
}

impl RemoteTier for MockRemoteTier {
    fn fetch_symbols(
        &mut self,
        request: &FetchSymbolsRequest,
    ) -> fsqlite_error::Result<Vec<SymbolRecord>> {
        self.fetch_calls = self.fetch_calls.saturating_add(1);
        Ok(self
            .object_symbols
            .get(&request.object_id)
            .cloned()
            .unwrap_or_default())
    }

    fn upload_segment(
        &mut self,
        _request: &UploadSegmentRequest,
    ) -> fsqlite_error::Result<UploadSegmentReceipt> {
        Ok(UploadSegmentReceipt {
            acked_stores: 1,
            deduplicated: false,
        })
    }

    fn segment_recoverable(&self, _segment_id: u64, _min_symbols_per_object: usize) -> bool {
        true
    }
}

fn object_id_from_u64(raw: u64) -> ObjectId {
    let mut bytes = [0_u8; 16];
    bytes[0..8].copy_from_slice(&raw.to_le_bytes());
    bytes[8..16].copy_from_slice(&raw.to_le_bytes());
    ObjectId::from_bytes(bytes)
}

fn remote_cap(seed: u8) -> RemoteCap {
    RemoteCap::from_bytes([seed; 16])
}

fn make_symbol_records(
    object_id: ObjectId,
    payload: &[u8],
    symbol_size: usize,
    repair_symbols: usize,
) -> Vec<SymbolRecord> {
    let symbol_size_u32 = u32::try_from(symbol_size).expect("symbol_size fits u32");
    let transfer_len_u64 = u64::try_from(payload.len()).expect("payload length fits u64");
    let oti = Oti {
        f: transfer_len_u64,
        al: 1,
        t: symbol_size_u32,
        z: 1,
        n: 1,
    };

    let source_symbols = payload.len().div_ceil(symbol_size);
    let mut out = Vec::new();

    for idx in 0..source_symbols {
        let start = idx * symbol_size;
        let end = (start + symbol_size).min(payload.len());
        let mut symbol = vec![0_u8; symbol_size];
        symbol[..end - start].copy_from_slice(&payload[start..end]);
        let esi = u32::try_from(idx).expect("source esi fits u32");
        let flags = if idx == 0 {
            SymbolRecordFlags::SYSTEMATIC_RUN_START
        } else {
            SymbolRecordFlags::empty()
        };
        out.push(SymbolRecord::new(object_id, oti, esi, symbol, flags));
    }

    for repair_idx in 0..repair_symbols {
        let repair_esi_usize = source_symbols.saturating_add(repair_idx);
        let esi = u32::try_from(repair_esi_usize).expect("repair esi fits u32");
        let mut symbol = vec![0_u8; symbol_size];
        let esi_low = u8::try_from(esi & 0xFF).expect("masked to u8");
        for (offset, byte) in symbol.iter_mut().enumerate() {
            let offset_low = u8::try_from(offset & 0xFF).expect("masked to u8");
            *byte = esi_low ^ offset_low;
        }
        out.push(SymbolRecord::new(
            object_id,
            oti,
            esi,
            symbol,
            SymbolRecordFlags::empty(),
        ));
    }

    out
}

#[test]
fn test_bd_cfj0_unit_compliance_gate() -> Result<(), String> {
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

    Ok(())
}

proptest! {
    #[test]
    fn prop_bd_cfj0_structure_compliance(missing_index in 0usize..REQUIRED_TOKENS.len()) {
        let mut synthetic = format!(
            "## Unit Test Requirements\n- {}\n- {}\n\n## E2E Test\n- {}\n\n## Logging Requirements\n- DEBUG: stage progress\n- INFO: summary\n- WARN: degraded mode\n- ERROR: terminal failure\n- Reference: {}\n",
            UNIT_TEST_IDS[0],
            UNIT_TEST_IDS[1],
            E2E_TEST_IDS[0],
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

#[test]
fn test_time_travel_marker_resolution_smoke() {
    let base_ts = 1_700_000_000_000_000_000_u64;
    let mut commit_log = CommitLog::new(CommitSeq::new(10));

    commit_log.append(commit_record(100, 10, base_ts + 1_000_000_000));
    commit_log.append(commit_record(101, 11, base_ts + 2_000_000_000));
    commit_log.append(commit_record(102, 12, base_ts + 3_000_000_000));

    let resolved = resolve_timestamp_via_commit_log(&commit_log, base_ts + 2_500_000_000)
        .expect("timestamp should resolve to commit 11");
    assert_eq!(resolved, CommitSeq::new(11));

    let unresolved = resolve_timestamp_via_commit_log(&commit_log, base_ts + 500_000_000);
    assert!(
        unresolved.is_err(),
        "timestamp before first marker must fail"
    );

    let target = TimeTravelTarget::CommitSequence(resolved);
    assert_eq!(target, TimeTravelTarget::CommitSequence(CommitSeq::new(11)));
}

#[test]
fn test_time_travel_tiered_storage_fetch() {
    let object_id = object_id_from_u64(42);
    let payload = b"time-travel-tiered-storage-fetch";
    let full_records = make_symbol_records(object_id, payload, 8, 2);

    let mut local_partial = full_records.clone();
    local_partial.retain(|record| record.esi == 0 || record.esi == 2);

    let mut storage = TieredStorage::new(DurabilityMode::local());
    storage.insert_l2_segment(42, local_partial);

    let mut remote = MockRemoteTier::default();
    remote.set_object_symbols(object_id, full_records);

    let cx = Cx::<cap::All>::new();
    let outcome = storage
        .fetch_object(&cx, object_id, 101, Some(&mut remote), Some(remote_cap(7)))
        .expect("tiered storage fetch should recover remote historical symbols");

    assert_eq!(outcome.bytes, payload);
    assert!(outcome.remote_used);
    assert!(outcome.write_back_count > 0);
    assert_eq!(remote.fetch_calls(), 1);
}

#[test]
fn test_time_travel_cx_budget_enforcement() {
    let object_id = object_id_from_u64(43);
    let payload = b"time-travel-cx-budget-enforcement";
    let full_records = make_symbol_records(object_id, payload, 8, 2);

    let mut local_partial = full_records.clone();
    local_partial.retain(|record| record.esi == 0 || record.esi == 2);

    let mut storage = TieredStorage::new(DurabilityMode::local());
    storage.insert_l2_segment(43, local_partial);

    let mut remote = MockRemoteTier::default();
    remote.set_object_symbols(object_id, full_records);

    let cx = Cx::<cap::All>::new();
    cx.cancel();

    let result = storage.fetch_object(&cx, object_id, 102, Some(&mut remote), Some(remote_cap(8)));
    assert!(matches!(result, Err(FrankenError::Busy)));
    assert_eq!(remote.fetch_calls(), 0);
}

#[test]
fn test_e2e_bd_cfj0_compliance() -> Result<(), String> {
    let description = load_issue_description(BEAD_ID)?;
    let evaluation = evaluate_description(&description);

    eprintln!(
        "INFO bead_id={BEAD_ID} case=e2e_summary missing_unit_ids={} missing_e2e_ids={} missing_log_levels={} missing_log_standard_ref={}",
        evaluation.missing_unit_ids.len(),
        evaluation.missing_e2e_ids.len(),
        evaluation.missing_log_levels.len(),
        evaluation.missing_log_standard_ref
    );

    for id in &evaluation.missing_unit_ids {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_unit_id id={id}");
    }
    for id in &evaluation.missing_e2e_ids {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_e2e_id id={id}");
    }
    for level in &evaluation.missing_log_levels {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_log_level level={level}");
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
