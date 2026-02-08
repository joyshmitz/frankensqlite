//! Deterministic harness logging and repro-bundle utilities (`bd-1fpm`).
//!
//! This module defines a single logging standard for test runners:
//! - `meta.json` for run metadata
//! - `events.jsonl` for structured lifecycle events
//! - `stdout.log` / `stderr.log` for text streams
//! - optional engine artifacts (DB/WAL/SHM, oracle diffs, etc.)

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use fsqlite_error::{FrankenError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::{error, info, warn};

/// Version of the harness logging schema.
pub const LOG_SCHEMA_VERSION: u32 = 1;

/// Files that must be present in every repro bundle.
pub const REQUIRED_BUNDLE_FILES: [&str; 4] =
    ["meta.json", "events.jsonl", "stdout.log", "stderr.log"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleEventKind {
    RunStart,
    Setup,
    Step,
    Assertion,
    Teardown,
    RunEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Passed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleMeta {
    pub schema_version: u32,
    pub suite: String,
    pub case_id: String,
    pub seed: u64,
    pub harness_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessEvent {
    pub kind: LifecycleEventKind,
    pub status: Option<RunStatus>,
    pub step: u64,
    pub message: String,
    pub payload: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConformanceDiff {
    pub case_id: String,
    pub sql: String,
    pub params: String,
    pub oracle_result: String,
    pub franken_result: String,
    pub diff: String,
}

#[derive(Debug)]
pub struct ReproBundle {
    root: PathBuf,
    events_file: File,
    next_step: u64,
}

impl ReproBundle {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn emit_event(
        &mut self,
        kind: LifecycleEventKind,
        message: impl Into<String>,
        payload: BTreeMap<String, Value>,
    ) -> Result<()> {
        let event = HarnessEvent {
            kind,
            status: None,
            step: self.next_step,
            message: message.into(),
            payload,
        };
        self.next_step = self.next_step.saturating_add(1);
        self.write_event_line(&event)
    }

    pub fn append_stdout(&self, text: &str) -> Result<()> {
        append_line(self.root.join("stdout.log"), text)
    }

    pub fn append_stderr(&self, text: &str) -> Result<()> {
        append_line(self.root.join("stderr.log"), text)
    }

    pub fn write_artifact_json<T: Serialize>(
        &self,
        relative_path: &str,
        value: &T,
    ) -> Result<PathBuf> {
        let artifact_path = self.root.join(relative_path);
        if let Some(parent) = artifact_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let bytes = serde_json::to_vec_pretty(value)
            .map_err(|err| internal_error(format!("failed to serialize artifact JSON: {err}")))?;
        fs::write(&artifact_path, bytes)?;
        Ok(artifact_path)
    }

    pub fn record_conformance_diff(&mut self, diff: &ConformanceDiff) -> Result<PathBuf> {
        let artifact_path = self.write_artifact_json("oracle_diff.json", diff)?;

        let mut payload = BTreeMap::new();
        payload.insert("case_id".to_string(), Value::String(diff.case_id.clone()));
        payload.insert("sql".to_string(), Value::String(diff.sql.clone()));
        payload.insert("params".to_string(), Value::String(diff.params.clone()));
        payload.insert(
            "oracle_result".to_string(),
            Value::String(diff.oracle_result.clone()),
        );
        payload.insert(
            "franken_result".to_string(),
            Value::String(diff.franken_result.clone()),
        );
        payload.insert("diff".to_string(), Value::String(diff.diff.clone()));

        self.emit_event(LifecycleEventKind::Assertion, "oracle_diff", payload)?;
        Ok(artifact_path)
    }

    pub fn finish(mut self, status: RunStatus) -> Result<PathBuf> {
        let event = HarnessEvent {
            kind: LifecycleEventKind::RunEnd,
            status: Some(status),
            step: self.next_step,
            message: "run_end".to_string(),
            payload: BTreeMap::new(),
        };
        self.write_event_line(&event)?;
        self.events_file.flush()?;
        info!(
            suite = %self.root.display(),
            status = ?status,
            "harness repro bundle finalized"
        );
        Ok(self.root)
    }

    fn write_event_line(&mut self, event: &HarnessEvent) -> Result<()> {
        let encoded = serde_json::to_string(event)
            .map_err(|err| internal_error(format!("failed to serialize harness event: {err}")))?;
        writeln!(self.events_file, "{encoded}")?;
        self.events_file.flush()?;
        Ok(())
    }
}

pub fn init_repro_bundle(
    base_dir: &Path,
    suite: &str,
    case_id: &str,
    seed: u64,
) -> Result<ReproBundle> {
    if suite.is_empty() {
        return Err(internal_error("suite must be non-empty"));
    }
    if case_id.is_empty() {
        return Err(internal_error("case_id must be non-empty"));
    }

    let bundle_name = bundle_dir_name(suite, case_id, seed);
    let root = base_dir.join(bundle_name);
    fs::create_dir_all(&root)?;

    let meta = BundleMeta {
        schema_version: LOG_SCHEMA_VERSION,
        suite: suite.to_string(),
        case_id: case_id.to_string(),
        seed,
        harness_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    write_json_file(root.join("meta.json"), &meta)?;

    File::create(root.join("stdout.log"))?;
    File::create(root.join("stderr.log"))?;

    let events_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join("events.jsonl"))?;

    let mut bundle = ReproBundle {
        root,
        events_file,
        next_step: 0,
    };
    bundle.emit_event(LifecycleEventKind::RunStart, "run_start", BTreeMap::new())?;

    info!(
        suite = suite,
        case_id = case_id,
        seed = seed,
        root = %bundle.root.display(),
        "harness repro bundle initialized"
    );

    Ok(bundle)
}

pub fn validate_required_files(bundle_root: &Path) -> Result<()> {
    let missing: Vec<&str> = REQUIRED_BUNDLE_FILES
        .iter()
        .copied()
        .filter(|name| !bundle_root.join(name).is_file())
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    error!(
        bundle = %bundle_root.display(),
        missing_count = missing.len(),
        "missing required repro bundle files"
    );
    Err(internal_error(format!(
        "missing required bundle files: {}",
        missing.join(", ")
    )))
}

pub fn validate_bundle_meta(bundle_root: &Path) -> Result<BundleMeta> {
    let meta_path = bundle_root.join("meta.json");
    let bytes = fs::read(meta_path)?;
    let meta: BundleMeta = serde_json::from_slice(&bytes)
        .map_err(|err| internal_error(format!("meta.json parse failure: {err}")))?;

    if meta.schema_version != LOG_SCHEMA_VERSION {
        warn!(
            expected = LOG_SCHEMA_VERSION,
            found = meta.schema_version,
            "bundle schema version mismatch"
        );
        return Err(internal_error(format!(
            "unsupported schema version: expected {LOG_SCHEMA_VERSION}, got {}",
            meta.schema_version
        )));
    }

    if meta.suite.is_empty() || meta.case_id.is_empty() {
        return Err(internal_error(
            "meta.json must include non-empty suite and case_id",
        ));
    }

    Ok(meta)
}

pub fn validate_events_jsonl(bundle_root: &Path) -> Result<Vec<HarnessEvent>> {
    let events_path = bundle_root.join("events.jsonl");
    let file = File::open(events_path)?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for (line_no, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() {
            return Err(internal_error(format!(
                "events.jsonl has empty line at {}",
                line_no + 1
            )));
        }
        let event: HarnessEvent = serde_json::from_str(&line).map_err(|err| {
            internal_error(format!(
                "events.jsonl parse failure at line {}: {err}",
                line_no + 1
            ))
        })?;
        if event.message.is_empty() {
            return Err(internal_error(format!(
                "events.jsonl has empty message at line {}",
                line_no + 1
            )));
        }
        events.push(event);
    }

    if events.is_empty() {
        return Err(internal_error(
            "events.jsonl must contain at least one event",
        ));
    }

    Ok(events)
}

pub fn validate_bundle(bundle_root: &Path) -> Result<()> {
    validate_required_files(bundle_root)?;
    let _meta = validate_bundle_meta(bundle_root)?;
    let events = validate_events_jsonl(bundle_root)?;

    if events.first().map(|event| event.kind) != Some(LifecycleEventKind::RunStart) {
        return Err(internal_error(
            "events.jsonl must start with a run_start event",
        ));
    }
    if events.last().map(|event| event.kind) != Some(LifecycleEventKind::RunEnd) {
        return Err(internal_error("events.jsonl must end with a run_end event"));
    }

    Ok(())
}

fn append_line(path: PathBuf, text: &str) -> Result<()> {
    let mut file = OpenOptions::new().append(true).create(true).open(path)?;
    writeln!(file, "{text}")?;
    Ok(())
}

fn write_json_file<T: Serialize>(path: PathBuf, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|err| internal_error(format!("failed to serialize JSON: {err}")))?;
    fs::write(path, bytes)?;
    Ok(())
}

fn bundle_dir_name(suite: &str, case_id: &str, seed: u64) -> String {
    format!(
        "{}-{}-seed-{seed}",
        sanitize_segment(suite),
        sanitize_segment(case_id)
    )
}

fn sanitize_segment(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn internal_error(message: impl Into<String>) -> FrankenError {
    FrankenError::Internal(message.into())
}
