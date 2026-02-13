//! Unified E2E log event schema and scenario coverage validation (bd-1dp9.7.2).
//!
//! Defines the canonical structured log event schema for all E2E test scripts
//! in the FrankenSQLite workspace. Provides schema validation, scenario coverage
//! checking, and log quality assessment tied to the traceability matrix from
//! bd-mblr.4.5.1.
//!
//! # Schema Version
//!
//! The current schema version is `1.0.0`. All E2E scripts should emit events
//! conforming to this schema. The schema includes required fields (run_id,
//! timestamp, phase, event_type) and recommended fields (scenario_id, seed,
//! backend, artifact_hash).
//!
//! # Coverage Assessment
//!
//! The module cross-references the traceability matrix to compute which
//! parity-critical scenarios have E2E script coverage and which have gaps.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::e2e_traceability::{self, TraceabilityMatrix};
use crate::parity_taxonomy::FeatureCategory;

#[allow(dead_code)]
const BEAD_ID: &str = "bd-1dp9.7.2";

/// Schema version for the unified E2E log format.
pub const LOG_SCHEMA_VERSION: &str = "1.0.0";

// ─── Log Event Schema ───────────────────────────────────────────────────

/// Required fields for every E2E log event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LogEventSchema {
    /// Unique run identifier (format: `{bead_id}-{timestamp}-{pid}`).
    pub run_id: String,
    /// ISO 8601 timestamp (UTC).
    pub timestamp: String,
    /// Execution phase (e.g. `setup`, `execute`, `validate`, `teardown`).
    pub phase: LogPhase,
    /// Event type classification.
    pub event_type: LogEventType,
    /// Scenario ID from traceability matrix (optional, recommended).
    pub scenario_id: Option<String>,
    /// Deterministic seed used for this run (optional, recommended).
    pub seed: Option<u64>,
    /// Backend under test (optional).
    pub backend: Option<String>,
    /// SHA-256 hash of output artifact (optional).
    pub artifact_hash: Option<String>,
    /// Structured key-value context fields.
    pub context: BTreeMap<String, String>,
}

/// Execution phase markers for log events.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogPhase {
    /// Initial setup (create tables, seed data).
    Setup,
    /// Main test execution.
    Execute,
    /// Result validation and comparison.
    Validate,
    /// Cleanup and resource release.
    Teardown,
    /// Summary/report generation.
    Report,
}

/// Classification of log event types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogEventType {
    /// Test started.
    Start,
    /// Test passed.
    Pass,
    /// Test failed.
    Fail,
    /// Test skipped with rationale.
    Skip,
    /// Informational event.
    Info,
    /// Warning (non-fatal issue).
    Warn,
    /// Error (fatal issue).
    Error,
    /// First divergence point detected.
    FirstDivergence,
    /// Artifact generated (hash available).
    ArtifactGenerated,
}

/// Schema field requirement level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FieldRequirement {
    /// Must be present in every event.
    Required,
    /// Should be present when applicable.
    Recommended,
    /// May be present for additional context.
    Optional,
}

/// Description of a schema field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FieldSpec {
    pub name: String,
    pub description: String,
    pub requirement: FieldRequirement,
    pub example: String,
}

// ─── Schema Documentation ───────────────────────────────────────────────

/// Build the canonical field specification for the unified log schema.
#[must_use]
pub fn build_field_specs() -> Vec<FieldSpec> {
    vec![
        FieldSpec {
            name: "run_id".to_owned(),
            description: "Unique run identifier for log correlation".to_owned(),
            requirement: FieldRequirement::Required,
            example: "bd-mblr-20260213T050000Z-12345".to_owned(),
        },
        FieldSpec {
            name: "timestamp".to_owned(),
            description: "ISO 8601 UTC timestamp of the event".to_owned(),
            requirement: FieldRequirement::Required,
            example: "2026-02-13T05:00:00.000Z".to_owned(),
        },
        FieldSpec {
            name: "phase".to_owned(),
            description: "Execution phase (setup/execute/validate/teardown/report)".to_owned(),
            requirement: FieldRequirement::Required,
            example: "execute".to_owned(),
        },
        FieldSpec {
            name: "event_type".to_owned(),
            description: "Event classification (start/pass/fail/skip/info/warn/error)".to_owned(),
            requirement: FieldRequirement::Required,
            example: "pass".to_owned(),
        },
        FieldSpec {
            name: "scenario_id".to_owned(),
            description: "Scenario ID from traceability matrix (CATEGORY-NUMBER)".to_owned(),
            requirement: FieldRequirement::Recommended,
            example: "MVCC-3".to_owned(),
        },
        FieldSpec {
            name: "seed".to_owned(),
            description: "Deterministic seed used for reproducibility".to_owned(),
            requirement: FieldRequirement::Recommended,
            example: "6148914689804861784".to_owned(),
        },
        FieldSpec {
            name: "backend".to_owned(),
            description: "Backend under test (fsqlite/rusqlite/both)".to_owned(),
            requirement: FieldRequirement::Recommended,
            example: "fsqlite".to_owned(),
        },
        FieldSpec {
            name: "artifact_hash".to_owned(),
            description: "SHA-256 hash of generated artifact".to_owned(),
            requirement: FieldRequirement::Optional,
            example: "a1b2c3d4...".to_owned(),
        },
        FieldSpec {
            name: "context".to_owned(),
            description: "Free-form key-value pairs for additional context".to_owned(),
            requirement: FieldRequirement::Optional,
            example: "{\"table_count\": \"5\", \"concurrency\": \"4\"}".to_owned(),
        },
    ]
}

// ─── Scenario Coverage Assessment ───────────────────────────────────────

/// Category of parity-critical scenario.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScenarioCriticality {
    /// Must pass for any release.
    Critical,
    /// Should pass but degraded mode acceptable.
    Important,
    /// Nice to have.
    Standard,
}

/// A parity-critical scenario and its coverage status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CriticalScenario {
    /// Scenario ID.
    pub scenario_id: String,
    /// Feature category this scenario validates.
    pub category: FeatureCategory,
    /// Criticality level.
    pub criticality: ScenarioCriticality,
    /// Description of what this scenario validates.
    pub description: String,
    /// Whether this scenario has E2E script coverage.
    pub covered: bool,
    /// Script paths that cover this scenario.
    pub covering_scripts: Vec<String>,
    /// Replay command for this scenario.
    pub replay_command: Option<String>,
}

/// Assessment of E2E scenario coverage completeness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioCoverageReport {
    /// Schema version.
    pub schema_version: String,
    /// Bead ID.
    pub bead_id: String,
    /// Log schema version.
    pub log_schema_version: String,
    /// All critical scenarios with coverage status.
    pub scenarios: Vec<CriticalScenario>,
    /// Coverage statistics.
    pub stats: CoverageReportStats,
}

/// Statistics from the coverage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReportStats {
    pub total_scenarios: usize,
    pub covered_scenarios: usize,
    pub uncovered_scenarios: usize,
    pub critical_covered: usize,
    pub critical_total: usize,
    pub important_covered: usize,
    pub important_total: usize,
    pub coverage_pct: f64,
    pub by_category: BTreeMap<String, CategoryCoverageStats>,
}

/// Per-category coverage stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryCoverageStats {
    pub total: usize,
    pub covered: usize,
    pub pct: f64,
}

impl ScenarioCoverageReport {
    /// Validate the report.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // No duplicate scenario IDs
        let mut seen = BTreeSet::new();
        for s in &self.scenarios {
            if !seen.insert(&s.scenario_id) {
                errors.push(format!("Duplicate scenario ID: {}", s.scenario_id));
            }
        }

        // Every covered scenario must have at least one covering script
        for s in &self.scenarios {
            if s.covered && s.covering_scripts.is_empty() {
                errors.push(format!(
                    "Scenario {} marked covered but has no covering scripts",
                    s.scenario_id
                ));
            }
        }

        // Critical scenarios should have replay commands
        for s in &self.scenarios {
            if s.covered
                && s.criticality == ScenarioCriticality::Critical
                && s.replay_command.is_none()
            {
                errors.push(format!(
                    "Critical scenario {} lacks replay command",
                    s.scenario_id
                ));
            }
        }

        // Stats consistency
        let actual_covered = self.scenarios.iter().filter(|s| s.covered).count();
        if actual_covered != self.stats.covered_scenarios {
            errors.push(format!(
                "Stats mismatch: counted {} covered but stats says {}",
                actual_covered, self.stats.covered_scenarios
            ));
        }

        errors
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// ─── Build the Coverage Report ──────────────────────────────────────────

/// Build the parity-critical scenario coverage report by cross-referencing
/// the traceability matrix with the canonical critical scenario list.
#[must_use]
pub fn build_coverage_report() -> ScenarioCoverageReport {
    let matrix = e2e_traceability::build_canonical_inventory();
    let critical_scenarios = build_critical_scenario_list();

    let scenarios = assess_coverage(&matrix, critical_scenarios);
    let stats = compute_stats(&scenarios);

    ScenarioCoverageReport {
        schema_version: "1.0.0".to_owned(),
        bead_id: BEAD_ID.to_owned(),
        log_schema_version: LOG_SCHEMA_VERSION.to_owned(),
        scenarios,
        stats,
    }
}

fn build_critical_scenario_list() -> Vec<(String, FeatureCategory, ScenarioCriticality, String)> {
    vec![
        // SQL Grammar — Critical
        (
            "SQL-1".to_owned(),
            FeatureCategory::SqlGrammar,
            ScenarioCriticality::Critical,
            "DDL statement compliance".to_owned(),
        ),
        (
            "SQL-2".to_owned(),
            FeatureCategory::SqlGrammar,
            ScenarioCriticality::Critical,
            "SELECT statement compliance".to_owned(),
        ),
        (
            "SQL-3".to_owned(),
            FeatureCategory::SqlGrammar,
            ScenarioCriticality::Critical,
            "Full SQL roundtrip".to_owned(),
        ),
        (
            "SQL-4".to_owned(),
            FeatureCategory::SqlGrammar,
            ScenarioCriticality::Important,
            "VACUUM and PRAGMA compliance".to_owned(),
        ),
        (
            "SQL-5".to_owned(),
            FeatureCategory::SqlGrammar,
            ScenarioCriticality::Standard,
            "Query pipeline compliance".to_owned(),
        ),
        (
            "SQL-6".to_owned(),
            FeatureCategory::SqlGrammar,
            ScenarioCriticality::Standard,
            "SQL pattern coverage (broad)".to_owned(),
        ),
        // Concurrency — Critical
        (
            "CON-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "Concurrent-writer compliance gate".to_owned(),
        ),
        (
            "CON-3".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "Concurrent multi-thread writes".to_owned(),
        ),
        (
            "CON-5".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "MVCC concurrent writer stress".to_owned(),
        ),
        (
            "CON-6".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "Deterministic concurrency".to_owned(),
        ),
        (
            "CON-7".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "MVCC writer stress (harness)".to_owned(),
        ),
        // MVCC — Critical
        (
            "MVCC-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "Phase 5 MVCC compliance".to_owned(),
        ),
        (
            "MVCC-2".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "MVCC isolation validation".to_owned(),
        ),
        (
            "MVCC-3".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "Concurrent write correctness".to_owned(),
        ),
        (
            "MVCC-4".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "MVCC writer stress (E2E)".to_owned(),
        ),
        (
            "MVCC-5".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "MVCC stress (harness)".to_owned(),
        ),
        (
            "MVCC-7".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Standard,
            "Time-travel queries".to_owned(),
        ),
        // SSI — Critical
        (
            "SSI-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "SSI write-skew detection".to_owned(),
        ),
        (
            "SSI-2".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "SSI write-skew prevention".to_owned(),
        ),
        (
            "SSI-3".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "Phase 6 SSI compliance".to_owned(),
        ),
        // Transaction — Critical
        (
            "TXN-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "Transaction semantics".to_owned(),
        ),
        (
            "TXN-2".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "Savepoint semantics".to_owned(),
        ),
        (
            "TXN-3".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "Transaction control harness".to_owned(),
        ),
        // Recovery — Critical
        (
            "REC-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "WAL replay after crash".to_owned(),
        ),
        (
            "REC-2".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "Single-page recovery".to_owned(),
        ),
        (
            "REC-3".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "WAL corruption recovery".to_owned(),
        ),
        (
            "REC-4".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "Crash recovery (harness)".to_owned(),
        ),
        // WAL — Critical
        (
            "WAL-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Critical,
            "WAL replay correctness".to_owned(),
        ),
        (
            "WAL-2".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "WAL integrity after crash".to_owned(),
        ),
        (
            "WAL-3".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "WAL checksum chain".to_owned(),
        ),
        // Compatibility — Critical
        (
            "COMPAT-1".to_owned(),
            FeatureCategory::FileFormat,
            ScenarioCriticality::Critical,
            "Real database integrity".to_owned(),
        ),
        (
            "COMPAT-3".to_owned(),
            FeatureCategory::FileFormat,
            ScenarioCriticality::Critical,
            "File format compatibility".to_owned(),
        ),
        (
            "COMPAT-4".to_owned(),
            FeatureCategory::FileFormat,
            ScenarioCriticality::Important,
            "Behavioral quirks compat".to_owned(),
        ),
        (
            "COMPAT-5".to_owned(),
            FeatureCategory::FileFormat,
            ScenarioCriticality::Important,
            "File format versioning".to_owned(),
        ),
        // Extensions — Important
        (
            "EXT-1".to_owned(),
            FeatureCategory::Extensions,
            ScenarioCriticality::Important,
            "FTS3 compatibility".to_owned(),
        ),
        (
            "EXT-2".to_owned(),
            FeatureCategory::Extensions,
            ScenarioCriticality::Important,
            "FTS5 compliance".to_owned(),
        ),
        (
            "EXT-3".to_owned(),
            FeatureCategory::Extensions,
            ScenarioCriticality::Important,
            "FTS3/FTS4 backward compat".to_owned(),
        ),
        (
            "EXT-4".to_owned(),
            FeatureCategory::Extensions,
            ScenarioCriticality::Important,
            "JSON1 extension".to_owned(),
        ),
        // Functions — Standard
        (
            "FUN-1".to_owned(),
            FeatureCategory::BuiltinFunctions,
            ScenarioCriticality::Standard,
            "Date/time functions".to_owned(),
        ),
        // Performance — Standard
        (
            "PERF-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Standard,
            "ARC warmup".to_owned(),
        ),
        (
            "PERF-2".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Standard,
            "SSI performance".to_owned(),
        ),
        (
            "PERF-3".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Standard,
            "B-tree hotspot".to_owned(),
        ),
        // FEC — Important
        (
            "FEC-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "WAL FEC group metadata".to_owned(),
        ),
        (
            "FEC-2".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "WAL FEC repair symbols".to_owned(),
        ),
        (
            "FEC-3".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Standard,
            "RaptorQ E2E integration".to_owned(),
        ),
        // Correctness — Critical
        (
            "COR-1".to_owned(),
            FeatureCategory::SqlGrammar,
            ScenarioCriticality::Critical,
            "Sequential insert correctness".to_owned(),
        ),
        (
            "COR-2".to_owned(),
            FeatureCategory::SqlGrammar,
            ScenarioCriticality::Critical,
            "Mixed DML correctness".to_owned(),
        ),
        // Seed — Important
        (
            "SEED-1".to_owned(),
            FeatureCategory::StorageTransaction,
            ScenarioCriticality::Important,
            "Seed reproducibility".to_owned(),
        ),
        // Infrastructure — Standard
        (
            "INFRA-5".to_owned(),
            FeatureCategory::ApiCli,
            ScenarioCriticality::Standard,
            "Workspace layering".to_owned(),
        ),
        (
            "INFRA-6".to_owned(),
            FeatureCategory::ApiCli,
            ScenarioCriticality::Standard,
            "Logging standard".to_owned(),
        ),
    ]
}

fn assess_coverage(
    matrix: &TraceabilityMatrix,
    critical_scenarios: Vec<(String, FeatureCategory, ScenarioCriticality, String)>,
) -> Vec<CriticalScenario> {
    // Build reverse index: scenario_id -> script paths
    let mut scenario_to_scripts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut scenario_to_command: BTreeMap<String, String> = BTreeMap::new();

    for script in &matrix.scripts {
        for sid in &script.scenario_ids {
            scenario_to_scripts
                .entry(sid.clone())
                .or_default()
                .push(script.path.clone());
            scenario_to_command
                .entry(sid.clone())
                .or_insert_with(|| script.invocation.command.clone());
        }
    }

    critical_scenarios
        .into_iter()
        .map(|(id, category, criticality, description)| {
            let scripts = scenario_to_scripts.get(&id).cloned().unwrap_or_default();
            let covered = !scripts.is_empty();
            let replay_command = scenario_to_command.get(&id).cloned();

            CriticalScenario {
                scenario_id: id,
                category,
                criticality,
                description,
                covered,
                covering_scripts: scripts,
                replay_command,
            }
        })
        .collect()
}

fn compute_stats(scenarios: &[CriticalScenario]) -> CoverageReportStats {
    let total = scenarios.len();
    let covered = scenarios.iter().filter(|s| s.covered).count();
    let uncovered = total - covered;

    let critical_total = scenarios
        .iter()
        .filter(|s| s.criticality == ScenarioCriticality::Critical)
        .count();
    let critical_covered = scenarios
        .iter()
        .filter(|s| s.criticality == ScenarioCriticality::Critical && s.covered)
        .count();

    let important_total = scenarios
        .iter()
        .filter(|s| s.criticality == ScenarioCriticality::Important)
        .count();
    let important_covered = scenarios
        .iter()
        .filter(|s| s.criticality == ScenarioCriticality::Important && s.covered)
        .count();

    let coverage_pct = if total > 0 {
        truncate_f64(covered as f64 / total as f64, 4)
    } else {
        0.0
    };

    // Per-category stats
    let mut by_category: BTreeMap<String, CategoryCoverageStats> = BTreeMap::new();
    for cat in FeatureCategory::ALL {
        let cat_scenarios: Vec<_> = scenarios.iter().filter(|s| s.category == cat).collect();
        let cat_total = cat_scenarios.len();
        if cat_total > 0 {
            let cat_covered = cat_scenarios.iter().filter(|s| s.covered).count();
            by_category.insert(
                format!("{cat:?}"),
                CategoryCoverageStats {
                    total: cat_total,
                    covered: cat_covered,
                    pct: truncate_f64(cat_covered as f64 / cat_total as f64, 4),
                },
            );
        }
    }

    CoverageReportStats {
        total_scenarios: total,
        covered_scenarios: covered,
        uncovered_scenarios: uncovered,
        critical_covered,
        critical_total,
        important_covered,
        important_total,
        coverage_pct,
        by_category,
    }
}

fn truncate_f64(value: f64, decimals: u32) -> f64 {
    let exp = i32::try_from(decimals).unwrap_or(6);
    let factor = 10_f64.powi(exp);
    (value * factor).trunc() / factor
}

// ─── Log Quality Validator ──────────────────────────────────────────────

/// Validate that a log event conforms to the schema.
pub fn validate_log_event(event: &LogEventSchema) -> Vec<String> {
    let mut errors = Vec::new();

    if event.run_id.is_empty() {
        errors.push("run_id is empty".to_owned());
    }

    if event.timestamp.is_empty() {
        errors.push("timestamp is empty".to_owned());
    }

    // Scenario ID should follow convention if present
    if let Some(ref sid) = event.scenario_id {
        if !sid.contains('-') {
            errors.push(format!(
                "scenario_id '{}' doesn't follow CATEGORY-NUMBER convention",
                sid
            ));
        }
    }

    errors
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_report_builds() {
        let report = build_coverage_report();
        assert!(!report.scenarios.is_empty());
        assert_eq!(report.schema_version, "1.0.0");
        assert_eq!(report.bead_id, BEAD_ID);
    }

    #[test]
    fn coverage_report_validates() {
        let report = build_coverage_report();
        let errors = report.validate();
        assert!(
            errors.is_empty(),
            "Validation errors:\n{}",
            errors.join("\n")
        );
    }

    #[test]
    fn critical_scenarios_have_coverage() {
        let report = build_coverage_report();
        // All critical scenarios should be covered
        for s in &report.scenarios {
            if s.criticality == ScenarioCriticality::Critical {
                assert!(
                    s.covered,
                    "Critical scenario {} is not covered: {}",
                    s.scenario_id, s.description
                );
            }
        }
    }

    #[test]
    fn critical_scenarios_have_replay() {
        let report = build_coverage_report();
        for s in &report.scenarios {
            if s.criticality == ScenarioCriticality::Critical && s.covered {
                assert!(
                    s.replay_command.is_some(),
                    "Critical scenario {} lacks replay command",
                    s.scenario_id
                );
            }
        }
    }

    #[test]
    fn coverage_pct_is_high() {
        let report = build_coverage_report();
        assert!(
            report.stats.coverage_pct >= 0.9,
            "Expected >= 90% coverage, got {:.1}%",
            report.stats.coverage_pct * 100.0
        );
    }

    #[test]
    fn field_specs_complete() {
        let specs = build_field_specs();
        let required: Vec<_> = specs
            .iter()
            .filter(|s| s.requirement == FieldRequirement::Required)
            .collect();
        assert!(required.len() >= 4, "Need at least 4 required fields");
    }

    #[test]
    fn log_event_validation() {
        let good_event = LogEventSchema {
            run_id: "test-run-001".to_owned(),
            timestamp: "2026-02-13T05:00:00Z".to_owned(),
            phase: LogPhase::Execute,
            event_type: LogEventType::Pass,
            scenario_id: Some("MVCC-3".to_owned()),
            seed: Some(42),
            backend: Some("fsqlite".to_owned()),
            artifact_hash: None,
            context: BTreeMap::new(),
        };

        let errors = validate_log_event(&good_event);
        assert!(
            errors.is_empty(),
            "Good event should validate: {:?}",
            errors
        );

        let bad_event = LogEventSchema {
            run_id: String::new(),
            timestamp: String::new(),
            phase: LogPhase::Execute,
            event_type: LogEventType::Pass,
            scenario_id: Some("invalid".to_owned()),
            seed: None,
            backend: None,
            artifact_hash: None,
            context: BTreeMap::new(),
        };

        let errors = validate_log_event(&bad_event);
        assert!(!errors.is_empty(), "Bad event should have errors");
    }

    #[test]
    fn json_roundtrip() {
        let report = build_coverage_report();
        let json = report.to_json().expect("serialize");
        let deserialized: ScenarioCoverageReport =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.scenarios.len(), report.scenarios.len());
    }

    #[test]
    fn no_duplicate_scenario_ids() {
        let report = build_coverage_report();
        let mut seen = BTreeSet::new();
        for s in &report.scenarios {
            assert!(seen.insert(&s.scenario_id), "Duplicate: {}", s.scenario_id);
        }
    }

    #[test]
    fn stats_consistency() {
        let report = build_coverage_report();
        assert_eq!(
            report.stats.covered_scenarios + report.stats.uncovered_scenarios,
            report.stats.total_scenarios,
        );
        assert_eq!(
            report.stats.critical_covered + report.stats.critical_total
                - report.stats.critical_total,
            report.stats.critical_covered,
        );
    }

    #[test]
    fn category_coverage_present() {
        let report = build_coverage_report();
        // At least some categories should have coverage info
        assert!(
            !report.stats.by_category.is_empty(),
            "Should have per-category stats"
        );
    }

    #[test]
    fn deterministic_report() {
        let r1 = build_coverage_report();
        let r2 = build_coverage_report();
        assert_eq!(r1.scenarios.len(), r2.scenarios.len());
        assert_eq!(r1.stats.coverage_pct, r2.stats.coverage_pct);
    }
}
