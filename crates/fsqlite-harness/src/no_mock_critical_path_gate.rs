//! No-mock critical-path CI gate (bd-mblr.3.4.1).
//!
//! Enforces that every invariant in a critical category (SqlGrammar,
//! VdbeOpcodes, StorageTransaction) has at least one non-mock evidence
//! entry exercising real components. Mock-only coverage of critical-path
//! invariants is a blocking failure.
//!
//! # Architecture
//!
//! The gate pulls data from:
//! 1. [`UnitMatrix`] — per-category tests with invariant lists
//! 2. [`NoMockEvidenceMap`] — real-component evidence for invariants
//!
//! It produces a [`NoMockCriticalPathReport`] with:
//! - Per-invariant pass/fail for critical categories
//! - Violations with actionable remediation
//! - Machine-readable JSON and human-readable markdown
//!
//! # Upstream Dependencies
//!
//! - [`unit_matrix`](crate::unit_matrix) (bd-1dp9.7.1)
//! - [`no_mock_evidence`](crate::no_mock_evidence) (bd-mblr.6.7)
//! - [`parity_taxonomy`](crate::parity_taxonomy) (bd-1dp9.1.1)
//!
//! # Downstream Consumers
//!
//! - **bd-mblr.3.5.1**: Machine-readable validation manifest
//! - **bd-mblr.3.5**: Unified quality evidence rollup
//! - **bd-mblr.3.3**: Flake budget and quarantine workflow

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

use crate::no_mock_evidence::{NoMockEvidenceMap, build_evidence_map};
use crate::parity_taxonomy::FeatureCategory;
use crate::unit_matrix::{UnitMatrix, build_canonical_matrix};

const BEAD_ID: &str = "bd-mblr.3.4.1";

/// Schema version for report compatibility.
pub const NO_MOCK_GATE_SCHEMA_VERSION: &str = "1.0.0";

/// Default critical categories that require non-mock evidence.
///
/// Kept explicit for auditability, then validated against taxonomy-derived
/// critical categories at runtime to fail closed on drift.
pub const DEFAULT_CRITICAL_CATEGORIES: [FeatureCategory; 3] = [
    FeatureCategory::SqlGrammar,
    FeatureCategory::VdbeOpcodes,
    FeatureCategory::StorageTransaction,
];

/// Any category at or above this global weight is treated as critical.
pub const MIN_CRITICAL_GLOBAL_WEIGHT: f64 = 0.15;

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// Overall verdict of the no-mock critical-path gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoMockVerdict {
    /// All critical-path invariants have non-mock evidence.
    Pass,
    /// One or more critical-path invariants lack non-mock evidence.
    Fail,
    /// All critical invariants have evidence, but some are exceptions.
    PassWithWarnings,
}

impl std::fmt::Display for NoMockVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::PassWithWarnings => write!(f, "PASS_WITH_WARNINGS"),
        }
    }
}

// ---------------------------------------------------------------------------
// Violation
// ---------------------------------------------------------------------------

/// Severity of a no-mock gate violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoMockViolationSeverity {
    /// Warning: exception is documented but non-mock test is preferred.
    Warning,
    /// Blocking: critical invariant has no non-mock evidence at all.
    Blocking,
}

impl std::fmt::Display for NoMockViolationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Warning => write!(f, "warning"),
            Self::Blocking => write!(f, "blocking"),
        }
    }
}

/// A single violation found by the no-mock critical-path gate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NoMockViolation {
    /// Which invariant is affected.
    pub invariant: String,
    /// The matrix test ID referencing this invariant.
    pub matrix_test_id: String,
    /// The feature category of this invariant.
    pub category: String,
    /// Severity of the violation.
    pub severity: NoMockViolationSeverity,
    /// Why this is a violation.
    pub reason: String,
    /// Actionable remediation instruction.
    pub remediation: String,
}

// ---------------------------------------------------------------------------
// Per-category result
// ---------------------------------------------------------------------------

/// Result for a single critical category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalCategoryResult {
    /// Feature category.
    pub category: FeatureCategory,
    /// Display name.
    pub display_name: String,
    /// Total invariants in this category.
    pub total_invariants: usize,
    /// Invariants with non-mock evidence (not exceptions).
    pub real_evidence_count: usize,
    /// Invariants with exception-only evidence.
    pub exception_count: usize,
    /// Invariants with no evidence at all.
    pub missing_count: usize,
    /// Coverage percentage (real evidence / total).
    pub coverage_pct: f64,
    /// Whether this category passes the gate.
    pub passes: bool,
}

// ---------------------------------------------------------------------------
// Gate report
// ---------------------------------------------------------------------------

/// Complete report produced by the no-mock critical-path gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoMockCriticalPathReport {
    /// Schema version.
    pub schema_version: String,
    /// Bead ID.
    pub bead_id: String,
    /// Overall verdict.
    pub verdict: NoMockVerdict,
    /// Total critical-path invariants across all critical categories.
    pub total_critical_invariants: usize,
    /// Invariants with real (non-mock, non-exception) evidence.
    pub real_evidence_count: usize,
    /// Invariants covered by exceptions.
    pub exception_count: usize,
    /// Invariants with no evidence.
    pub missing_evidence_count: usize,
    /// Coverage percentage (real + exception / total).
    pub coverage_pct: f64,
    /// Per-category results.
    pub categories: Vec<CriticalCategoryResult>,
    /// All violations found.
    pub violations: Vec<NoMockViolation>,
    /// Blocking violation count.
    pub blocking_count: usize,
    /// Warning count.
    pub warning_count: usize,
    /// Human-readable summary.
    pub summary: String,
}

impl NoMockCriticalPathReport {
    /// Serialize to deterministic JSON.
    ///
    /// # Errors
    ///
    /// Returns `Err` if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the JSON is malformed.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Render a human-readable markdown summary.
    #[must_use]
    pub fn render_summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# No-Mock Critical-Path Gate Report");
        let _ = writeln!(out);
        let _ = writeln!(out, "**Verdict**: {}", self.verdict);
        let _ = writeln!(
            out,
            "**Critical Invariants**: {} total, {} with real evidence, {} exceptions, {} missing",
            self.total_critical_invariants,
            self.real_evidence_count,
            self.exception_count,
            self.missing_evidence_count
        );
        let _ = writeln!(out, "**Coverage**: {:.1}%", self.coverage_pct * 100.0);
        let _ = writeln!(out);

        let _ = writeln!(out, "## Per-Category Results");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| Category | Total | Real Evidence | Exceptions | Missing | Coverage | Status |"
        );
        let _ = writeln!(
            out,
            "|----------|-------|---------------|------------|---------|----------|--------|"
        );
        for cat in &self.categories {
            let status = if cat.passes { "PASS" } else { "FAIL" };
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} | {:.1}% | {} |",
                cat.display_name,
                cat.total_invariants,
                cat.real_evidence_count,
                cat.exception_count,
                cat.missing_count,
                cat.coverage_pct * 100.0,
                status,
            );
        }

        if !self.violations.is_empty() {
            let _ = writeln!(out);
            let _ = writeln!(out, "## Violations");
            let _ = writeln!(out);
            for v in &self.violations {
                let _ = writeln!(
                    out,
                    "- **[{}]** `{}` ({}): {}",
                    v.severity, v.matrix_test_id, v.category, v.reason
                );
                let _ = writeln!(out, "  - Remediation: {}", v.remediation);
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluate the no-mock critical-path gate.
///
/// Builds the unit matrix and evidence map, cross-references them to
/// ensure every invariant in a critical category has non-mock evidence.
#[must_use]
pub fn evaluate_no_mock_critical_path_gate(
    critical_categories: &[FeatureCategory],
) -> NoMockCriticalPathReport {
    let matrix = build_canonical_matrix();
    let evidence_map = build_evidence_map();
    evaluate_with_data(&matrix, &evidence_map, critical_categories)
}

fn taxonomy_critical_categories() -> Vec<FeatureCategory> {
    FeatureCategory::ALL
        .into_iter()
        .filter(|category| category.global_weight() >= MIN_CRITICAL_GLOBAL_WEIGHT)
        .collect()
}

fn normalize_critical_categories(
    critical_categories: &[FeatureCategory],
) -> (Vec<FeatureCategory>, Vec<NoMockViolation>) {
    let mut configured_categories: BTreeSet<FeatureCategory> = BTreeSet::new();
    let mut duplicate_categories: BTreeSet<FeatureCategory> = BTreeSet::new();

    for category in critical_categories {
        if !configured_categories.insert(*category) {
            duplicate_categories.insert(*category);
        }
    }

    let mut violations = Vec::new();

    for duplicate in duplicate_categories {
        violations.push(NoMockViolation {
            invariant: format!("critical-category-config:{}", duplicate.prefix()),
            matrix_test_id: format!("CFG-NO-MOCK-DUP-{}", duplicate.prefix()),
            category: duplicate.display_name().to_owned(),
            severity: NoMockViolationSeverity::Blocking,
            reason: format!(
                "Critical category configured multiple times: {}",
                duplicate.display_name()
            ),
            remediation: format!(
                "Remove duplicate FeatureCategory::{duplicate:?} from the no-mock critical \
                 category configuration."
            ),
        });
    }

    for expected in taxonomy_critical_categories() {
        if !configured_categories.contains(&expected) {
            violations.push(NoMockViolation {
                invariant: format!("critical-category-config:{}", expected.prefix()),
                matrix_test_id: format!("CFG-NO-MOCK-MISSING-{}", expected.prefix()),
                category: expected.display_name().to_owned(),
                severity: NoMockViolationSeverity::Blocking,
                reason: format!(
                    "Taxonomy marks this category as critical (global_weight={:.2}), but the \
                     no-mock gate configuration omits it",
                    expected.global_weight()
                ),
                remediation: format!(
                    "Add FeatureCategory::{expected:?} to the no-mock critical category \
                     configuration so newly critical paths cannot bypass enforcement."
                ),
            });
        }
    }

    (configured_categories.into_iter().collect(), violations)
}

/// Evaluate the gate with pre-built data (for testing).
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn evaluate_with_data(
    matrix: &UnitMatrix,
    evidence_map: &NoMockEvidenceMap,
    critical_categories: &[FeatureCategory],
) -> NoMockCriticalPathReport {
    let (critical_categories, mut violations) = normalize_critical_categories(critical_categories);

    // Build evidence lookup: matrix_test_id -> Vec<evidence entries>
    let mut evidence_by_test: BTreeMap<String, Vec<&crate::no_mock_evidence::NoMockEvidenceEntry>> =
        BTreeMap::new();
    for entry in &evidence_map.entries {
        evidence_by_test
            .entry(entry.matrix_test_id.clone())
            .or_default()
            .push(entry);
    }

    let mut category_results = Vec::new();
    let mut total_critical = 0_usize;
    let mut total_real = 0_usize;
    let mut total_exception = 0_usize;
    let mut total_missing = 0_usize;

    for cat in critical_categories {
        let mut cat_total = 0_usize;
        let mut cat_real = 0_usize;
        let mut cat_exception = 0_usize;
        let mut cat_missing = 0_usize;

        // Find all tests in this category
        let category_tests: Vec<_> = matrix.tests.iter().filter(|t| t.category == cat).collect();

        for test in &category_tests {
            // Each test may have multiple invariants; each needs evidence
            for invariant in &test.invariants {
                cat_total += 1;

                // Look for evidence matching this test ID and invariant
                let evidence_entries = evidence_by_test.get(&test.test_id);

                let matching: Vec<_> = evidence_entries
                    .into_iter()
                    .flatten()
                    .filter(|e| e.invariant == *invariant)
                    .collect();

                if matching.is_empty() {
                    // No evidence at all — blocking failure
                    cat_missing += 1;
                    violations.push(NoMockViolation {
                        invariant: invariant.clone(),
                        matrix_test_id: test.test_id.clone(),
                        category: cat.display_name().to_owned(),
                        severity: NoMockViolationSeverity::Blocking,
                        reason: format!(
                            "Critical-path invariant has no non-mock evidence: \"{invariant}\""
                        ),
                        remediation: format!(
                            "Add a NoMockEvidenceEntry for test {} invariant \"{}\" \
                             in no_mock_evidence.rs with real_components listing \
                             the actual types exercised",
                            test.test_id, invariant
                        ),
                    });
                } else {
                    // Check if all evidence is exception-only
                    let has_real = matching.iter().any(|e| !e.is_exception);
                    let has_exception = matching.iter().any(|e| e.is_exception);

                    if has_real {
                        cat_real += 1;
                    } else if has_exception {
                        // Exception-only: warn but don't block
                        cat_exception += 1;
                        violations.push(NoMockViolation {
                            invariant: invariant.clone(),
                            matrix_test_id: test.test_id.clone(),
                            category: cat.display_name().to_owned(),
                            severity: NoMockViolationSeverity::Warning,
                            reason: format!(
                                "Critical-path invariant covered only by exception: \
                                 \"{invariant}\""
                            ),
                            remediation: format!(
                                "Consider adding a real (non-exception) test for {} \
                                 invariant \"{}\". Current exception rationale: {}",
                                test.test_id,
                                invariant,
                                matching
                                    .iter()
                                    .find_map(|e| e.exception_rationale.as_deref())
                                    .unwrap_or("(none)")
                            ),
                        });
                    } else {
                        // Shouldn't happen, but treat as real evidence
                        cat_real += 1;
                    }
                }
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let coverage_pct = if cat_total > 0 {
            (cat_real + cat_exception) as f64 / cat_total as f64
        } else {
            1.0
        };

        let passes = cat_missing == 0;

        category_results.push(CriticalCategoryResult {
            category: cat,
            display_name: cat.display_name().to_owned(),
            total_invariants: cat_total,
            real_evidence_count: cat_real,
            exception_count: cat_exception,
            missing_count: cat_missing,
            coverage_pct,
            passes,
        });

        total_critical += cat_total;
        total_real += cat_real;
        total_exception += cat_exception;
        total_missing += cat_missing;
    }

    let blocking_count = violations
        .iter()
        .filter(|v| v.severity == NoMockViolationSeverity::Blocking)
        .count();
    let warning_count = violations
        .iter()
        .filter(|v| v.severity == NoMockViolationSeverity::Warning)
        .count();

    let verdict = if blocking_count > 0 {
        NoMockVerdict::Fail
    } else if warning_count > 0 {
        NoMockVerdict::PassWithWarnings
    } else {
        NoMockVerdict::Pass
    };

    #[allow(clippy::cast_precision_loss)]
    let coverage_pct = if total_critical > 0 {
        (total_real + total_exception) as f64 / total_critical as f64
    } else {
        1.0
    };

    let summary = format!(
        "No-mock critical-path gate: {} — {total_critical} critical invariants, \
         {total_real} with real evidence, {total_exception} exceptions, \
         {total_missing} missing ({blocking_count} blocking, {warning_count} warnings)",
        verdict
    );

    NoMockCriticalPathReport {
        schema_version: NO_MOCK_GATE_SCHEMA_VERSION.to_owned(),
        bead_id: BEAD_ID.to_owned(),
        verdict,
        total_critical_invariants: total_critical,
        real_evidence_count: total_real,
        exception_count: total_exception,
        missing_evidence_count: total_missing,
        coverage_pct,
        categories: category_results,
        violations,
        blocking_count,
        warning_count,
        summary,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::no_mock_evidence::{EvidenceStats, NoMockEvidenceEntry, NoMockEvidenceMap};
    use crate::unit_matrix::{FailureDiagnostics, UnitTestEntry};

    fn make_test_matrix(entries: Vec<UnitTestEntry>) -> UnitMatrix {
        UnitMatrix {
            schema_version: "1.0.0".to_owned(),
            bead_id: "test".to_owned(),
            root_seed: 42,
            tests: entries,
            coverage: Vec::new(),
        }
    }

    fn make_evidence_map(entries: Vec<NoMockEvidenceEntry>) -> NoMockEvidenceMap {
        let total = entries.len();
        let exceptions = entries.iter().filter(|e| e.is_exception).count();
        NoMockEvidenceMap {
            schema_version: "1.0.0".to_owned(),
            bead_id: "test".to_owned(),
            entries,
            stats: EvidenceStats {
                total_invariants: total,
                evidenced_count: total,
                exception_count: exceptions,
                coverage_pct: 1.0,
                distinct_components: 1,
                distinct_tests: 1,
            },
        }
    }

    fn make_test_entry(
        test_id: &str,
        category: FeatureCategory,
        invariants: Vec<&str>,
    ) -> UnitTestEntry {
        UnitTestEntry {
            test_id: test_id.to_owned(),
            category,
            crate_name: "fsqlite-harness".to_owned(),
            module_path: "tests".to_owned(),
            description: "test".to_owned(),
            invariants: invariants.into_iter().map(String::from).collect(),
            seed: 42,
            property_based: false,
            failure_diagnostics: FailureDiagnostics {
                dump_targets: Vec::new(),
                log_spans: Vec::new(),
                related_beads: Vec::new(),
            },
        }
    }

    fn make_evidence(test_id: &str, invariant: &str, is_exception: bool) -> NoMockEvidenceEntry {
        NoMockEvidenceEntry {
            matrix_test_id: test_id.to_owned(),
            invariant: invariant.to_owned(),
            test_function: "test_fn".to_owned(),
            test_crate: "fsqlite-harness".to_owned(),
            test_module: "tests".to_owned(),
            real_components: vec!["SimplePager".to_owned()],
            rationale: "tests real pager".to_owned(),
            is_exception,
            exception_rationale: if is_exception {
                Some("hardware-level crash recovery".to_owned())
            } else {
                None
            },
        }
    }

    #[test]
    fn pass_when_all_critical_invariants_have_real_evidence() {
        let matrix = make_test_matrix(vec![
            make_test_entry(
                "UT-SQL-001",
                FeatureCategory::SqlGrammar,
                vec!["parse SELECT"],
            ),
            make_test_entry(
                "UT-VDBE-001",
                FeatureCategory::VdbeOpcodes,
                vec!["opcode Goto"],
            ),
        ]);
        let evidence = make_evidence_map(vec![
            make_evidence("UT-SQL-001", "parse SELECT", false),
            make_evidence("UT-VDBE-001", "opcode Goto", false),
        ]);

        let report = evaluate_with_data(&matrix, &evidence, &DEFAULT_CRITICAL_CATEGORIES);
        assert_eq!(report.verdict, NoMockVerdict::Pass);
        assert_eq!(report.blocking_count, 0);
        assert_eq!(report.warning_count, 0);
    }

    #[test]
    fn fail_when_critical_invariant_has_no_evidence() {
        let matrix = make_test_matrix(vec![make_test_entry(
            "UT-STOR-001",
            FeatureCategory::StorageTransaction,
            vec!["WAL checkpoint"],
        )]);
        let evidence = make_evidence_map(vec![]);

        let report = evaluate_with_data(&matrix, &evidence, &DEFAULT_CRITICAL_CATEGORIES);
        assert_eq!(report.verdict, NoMockVerdict::Fail);
        assert_eq!(report.blocking_count, 1);
        assert!(report.violations[0].reason.contains("no non-mock evidence"));
    }

    #[test]
    fn warn_when_critical_invariant_has_only_exception() {
        let matrix = make_test_matrix(vec![make_test_entry(
            "UT-STOR-001",
            FeatureCategory::StorageTransaction,
            vec!["crash recovery"],
        )]);
        let evidence =
            make_evidence_map(vec![make_evidence("UT-STOR-001", "crash recovery", true)]);

        let report = evaluate_with_data(&matrix, &evidence, &DEFAULT_CRITICAL_CATEGORIES);
        assert_eq!(report.verdict, NoMockVerdict::PassWithWarnings);
        assert_eq!(report.blocking_count, 0);
        assert_eq!(report.warning_count, 1);
    }

    #[test]
    fn non_critical_categories_are_ignored() {
        let matrix = make_test_matrix(vec![make_test_entry(
            "UT-EXT-001",
            FeatureCategory::Extensions,
            vec!["JSON extract"],
        )]);
        // No evidence for this non-critical invariant
        let evidence = make_evidence_map(vec![]);

        let report = evaluate_with_data(&matrix, &evidence, &DEFAULT_CRITICAL_CATEGORIES);
        assert_eq!(report.verdict, NoMockVerdict::Pass);
        assert_eq!(report.total_critical_invariants, 0);
    }

    #[test]
    fn multiple_violations_across_categories() {
        let matrix = make_test_matrix(vec![
            make_test_entry(
                "UT-SQL-001",
                FeatureCategory::SqlGrammar,
                vec!["parse SELECT", "parse INSERT"],
            ),
            make_test_entry(
                "UT-VDBE-001",
                FeatureCategory::VdbeOpcodes,
                vec!["opcode Halt"],
            ),
        ]);
        // Only evidence for parse SELECT, missing parse INSERT and opcode Halt
        let evidence = make_evidence_map(vec![make_evidence("UT-SQL-001", "parse SELECT", false)]);

        let report = evaluate_with_data(&matrix, &evidence, &DEFAULT_CRITICAL_CATEGORIES);
        assert_eq!(report.verdict, NoMockVerdict::Fail);
        assert_eq!(report.blocking_count, 2);
        assert_eq!(report.missing_evidence_count, 2);
    }

    #[test]
    fn report_json_roundtrip() {
        let matrix = make_test_matrix(vec![make_test_entry(
            "UT-SQL-001",
            FeatureCategory::SqlGrammar,
            vec!["parse SELECT"],
        )]);
        let evidence = make_evidence_map(vec![make_evidence("UT-SQL-001", "parse SELECT", false)]);

        let report = evaluate_with_data(&matrix, &evidence, &DEFAULT_CRITICAL_CATEGORIES);
        let json = report.to_json().expect("serialize");
        let restored = NoMockCriticalPathReport::from_json(&json).expect("deserialize");
        assert_eq!(restored.verdict, report.verdict);
        assert_eq!(
            restored.total_critical_invariants,
            report.total_critical_invariants
        );
        assert_eq!(restored.blocking_count, report.blocking_count);
    }

    #[test]
    fn report_renders_markdown() {
        let matrix = make_test_matrix(vec![make_test_entry(
            "UT-SQL-001",
            FeatureCategory::SqlGrammar,
            vec!["parse SELECT"],
        )]);
        let evidence = make_evidence_map(vec![]);

        let report = evaluate_with_data(&matrix, &evidence, &DEFAULT_CRITICAL_CATEGORIES);
        let md = report.render_summary();
        assert!(md.contains("# No-Mock Critical-Path Gate Report"));
        assert!(md.contains("FAIL"));
        assert!(md.contains("Violations"));
    }

    #[test]
    fn empty_matrix_passes() {
        let matrix = make_test_matrix(vec![]);
        let evidence = make_evidence_map(vec![]);

        let report = evaluate_with_data(&matrix, &evidence, &DEFAULT_CRITICAL_CATEGORIES);
        assert_eq!(report.verdict, NoMockVerdict::Pass);
        assert_eq!(report.total_critical_invariants, 0);
    }

    #[test]
    fn default_critical_categories_match_taxonomy_threshold() {
        assert_eq!(
            taxonomy_critical_categories(),
            DEFAULT_CRITICAL_CATEGORIES.to_vec()
        );
    }

    #[test]
    fn fail_when_taxonomy_critical_category_is_missing_from_configuration() {
        let matrix = make_test_matrix(vec![make_test_entry(
            "UT-SQL-001",
            FeatureCategory::SqlGrammar,
            vec!["parse SELECT"],
        )]);
        let evidence = make_evidence_map(vec![make_evidence("UT-SQL-001", "parse SELECT", false)]);
        let configured = [FeatureCategory::SqlGrammar, FeatureCategory::VdbeOpcodes];

        let report = evaluate_with_data(&matrix, &evidence, &configured);
        assert_eq!(report.verdict, NoMockVerdict::Fail);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.matrix_test_id == "CFG-NO-MOCK-MISSING-STOR")
        );
    }

    #[test]
    fn fail_when_critical_category_configuration_has_duplicates() {
        let matrix = make_test_matrix(vec![make_test_entry(
            "UT-SQL-001",
            FeatureCategory::SqlGrammar,
            vec!["parse SELECT"],
        )]);
        let evidence = make_evidence_map(vec![make_evidence("UT-SQL-001", "parse SELECT", false)]);
        let configured = [
            FeatureCategory::SqlGrammar,
            FeatureCategory::SqlGrammar,
            FeatureCategory::VdbeOpcodes,
            FeatureCategory::StorageTransaction,
        ];

        let report = evaluate_with_data(&matrix, &evidence, &configured);
        assert_eq!(report.verdict, NoMockVerdict::Fail);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.matrix_test_id == "CFG-NO-MOCK-DUP-SQL")
        );
    }

    #[test]
    fn full_gate_evaluation_runs() {
        // Integration test using the canonical matrix and evidence map
        let report = evaluate_no_mock_critical_path_gate(&DEFAULT_CRITICAL_CATEGORIES);
        // Should produce a valid report regardless of current evidence state
        assert!(!report.schema_version.is_empty());
        assert_eq!(report.bead_id, BEAD_ID);
    }
}
