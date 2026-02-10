//! Guided corruption + recovery walkthrough (bd-1w6k.7.5).
//!
//! Produces a single-command, human-readable narrative that walks through
//! corruption scenarios with step-by-step commentary.  Designed to be
//! compelling for demos and documentation.
//!
//! The walkthrough selects representative scenarios from the catalog and
//! runs both the C SQLite and FrankenSQLite paths side by side, printing
//! a clear narrative to stdout.

use std::fmt::Write;
use std::time::Instant;

use crate::corruption_demo_sqlite::{SqliteCorruptionResult, run_sqlite_corruption_scenario};
use crate::corruption_scenarios::{CorruptionScenario, ExpectedFsqliteBehavior, scenario_catalog};
use crate::fsqlite_recovery_demo::{FsqliteRecoveryReport, run_scenario as run_fsqlite_scenario};

// ── Public types ─────────────────────────────────────────────────────────

/// The result of running the full walkthrough.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WalkthroughReport {
    /// Per-scenario narrative sections.
    pub sections: Vec<WalkthroughSection>,
    /// Total wall-clock time for the entire walkthrough.
    pub total_elapsed_ms: u64,
    /// Whether every scenario's outcome matched expectations.
    pub all_passed: bool,
}

/// One section of the walkthrough narrative.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct WalkthroughSection {
    /// Scenario name.
    pub name: String,
    /// The narrative description of what this scenario demonstrates.
    pub description: String,
    /// Step-by-step log lines (human-readable).
    pub steps: Vec<String>,
    /// Whether C SQLite opened the database after corruption.
    pub sqlite_opened: bool,
    /// Whether C SQLite passed integrity check.
    pub sqlite_integrity_ok: bool,
    /// Rows C SQLite recovered (None if open failed).
    pub sqlite_rows_recovered: Option<usize>,
    /// Whether FrankenSQLite successfully recovered.
    pub fsqlite_recovered: bool,
    /// Pages FrankenSQLite recovered.
    pub fsqlite_pages_recovered: usize,
    /// FrankenSQLite verdict string.
    pub fsqlite_verdict: String,
    /// Whether the outcome matched expectations.
    pub passed: bool,
    /// Wall-clock time for this scenario.
    pub elapsed_ms: u64,
}

// ── Main entry point ─────────────────────────────────────────────────────

/// Run the full guided walkthrough, returning a structured report.
///
/// Selects 4 representative scenarios from the catalog:
/// 1. WAL corruption within FEC tolerance (recovery succeeds)
/// 2. Single-bit WAL corruption (subtle bitrot, recovery succeeds)
/// 3. WAL corruption beyond FEC capacity (both engines lose data)
/// 4. Database header zeroed (catastrophic, no recovery)
#[must_use]
pub fn run_walkthrough() -> WalkthroughReport {
    let catalog = scenario_catalog();
    let selected_names = [
        "wal_corrupt_within_tolerance",
        "wal_single_bit_flip",
        "wal_corrupt_beyond_tolerance",
        "db_header_zeroed",
    ];

    let selected: Vec<&CorruptionScenario> = selected_names
        .iter()
        .filter_map(|name| catalog.iter().find(|s| s.name == *name))
        .collect();

    let overall_start = Instant::now();
    let mut sections = Vec::with_capacity(selected.len());
    let mut all_passed = true;

    for (i, scenario) in selected.iter().enumerate() {
        let section = run_walkthrough_scenario(scenario, i + 1);
        if !section.passed {
            all_passed = false;
        }
        sections.push(section);
    }

    WalkthroughReport {
        sections,
        #[allow(clippy::cast_possible_truncation)]
        total_elapsed_ms: overall_start.elapsed().as_millis() as u64,
        all_passed,
    }
}

/// Format the walkthrough report as a human-readable narrative.
#[must_use]
pub fn format_walkthrough(report: &WalkthroughReport) -> String {
    let mut out = String::with_capacity(4096);

    let _ = writeln!(
        out,
        "=== FrankenSQLite Corruption + Recovery Walkthrough ==="
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "This walkthrough demonstrates how FrankenSQLite's WAL-FEC"
    );
    let _ = writeln!(
        out,
        "recovery compares to C SQLite when databases are corrupted."
    );
    let _ = writeln!(
        out,
        "Each scenario creates a fresh database, injects corruption,"
    );
    let _ = writeln!(out, "and shows the outcome for both engines.");
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", "-".repeat(60));

    for (i, section) in report.sections.iter().enumerate() {
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "--- Scenario {}: {} ---",
            i + 1,
            section.name.replace('_', " ")
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "  {}", section.description);
        let _ = writeln!(out);

        for step in &section.steps {
            let _ = writeln!(out, "  {step}");
        }

        let _ = writeln!(out);
        let status = if section.passed { "PASS" } else { "FAIL" };
        let _ = writeln!(out, "  Result: [{status}] ({}ms)", section.elapsed_ms);
        let _ = writeln!(out, "{}", "-".repeat(60));
    }

    let _ = writeln!(out);
    let overall = if report.all_passed {
        "ALL SCENARIOS PASSED"
    } else {
        "SOME SCENARIOS FAILED"
    };
    let _ = writeln!(
        out,
        "=== {overall} ({}/{}  in {}ms) ===",
        report.sections.iter().filter(|s| s.passed).count(),
        report.sections.len(),
        report.total_elapsed_ms,
    );

    out
}

// ── Per-scenario runner ──────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn run_walkthrough_scenario(scenario: &CorruptionScenario, number: usize) -> WalkthroughSection {
    let start = Instant::now();
    let mut steps = Vec::new();

    steps.push(format!(
        "[Step 1] Setting up: {}-row WAL-mode database (seed={})",
        scenario.setup_row_count, scenario.seed
    ));

    if scenario.setup_wal_fec {
        steps.push(format!(
            "[Step 2] Building WAL-FEC sidecar with R={} repair symbols",
            scenario.setup_repair_symbols
        ));
    } else {
        steps.push("[Step 2] No WAL-FEC sidecar (recovery not available)".to_owned());
    }

    steps.push(format!(
        "[Step 3] Injecting corruption: {:?} on {:?}",
        scenario.pattern, scenario.target
    ));

    // ── Run C SQLite side ────────────────────────────────────────────
    steps.push(format!("[Step 4] Running C SQLite (scenario #{number})..."));

    let sqlite_result = run_sqlite_side(scenario);

    match &sqlite_result {
        Some(r) if r.open_succeeded => {
            let integrity_str = if r.integrity_ok { "PASS" } else { "FAIL" };
            let rows_str = r
                .rows_recovered
                .map_or_else(|| "N/A".to_owned(), |n| n.to_string());
            steps.push(format!(
                "         C SQLite: opened=yes, integrity={integrity_str}, rows_recovered={rows_str}"
            ));
            if !r.integrity_ok {
                steps
                    .push("         C SQLite detected corruption but cannot self-heal.".to_owned());
            }
        }
        Some(r) => {
            steps.push(format!(
                "         C SQLite: opened=NO (error: {})",
                r.error.as_deref().unwrap_or("unknown")
            ));
            steps.push("         C SQLite cannot read the database at all.".to_owned());
        }
        None => {
            steps.push("         C SQLite: setup or injection failed.".to_owned());
        }
    }

    // ── Run FrankenSQLite side ───────────────────────────────────────
    steps.push(format!(
        "[Step 5] Running FrankenSQLite recovery (scenario #{number})..."
    ));

    let fsqlite_report = run_fsqlite_scenario(scenario);

    if fsqlite_report.recovery_succeeded {
        steps.push(format!(
            "         FrankenSQLite: RECOVERED {} pages!",
            fsqlite_report.pages_recovered
        ));
        steps.push(format!("         Verdict: {}", fsqlite_report.verdict));
    } else {
        steps.push(format!(
            "         FrankenSQLite: recovery_succeeded=false, verdict={}",
            fsqlite_report.verdict
        ));
    }

    // ── Narrative summary ────────────────────────────────────────────
    steps.push(String::new());
    steps.push(format_narrative_summary(
        scenario,
        sqlite_result.as_ref(),
        &fsqlite_report,
    ));

    #[allow(clippy::cast_possible_truncation)]
    let elapsed_ms = start.elapsed().as_millis() as u64;

    WalkthroughSection {
        name: scenario.name.to_owned(),
        description: scenario.description.to_owned(),
        steps,
        sqlite_opened: sqlite_result.as_ref().is_some_and(|r| r.open_succeeded),
        sqlite_integrity_ok: sqlite_result.as_ref().is_some_and(|r| r.integrity_ok),
        sqlite_rows_recovered: sqlite_result.as_ref().and_then(|r| r.rows_recovered),
        fsqlite_recovered: fsqlite_report.recovery_succeeded,
        fsqlite_pages_recovered: fsqlite_report.pages_recovered,
        fsqlite_verdict: fsqlite_report.verdict.clone(),
        passed: fsqlite_report.passed,
        elapsed_ms,
    }
}

fn run_sqlite_side(scenario: &CorruptionScenario) -> Option<SqliteCorruptionResult> {
    let dir = tempfile::tempdir().ok()?;
    run_sqlite_corruption_scenario(scenario, dir.path()).ok()
}

fn format_narrative_summary(
    scenario: &CorruptionScenario,
    sqlite_result: Option<&SqliteCorruptionResult>,
    fsqlite_report: &FsqliteRecoveryReport,
) -> String {
    let mut summary = String::new();
    let _ = write!(summary, "[Summary] ");

    match scenario.expected_fsqlite {
        ExpectedFsqliteBehavior::FullRecovery => {
            let sqlite_lost = sqlite_result
                .and_then(|r| {
                    r.rows_recovered
                        .map(|recovered| r.rows_inserted.saturating_sub(recovered))
                })
                .unwrap_or(0);

            if fsqlite_report.recovery_succeeded {
                let _ = write!(
                    summary,
                    "C SQLite lost ~{sqlite_lost} rows. \
                     FrankenSQLite recovered all {} pages via WAL-FEC.",
                    fsqlite_report.pages_recovered
                );
            } else {
                let _ = write!(
                    summary,
                    "Unexpected: FrankenSQLite failed to recover despite sufficient FEC symbols."
                );
            }
        }
        ExpectedFsqliteBehavior::RepairExceedsCapacity => {
            let _ = write!(
                summary,
                "Corruption exceeds FEC repair capacity. \
                 Both engines lose data (graceful degradation)."
            );
        }
        ExpectedFsqliteBehavior::RecoveryDisabled => {
            let _ = write!(
                summary,
                "Recovery toggle disabled. Both engines behave identically (truncation)."
            );
        }
        ExpectedFsqliteBehavior::SidecarDamaged => {
            let _ = write!(
                summary,
                "WAL-FEC sidecar itself is damaged. Falls back to truncation."
            );
        }
    }

    summary
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walkthrough_runs_without_panic() {
        let report = run_walkthrough();
        assert_eq!(report.sections.len(), 4, "should have 4 scenarios");
    }

    #[test]
    fn walkthrough_sections_have_steps() {
        let report = run_walkthrough();
        for section in &report.sections {
            assert!(
                !section.steps.is_empty(),
                "section '{}' should have steps",
                section.name
            );
            assert!(
                !section.description.is_empty(),
                "section '{}' should have description",
                section.name
            );
        }
    }

    #[test]
    fn walkthrough_recoverable_scenarios_recover() {
        let report = run_walkthrough();
        // First two scenarios should show FrankenSQLite recovery.
        let within = report
            .sections
            .iter()
            .find(|s| s.name == "wal_corrupt_within_tolerance");
        if let Some(s) = within {
            assert!(s.fsqlite_recovered, "within-tolerance should recover");
            assert!(s.fsqlite_pages_recovered > 0);
        }

        let bitflip = report
            .sections
            .iter()
            .find(|s| s.name == "wal_single_bit_flip");
        if let Some(s) = bitflip {
            assert!(s.fsqlite_recovered, "bitflip should recover");
        }
    }

    #[test]
    fn walkthrough_beyond_tolerance_does_not_recover() {
        let report = run_walkthrough();
        let beyond = report
            .sections
            .iter()
            .find(|s| s.name == "wal_corrupt_beyond_tolerance");
        if let Some(s) = beyond {
            assert!(!s.fsqlite_recovered, "beyond-tolerance should not recover");
        }
    }

    #[test]
    fn walkthrough_format_produces_output() {
        let report = run_walkthrough();
        let text = format_walkthrough(&report);
        assert!(text.contains("Walkthrough"), "should have title");
        assert!(text.contains("Scenario 1"), "should have scenario 1");
        assert!(text.contains("Scenario 4"), "should have scenario 4");
        assert!(text.contains("[Step 1]"), "should have step markers");
    }

    #[test]
    fn walkthrough_serialization_roundtrip() {
        let report = run_walkthrough();
        let json = serde_json::to_string(&report).expect("serialize");
        let deser: WalkthroughReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.sections.len(), report.sections.len());
        assert_eq!(deser.all_passed, report.all_passed);
    }

    #[test]
    fn walkthrough_section_names_match_catalog() {
        let report = run_walkthrough();
        let catalog = scenario_catalog();
        for section in &report.sections {
            assert!(
                catalog.iter().any(|s| s.name == section.name),
                "section '{}' should match a catalog scenario",
                section.name
            );
        }
    }
}
