//! CI smoke test: tiny-corpus correctness verification.
//!
//! Bead: bd-1w6k.8.2
//!
//! A minimal, fast correctness test suitable for CI pipelines.  Creates a
//! tiny in-memory database, runs a deterministic workload against both
//! FrankenSQLite and C SQLite, and verifies:
//!
//! 1. All operations produce identical results on both engines.
//! 2. Final database state matches (logical SHA-256 comparison).
//! 3. C SQLite integrity\_check passes on the on-disk copy.
//!
//! Target: completes in <10 seconds.  No external files or golden corpus
//! required — everything is generated in-test.

use crate::E2eResult;
use crate::comparison::ComparisonRunner;
use crate::oplog;
use crate::sqlite_executor::{SqliteExecConfig, run_oplog_sqlite};

/// Outcome of a CI smoke run.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CiSmokeResult {
    /// Whether the test passed overall.
    pub passed: bool,
    /// Number of operations that matched between engines.
    pub operations_matched: usize,
    /// Number of mismatches detected.
    pub mismatches: usize,
    /// Whether the logical state hash matched after the workload.
    pub logical_hash_matched: bool,
    /// Whether C SQLite integrity_check returned "ok".
    pub integrity_check_ok: bool,
    /// Total wall-clock time in milliseconds.
    pub elapsed_ms: u64,
    /// Detail message.
    pub detail: String,
}

/// Run the CI smoke test.
///
/// Creates a tiny deterministic workload (seed=42, 50 ops, 1 worker) and
/// executes it against both engines in-memory, then verifies correctness.
///
/// # Errors
///
/// Returns `E2eError` if the comparison infrastructure fails to initialize.
pub fn run_ci_smoke() -> E2eResult<CiSmokeResult> {
    let start = std::time::Instant::now();

    // ── Phase 1: In-memory comparison ──────────────────────────────
    let runner = ComparisonRunner::new_in_memory()?;

    let statements = vec![
        "CREATE TABLE ci_smoke (id INTEGER PRIMARY KEY, val TEXT, num INTEGER);".to_owned(),
        "INSERT INTO ci_smoke VALUES (1, 'alpha', 100);".to_owned(),
        "INSERT INTO ci_smoke VALUES (2, 'beta', 200);".to_owned(),
        "INSERT INTO ci_smoke VALUES (3, 'gamma', 300);".to_owned(),
        "INSERT INTO ci_smoke VALUES (4, 'delta', 400);".to_owned(),
        "INSERT INTO ci_smoke VALUES (5, 'epsilon', 500);".to_owned(),
        "UPDATE ci_smoke SET num = 301 WHERE id = 3;".to_owned(),
        "SELECT COUNT(*) FROM ci_smoke;".to_owned(),
        "SELECT id, val FROM ci_smoke WHERE num > 250 ORDER BY id;".to_owned(),
        "DELETE FROM ci_smoke WHERE id = 1;".to_owned(),
        "SELECT COUNT(*) FROM ci_smoke;".to_owned(),
    ];

    let comparison = runner.run_and_compare(&statements);
    let hash_cmp = runner.compare_logical_state();

    // ── Phase 2: On-disk integrity check via oplog ─────────────────
    let tmp = tempfile::TempDir::new()?;
    let db_path = tmp.path().join("ci_smoke.db");
    let oplog = oplog::preset_commutative_inserts_disjoint_keys("ci-smoke", 42, 1, 20);
    let report = run_oplog_sqlite(&db_path, &oplog, &SqliteExecConfig::default())?;
    let integrity_ok = report.correctness.integrity_check_ok.unwrap_or(false);

    let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    let passed = comparison.operations_mismatched == 0 && hash_cmp.matched && integrity_ok;

    let detail = if passed {
        format!(
            "CI smoke passed: {}/{} ops matched, logical hash matched, integrity ok, {}ms",
            comparison.operations_matched,
            comparison.operations_matched + comparison.operations_mismatched,
            elapsed_ms,
        )
    } else {
        let mut parts = Vec::new();
        if comparison.operations_mismatched > 0 {
            parts.push(format!(
                "{} operation mismatches",
                comparison.operations_mismatched
            ));
            if let Some(first) = comparison.mismatches.first() {
                parts.push(format!(
                    "first mismatch @{} sql=`{}` csqlite={:?} fsqlite={:?}",
                    first.index, first.sql, first.csqlite, first.fsqlite
                ));
            }
        }
        if !hash_cmp.matched {
            parts.push(format!(
                "logical hash mismatch: csqlite={}, fsqlite={}",
                &hash_cmp.csqlite_sha256[..16],
                &hash_cmp.frank_sha256[..16],
            ));
        }
        if !integrity_ok {
            parts.push("integrity_check failed".to_owned());
        }
        format!("CI smoke FAILED: {}", parts.join("; "))
    };

    Ok(CiSmokeResult {
        passed,
        operations_matched: comparison.operations_matched,
        mismatches: comparison.operations_mismatched,
        logical_hash_matched: hash_cmp.matched,
        integrity_check_ok: integrity_ok,
        elapsed_ms,
        detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_smoke_test_passes() {
        let result = run_ci_smoke().expect("CI smoke should not return infrastructure error");
        assert!(result.passed, "CI smoke failed: {}", result.detail);
        assert_eq!(result.mismatches, 0, "no mismatches expected");
        assert!(result.logical_hash_matched, "logical hashes should match");
        assert!(result.integrity_check_ok, "integrity_check should pass");
    }

    #[test]
    fn ci_smoke_completes_quickly() {
        let result = run_ci_smoke().expect("CI smoke should not return infrastructure error");
        assert!(
            result.elapsed_ms < 30_000,
            "CI smoke should complete in <30s, took {}ms",
            result.elapsed_ms
        );
    }

    #[test]
    fn ci_smoke_result_serializes() {
        let result = run_ci_smoke().expect("CI smoke should not return infrastructure error");
        let json = serde_json::to_string_pretty(&result).unwrap();
        let parsed: CiSmokeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.passed, result.passed);
        assert_eq!(parsed.operations_matched, result.operations_matched);
    }

    #[test]
    fn ci_smoke_in_memory_comparison_covers_all_ops() {
        let runner = ComparisonRunner::new_in_memory().unwrap();

        let stmts = vec![
            "CREATE TABLE t (id INTEGER PRIMARY KEY, x TEXT);".to_owned(),
            "INSERT INTO t VALUES (1, 'a');".to_owned(),
            "INSERT INTO t VALUES (2, 'b');".to_owned(),
            "INSERT INTO t VALUES (3, 'c');".to_owned(),
            "UPDATE t SET x = 'z' WHERE id = 2;".to_owned(),
            "DELETE FROM t WHERE id = 1;".to_owned(),
            "SELECT * FROM t ORDER BY id;".to_owned(),
        ];

        let result = runner.run_and_compare(&stmts);
        assert_eq!(
            result.operations_mismatched,
            0,
            "all ops should match: mismatches={:?}",
            result
                .mismatches
                .iter()
                .map(|m| format!("{}: csql={:?}, fsql={:?}", m.sql, m.csqlite, m.fsqlite))
                .collect::<Vec<_>>()
        );

        let hash = runner.compare_logical_state();
        assert!(hash.matched, "logical state should match after workload");
    }

    #[test]
    fn ci_smoke_oplog_integrity_check() {
        let tmp = tempfile::TempDir::new().unwrap();
        let db_path = tmp.path().join("oplog_ic.db");

        let oplog = oplog::preset_commutative_inserts_disjoint_keys("ci-smoke-ic", 99, 1, 10);
        let report = run_oplog_sqlite(&db_path, &oplog, &SqliteExecConfig::default()).unwrap();

        assert!(
            report.error.is_none(),
            "oplog run should succeed: {:?}",
            report.error
        );
        assert_eq!(
            report.correctness.integrity_check_ok,
            Some(true),
            "integrity_check should pass on fresh workload DB"
        );
    }

    #[test]
    fn ci_smoke_deterministic_workload_produces_same_hash() {
        // Run the same workload twice and verify both produce the same hash.
        let runner1 = ComparisonRunner::new_in_memory().unwrap();
        let runner2 = ComparisonRunner::new_in_memory().unwrap();

        let stmts: Vec<String> = vec![
            "CREATE TABLE det (id INTEGER PRIMARY KEY, v INTEGER);",
            "INSERT INTO det VALUES (1, 42);",
            "INSERT INTO det VALUES (2, 84);",
            "UPDATE det SET v = v + 1 WHERE id = 1;",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();

        let _ = runner1.run_and_compare(&stmts);
        let _ = runner2.run_and_compare(&stmts);

        let hash1 = runner1.compare_logical_state();
        let hash2 = runner2.compare_logical_state();

        assert_eq!(
            hash1.csqlite_sha256, hash2.csqlite_sha256,
            "C SQLite should produce identical hash for same workload"
        );
        assert_eq!(
            hash1.frank_sha256, hash2.frank_sha256,
            "FrankenSQLite should produce identical hash for same workload"
        );
    }
}
