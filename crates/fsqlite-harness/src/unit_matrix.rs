//! Deterministic unit matrix expansion mapped to parity taxonomy (bd-1dp9.7.1).
//!
//! Maps each [`FeatureCategory`] bucket from the parity taxonomy to a set of
//! unit test entries with deterministic seed strategy, invariant assertions,
//! and failure diagnostics contracts.
//!
//! # Architecture
//!
//! The unit matrix connects:
//! 1. **Taxonomy buckets** — the 9 feature categories from bd-1dp9.1.1
//! 2. **Test entries** — concrete unit test specifications with seed, crate, and invariant info
//! 3. **Coverage report** — per-bucket fill percentages and missing areas
//!
//! Seeds are derived using the [`SeedTaxonomy`] mechanism (xxh3-based derivation)
//! for cross-platform determinism.

use serde::{Deserialize, Serialize};

use crate::parity_taxonomy::FeatureCategory;
use crate::seed_taxonomy::SeedTaxonomy;

#[allow(dead_code)]
const BEAD_ID: &str = "bd-1dp9.7.1";

/// Root seed for unit matrix test derivation.
/// ASCII for "UNITMATRIX" truncated to u64.
const UNIT_MATRIX_ROOT_SEED: u64 = 0x554E_4954_4D41_5458;

// ─── Core Types ─────────────────────────────────────────────────────────

/// A single unit test entry in the coverage matrix.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct UnitTestEntry {
    /// Stable test identifier (e.g. `UT-SQL-001`).
    pub test_id: String,
    /// Feature category this test covers.
    pub category: FeatureCategory,
    /// Crate containing the test.
    pub crate_name: String,
    /// Module path within the crate (e.g. `parser::select`).
    pub module_path: String,
    /// Human-readable description of what is tested.
    pub description: String,
    /// Invariant assertions this test validates.
    pub invariants: Vec<String>,
    /// Deterministic seed for this test (derived from category + test_id).
    pub seed: u64,
    /// Whether this test uses property-based testing (proptest).
    pub property_based: bool,
    /// Failure diagnostics: what to inspect on failure.
    pub failure_diagnostics: FailureDiagnostics,
}

/// Failure diagnostics contract: what information to collect on test failure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FailureDiagnostics {
    /// Key data structures to dump on failure.
    pub dump_targets: Vec<String>,
    /// Related log spans to check.
    pub log_spans: Vec<String>,
    /// Related beads for context.
    pub related_beads: Vec<String>,
}

/// Coverage status for a single taxonomy bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketCoverage {
    /// Feature category.
    pub category: FeatureCategory,
    /// Number of unit tests covering this bucket.
    pub test_count: usize,
    /// Number of invariants asserted across all tests.
    pub invariant_count: usize,
    /// Number of property-based tests.
    pub property_test_count: usize,
    /// Crates contributing tests to this bucket.
    pub contributing_crates: Vec<String>,
    /// Areas within this bucket lacking coverage.
    pub missing_coverage: Vec<String>,
    /// Fill percentage (0.0 to 1.0).
    pub fill_pct: f64,
}

/// The complete unit test coverage matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitMatrix {
    /// Schema version.
    pub schema_version: String,
    /// Bead ID.
    pub bead_id: String,
    /// Root seed used for derivation.
    pub root_seed: u64,
    /// All unit test entries.
    pub tests: Vec<UnitTestEntry>,
    /// Per-bucket coverage report.
    pub coverage: Vec<BucketCoverage>,
}

impl UnitMatrix {
    /// Validate structural invariants.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // 1. No duplicate test IDs
        let mut seen = std::collections::BTreeSet::new();
        for t in &self.tests {
            if !seen.insert(&t.test_id) {
                errors.push(format!("Duplicate test ID: {}", t.test_id));
            }
        }

        // 2. Every test must have at least one invariant
        for t in &self.tests {
            if t.invariants.is_empty() {
                errors.push(format!("Test {} has no invariants", t.test_id));
            }
        }

        // 3. Every category must have at least one test
        for cat in FeatureCategory::ALL {
            let count = self.tests.iter().filter(|t| t.category == cat).count();
            if count == 0 {
                errors.push(format!("Category {cat:?} has no unit tests"));
            }
        }

        // 4. Coverage report must have entry for each category
        for cat in FeatureCategory::ALL {
            if !self.coverage.iter().any(|c| c.category == cat) {
                errors.push(format!("No coverage entry for {cat:?}"));
            }
        }

        // 5. Seeds must be non-zero
        for t in &self.tests {
            if t.seed == 0 {
                errors.push(format!("Test {} has zero seed", t.test_id));
            }
        }

        errors
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Compute overall fill percentage across all buckets.
    #[must_use]
    pub fn overall_fill_pct(&self) -> f64 {
        if self.coverage.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.coverage.iter().map(|c| c.fill_pct).sum();
        truncate_f64(sum / self.coverage.len() as f64, 4)
    }
}

/// Truncate f64 to N decimal places for cross-platform determinism.
fn truncate_f64(value: f64, decimals: u32) -> f64 {
    let exp = i32::try_from(decimals).unwrap_or(6);
    let factor = 10_f64.powi(exp);
    (value * factor).trunc() / factor
}

/// Derive a deterministic seed for a unit test.
fn derive_test_seed(category: FeatureCategory, test_id: &str) -> u64 {
    let scope = format!("{category:?}::{test_id}");
    let taxonomy = SeedTaxonomy::derive(UNIT_MATRIX_ROOT_SEED, &scope);
    taxonomy.schedule
}

// ─── Test Entry Builders ────────────────────────────────────────────────

struct TestEntryBuilder {
    entry: UnitTestEntry,
}

impl TestEntryBuilder {
    fn new(test_id: &str, category: FeatureCategory, crate_name: &str, description: &str) -> Self {
        let seed = derive_test_seed(category, test_id);
        Self {
            entry: UnitTestEntry {
                test_id: test_id.to_owned(),
                category,
                crate_name: crate_name.to_owned(),
                module_path: String::new(),
                description: description.to_owned(),
                invariants: Vec::new(),
                seed,
                property_based: false,
                failure_diagnostics: FailureDiagnostics {
                    dump_targets: Vec::new(),
                    log_spans: Vec::new(),
                    related_beads: Vec::new(),
                },
            },
        }
    }

    #[must_use]
    fn module(mut self, path: &str) -> Self {
        path.clone_into(&mut self.entry.module_path);
        self
    }

    #[must_use]
    fn invariants(mut self, invs: &[&str]) -> Self {
        self.entry.invariants = invs.iter().map(|s| (*s).to_owned()).collect();
        self
    }

    #[must_use]
    fn property_based(mut self) -> Self {
        self.entry.property_based = true;
        self
    }

    #[must_use]
    fn diagnostics(mut self, dumps: &[&str], spans: &[&str], beads: &[&str]) -> Self {
        self.entry.failure_diagnostics.dump_targets =
            dumps.iter().map(|s| (*s).to_owned()).collect();
        self.entry.failure_diagnostics.log_spans = spans.iter().map(|s| (*s).to_owned()).collect();
        self.entry.failure_diagnostics.related_beads =
            beads.iter().map(|s| (*s).to_owned()).collect();
        self
    }

    fn build(self) -> UnitTestEntry {
        self.entry
    }
}

// ─── Canonical Matrix Builder ───────────────────────────────────────────

/// Build the canonical unit test coverage matrix mapped to parity taxonomy.
#[allow(clippy::too_many_lines, clippy::vec_init_then_push)]
pub fn build_canonical_matrix() -> UnitMatrix {
    let mut tests = Vec::new();

    build_sql_grammar_tests(&mut tests);
    build_vdbe_tests(&mut tests);
    build_storage_txn_tests(&mut tests);
    build_pragma_tests(&mut tests);
    build_builtin_function_tests(&mut tests);
    build_extension_tests(&mut tests);
    build_type_system_tests(&mut tests);
    build_file_format_tests(&mut tests);
    build_api_cli_tests(&mut tests);

    let coverage = compute_coverage(&tests);

    UnitMatrix {
        schema_version: "1.0.0".to_owned(),
        bead_id: BEAD_ID.to_owned(),
        root_seed: UNIT_MATRIX_ROOT_SEED,
        tests,
        coverage,
    }
}

#[allow(clippy::too_many_lines)]
fn build_sql_grammar_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::SqlGrammar;
    let crate_name = "fsqlite-parser";

    tests.push(
        TestEntryBuilder::new(
            "UT-SQL-001",
            cat,
            crate_name,
            "SELECT with all clause types",
        )
        .module("parser::select")
        .invariants(&[
            "SELECT with WHERE, GROUP BY, HAVING, ORDER BY, LIMIT parses",
            "AST round-trips to equivalent SQL",
        ])
        .diagnostics(&["ast_tree"], &["parse_select"], &["bd-2d6i"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-SQL-002",
            cat,
            crate_name,
            "INSERT variants (VALUES, SELECT, DEFAULT)",
        )
        .module("parser::insert")
        .invariants(&[
            "INSERT INTO ... VALUES parses correctly",
            "INSERT INTO ... SELECT sub-query parses",
            "INSERT OR REPLACE recognized",
        ])
        .diagnostics(&["ast_tree"], &["parse_insert"], &["bd-340i"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-SQL-003",
            cat,
            crate_name,
            "UPDATE with complex WHERE and SET",
        )
        .module("parser::update")
        .invariants(&[
            "UPDATE SET with multiple columns",
            "UPDATE with subquery in WHERE",
        ])
        .diagnostics(&["ast_tree"], &["parse_update"], &["bd-340i"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new("UT-SQL-004", cat, crate_name, "DELETE with WHERE and LIMIT")
            .module("parser::delete")
            .invariants(&[
                "DELETE FROM with WHERE clause",
                "DELETE with ORDER BY + LIMIT",
            ])
            .diagnostics(&["ast_tree"], &["parse_delete"], &["bd-340i"])
            .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-SQL-005",
            cat,
            crate_name,
            "CREATE TABLE with constraints",
        )
        .module("parser::ddl")
        .invariants(&[
            "PRIMARY KEY, UNIQUE, NOT NULL, DEFAULT constraints parse",
            "FOREIGN KEY references parse",
            "CHECK constraint expressions parse",
        ])
        .diagnostics(&["ast_tree"], &["parse_create_table"], &["bd-3kin"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-SQL-006",
            cat,
            crate_name,
            "Expression operator precedence (Pratt)",
        )
        .module("parser::expr")
        .invariants(&[
            "Arithmetic precedence: * before +",
            "Boolean precedence: AND before OR",
            "BETWEEN, IN, LIKE, GLOB operators parse",
        ])
        .property_based()
        .diagnostics(
            &["ast_tree", "precedence_table"],
            &["parse_expr"],
            &["bd-2832"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-SQL-007",
            cat,
            crate_name,
            "JOIN types (INNER, LEFT, CROSS, NATURAL)",
        )
        .module("parser::select")
        .invariants(&[
            "All JOIN types parse with ON clause",
            "USING clause parses",
            "Multi-table joins compose correctly",
        ])
        .diagnostics(&["ast_tree"], &["parse_join"], &["bd-2d6i"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-SQL-008",
            cat,
            crate_name,
            "Compound queries (UNION, INTERSECT, EXCEPT)",
        )
        .module("parser::compound")
        .invariants(&[
            "UNION ALL preserves duplicates in AST",
            "INTERSECT and EXCEPT parse correctly",
            "Compound with ORDER BY on outer query",
        ])
        .diagnostics(&["ast_tree"], &["parse_compound"], &["bd-2d6i"])
        .build(),
    );
}

#[allow(clippy::too_many_lines)]
fn build_vdbe_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::VdbeOpcodes;
    let crate_name = "fsqlite-vdbe";

    tests.push(
        TestEntryBuilder::new(
            "UT-VDBE-001",
            cat,
            crate_name,
            "Arithmetic opcodes (Add, Subtract, Multiply, Divide)",
        )
        .module("vdbe::arithmetic")
        .invariants(&[
            "Integer arithmetic produces correct results",
            "Division by zero handled per SQLite semantics",
            "Remainder opcode matches SQLite behavior",
        ])
        .diagnostics(&["register_state"], &["vdbe_step"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-VDBE-002",
            cat,
            crate_name,
            "Comparison and branching (Eq, Ne, Lt, Le, Gt, Ge)",
        )
        .module("vdbe::comparison")
        .invariants(&[
            "Type affinity comparisons match SQLite",
            "NULL comparisons follow SQL semantics",
            "Branch targets resolve correctly",
        ])
        .diagnostics(&["register_state", "pc"], &["vdbe_step"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-VDBE-003",
            cat,
            crate_name,
            "Cursor operations (OpenRead, OpenWrite, Seek)",
        )
        .module("vdbe::cursor")
        .invariants(&[
            "Cursor opens on correct root page",
            "SeekGE/SeekGT/SeekLE/SeekLT honor index ordering",
            "Cursor rewind/last work on empty tables",
        ])
        .diagnostics(
            &["cursor_state", "btree_page"],
            &["vdbe_cursor"],
            &["bd-25q8"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-VDBE-004",
            cat,
            crate_name,
            "Transaction control opcodes (Transaction, Savepoint, AutoCommit)",
        )
        .module("vdbe::transaction")
        .invariants(&[
            "Transaction opcode acquires correct lock level",
            "Savepoint creates undo barrier",
            "AutoCommit finalizes or rolls back",
        ])
        .diagnostics(&["txn_state"], &["vdbe_txn"], &["bd-7pxb"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-VDBE-005",
            cat,
            crate_name,
            "Row construction (MakeRecord, Column, ResultRow)",
        )
        .module("vdbe::record")
        .invariants(&[
            "MakeRecord encodes type header correctly",
            "Column extracts correct field by index",
            "ResultRow produces output row",
        ])
        .diagnostics(&["register_state", "record_bytes"], &["vdbe_step"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-VDBE-006",
            cat,
            crate_name,
            "Aggregate opcodes (AggStep, AggFinal)",
        )
        .module("vdbe::aggregate")
        .invariants(&[
            "SUM/COUNT/AVG accumulate correctly",
            "GROUP BY bucketing produces correct groups",
            "Empty group returns NULL for aggregates",
        ])
        .diagnostics(&["agg_state"], &["vdbe_agg"], &[])
        .build(),
    );
}

#[allow(clippy::too_many_lines)]
fn build_storage_txn_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::StorageTransaction;

    tests.push(
        TestEntryBuilder::new(
            "UT-STOR-001",
            cat,
            "fsqlite-wal",
            "WAL frame write and read-back",
        )
        .module("wal::frame")
        .invariants(&[
            "Written frame reads back identically",
            "Frame checksum validates",
            "Salt values propagate correctly",
        ])
        .diagnostics(&["wal_header", "frame_bytes"], &["wal_write"], &["bd-2fas"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-STOR-002",
            cat,
            "fsqlite-pager",
            "Page cache eviction (ARC policy)",
        )
        .module("pager::cache")
        .invariants(&[
            "Eviction respects ARC ghost lists",
            "Cache hit rate tracks correctly",
            "Dirty pages flushed before eviction",
        ])
        .diagnostics(
            &["cache_stats", "arc_state"],
            &["pager_evict"],
            &["bd-2zoa"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-STOR-003",
            cat,
            "fsqlite-btree",
            "B-tree page split and merge",
        )
        .module("btree::split")
        .invariants(&[
            "Split preserves key ordering",
            "Merge reclaims space correctly",
            "Parent pointers updated after split",
        ])
        .diagnostics(
            &["btree_page", "cell_array"],
            &["btree_split"],
            &["bd-25q8"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-STOR-004",
            cat,
            "fsqlite-mvcc",
            "MVCC page version chain",
        )
        .module("mvcc::version")
        .invariants(&[
            "Version chain grows with concurrent writers",
            "Snapshot reads see correct version",
            "Garbage collection reclaims old versions",
        ])
        .diagnostics(
            &["version_chain", "snapshot_id"],
            &["mvcc_read"],
            &["bd-2npr"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-STOR-005",
            cat,
            "fsqlite-mvcc",
            "SSI rw-antidependency detection",
        )
        .module("mvcc::ssi")
        .invariants(&[
            "Write-skew detected and aborted",
            "No false positives on disjoint writes",
            "First-committer-wins enforced",
        ])
        .diagnostics(
            &["conflict_graph", "txn_state"],
            &["ssi_check"],
            &["bd-2d3i.1"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-STOR-006",
            cat,
            "fsqlite-wal",
            "Checkpoint (PASSIVE, FULL, RESTART, TRUNCATE)",
        )
        .module("wal::checkpoint")
        .invariants(&[
            "PASSIVE checkpoint transfers frames without blocking",
            "FULL checkpoint waits for readers to drain",
            "TRUNCATE resets WAL to zero length",
        ])
        .diagnostics(&["wal_state", "checkpoint_info"], &["wal_checkpoint"], &[])
        .build(),
    );
}

fn build_pragma_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::Pragma;
    let crate_name = "fsqlite-core";

    tests.push(
        TestEntryBuilder::new(
            "UT-PRAGMA-001",
            cat,
            crate_name,
            "journal_mode pragma transitions",
        )
        .module("connection::pragma")
        .invariants(&[
            "journal_mode=WAL switches to WAL mode",
            "journal_mode=DELETE switches to rollback journal",
            "Invalid journal_mode is ignored (no crash)",
        ])
        .diagnostics(&["pragma_state"], &["pragma_exec"], &["bd-1mrj"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-PRAGMA-002",
            cat,
            crate_name,
            "page_size and cache_size pragmas",
        )
        .module("connection::pragma")
        .invariants(&[
            "page_size must be power of 2 in [512, 65536]",
            "cache_size negative means kibibytes",
            "Page size change requires VACUUM",
        ])
        .diagnostics(
            &["pragma_state", "pager_config"],
            &["pragma_exec"],
            &["bd-1mrj"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new("UT-PRAGMA-003", cat, crate_name, "integrity_check pragma")
            .module("connection::pragma")
            .invariants(&[
                "Returns 'ok' on uncorrupted database",
                "Detects page corruption",
                "Reports specific corruption details",
            ])
            .diagnostics(&["btree_state", "page_checksum"], &["integrity_check"], &[])
            .build(),
    );
}

fn build_builtin_function_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::BuiltinFunctions;
    let crate_name = "fsqlite-func";

    tests.push(
        TestEntryBuilder::new(
            "UT-FUN-001",
            cat,
            crate_name,
            "String functions (length, substr, replace, trim)",
        )
        .module("func::string")
        .invariants(&[
            "length() counts UTF-8 characters not bytes",
            "substr() handles negative start",
            "replace() handles overlapping patterns",
        ])
        .diagnostics(&["func_args", "result_value"], &["func_exec"], &["bd-2zg1"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-FUN-002",
            cat,
            crate_name,
            "Math functions (abs, max, min, round)",
        )
        .module("func::math")
        .invariants(&[
            "abs(NULL) returns NULL",
            "max/min with mixed types uses affinity",
            "round() handles negative decimal places",
        ])
        .diagnostics(&["func_args", "result_value"], &["func_exec"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-FUN-003",
            cat,
            crate_name,
            "Aggregate functions (sum, count, avg, group_concat)",
        )
        .module("func::aggregate")
        .invariants(&[
            "count(*) includes NULLs, count(col) excludes NULLs",
            "sum() returns integer for integer inputs",
            "group_concat() respects separator argument",
        ])
        .diagnostics(&["agg_state", "result_value"], &["func_agg"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-FUN-004",
            cat,
            crate_name,
            "Date/time functions (date, time, datetime, julianday)",
        )
        .module("func::datetime")
        .invariants(&[
            "date('now') returns current date in YYYY-MM-DD",
            "time modifiers apply correctly",
            "julianday() round-trips with datetime()",
        ])
        .diagnostics(&["func_args", "result_value"], &["func_exec"], &["bd-3lhq"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-FUN-005",
            cat,
            crate_name,
            "Type inspection (typeof, coalesce, nullif, ifnull)",
        )
        .module("func::typecheck")
        .invariants(&[
            "typeof() returns correct type string",
            "coalesce() returns first non-NULL",
            "nullif(a,b) returns NULL when a==b",
        ])
        .diagnostics(&["func_args", "result_value"], &["func_exec"], &[])
        .build(),
    );
}

fn build_extension_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::Extensions;

    tests.push(
        TestEntryBuilder::new(
            "UT-EXT-001",
            cat,
            "fsqlite-ext-json",
            "JSON extraction (json_extract, json_type)",
        )
        .module("json::extract")
        .invariants(&[
            "json_extract() with path returns correct value",
            "json_type() returns correct type string",
            "Invalid JSON returns error not crash",
        ])
        .diagnostics(&["json_doc", "result_value"], &["json_func"], &["bd-3cvl"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-EXT-002",
            cat,
            "fsqlite-ext-fts5",
            "FTS5 tokenization and matching",
        )
        .module("fts5::tokenize")
        .invariants(&[
            "Porter stemmer tokenizes correctly",
            "MATCH operator returns ranked results",
            "highlight() produces correct spans",
        ])
        .diagnostics(
            &["token_stream", "match_result"],
            &["fts5_query"],
            &["bd-316x"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-EXT-003",
            cat,
            "fsqlite-ext-fts3",
            "FTS3/FTS4 backward compatibility",
        )
        .module("fts3::compat")
        .invariants(&[
            "FTS3 content table accessible",
            "FTS4 content= option parses",
            "matchinfo() returns valid BLOB",
        ])
        .diagnostics(&["fts_state"], &["fts3_query"], &["bd-2xl9"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-EXT-004",
            cat,
            "fsqlite-ext-rtree",
            "R-tree spatial queries",
        )
        .module("rtree::query")
        .invariants(&[
            "Bounding box containment query returns correct results",
            "R-tree insert maintains tree invariants",
            "Empty R-tree query returns zero rows",
        ])
        .diagnostics(&["rtree_node"], &["rtree_search"], &[])
        .build(),
    );
}

fn build_type_system_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::TypeSystem;
    let crate_name = "fsqlite-types";

    tests.push(
        TestEntryBuilder::new(
            "UT-TYPE-001",
            cat,
            crate_name,
            "Type affinity determination (column declaration)",
        )
        .module("types::affinity")
        .invariants(&[
            "INTEGER affinity for INT, INTEGER, TINYINT, etc.",
            "TEXT affinity for CHAR, VARCHAR, CLOB, etc.",
            "REAL affinity for FLOAT, DOUBLE, REAL",
            "BLOB affinity for BLOB or no type",
            "NUMERIC affinity as default",
        ])
        .diagnostics(&["column_type", "affinity"], &["type_resolve"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-TYPE-002",
            cat,
            crate_name,
            "Type coercion in comparisons",
        )
        .module("types::coercion")
        .invariants(&[
            "Integer vs text comparison uses affinity rules",
            "NULL comparisons always yield NULL/false",
            "BLOB comparisons are bytewise",
        ])
        .diagnostics(
            &["lhs_value", "rhs_value", "result"],
            &["type_coerce"],
            &["bd-22l4"],
        )
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-TYPE-003",
            cat,
            crate_name,
            "Collation sequences (BINARY, NOCASE, RTRIM)",
        )
        .module("types::collation")
        .invariants(&[
            "BINARY compares bytes exactly",
            "NOCASE folds ASCII case",
            "RTRIM ignores trailing spaces",
        ])
        .diagnostics(&["collation_name", "cmp_result"], &["collation_cmp"], &[])
        .build(),
    );
}

fn build_file_format_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::FileFormat;

    tests.push(
        TestEntryBuilder::new(
            "UT-FMT-001",
            cat,
            "fsqlite-pager",
            "Database header 100-byte layout",
        )
        .module("pager::header")
        .invariants(&[
            "Magic string 'SQLite format 3\\0' at offset 0",
            "Page size at offset 16-17 is power of 2",
            "Schema format number at offset 44",
            "Header checksum valid",
        ])
        .diagnostics(&["header_bytes"], &["header_read"], &["bd-1uzb"])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-FMT-002",
            cat,
            "fsqlite-btree",
            "B-tree page cell encoding",
        )
        .module("btree::cell")
        .invariants(&[
            "Varint encoding matches SQLite spec",
            "Cell pointer array sorted in key order",
            "Overflow page chain terminates correctly",
        ])
        .diagnostics(&["page_bytes", "cell_data"], &["btree_read"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-FMT-003",
            cat,
            "fsqlite-btree",
            "Record format encoding/decoding",
        )
        .module("btree::record")
        .invariants(&[
            "Record header varint count matches field count",
            "Serial type codes map to correct storage size",
            "NULL, integer, float, text, blob all round-trip",
        ])
        .property_based()
        .diagnostics(&["record_bytes", "decoded_values"], &["record_decode"], &[])
        .build(),
    );
}

fn build_api_cli_tests(tests: &mut Vec<UnitTestEntry>) {
    let cat = FeatureCategory::ApiCli;
    let crate_name = "fsqlite-core";

    tests.push(
        TestEntryBuilder::new(
            "UT-API-001",
            cat,
            crate_name,
            "Connection open/close lifecycle",
        )
        .module("connection::lifecycle")
        .invariants(&[
            "Open creates valid connection handle",
            "Close releases all resources",
            "Double-close does not panic",
        ])
        .diagnostics(&["conn_state"], &["conn_open"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-API-002",
            cat,
            crate_name,
            "Prepared statement lifecycle (prepare/step/finalize)",
        )
        .module("connection::stmt")
        .invariants(&[
            "Prepare returns valid statement handle",
            "Step returns Row or Done",
            "Finalize releases compiled query",
            "Parameter binding works for all types",
        ])
        .diagnostics(&["stmt_state", "bind_values"], &["stmt_prepare"], &[])
        .build(),
    );

    tests.push(
        TestEntryBuilder::new(
            "UT-API-003",
            cat,
            crate_name,
            "Error reporting (error codes, messages)",
        )
        .module("connection::error")
        .invariants(&[
            "Syntax error returns SQLITE_ERROR with message",
            "Constraint violation returns SQLITE_CONSTRAINT",
            "Busy returns SQLITE_BUSY with retry hint",
        ])
        .diagnostics(&["error_code", "error_msg"], &["conn_exec"], &[])
        .build(),
    );
}

fn compute_coverage(tests: &[UnitTestEntry]) -> Vec<BucketCoverage> {
    FeatureCategory::ALL
        .iter()
        .map(|&cat| {
            let bucket_tests: Vec<_> = tests.iter().filter(|t| t.category == cat).collect();
            let test_count = bucket_tests.len();
            let invariant_count: usize = bucket_tests.iter().map(|t| t.invariants.len()).sum();
            let property_test_count = bucket_tests.iter().filter(|t| t.property_based).count();

            let mut crates: Vec<String> = bucket_tests
                .iter()
                .map(|t| t.crate_name.clone())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            crates.sort();

            let (missing_coverage, fill_pct) = assess_bucket_coverage(cat, test_count);

            BucketCoverage {
                category: cat,
                test_count,
                invariant_count,
                property_test_count,
                contributing_crates: crates,
                missing_coverage,
                fill_pct,
            }
        })
        .collect()
}

/// Assess coverage completeness for a category.
/// Returns (missing areas, fill percentage).
fn assess_bucket_coverage(cat: FeatureCategory, test_count: usize) -> (Vec<String>, f64) {
    // Target test counts per category (based on feature surface area).
    let (target, missing) = match cat {
        FeatureCategory::SqlGrammar => (
            12,
            vec![
                "CTE (common table expressions)",
                "Window functions",
                "UPSERT (INSERT ON CONFLICT)",
                "ALTER TABLE variants",
            ],
        ),
        FeatureCategory::VdbeOpcodes => (
            10,
            vec![
                "String opcodes (Concat, Function)",
                "Index opcodes (IdxInsert, IdxDelete)",
                "Bloom filter opcodes",
                "Virtual table opcodes",
            ],
        ),
        FeatureCategory::StorageTransaction => (
            10,
            vec![
                "WAL FEC repair cycle",
                "Deadlock-free lock ordering",
                "Shared lock table contention",
                "Page recycling (freelist)",
            ],
        ),
        FeatureCategory::Pragma => (
            6,
            vec![
                "wal_autocheckpoint",
                "foreign_keys enforcement",
                "compile_options listing",
            ],
        ),
        FeatureCategory::BuiltinFunctions => (
            8,
            vec![
                "Hex/unhex/zeroblob",
                "Unicode functions",
                "Random/randomblob",
            ],
        ),
        FeatureCategory::Extensions => (6, vec!["Session extension changeset", "ICU collation"]),
        FeatureCategory::TypeSystem => {
            (5, vec!["UTF-16 encoding paths", "Numeric text recognition"])
        }
        FeatureCategory::FileFormat => (5, vec!["Freelist page format", "Pointer map pages"]),
        FeatureCategory::ApiCli => (5, vec!["Blob I/O API", "Unlock notification"]),
    };

    let fill = truncate_f64(test_count as f64 / f64::from(target), 4);
    let fill_clamped = if fill > 1.0 { 1.0 } else { fill };
    let missing_strs = missing.into_iter().map(String::from).collect();
    (missing_strs, fill_clamped)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_matrix_builds() {
        let matrix = build_canonical_matrix();
        assert!(!matrix.tests.is_empty());
        assert_eq!(matrix.schema_version, "1.0.0");
        assert_eq!(matrix.bead_id, "bd-1dp9.7.1");
    }

    #[test]
    fn canonical_matrix_validates() {
        let matrix = build_canonical_matrix();
        let errors = matrix.validate();
        assert!(
            errors.is_empty(),
            "Validation errors:\n{}",
            errors.join("\n")
        );
    }

    #[test]
    fn all_categories_have_tests() {
        let matrix = build_canonical_matrix();
        for cat in FeatureCategory::ALL {
            let count = matrix.tests.iter().filter(|t| t.category == cat).count();
            assert!(
                count > 0,
                "Category {cat:?} has no unit tests in the matrix"
            );
        }
    }

    #[test]
    fn all_tests_have_invariants() {
        let matrix = build_canonical_matrix();
        for t in &matrix.tests {
            assert!(
                !t.invariants.is_empty(),
                "Test {} has no invariants",
                t.test_id
            );
        }
    }

    #[test]
    fn test_ids_are_unique() {
        let matrix = build_canonical_matrix();
        let mut seen = std::collections::BTreeSet::new();
        for t in &matrix.tests {
            assert!(seen.insert(&t.test_id), "Duplicate test ID: {}", t.test_id);
        }
    }

    #[test]
    fn seeds_are_deterministic() {
        let m1 = build_canonical_matrix();
        let m2 = build_canonical_matrix();
        for (t1, t2) in m1.tests.iter().zip(m2.tests.iter()) {
            assert_eq!(
                t1.seed, t2.seed,
                "Seed mismatch for {}: {} vs {}",
                t1.test_id, t1.seed, t2.seed
            );
        }
    }

    #[test]
    fn seeds_are_nonzero() {
        let matrix = build_canonical_matrix();
        for t in &matrix.tests {
            assert_ne!(t.seed, 0, "Test {} has zero seed", t.test_id);
        }
    }

    #[test]
    fn seeds_are_distinct() {
        let matrix = build_canonical_matrix();
        let seeds: std::collections::BTreeSet<_> = matrix.tests.iter().map(|t| t.seed).collect();
        assert_eq!(
            seeds.len(),
            matrix.tests.len(),
            "Some tests share the same seed"
        );
    }

    #[test]
    fn coverage_report_complete() {
        let matrix = build_canonical_matrix();
        assert_eq!(
            matrix.coverage.len(),
            FeatureCategory::ALL.len(),
            "Coverage report should have entry for each category"
        );
        for c in &matrix.coverage {
            assert!(
                c.test_count > 0,
                "Coverage for {:?} shows 0 tests",
                c.category
            );
            assert!(
                c.fill_pct > 0.0,
                "Coverage for {:?} shows 0% fill",
                c.category
            );
            assert!(
                c.fill_pct <= 1.0,
                "Coverage for {:?} exceeds 100%: {}",
                c.category,
                c.fill_pct
            );
        }
    }

    #[test]
    fn overall_fill_positive() {
        let matrix = build_canonical_matrix();
        let fill = matrix.overall_fill_pct();
        assert!(fill > 0.0, "Overall fill should be positive: {fill}");
        assert!(fill <= 1.0, "Overall fill should be at most 1.0: {fill}");
    }

    #[test]
    fn json_roundtrip() {
        let matrix = build_canonical_matrix();
        let json = matrix.to_json().expect("serialize");
        let deserialized: UnitMatrix = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.tests.len(), matrix.tests.len());
        assert_eq!(deserialized.coverage.len(), matrix.coverage.len());
    }

    #[test]
    fn coverage_missing_areas_populated() {
        let matrix = build_canonical_matrix();
        let total_missing: usize = matrix
            .coverage
            .iter()
            .map(|c| c.missing_coverage.len())
            .sum();
        assert!(
            total_missing > 0,
            "Should have missing coverage areas identified"
        );
    }

    #[test]
    fn property_based_tests_flagged() {
        let matrix = build_canonical_matrix();
        let prop_count = matrix.tests.iter().filter(|t| t.property_based).count();
        assert!(
            prop_count >= 2,
            "Expected at least 2 property-based tests, found {prop_count}"
        );
    }
}
