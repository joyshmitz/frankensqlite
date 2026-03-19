//! Integration tests for operation baseline capture and regression detection.
//!
//! Bead: bd-1lsfu.1
//!
//! Tests that:
//! 1. All 9 operations can be measured via `measure_operation`.
//! 2. Baseline JSON roundtrips correctly.
//! 3. Regression detection works with configurable thresholds.
//! 4. Baselines can be captured for both FrankenSQLite and C SQLite.

use fsqlite_core::connection::{
    hot_path_profile_snapshot, reset_hot_path_profile, set_hot_path_profile_enabled,
};
use fsqlite_e2e::baseline::{
    BaselineReport, DEFAULT_REGRESSION_THRESHOLD, LatencyStats, Operation, OperationBaseline,
    RegressionResult, measure_operation,
};
use fsqlite_types::SqliteValue;
use std::sync::{Mutex, OnceLock};

// ─── Baseline module unit integration tests ─────────────────────────────

#[test]
fn all_nine_operations_have_unique_names() {
    let ops = Operation::all();
    assert_eq!(ops.len(), 9);
    let mut names: Vec<&str> = ops.iter().map(Operation::display_name).collect();
    let orig_len = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), orig_len, "duplicate operation display names");
}

#[test]
fn baseline_report_json_schema_version() {
    let report = BaselineReport::new("test");
    let json = report.to_pretty_json().unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(
        parsed["schema_version"],
        "fsqlite-e2e.operation_baseline.v1"
    );
    assert!(parsed["methodology"]["version"].is_string());
    assert!(parsed["environment"]["arch"].is_string());
}

#[test]
fn regression_threshold_default_is_ten_percent() {
    assert!((DEFAULT_REGRESSION_THRESHOLD - 0.10).abs() < f64::EPSILON);
}

#[test]
fn regression_check_empty_reports() {
    let old = BaselineReport::new("test");
    let current = BaselineReport::new("test");
    let results = old.check_regression(&current, DEFAULT_REGRESSION_THRESHOLD);
    assert!(results.is_empty());
}

#[test]
fn regression_check_missing_operation_in_current() {
    let mut old = BaselineReport::new("test");
    old.baselines.push(OperationBaseline {
        operation: Operation::PointLookup,
        engine: "frankensqlite".to_owned(),
        row_count: 1000,
        iterations: 100,
        warmup_iterations: 10,
        latency: LatencyStats {
            p50_micros: 50,
            p95_micros: 100,
            p99_micros: 200,
            max_micros: 500,
        },
        throughput_ops_per_sec: 20000.0,
    });

    let current = BaselineReport::new("test");
    let results = old.check_regression(&current, DEFAULT_REGRESSION_THRESHOLD);
    // Missing operation = no comparison = no regression.
    assert!(results.is_empty());
}

#[test]
fn regression_check_exact_match() {
    let baseline = OperationBaseline {
        operation: Operation::Aggregation,
        engine: "frankensqlite".to_owned(),
        row_count: 1000,
        iterations: 100,
        warmup_iterations: 10,
        latency: LatencyStats {
            p50_micros: 100,
            p95_micros: 200,
            p99_micros: 300,
            max_micros: 500,
        },
        throughput_ops_per_sec: 10000.0,
    };

    let mut old = BaselineReport::new("test");
    old.baselines.push(baseline.clone());
    let mut current = BaselineReport::new("test");
    current.baselines.push(baseline);

    let results = old.check_regression(&current, DEFAULT_REGRESSION_THRESHOLD);
    assert_eq!(results.len(), 1);
    assert!(!results[0].regressed);
    assert!((results[0].change_pct).abs() < 0.01);
}

#[test]
fn regression_result_summary_contains_key_info() {
    let result = RegressionResult {
        operation: Operation::BatchInsert,
        engine: "frankensqlite".to_owned(),
        baseline_p50_micros: 1000,
        current_p50_micros: 1200,
        change_pct: 20.0,
        regressed: true,
    };
    let summary = result.summary();
    assert!(summary.contains("REGRESSION"));
    assert!(summary.contains("batch_insert"));
    assert!(summary.contains("frankensqlite"));
    assert!(summary.contains("1000"));
    assert!(summary.contains("1200"));
}

// ─── Live operation measurement tests ───────────────────────────────────

#[test]
fn measure_noop_operation() {
    let (stats, throughput) = measure_operation(2, 10, || {
        // No-op: should be very fast.
    });
    // Stats should be valid (non-panicking, sane values).
    assert!(stats.p50_micros <= stats.p95_micros);
    assert!(stats.p95_micros <= stats.p99_micros);
    assert!(stats.p99_micros <= stats.max_micros);
    assert!(throughput > 0.0);
}

#[test]
fn measure_frankensqlite_point_lookup() {
    let conn = fsqlite::Connection::open(":memory:").unwrap();
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .unwrap();
    conn.execute("BEGIN").unwrap();
    for i in 0..100_i64 {
        conn.execute(&format!("INSERT INTO t VALUES ({i}, 'v{i}')"))
            .unwrap();
    }
    conn.execute("COMMIT").unwrap();

    let mut id = 1_i64;
    let (stats, throughput) = measure_operation(5, 50, || {
        let rows = conn
            .query(&format!("SELECT * FROM t WHERE id = {id}"))
            .unwrap();
        assert_eq!(rows.len(), 1);
        id = (id % 100) + 1;
    });

    assert!(stats.p50_micros >= 1, "point lookup should take >= 1us");
    assert!(throughput > 0.0);
}

#[test]
fn measure_frankensqlite_sequential_scan() {
    let conn = fsqlite::Connection::open(":memory:").unwrap();
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        .unwrap();
    conn.execute("BEGIN").unwrap();
    for i in 0..200_i64 {
        conn.execute(&format!("INSERT INTO t VALUES ({i}, 'v{i}')"))
            .unwrap();
    }
    conn.execute("COMMIT").unwrap();

    let (stats, throughput) = measure_operation(3, 20, || {
        let rows = conn.query("SELECT * FROM t").unwrap();
        assert_eq!(rows.len(), 200);
    });

    assert!(stats.p50_micros >= 1);
    assert!(throughput > 0.0);
}

#[test]
fn measure_csqlite_point_lookup() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);")
        .unwrap();
    conn.execute_batch("BEGIN").unwrap();
    {
        let mut stmt = conn
            .prepare("INSERT INTO t VALUES (?1, ('v' || ?1))")
            .unwrap();
        for i in 0..100_i64 {
            stmt.execute(rusqlite::params![i]).unwrap();
        }
    }
    conn.execute_batch("COMMIT").unwrap();

    let mut stmt = conn.prepare("SELECT * FROM t WHERE id = ?1").unwrap();
    let mut id = 1_i64;
    let (stats, throughput) = measure_operation(5, 50, || {
        let rows: Vec<(i64,)> = stmt
            .query_map(rusqlite::params![id], |row| Ok((row.get(0).unwrap(),)))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 1);
        id = (id % 100) + 1;
    });

    // p50 is u64, always >= 0; just confirm stats are valid.
    assert!(stats.p95_micros >= stats.p50_micros);
    assert!(throughput > 0.0);
}

// ─── Baseline save/load integration ─────────────────────────────────────

#[test]
fn save_load_roundtrip_with_all_operations() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full_baseline.json");

    let mut report = BaselineReport::new("test");
    for op in Operation::all() {
        report.baselines.push(OperationBaseline {
            operation: op,
            engine: "frankensqlite".to_owned(),
            row_count: 1000,
            iterations: 100,
            warmup_iterations: 10,
            latency: LatencyStats {
                p50_micros: 100,
                p95_micros: 200,
                p99_micros: 300,
                max_micros: 500,
            },
            throughput_ops_per_sec: 10000.0,
        });
    }

    fsqlite_e2e::baseline::save_baseline(&report, &path).unwrap();
    let loaded = fsqlite_e2e::baseline::load_baseline(&path).unwrap();

    assert_eq!(loaded.baselines.len(), 9);
    for (i, op) in Operation::all().iter().enumerate() {
        assert_eq!(loaded.baselines[i].operation, *op);
    }
}

// ─── Full 9-operation baseline capture ──────────────────────────────────

#[test]
#[allow(clippy::too_many_lines)]
fn capture_all_nine_baselines_frankensqlite() {
    let conn = fsqlite::Connection::open(":memory:").unwrap();

    // Setup: create main table with 200 rows.
    conn.execute(
        "CREATE TABLE bench (id INTEGER PRIMARY KEY, name TEXT, category TEXT, score INTEGER)",
    )
    .unwrap();
    conn.execute("BEGIN").unwrap();
    for i in 1..=200_i64 {
        conn.execute(&format!(
            "INSERT INTO bench VALUES ({i}, 'name_{i}', 'cat_{}', {})",
            i % 10,
            i * 7,
        ))
        .unwrap();
    }
    conn.execute("COMMIT").unwrap();

    // Setup: create join table with 100 rows.
    conn.execute("CREATE TABLE bench2 (id INTEGER PRIMARY KEY, bench_id INTEGER, label TEXT)")
        .unwrap();
    conn.execute("BEGIN").unwrap();
    for i in 1..=100_i64 {
        conn.execute(&format!(
            "INSERT INTO bench2 VALUES ({i}, {}, 'label_{i}')",
            i * 2,
        ))
        .unwrap();
    }
    conn.execute("COMMIT").unwrap();

    let mut report = BaselineReport::new("test");
    let warmup = 3_u32;
    let iters = 20_u32;

    // 1. Sequential scan.
    let (lat, thr) = measure_operation(warmup, iters, || {
        let rows = conn.query("SELECT * FROM bench").unwrap();
        assert_eq!(rows.len(), 200);
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::SequentialScan,
        engine: "frankensqlite".to_owned(),
        row_count: 200,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // 2. Point lookup.
    let mut id = 1_i64;
    let (lat, thr) = measure_operation(warmup, iters, || {
        let rows = conn
            .query(&format!("SELECT * FROM bench WHERE id = {id}"))
            .unwrap();
        assert_eq!(rows.len(), 1);
        id = (id % 200) + 1;
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::PointLookup,
        engine: "frankensqlite".to_owned(),
        row_count: 200,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // 3. Range scan.
    let (lat, thr) = measure_operation(warmup, iters, || {
        let rows = conn
            .query("SELECT * FROM bench WHERE id >= 50 AND id < 100")
            .unwrap();
        assert_eq!(rows.len(), 50);
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::RangeScan,
        engine: "frankensqlite".to_owned(),
        row_count: 200,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // 4. Single-row insert (into a separate disposable table per measurement).
    let conn4 = fsqlite::Connection::open(":memory:").unwrap();
    conn4
        .execute("CREATE TABLE ins_test (id INTEGER PRIMARY KEY, val TEXT)")
        .unwrap();
    let mut insert_id = 1_i64;
    let (lat, thr) = measure_operation(warmup, iters, || {
        conn4
            .execute(&format!(
                "INSERT INTO ins_test VALUES ({insert_id}, 'val_{insert_id}')"
            ))
            .unwrap();
        insert_id += 1;
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::SingleRowInsert,
        engine: "frankensqlite".to_owned(),
        row_count: 0,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // 5. Batch insert.
    let (lat, thr) = measure_operation(warmup, iters, || {
        let batch_conn = fsqlite::Connection::open(":memory:").unwrap();
        batch_conn
            .execute("CREATE TABLE batch_t (id INTEGER PRIMARY KEY, val TEXT)")
            .unwrap();
        batch_conn.execute("BEGIN").unwrap();
        for j in 1..=100_i64 {
            batch_conn
                .execute(&format!("INSERT INTO batch_t VALUES ({j}, 'v{j}')"))
                .unwrap();
        }
        batch_conn.execute("COMMIT").unwrap();
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::BatchInsert,
        engine: "frankensqlite".to_owned(),
        row_count: 100,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // 6. Single-row update.
    let mut upd_id = 1_i64;
    let (lat, thr) = measure_operation(warmup, iters, || {
        conn.execute(&format!(
            "UPDATE bench SET score = {} WHERE id = {upd_id}",
            upd_id * 13,
        ))
        .unwrap();
        upd_id = (upd_id % 200) + 1;
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::SingleRowUpdate,
        engine: "frankensqlite".to_owned(),
        row_count: 200,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // 7. Single-row delete (use a disposable table).
    let conn7 = fsqlite::Connection::open(":memory:").unwrap();
    conn7
        .execute("CREATE TABLE del_test (id INTEGER PRIMARY KEY, val TEXT)")
        .unwrap();
    for j in 1..=1000_i64 {
        conn7
            .execute(&format!("INSERT INTO del_test VALUES ({j}, 'v{j}')"))
            .unwrap();
    }
    let mut del_id = 1_i64;
    let (lat, thr) = measure_operation(warmup, iters, || {
        conn7
            .execute(&format!("DELETE FROM del_test WHERE id = {del_id}"))
            .unwrap();
        del_id += 1;
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::SingleRowDelete,
        engine: "frankensqlite".to_owned(),
        row_count: 1000,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // 8. 2-way equi-join.
    let (lat, thr) = measure_operation(warmup, iters, || {
        let rows = conn
            .query(
                "SELECT bench.id, bench.name, bench2.label \
                 FROM bench INNER JOIN bench2 ON bench.id = bench2.bench_id",
            )
            .unwrap();
        assert!(!rows.is_empty());
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::TwoWayEquiJoin,
        engine: "frankensqlite".to_owned(),
        row_count: 200,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // 9. Aggregation.
    let (lat, thr) = measure_operation(warmup, iters, || {
        let rows = conn
            .query("SELECT COUNT(*), SUM(score), AVG(score) FROM bench")
            .unwrap();
        assert_eq!(rows.len(), 1);
    });
    report.baselines.push(OperationBaseline {
        operation: Operation::Aggregation,
        engine: "frankensqlite".to_owned(),
        row_count: 200,
        iterations: iters,
        warmup_iterations: warmup,
        latency: lat,
        throughput_ops_per_sec: thr,
    });

    // Verify we captured all 9.
    assert_eq!(report.baselines.len(), 9);

    // Verify JSON roundtrip.
    let json = report.to_pretty_json().unwrap();
    let parsed = BaselineReport::from_json(&json).unwrap();
    assert_eq!(parsed.baselines.len(), 9);

    // Verify no regressions against self (should all be exact match).
    let results = report.check_regression(&parsed, DEFAULT_REGRESSION_THRESHOLD);
    for r in &results {
        assert!(
            !r.regressed,
            "self-comparison should not regress: {}",
            r.summary()
        );
    }
}

// ─── Manual perf probes ────────────────────────────────────────────────

fn median_f64(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    values[values.len() / 2]
}

fn manual_hot_path_profile_mutex() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct ManualHotPathProfileGuard {
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl ManualHotPathProfileGuard {
    fn new() -> Self {
        let guard = manual_hot_path_profile_mutex().lock().unwrap();
        reset_hot_path_profile();
        set_hot_path_profile_enabled(true);
        Self { _guard: guard }
    }
}

impl Drop for ManualHotPathProfileGuard {
    fn drop(&mut self) {
        reset_hot_path_profile();
        set_hot_path_profile_enabled(false);
    }
}

#[derive(Debug)]
struct PrepareCacheProbeRun {
    rows_per_sec: f64,
    parse_cache_hits: u64,
    parse_cache_misses: u64,
    compiled_cache_hits: u64,
    compiled_cache_misses: u64,
    prepared_cache_hits: u64,
    prepared_cache_misses: u64,
}

#[derive(Debug)]
struct DecodeCacheProbeRun {
    rows_per_sec: f64,
    decode_cache_hits: u64,
    decode_cache_misses: u64,
    decode_cache_invalidations_position: u64,
    decode_cache_invalidations_write: u64,
    decode_cache_invalidations_pseudo: u64,
}

fn run_fsqlite_prepare_cache_probe<'a, I>(sqls: I, row_count: i64) -> PrepareCacheProbeRun
where
    I: IntoIterator<Item = &'a str>,
{
    const CREATE_TABLE: &str =
        "CREATE TABLE bench (id INTEGER PRIMARY KEY, data TEXT, value REAL);";

    fn apply_pragmas_fsqlite(conn: &fsqlite::Connection) {
        for pragma in [
            "PRAGMA page_size = 4096;",
            "PRAGMA journal_mode = WAL;",
            "PRAGMA synchronous = NORMAL;",
            "PRAGMA cache_size = -64000;",
        ] {
            let _ = conn.execute(pragma);
        }
    }

    let _profile_guard = ManualHotPathProfileGuard::new();
    let conn = fsqlite::Connection::open(":memory:").unwrap();
    apply_pragmas_fsqlite(&conn);
    conn.execute(CREATE_TABLE).unwrap();

    reset_hot_path_profile();
    let start = std::time::Instant::now();
    for (idx, sql) in sqls.into_iter().enumerate() {
        let stmt = conn.prepare(sql).unwrap();
        stmt.execute_with_params(&[SqliteValue::Integer(i64::try_from(idx).unwrap())])
            .unwrap();
    }
    let elapsed = start.elapsed();
    let profile = hot_path_profile_snapshot();

    let rows = conn.query("SELECT COUNT(*) FROM bench").unwrap();
    assert_eq!(rows[0].values()[0], SqliteValue::Integer(row_count));

    PrepareCacheProbeRun {
        rows_per_sec: row_count as f64 / elapsed.as_secs_f64(),
        parse_cache_hits: profile.parser.parse_cache_hits,
        parse_cache_misses: profile.parser.parse_cache_misses,
        compiled_cache_hits: profile.parser.compiled_cache_hits,
        compiled_cache_misses: profile.parser.compiled_cache_misses,
        prepared_cache_hits: profile.parser.prepared_cache_hits,
        prepared_cache_misses: profile.parser.prepared_cache_misses,
    }
}

fn run_fsqlite_decode_cache_probe(sql: &str, iterations: usize) -> DecodeCacheProbeRun {
    const CREATE_TABLE: &str = "CREATE TABLE bench (id INTEGER PRIMARY KEY, data TEXT);";
    const INSERT_SQL: &str = "INSERT INTO bench VALUES (?1, ?2);";

    fn apply_pragmas_fsqlite(conn: &fsqlite::Connection) {
        for pragma in [
            "PRAGMA page_size = 4096;",
            "PRAGMA journal_mode = WAL;",
            "PRAGMA synchronous = NORMAL;",
            "PRAGMA cache_size = -64000;",
        ] {
            let _ = conn.execute(pragma);
        }
    }

    let _profile_guard = ManualHotPathProfileGuard::new();
    let conn = fsqlite::Connection::open(":memory:").unwrap();
    apply_pragmas_fsqlite(&conn);
    conn.execute(CREATE_TABLE).unwrap();
    conn.execute_with_params(
        INSERT_SQL,
        &[
            SqliteValue::Integer(1),
            SqliteValue::Text("decode-cache-hot-row".into()),
        ],
    )
    .unwrap();

    reset_hot_path_profile();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let rows = conn.query(sql).unwrap();
        assert_eq!(rows.len(), 1, "probe query should return exactly one row");
    }
    let elapsed = start.elapsed();

    let profile = hot_path_profile_snapshot();
    DecodeCacheProbeRun {
        rows_per_sec: iterations as f64 / elapsed.as_secs_f64(),
        decode_cache_hits: profile.vdbe.decode_cache_hits_total,
        decode_cache_misses: profile.vdbe.decode_cache_misses_total,
        decode_cache_invalidations_position: profile.vdbe.decode_cache_invalidations_position_total,
        decode_cache_invalidations_write: profile.vdbe.decode_cache_invalidations_write_total,
        decode_cache_invalidations_pseudo: profile.vdbe.decode_cache_invalidations_pseudo_total,
    }
}

#[test]
#[ignore = "manual perf probe; run via rch when investigating write throughput"]
fn manual_perf_probe_write_10k_autocommit_prepared_and_ad_hoc() {
    const ROW_COUNT: i64 = 10_000;
    const MEASURED_RUNS: usize = 3;
    const CREATE_TABLE: &str =
        "CREATE TABLE bench (id INTEGER PRIMARY KEY, data TEXT, value REAL);";
    const INSERT_SQL: &str = "INSERT INTO bench VALUES (?1, ('data_' || ?1), (?1 * 0.137));";

    fn apply_pragmas_csqlite(conn: &rusqlite::Connection) {
        conn.execute_batch(
            "PRAGMA page_size = 4096;\
             PRAGMA journal_mode = WAL;\
             PRAGMA synchronous = NORMAL;\
             PRAGMA cache_size = -64000;",
        )
        .ok();
    }

    fn apply_pragmas_fsqlite(conn: &fsqlite::Connection) {
        for pragma in [
            "PRAGMA page_size = 4096;",
            "PRAGMA journal_mode = WAL;",
            "PRAGMA synchronous = NORMAL;",
            "PRAGMA cache_size = -64000;",
        ] {
            let _ = conn.execute(pragma);
        }
    }

    fn run_csqlite_prepared_once() -> f64 {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        apply_pragmas_csqlite(&conn);
        conn.execute_batch(CREATE_TABLE).unwrap();
        let start = std::time::Instant::now();
        let mut stmt = conn.prepare(INSERT_SQL).unwrap();
        for i in 0..ROW_COUNT {
            stmt.execute(rusqlite::params![i]).unwrap();
        }
        let elapsed = start.elapsed();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM bench", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, ROW_COUNT);
        ROW_COUNT as f64 / elapsed.as_secs_f64()
    }

    fn run_csqlite_ad_hoc_once() -> f64 {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        apply_pragmas_csqlite(&conn);
        conn.execute_batch(CREATE_TABLE).unwrap();
        let start = std::time::Instant::now();
        for i in 0..ROW_COUNT {
            conn.execute(INSERT_SQL, rusqlite::params![i]).unwrap();
        }
        let elapsed = start.elapsed();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM bench", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, ROW_COUNT);
        ROW_COUNT as f64 / elapsed.as_secs_f64()
    }

    fn run_fsqlite_prepared_once() -> f64 {
        let conn = fsqlite::Connection::open(":memory:").unwrap();
        apply_pragmas_fsqlite(&conn);
        conn.execute(CREATE_TABLE).unwrap();
        let stmt = conn.prepare(INSERT_SQL).unwrap();
        let start = std::time::Instant::now();
        for i in 0..ROW_COUNT {
            stmt.execute_with_params(&[fsqlite_types::value::SqliteValue::Integer(i)])
                .unwrap();
        }
        let elapsed = start.elapsed();
        let rows = conn.query("SELECT COUNT(*) FROM bench").unwrap();
        assert_eq!(
            rows[0].values()[0],
            fsqlite_types::value::SqliteValue::Integer(ROW_COUNT)
        );
        ROW_COUNT as f64 / elapsed.as_secs_f64()
    }

    fn run_fsqlite_ad_hoc_once() -> f64 {
        let conn = fsqlite::Connection::open(":memory:").unwrap();
        apply_pragmas_fsqlite(&conn);
        conn.execute(CREATE_TABLE).unwrap();
        let start = std::time::Instant::now();
        for i in 0..ROW_COUNT {
            conn.execute_with_params(INSERT_SQL, &[fsqlite_types::value::SqliteValue::Integer(i)])
                .unwrap();
        }
        let elapsed = start.elapsed();
        let rows = conn.query("SELECT COUNT(*) FROM bench").unwrap();
        assert_eq!(
            rows[0].values()[0],
            fsqlite_types::value::SqliteValue::Integer(ROW_COUNT)
        );
        ROW_COUNT as f64 / elapsed.as_secs_f64()
    }

    let csqlite_prepared: Vec<f64> = (0..MEASURED_RUNS)
        .map(|_| run_csqlite_prepared_once())
        .collect();
    let csqlite_ad_hoc: Vec<f64> = (0..MEASURED_RUNS)
        .map(|_| run_csqlite_ad_hoc_once())
        .collect();
    let fsqlite_prepared: Vec<f64> = (0..MEASURED_RUNS)
        .map(|_| run_fsqlite_prepared_once())
        .collect();
    let fsqlite_ad_hoc: Vec<f64> = (0..MEASURED_RUNS)
        .map(|_| run_fsqlite_ad_hoc_once())
        .collect();

    let csqlite_prepared_median = median_f64(csqlite_prepared.clone());
    let csqlite_ad_hoc_median = median_f64(csqlite_ad_hoc.clone());
    let fsqlite_prepared_median = median_f64(fsqlite_prepared.clone());
    let fsqlite_ad_hoc_median = median_f64(fsqlite_ad_hoc.clone());

    eprintln!(
        "manual_perf_probe.write_10k_autocommit.csqlite_prepared.samples={csqlite_prepared:?} median_rows_per_sec={csqlite_prepared_median:.1}"
    );
    eprintln!(
        "manual_perf_probe.write_10k_autocommit.csqlite_ad_hoc.samples={csqlite_ad_hoc:?} median_rows_per_sec={csqlite_ad_hoc_median:.1}"
    );
    eprintln!(
        "manual_perf_probe.write_10k_autocommit.fsqlite_prepared.samples={fsqlite_prepared:?} median_rows_per_sec={fsqlite_prepared_median:.1} ratio_vs_csqlite={:.4}",
        fsqlite_prepared_median / csqlite_prepared_median
    );
    eprintln!(
        "manual_perf_probe.write_10k_autocommit.fsqlite_ad_hoc.samples={fsqlite_ad_hoc:?} median_rows_per_sec={fsqlite_ad_hoc_median:.1} ratio_vs_csqlite={:.4}",
        fsqlite_ad_hoc_median / csqlite_prepared_median
    );

    assert!(csqlite_prepared_median > 0.0);
    assert!(csqlite_ad_hoc_median > 0.0);
    assert!(fsqlite_prepared_median > 0.0);
    assert!(fsqlite_ad_hoc_median > 0.0);
}

#[test]
#[ignore = "manual perf probe; run via rch when investigating repeated prepare reuse"]
fn manual_perf_probe_prepare_cache_reuse_vs_unique_sql_variants() {
    const ROW_COUNT: i64 = 10_000;
    const INSERT_SQL: &str = "INSERT INTO bench VALUES (?1, ('data_' || ?1), (?1 * 0.137))";

    let unique_sqls: Vec<String> = (0..ROW_COUNT)
        .map(|i| format!("{INSERT_SQL} -- prepare-cache-miss-{i}"))
        .collect();

    let reused = run_fsqlite_prepare_cache_probe(
        std::iter::repeat_n(INSERT_SQL, usize::try_from(ROW_COUNT).unwrap()),
        ROW_COUNT,
    );
    let unique = run_fsqlite_prepare_cache_probe(unique_sqls.iter().map(String::as_str), ROW_COUNT);

    eprintln!(
        "manual_perf_probe.prepare_cache_reuse.reused rows_per_sec={:.1} parse_cache_hit={} parse_cache_miss={} compiled_cache_hit={} compiled_cache_miss={} prepared_cache_hit={} prepared_cache_miss={}",
        reused.rows_per_sec,
        reused.parse_cache_hits,
        reused.parse_cache_misses,
        reused.compiled_cache_hits,
        reused.compiled_cache_misses,
        reused.prepared_cache_hits,
        reused.prepared_cache_misses,
    );
    eprintln!(
        "manual_perf_probe.prepare_cache_reuse.unique_sql rows_per_sec={:.1} parse_cache_hit={} parse_cache_miss={} compiled_cache_hit={} compiled_cache_miss={} prepared_cache_hit={} prepared_cache_miss={}",
        unique.rows_per_sec,
        unique.parse_cache_hits,
        unique.parse_cache_misses,
        unique.compiled_cache_hits,
        unique.compiled_cache_misses,
        unique.prepared_cache_hits,
        unique.prepared_cache_misses,
    );
    eprintln!(
        "manual_perf_probe.prepare_cache_reuse.ratio reused_vs_unique={:.4}",
        reused.rows_per_sec / unique.rows_per_sec
    );

    assert!(reused.rows_per_sec > 0.0);
    assert!(unique.rows_per_sec > 0.0);
    assert_eq!(reused.prepared_cache_misses, 1);
    assert!(
        reused.prepared_cache_hits >= u64::try_from(ROW_COUNT - 1).unwrap(),
        "stable SQL should hit the prepared cache after the first prepare: {reused:?}"
    );
    assert_eq!(unique.prepared_cache_hits, 0);
    assert_eq!(
        unique.prepared_cache_misses,
        u64::try_from(ROW_COUNT).unwrap()
    );
    assert!(
        reused.rows_per_sec > unique.rows_per_sec,
        "prepared-cache reuse should outperform forced unique-SQL misses: reused={reused:?} unique={unique:?}"
    );
}

#[test]
#[ignore = "manual perf probe; run via rch when investigating record-decode cache reuse"]
fn manual_perf_probe_record_decode_cache_repeated_column_reads() {
    const ITERATIONS: usize = 10_000;
    const REPEATED_COLUMN_SQL: &str =
        "SELECT data, data, data, data, data FROM bench WHERE id = 1;";

    let repeated = run_fsqlite_decode_cache_probe(REPEATED_COLUMN_SQL, ITERATIONS);

    eprintln!(
        "manual_perf_probe.record_decode_cache.repeated_column_reads rows_per_sec={:.1} decode_cache_hit={} decode_cache_miss={} invalidation_position={} invalidation_write={} invalidation_pseudo={}",
        repeated.rows_per_sec,
        repeated.decode_cache_hits,
        repeated.decode_cache_misses,
        repeated.decode_cache_invalidations_position,
        repeated.decode_cache_invalidations_write,
        repeated.decode_cache_invalidations_pseudo,
    );

    assert!(repeated.rows_per_sec > 0.0);
    assert!(
        repeated.decode_cache_hits > repeated.decode_cache_misses,
        "repeated-column query should produce more decode-cache hits than misses: {repeated:?}"
    );
    assert_eq!(repeated.decode_cache_invalidations_position, 0);
    assert_eq!(repeated.decode_cache_invalidations_write, 0);
    assert_eq!(repeated.decode_cache_invalidations_pseudo, 0);
}
