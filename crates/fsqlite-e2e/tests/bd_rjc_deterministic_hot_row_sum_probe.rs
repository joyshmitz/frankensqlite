//! E2E Test: bd-rjc deterministic hot-row conservation/integrity probe.
//!
//! This test drives two writers against one hot row with a deterministic start
//! gate and bounded retries, then verifies:
//! - committed increment count equals the stored final value;
//! - `PRAGMA integrity_check` stays `ok`;
//! - `sqlite_master.rootpage` values remain valid.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use fsqlite_types::value::SqliteValue;

const WORKERS: usize = 2;
const ROUNDS_PER_WORKER: usize = 240;
const MAX_RETRIES_PER_ROUND: usize = 256;
const BUSY_TIMEOUT_MS: u64 = 5_000;
const RETRY_SLEEP_MS: u64 = 2;

#[derive(Debug, Default, Clone, Copy)]
struct WorkerStats {
    committed: i64,
    retries: u64,
}

#[test]
fn bd_rjc_deterministic_hot_row_sum_and_integrity_probe() {
    let dir = tempfile::tempdir().expect("create tempdir");
    let db_path = dir.path().join("bd_rjc_hot_row_probe.db");
    let db = db_path.to_string_lossy().to_string();

    {
        let conn = fsqlite::Connection::open(&db).expect("open setup db");
        conn.execute("PRAGMA journal_mode=WAL;")
            .expect("enable WAL mode");
        conn.execute(&format!("PRAGMA busy_timeout={BUSY_TIMEOUT_MS};"))
            .expect("set busy timeout");
        conn.execute("PRAGMA fsqlite.concurrent_mode=ON;")
            .expect("enable concurrent mode");
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v INTEGER NOT NULL);")
            .expect("create table");
        conn.execute("INSERT INTO t (id, v) VALUES (1, 0);")
            .expect("seed row");
    }

    let start_barrier = Arc::new(Barrier::new(WORKERS));
    let mut handles = Vec::with_capacity(WORKERS);

    for worker_id in 0..WORKERS {
        let db = db.clone();
        let start_barrier = Arc::clone(&start_barrier);
        handles.push(thread::spawn(move || -> Result<WorkerStats, String> {
            let conn = fsqlite::Connection::open(&db).map_err(|e| {
                format!("worker={worker_id}: open worker db failed: {e}")
            })?;
            conn.execute(&format!("PRAGMA busy_timeout={BUSY_TIMEOUT_MS};"))
                .map_err(|e| format!("worker={worker_id}: set worker busy timeout failed: {e}"))?;
            conn.execute("PRAGMA fsqlite.concurrent_mode=ON;")
                .map_err(|e| format!("worker={worker_id}: enable worker concurrent mode failed: {e}"))?;

            let mut stats = WorkerStats::default();
            start_barrier.wait();

            for round in 0..ROUNDS_PER_WORKER {
                let mut retries_this_round = 0_usize;
                loop {
                    match conn.execute("BEGIN CONCURRENT;") {
                        Ok(_) => {}
                        Err(err) if err.is_transient() => {
                            retries_this_round += 1;
                            stats.retries += 1;
                            if retries_this_round > MAX_RETRIES_PER_ROUND {
                                return Err(format!(
                                    "worker={worker_id} round={round}: exceeded retry budget at BEGIN ({retries_this_round})"
                                ));
                            }
                            thread::sleep(Duration::from_millis(RETRY_SLEEP_MS));
                            continue;
                        }
                        Err(err) => {
                            return Err(format!(
                                "worker={worker_id} round={round}: non-transient begin error: {err}"
                            ));
                        }
                    }

                    match conn.execute("UPDATE t SET v = v + 1 WHERE id = 1;") {
                        Ok(changes) => {
                            if changes != 1 {
                                let _ = conn.execute("ROLLBACK;");
                                return Err(format!(
                                    "worker={worker_id} round={round}: expected one-row update, got {changes}"
                                ));
                            }
                        }
                        Err(err) if err.is_transient() => {
                            retries_this_round += 1;
                            stats.retries += 1;
                            let _ = conn.execute("ROLLBACK;");
                            if retries_this_round > MAX_RETRIES_PER_ROUND {
                                return Err(format!(
                                    "worker={worker_id} round={round}: exceeded retry budget at UPDATE ({retries_this_round})"
                                ));
                            }
                            thread::sleep(Duration::from_millis(RETRY_SLEEP_MS));
                            continue;
                        }
                        Err(err) => {
                            let _ = conn.execute("ROLLBACK;");
                            return Err(format!(
                                "worker={worker_id} round={round}: non-transient update error: {err}"
                            ));
                        }
                    }

                    match conn.execute("COMMIT;") {
                        Ok(_) => {
                            stats.committed += 1;
                            break;
                        }
                        Err(err) if err.is_transient() => {
                            retries_this_round += 1;
                            stats.retries += 1;
                            let _ = conn.execute("ROLLBACK;");
                            if retries_this_round > MAX_RETRIES_PER_ROUND {
                                return Err(format!(
                                    "worker={worker_id} round={round}: exceeded retry budget at COMMIT ({retries_this_round})"
                                ));
                            }
                            thread::sleep(Duration::from_millis(RETRY_SLEEP_MS));
                        }
                        Err(err) => {
                            let _ = conn.execute("ROLLBACK;");
                            return Err(format!(
                                "worker={worker_id} round={round}: non-transient commit error: {err}"
                            ));
                        }
                    }
                }
            }

            Ok(stats)
        }));
    }

    let mut total_committed = 0_i64;
    let mut total_retries = 0_u64;
    for handle in handles {
        let stats = handle
            .join()
            .expect("worker must not panic")
            .unwrap_or_else(|msg| panic!("{msg}"));
        total_committed += stats.committed;
        total_retries += stats.retries;
    }

    let verifier = fsqlite::Connection::open(&db).expect("open verifier db");
    let final_value = query_single_integer(&verifier, "SELECT v FROM t WHERE id = 1;");
    assert_eq!(
        final_value, total_committed,
        "bd-rjc deterministic hot-row probe mismatch: final_value={final_value} expected_committed={total_committed} retries={total_retries}"
    );

    let integrity = query_single_text(&verifier, "PRAGMA integrity_check;");
    assert_eq!(
        integrity, "ok",
        "integrity_check failed after hot-row probe: {integrity}"
    );

    let schema_rows = verifier
        .query(
            "SELECT type, name, rootpage \
             FROM sqlite_master \
             WHERE type IN ('table', 'index') \
             ORDER BY name;",
        )
        .expect("query sqlite_master rootpages");
    assert!(
        !schema_rows.is_empty(),
        "expected at least one schema row in sqlite_master"
    );

    let mut saw_table_t = false;
    for row in &schema_rows {
        let obj_type = match row.get(0) {
            Some(SqliteValue::Text(value)) => value.clone(),
            Some(other) => panic!("expected text sqlite_master.type, got {other:?}"),
            None => panic!("sqlite_master row missing type"),
        };
        let name = match row.get(1) {
            Some(SqliteValue::Text(value)) => value.clone(),
            Some(other) => panic!("expected text sqlite_master.name, got {other:?}"),
            None => panic!("sqlite_master row missing name"),
        };
        let rootpage = match row.get(2) {
            Some(SqliteValue::Integer(value)) => *value,
            Some(other) => panic!("expected integer sqlite_master.rootpage, got {other:?}"),
            None => panic!("sqlite_master row missing rootpage"),
        };
        assert!(
            rootpage > 0,
            "invalid sqlite_master rootpage: type={obj_type} name={name} rootpage={rootpage}"
        );
        if obj_type == "table" && name == "t" {
            saw_table_t = true;
        }
    }
    assert!(saw_table_t, "sqlite_master missing table entry for 't'");
}

fn query_single_integer(conn: &fsqlite::Connection, sql: &str) -> i64 {
    let row = conn.query_row(sql).expect("query integer row");
    match row.get(0) {
        Some(SqliteValue::Integer(value)) => *value,
        Some(other) => panic!("expected integer row value, got {other:?} for `{sql}`"),
        None => panic!("missing integer row value for `{sql}`"),
    }
}

fn query_single_text(conn: &fsqlite::Connection, sql: &str) -> String {
    let row = conn.query_row(sql).expect("query text row");
    match row.get(0) {
        Some(SqliteValue::Text(value)) => value.clone(),
        Some(other) => panic!("expected text row value, got {other:?} for `{sql}`"),
        None => panic!("missing text row value for `{sql}`"),
    }
}
