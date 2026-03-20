//! Benchmark: Real persistent concurrent-writer throughput.
//!
//! Bead: bd-l9k8e.8 (C8)
//!
//! THIS IS THE ONLY BENCHMARK THAT MATTERS.
//!
//! FrankenSQLite's thesis: page-level MVCC enables concurrent writers where
//! SQLite serializes them.  This benchmark measures:
//!
//! - N writer threads (2, 4, 8, 16)
//! - Each writer INSERTs into a DIFFERENT table (guaranteeing different pages)
//! - File-backed database with WAL mode
//! - Prepared statements on both sides
//!
//! Success criterion: FrankenSQLite shows >1.5x throughput over SQLite at N>=4
//! writers for non-conflicting workloads.  Theoretical improvement is Nx.
//!
//! Metrics captured:
//! - Wall-clock throughput (ops/sec) at each thread count
//! - Per-thread commit latency histogram (p50, p95, p99, max)
//! - Conflict/retry count (SQLITE_BUSY retries for C SQLite)
//!
//! Known issues (2026-03-20):
//! - At 16 threads, database corruption can occur (page 0x00 type flag)
//! - FrankenSQLite p50 latency degrades at 8+ threads (internal contention)
//! - BusySnapshot conflicts appear even with separate tables (shared pages?)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use fsqlite::{FrankenError, SqliteValue};

const ROWS_PER_THREAD: i64 = 1000;
/// Maximum retries before giving up on a transaction (applies to both engines).
const MAX_TXN_RETRIES: u32 = 100;

// ─── PRAGMA helpers ─────────────────────────────────────────────────────

fn run_fsqlite_pragma(conn: &fsqlite::Connection, pragma: &str) {
    conn.execute(pragma)
        .unwrap_or_else(|error| panic!("failed to execute benchmark pragma `{pragma}`: {error:?}"));
}

fn apply_setup_pragmas_fsqlite(conn: &fsqlite::Connection) {
    for pragma in [
        "PRAGMA page_size = 4096;",
        "PRAGMA journal_mode = WAL;",
        "PRAGMA synchronous = NORMAL;",
        "PRAGMA cache_size = -64000;",
        "PRAGMA fsqlite.concurrent_mode = ON;",
    ] {
        run_fsqlite_pragma(conn, pragma);
    }
}

fn apply_session_pragmas_fsqlite(conn: &fsqlite::Connection) {
    for pragma in [
        "PRAGMA journal_mode = WAL;",
        "PRAGMA synchronous = NORMAL;",
        "PRAGMA cache_size = -64000;",
        "PRAGMA fsqlite.concurrent_mode = ON;",
    ] {
        run_fsqlite_pragma(conn, pragma);
    }
}

fn is_retryable_fsqlite_error(error: &FrankenError) -> bool {
    matches!(
        error,
        FrankenError::Busy | FrankenError::BusyRecovery | FrankenError::BusySnapshot { .. }
    )
}

fn is_duplicate_insert_after_retry(error: &FrankenError) -> bool {
    // Check for proper constraint errors
    if matches!(
        error,
        FrankenError::PrimaryKeyViolation | FrankenError::UniqueViolation { .. }
    ) {
        return true;
    }
    // Also check for VDBE constraint errors (code 19) wrapped as Internal
    if let FrankenError::Internal(msg) = error {
        if msg.contains("code 19:") && msg.contains("PRIMARY KEY") {
            return true;
        }
        if msg.contains("code 19:") && msg.contains("UNIQUE") {
            return true;
        }
    }
    false
}

fn is_corruption_error(error: &FrankenError) -> bool {
    matches!(
        error,
        FrankenError::DatabaseCorrupt { .. } | FrankenError::WalCorrupt { .. }
    )
}

fn create_table_sql(table_id: usize) -> String {
    format!(
        "CREATE TABLE IF NOT EXISTS bench_{table_id} (id INTEGER PRIMARY KEY, name TEXT, score INTEGER);"
    )
}

fn insert_sql(table_id: usize) -> String {
    format!("INSERT INTO bench_{table_id} VALUES (?1, ('t' || ?1), (?1 * 7));")
}

fn criterion_config() -> Criterion {
    Criterion::default().configure_from_args()
}

/// Compute percentiles from a sorted slice of latencies.
fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ─── C SQLite concurrent writers (file-backed WAL) ──────────────────────

fn bench_concurrent_csqlite_persistent(c: &mut Criterion, n_threads: usize, label: &str) {
    #[allow(clippy::cast_possible_wrap)]
    let total_rows = n_threads as u64 * ROWS_PER_THREAD as u64;
    let mut group = c.benchmark_group(label);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(45));
    group.throughput(Throughput::Elements(total_rows));

    group.bench_function("csqlite_concurrent_persistent", |b| {
        b.iter_batched(
            || {
                let tmp = tempfile::NamedTempFile::new().unwrap();
                let path = tmp.path().to_str().unwrap().to_owned();
                {
                    let setup = rusqlite::Connection::open(&path).unwrap();
                    setup
                        .execute_batch(
                            "PRAGMA page_size = 4096;\
                             PRAGMA journal_mode = WAL;\
                             PRAGMA synchronous = NORMAL;\
                             PRAGMA cache_size = -64000;",
                        )
                        .unwrap();
                    // Create separate tables for each thread
                    for tid in 0..n_threads {
                        setup.execute_batch(&create_table_sql(tid)).unwrap();
                    }
                }
                let retry_count = Arc::new(AtomicU64::new(0));
                (tmp, path, retry_count)
            },
            |(_tmp, path, retry_count)| {
                let barrier = Arc::new(Barrier::new(n_threads));
                let latencies: Arc<Vec<std::sync::Mutex<Vec<Duration>>>> = Arc::new(
                    (0..n_threads)
                        .map(|_| std::sync::Mutex::new(Vec::with_capacity(ROWS_PER_THREAD as usize)))
                        .collect(),
                );

                let handles: Vec<_> = (0..n_threads)
                    .map(|tid| {
                        let p = path.clone();
                        let bar = barrier.clone();
                        let retries = retry_count.clone();
                        let lat = latencies.clone();
                        thread::spawn(move || {
                            let conn = rusqlite::Connection::open(&p).unwrap();
                            conn.execute_batch(
                                "PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;",
                            )
                            .unwrap();
                            let insert_stmt = insert_sql(tid);
                            let mut stmt = conn.prepare(&insert_stmt).unwrap();
                            bar.wait();

                            // Each row is its own transaction for realistic commit latency
                            for i in 0..ROWS_PER_THREAD {
                                let start = Instant::now();
                                let mut begin_retries = 0u32;
                                loop {
                                    match conn.execute_batch("BEGIN IMMEDIATE") {
                                        Ok(()) => break,
                                        Err(e) => {
                                            let msg = e.to_string();
                                            if msg.contains("BUSY") || msg.contains("locked") {
                                                retries.fetch_add(1, Ordering::Relaxed);
                                                begin_retries += 1;
                                                if begin_retries >= MAX_TXN_RETRIES {
                                                    panic!("BEGIN failed after {MAX_TXN_RETRIES} retries: {e}");
                                                }
                                                std::thread::sleep(Duration::from_micros(100));
                                            } else {
                                                panic!("BEGIN failed: {e}");
                                            }
                                        }
                                    }
                                }
                                stmt.execute(rusqlite::params![i]).unwrap();
                                let mut commit_retries = 0u32;
                                loop {
                                    match conn.execute_batch("COMMIT") {
                                        Ok(()) => break,
                                        Err(e) => {
                                            let msg = e.to_string();
                                            if msg.contains("BUSY") || msg.contains("locked") {
                                                retries.fetch_add(1, Ordering::Relaxed);
                                                commit_retries += 1;
                                                if commit_retries >= MAX_TXN_RETRIES {
                                                    panic!("COMMIT failed after {MAX_TXN_RETRIES} retries: {e}");
                                                }
                                                std::thread::sleep(Duration::from_micros(100));
                                            } else {
                                                panic!("COMMIT failed: {e}");
                                            }
                                        }
                                    }
                                }
                                let elapsed = start.elapsed();
                                lat[tid].lock().unwrap().push(elapsed);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                // Report metrics
                let total_retries = retry_count.load(Ordering::Relaxed);
                let mut all_latencies: Vec<Duration> = latencies
                    .iter()
                    .flat_map(|m| m.lock().unwrap().clone())
                    .collect();
                all_latencies.sort();

                let p50 = percentile(&all_latencies, 50.0);
                let p95 = percentile(&all_latencies, 95.0);
                let p99 = percentile(&all_latencies, 99.0);
                let max = all_latencies.last().copied().unwrap_or(Duration::ZERO);

                eprintln!(
                    "[C SQLite {n_threads}t] retries={total_retries}, p50={:?}, p95={:?}, p99={:?}, max={:?}",
                    p50, p95, p99, max
                );
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // FrankenSQLite with real concurrent writers
    group.bench_function("frankensqlite_concurrent_persistent", |b| {
        b.iter_batched(
            || {
                let tmp = tempfile::NamedTempFile::new().unwrap();
                let path = tmp.path().to_str().unwrap().to_owned();
                {
                    // Setup: create tables using a single connection
                    let setup = fsqlite::Connection::open(&path).unwrap();
                    apply_setup_pragmas_fsqlite(&setup);
                    for tid in 0..n_threads {
                        setup.execute(&create_table_sql(tid)).unwrap();
                    }
                }
                let conflict_count = Arc::new(AtomicU64::new(0));
                (tmp, path, conflict_count)
            },
            |(_tmp, path, conflict_count)| {
                let barrier = Arc::new(Barrier::new(n_threads));
                let latencies: Arc<Vec<std::sync::Mutex<Vec<Duration>>>> = Arc::new(
                    (0..n_threads)
                        .map(|_| std::sync::Mutex::new(Vec::with_capacity(ROWS_PER_THREAD as usize)))
                        .collect(),
                );

                let handles: Vec<_> = (0..n_threads)
                    .map(|tid| {
                        let p = path.clone();
                        let bar = barrier.clone();
                        let conflicts = conflict_count.clone();
                        let lat = latencies.clone();
                        thread::spawn(move || {
                            let conn = fsqlite::Connection::open(&p).unwrap();
                            apply_session_pragmas_fsqlite(&conn);
                            let insert_stmt = insert_sql(tid);
                            let stmt = conn.prepare(&insert_stmt).unwrap();
                            bar.wait();

                            for i in 0..ROWS_PER_THREAD {
                                // Each thread writes to its own table, so row IDs can match
                                // the SQLite side exactly without cross-thread collisions.
                                let row_id = i;
                                let start = Instant::now();
                                let mut retry_count = 0u32;

                                'txn: loop {
                                    // BEGIN CONCURRENT with retry
                                    loop {
                                        match conn.execute("BEGIN CONCURRENT") {
                                            Ok(_) => break,
                                            Err(e) => {
                                                if is_retryable_fsqlite_error(&e) {
                                                    conflicts.fetch_add(1, Ordering::Relaxed);
                                                    retry_count += 1;
                                                    if retry_count >= MAX_TXN_RETRIES {
                                                        panic!(
                                                            "BEGIN CONCURRENT failed after {MAX_TXN_RETRIES} retries: {e:?}"
                                                        );
                                                    }
                                                    std::thread::sleep(Duration::from_micros(
                                                        100 * u64::from(retry_count),
                                                    ));
                                                } else {
                                                    panic!("BEGIN CONCURRENT failed: {e:?}");
                                                }
                                            }
                                        }
                                    }

                                    // INSERT
                                    if let Err(e) = stmt.execute_with_params(&[SqliteValue::Integer(row_id)]) {
                                        if is_duplicate_insert_after_retry(&e) {
                                            // Row already exists (from previous retry that actually committed)
                                            let _ = conn.execute("ROLLBACK");
                                            break 'txn;
                                        }
                                        if is_retryable_fsqlite_error(&e)
                                            || matches!(e, FrankenError::SerializationFailure { .. })
                                        {
                                            // Snapshot conflict — rollback and retry
                                            conflicts.fetch_add(1, Ordering::Relaxed);
                                            let _ = conn.execute("ROLLBACK");
                                            retry_count += 1;
                                            if retry_count >= MAX_TXN_RETRIES {
                                                panic!("INSERT failed after {MAX_TXN_RETRIES} retries: {e:?}");
                                            }
                                            std::thread::sleep(Duration::from_micros(100 * u64::from(retry_count)));
                                            continue 'txn;
                                        }
                                        if is_corruption_error(&e) {
                                            let _ = conn.execute("ROLLBACK");
                                            panic!("CORRUPTION DETECTED: {e:?}");
                                        }
                                        panic!("INSERT failed: {e:?}");
                                    }

                                    // COMMIT with retry
                                    match conn.execute("COMMIT") {
                                        Ok(_) => break 'txn,
                                        Err(e) => {
                                            if is_retryable_fsqlite_error(&e)
                                                || matches!(e, FrankenError::SerializationFailure { .. })
                                            {
                                                conflicts.fetch_add(1, Ordering::Relaxed);
                                                let _ = conn.execute("ROLLBACK");
                                                retry_count += 1;
                                                if retry_count >= MAX_TXN_RETRIES {
                                                    panic!("COMMIT failed after {MAX_TXN_RETRIES} retries: {e:?}");
                                                }
                                                std::thread::sleep(Duration::from_micros(100 * u64::from(retry_count)));
                                                // Loop back to BEGIN CONCURRENT
                                            } else {
                                                panic!("COMMIT failed: {e:?}");
                                            }
                                        }
                                    }
                                }

                                let elapsed = start.elapsed();
                                lat[tid].lock().unwrap().push(elapsed);
                            }
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }

                // Report metrics
                let total_conflicts = conflict_count.load(Ordering::Relaxed);
                let mut all_latencies: Vec<Duration> = latencies
                    .iter()
                    .flat_map(|m| m.lock().unwrap().clone())
                    .collect();
                all_latencies.sort();

                let p50 = percentile(&all_latencies, 50.0);
                let p95 = percentile(&all_latencies, 95.0);
                let p99 = percentile(&all_latencies, 99.0);
                let max = all_latencies.last().copied().unwrap_or(Duration::ZERO);

                eprintln!(
                    "[FrankenSQLite {n_threads}t] conflicts={total_conflicts}, p50={:?}, p95={:?}, p99={:?}, max={:?}",
                    p50, p95, p99, max
                );
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_persistent_2t(c: &mut Criterion) {
    bench_concurrent_csqlite_persistent(c, 2, "persistent_concurrent_write_2t");
}

fn bench_persistent_4t(c: &mut Criterion) {
    bench_concurrent_csqlite_persistent(c, 4, "persistent_concurrent_write_4t");
}

fn bench_persistent_8t(c: &mut Criterion) {
    bench_concurrent_csqlite_persistent(c, 8, "persistent_concurrent_write_8t");
}

fn bench_persistent_16t(c: &mut Criterion) {
    bench_concurrent_csqlite_persistent(c, 16, "persistent_concurrent_write_16t");
}

criterion_group!(
    name = persistent_concurrent_write;
    config = criterion_config();
    targets = bench_persistent_2t, bench_persistent_4t, bench_persistent_8t, bench_persistent_16t
);
criterion_main!(persistent_concurrent_write);
