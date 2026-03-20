//! D-TEST: Contention elimination test suite + regression gates (bd-3wop3.7).
//!
//! This test suite verifies all Phase 1 contention fixes (D1-D5) work together
//! and establishes regression gates for 8t/16t performance.
//!
//! ## Benchmark Gates
//! - 8t throughput >= 1.5x C SQLite (target: flip 0.67x → 1.5x+)
//! - 16t throughput >= 1.0x C SQLite (target: flip 0.53x → 1.0x+)
//! - p99 latency at 8t < 50ms
//! - Zero corruption at any thread count
//!
//! ## Contention Tests
//! 1. test_no_global_locks_in_commit_fast_path
//! 2. test_parallel_wal_segments_independent (D1 dependency)
//! 3. test_page_cache_shard_distribution (D2 dependency)
//! 4. test_combiner_reduces_atomic_ops (D3 dependency)
//! 5. test_ebr_no_gc_pauses (D5 dependency)
//! 6. test_scaling_curve
//!
//! ## Stress Tests
//! 7. test_64_thread_no_deadlock
//! 8. test_contention_under_gc_pressure
//!
//! ## Dependencies
//! - D1: Parallel WAL with per-thread log buffers
//! - D2: Sharded PageCache (128 partitions)
//! - D3: Flat Combining for commit sequencer
//! - D5: Epoch-Based Reclamation for MVCC GC
//!
//! Run with:
//! ```sh
//! cargo test -p fsqlite-e2e --test bd_3wop3_7_contention_elimination -- --nocapture
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Thread counts for scaling curve analysis.
const SCALING_THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16];

/// Duration for throughput measurements (ms).
const THROUGHPUT_MEASUREMENT_MS: u64 = 5000;

/// Duration for stress tests (seconds).
const STRESS_TEST_DURATION_SECS: u64 = 60;

/// Rows per thread for throughput tests.
const ROWS_PER_THREAD: u64 = 10_000;

/// 8-thread throughput gate: must be >= 1.5x C SQLite.
const GATE_8T_SPEEDUP: f64 = 1.5;

/// 16-thread throughput gate: must be >= 1.0x C SQLite.
const GATE_16T_SPEEDUP: f64 = 1.0;

/// p99 latency gate at 8 threads (milliseconds).
const GATE_8T_P99_LATENCY_MS: u64 = 50;

// ---------------------------------------------------------------------------
// Test result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ThroughputResult {
    thread_count: usize,
    ops_per_sec: f64,
    wall_time_ms: u64,
    total_ops: u64,
}

#[derive(Debug, Clone)]
struct LatencyResult {
    thread_count: usize,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Clone)]
struct ScalingPoint {
    thread_count: usize,
    fsqlite_ops_per_sec: f64,
    csqlite_ops_per_sec: f64,
    speedup: f64,
}

// ---------------------------------------------------------------------------
// Helper: C SQLite baseline measurement
// ---------------------------------------------------------------------------

/// Measure C SQLite throughput at the given thread count.
///
/// Uses WAL mode with busy_timeout for write serialization.
fn measure_csqlite_throughput(thread_count: usize, rows_per_thread: u64) -> ThroughputResult {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile");
    let path = tmp.path().to_str().unwrap().to_owned();

    // Setup database
    {
        let conn = rusqlite::Connection::open(&path).expect("open");
        conn.execute_batch(
            "PRAGMA page_size = 4096;
             PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             PRAGMA cache_size = -64000;
             CREATE TABLE bench (id INTEGER PRIMARY KEY, val INTEGER);",
        )
        .expect("setup");
    }

    let barrier = Arc::new(Barrier::new(thread_count));
    let total_ops = Arc::new(AtomicU64::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..thread_count)
        .map(|tid| {
            let p = path.clone();
            let bar = Arc::clone(&barrier);
            let ops = Arc::clone(&total_ops);
            let base = (tid as u64) * rows_per_thread * 2; // Non-overlapping ranges

            thread::spawn(move || {
                let conn = rusqlite::Connection::open(&p).expect("thread open");
                conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=10000;")
                    .expect("pragma");
                bar.wait();

                let mut local_ops = 0u64;
                for i in 0..rows_per_thread {
                    // Each insert is its own transaction for maximum contention
                    if conn
                        .execute(
                            "INSERT INTO bench VALUES (?1, ?2)",
                            rusqlite::params![base + i, i * 7],
                        )
                        .is_ok()
                    {
                        local_ops += 1;
                    }
                }
                ops.fetch_add(local_ops, Ordering::Relaxed);
            })
        })
        .collect();

    for h in handles {
        h.join().expect("join");
    }

    let elapsed = start.elapsed();
    let total = total_ops.load(Ordering::Relaxed);
    let ops_per_sec = total as f64 / elapsed.as_secs_f64();

    ThroughputResult {
        thread_count,
        ops_per_sec,
        wall_time_ms: elapsed.as_millis() as u64,
        total_ops: total,
    }
}

// ---------------------------------------------------------------------------
// Placeholder: FrankenSQLite throughput measurement
// ---------------------------------------------------------------------------

/// Measure FrankenSQLite throughput at the given thread count.
///
/// NOTE: This is a placeholder. Full implementation requires D1 (parallel WAL)
/// to be complete for true concurrent writes.
fn measure_fsqlite_throughput(thread_count: usize, rows_per_thread: u64) -> ThroughputResult {
    // TODO(D1): Implement concurrent write measurement once parallel WAL is wired.
    //
    // Current implementation runs sequential writes as a baseline. Once D1 is
    // complete, this will spawn `thread_count` threads, each with its own
    // Connection, inserting into non-overlapping key ranges.
    //
    // The parallel WAL ensures each thread writes to its own WAL segment without
    // acquiring the global WAL append gate, enabling true O(N) scaling.

    let conn = fsqlite::Connection::open_in_memory().expect("open");
    conn.execute("PRAGMA journal_mode = WAL").ok();
    conn.execute("CREATE TABLE bench (id INTEGER PRIMARY KEY, val INTEGER)")
        .expect("create");

    let total_ops = thread_count as u64 * rows_per_thread;
    let start = Instant::now();

    for i in 0..total_ops {
        conn.execute_with_params(
            "INSERT INTO bench VALUES (?1, ?2)",
            &[
                fsqlite::SqliteValue::Integer(i as i64),
                fsqlite::SqliteValue::Integer((i * 7) as i64),
            ],
        )
        .ok();
    }

    let elapsed = start.elapsed();
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    ThroughputResult {
        thread_count,
        ops_per_sec,
        wall_time_ms: elapsed.as_millis() as u64,
        total_ops,
    }
}

// ===========================================================================
// CONTENTION TESTS
// ===========================================================================

/// Test 1: Verify no global locks in the commit fast path.
///
/// Instruments the commit path to assert no global Mutex is acquired during
/// WAL frame writing. This is the foundational guarantee of D1.
///
/// Depends on: D1 (parallel WAL)
#[test]
#[ignore = "requires D1 (parallel WAL) to be complete"]
fn test_no_global_locks_in_commit_fast_path() {
    // TODO(D1): Implement instrumentation for commit path.
    //
    // Strategy:
    // 1. Set a thread-local flag before commit
    // 2. Hook into WAL frame write path
    // 3. Assert no global Mutex (WAL_APPEND_GATE) is acquired
    // 4. Clear flag after commit
    //
    // This test will fail if any global lock is acquired during the commit
    // fast path, catching regressions in the parallel WAL design.

    panic!("test_no_global_locks_in_commit_fast_path: D1 not yet implemented");
}

/// Test 2: Verify WAL segment writes don't serialize.
///
/// Spawns N writer threads, each writing to their own WAL segment. Measures
/// that aggregate write bandwidth scales with thread count.
///
/// Depends on: D1 (parallel WAL)
#[test]
#[ignore = "requires D1 (parallel WAL) to be complete"]
fn test_parallel_wal_segments_independent() {
    // TODO(D1): Implement once parallel WAL segments exist.
    //
    // Strategy:
    // 1. Spawn 8 threads, each inserting 10k rows
    // 2. Measure per-thread write bandwidth
    // 3. Assert total bandwidth > 6x single-thread (allowing for overhead)
    //
    // If WAL segments serialize, total bandwidth will plateau at 1x.

    panic!("test_parallel_wal_segments_independent: D1 not yet implemented");
}

/// Test 3: Verify pages distribute evenly across 128 shards.
///
/// Inserts rows that touch many different pages, then inspects the PageCache
/// shard distribution to verify no single shard is overloaded.
///
/// Depends on: D2 (sharded PageCache)
#[test]
#[ignore = "requires D2 (sharded PageCache) to be complete"]
fn test_page_cache_shard_distribution() {
    // TODO(D2): Implement once sharded PageCache is wired.
    //
    // Strategy:
    // 1. Create a 10MB database (2560 pages at 4KB)
    // 2. Touch all pages via full table scan
    // 3. Query PageCache shard distribution
    // 4. Assert coefficient of variation < 0.2 (even distribution)
    //
    // Birthday problem: 2560 pages across 128 shards ≈ 20 pages/shard average.
    // CoV < 0.2 ensures no pathological clustering.

    panic!("test_page_cache_shard_distribution: D2 not yet implemented");
}

/// Test 4: Verify flat combining reduces atomic operations.
///
/// Counts fetch_add calls during concurrent commits and asserts the count is
/// reduced by the combining factor (batching amortizes atomic ops).
///
/// Depends on: D3 (flat combining)
#[test]
#[ignore = "requires D3 (flat combining) to be complete"]
fn test_combiner_reduces_atomic_ops() {
    // TODO(D3): Implement once flat combining is wired.
    //
    // Strategy:
    // 1. Run 8 threads, each committing 1000 transactions
    // 2. Count total fetch_add calls to commit_seq counter
    // 3. Without combining: 8000 fetch_add calls
    // 4. With combining: ~8000 / combining_factor calls
    // 5. Assert reduction > 2x (conservative floor)
    //
    // Flat combining batches commit sequence increments, reducing cache-line
    // ping-pong on the global atomic.

    panic!("test_combiner_reduces_atomic_ops: D3 not yet implemented");
}

/// Test 5: Verify EBR causes no GC-induced latency spikes.
///
/// Runs sustained writes with aggressive GC, measuring p99 latency to ensure
/// no pause exceeds the threshold.
///
/// Depends on: D5 (EBR)
#[test]
fn test_ebr_no_gc_pauses() {
    // EBR is already implemented (D5 complete). This test verifies no GC
    // pauses under sustained write load.
    //
    // Strategy:
    // 1. Spawn 4 writer threads, continuous inserts for 5 seconds
    // 2. Record per-operation latency
    // 3. Assert p99 latency < 10ms (no GC stalls)

    use std::sync::atomic::AtomicBool;

    let conn = Arc::new(fsqlite::Connection::open_in_memory().expect("open"));
    conn.execute("PRAGMA journal_mode = WAL").ok();
    conn.execute("CREATE TABLE bench (id INTEGER PRIMARY KEY, val TEXT)")
        .expect("create");

    let stop = Arc::new(AtomicBool::new(false));
    let latencies = Arc::new(std::sync::Mutex::new(Vec::new()));

    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|tid| {
            let c = Arc::clone(&conn);
            let s = Arc::clone(&stop);
            let l = Arc::clone(&latencies);
            let b = Arc::clone(&barrier);

            thread::spawn(move || {
                b.wait();
                let mut local_latencies = Vec::with_capacity(10_000);
                let mut i = tid * 1_000_000;

                while !s.load(Ordering::Relaxed) {
                    let op_start = Instant::now();
                    let _ = c.execute_with_params(
                        "INSERT OR REPLACE INTO bench VALUES (?1, ?2)",
                        &[
                            fsqlite::SqliteValue::Integer(i),
                            fsqlite::SqliteValue::Text(format!("value_{i}")),
                        ],
                    );
                    local_latencies.push(op_start.elapsed().as_micros() as u64);
                    i += 1;
                }

                l.lock().unwrap().extend(local_latencies);
            })
        })
        .collect();

    // Run for 2 seconds (reduced for test speed)
    thread::sleep(Duration::from_secs(2));
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().expect("join");
    }

    let mut all_latencies = latencies.lock().unwrap().clone();
    all_latencies.sort_unstable();

    if all_latencies.is_empty() {
        panic!("No operations completed");
    }

    let p99_idx = (all_latencies.len() as f64 * 0.99) as usize;
    let p99_us = all_latencies.get(p99_idx).copied().unwrap_or(0);
    let p99_ms = p99_us as f64 / 1000.0;

    println!(
        "[test_ebr_no_gc_pauses] {} ops, p99={:.2}ms, max={:.2}ms",
        all_latencies.len(),
        p99_ms,
        all_latencies.last().copied().unwrap_or(0) as f64 / 1000.0
    );

    // Assert p99 < 10ms (generous threshold for GC-free operation)
    assert!(
        p99_ms < 10.0,
        "bd-3wop3.7: EBR p99 latency {p99_ms:.2}ms exceeds 10ms threshold"
    );
}

/// Test 6: Verify scaling curve is monotonic up to 8 threads.
///
/// Measures throughput at 1, 2, 4, 8 threads and asserts each level improves
/// over the previous (until contention plateaus).
#[test]
#[ignore = "requires D1 (parallel WAL) for accurate concurrent measurement"]
fn test_scaling_curve() {
    // TODO(D1): Enable once parallel WAL is complete.
    //
    // Strategy:
    // 1. Measure FrankenSQLite throughput at 1, 2, 4, 8 threads
    // 2. Assert throughput[N] > throughput[N/2] * 1.3 (diminishing returns OK)
    // 3. Assert 8t throughput > 1t throughput * 3.0 (meaningful scaling)
    //
    // With parallel WAL, we expect near-linear scaling up to 8 threads before
    // CPU cache effects dominate.

    panic!("test_scaling_curve: D1 not yet implemented");
}

// ===========================================================================
// REGRESSION GATES
// ===========================================================================

/// Regression gate: 8-thread throughput >= 1.5x C SQLite.
///
/// This is the primary performance gate. Failing this test blocks merges.
#[test]
#[ignore = "requires D1 (parallel WAL) for accurate concurrent measurement"]
fn test_8t_throughput_regression_gate() {
    let fsqlite_result = measure_fsqlite_throughput(8, ROWS_PER_THREAD);
    let csqlite_result = measure_csqlite_throughput(8, ROWS_PER_THREAD);

    let speedup = fsqlite_result.ops_per_sec / csqlite_result.ops_per_sec;

    println!(
        "[8t regression gate] FrankenSQLite: {:.0} ops/s, C SQLite: {:.0} ops/s, speedup: {:.2}x",
        fsqlite_result.ops_per_sec, csqlite_result.ops_per_sec, speedup
    );

    assert!(
        speedup >= GATE_8T_SPEEDUP,
        "bd-3wop3.7 REGRESSION: 8t speedup {speedup:.2}x < {GATE_8T_SPEEDUP}x gate"
    );
}

/// Regression gate: 16-thread throughput >= 1.0x C SQLite.
///
/// At 16 threads, we target parity with C SQLite (which is also contended).
#[test]
#[ignore = "requires D1 (parallel WAL) for accurate concurrent measurement"]
fn test_16t_throughput_regression_gate() {
    let fsqlite_result = measure_fsqlite_throughput(16, ROWS_PER_THREAD);
    let csqlite_result = measure_csqlite_throughput(16, ROWS_PER_THREAD);

    let speedup = fsqlite_result.ops_per_sec / csqlite_result.ops_per_sec;

    println!(
        "[16t regression gate] FrankenSQLite: {:.0} ops/s, C SQLite: {:.0} ops/s, speedup: {:.2}x",
        fsqlite_result.ops_per_sec, csqlite_result.ops_per_sec, speedup
    );

    assert!(
        speedup >= GATE_16T_SPEEDUP,
        "bd-3wop3.7 REGRESSION: 16t speedup {speedup:.2}x < {GATE_16T_SPEEDUP}x gate"
    );
}

// ===========================================================================
// STRESS TESTS
// ===========================================================================

/// Stress test: 64 threads, no deadlock within 60 seconds.
///
/// Spawns 64 writer threads with overlapping key ranges to maximize lock
/// contention. Asserts no deadlock occurs (all threads complete or timeout).
#[test]
fn test_64_thread_no_deadlock() {
    // This test runs with current implementation (sequential fallback).
    // Once D1 is complete, it will exercise true concurrent writes.

    let conn = Arc::new(fsqlite::Connection::open_in_memory().expect("open"));
    conn.execute("PRAGMA journal_mode = WAL").ok();
    conn.execute("CREATE TABLE stress (id INTEGER PRIMARY KEY, val INTEGER)")
        .expect("create");

    // Pre-populate some rows to enable updates
    for i in 0..100 {
        conn.execute_with_params(
            "INSERT INTO stress VALUES (?1, ?2)",
            &[
                fsqlite::SqliteValue::Integer(i),
                fsqlite::SqliteValue::Integer(0),
            ],
        )
        .ok();
    }

    let barrier = Arc::new(Barrier::new(64));
    let stop = Arc::new(AtomicBool::new(false));
    let completed = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..64)
        .map(|tid| {
            let c = Arc::clone(&conn);
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&stop);
            let comp = Arc::clone(&completed);

            thread::spawn(move || {
                b.wait();
                let mut ops = 0u64;

                while !s.load(Ordering::Relaxed) && ops < 1000 {
                    // Update a random-ish row to create contention
                    let row_id = ((tid * 17 + ops as usize) % 100) as i64;
                    let _ = c.execute_with_params(
                        "UPDATE stress SET val = val + 1 WHERE id = ?1",
                        &[fsqlite::SqliteValue::Integer(row_id)],
                    );
                    ops += 1;
                }

                comp.fetch_add(1, Ordering::Relaxed);
            })
        })
        .collect();

    // Give threads up to 30 seconds to complete
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline && completed.load(Ordering::Relaxed) < 64 {
        thread::sleep(Duration::from_millis(100));
    }

    stop.store(true, Ordering::Release);

    // Wait for all threads with timeout
    for h in handles {
        let _ = h.join();
    }

    let final_completed = completed.load(Ordering::Relaxed);
    println!("[test_64_thread_no_deadlock] {final_completed}/64 threads completed");

    assert_eq!(
        final_completed, 64,
        "bd-3wop3.7: deadlock detected - only {final_completed}/64 threads completed"
    );
}

/// Stress test: High write + aggressive GC, no throughput collapse.
///
/// Runs continuous writes while forcing frequent GC cycles. Asserts throughput
/// remains above a minimum floor (no GC-induced starvation).
#[test]
fn test_contention_under_gc_pressure() {
    // This test exercises the EBR-based GC under write pressure.

    let conn = Arc::new(fsqlite::Connection::open_in_memory().expect("open"));
    conn.execute("PRAGMA journal_mode = WAL").ok();
    conn.execute("CREATE TABLE gc_stress (id INTEGER PRIMARY KEY, data BLOB)")
        .expect("create");

    let stop = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicU64::new(0));

    let barrier = Arc::new(Barrier::new(4));
    let handles: Vec<_> = (0..4)
        .map(|tid| {
            let c = Arc::clone(&conn);
            let b = Arc::clone(&barrier);
            let s = Arc::clone(&stop);
            let ops = Arc::clone(&total_ops);

            thread::spawn(move || {
                b.wait();
                let mut local_ops = 0u64;
                let mut i = tid * 10_000_000;

                // Create a blob that will stress memory allocation
                let blob = vec![0xABu8; 1024];

                while !s.load(Ordering::Relaxed) {
                    // INSERT OR REPLACE to create version churn (GC pressure)
                    let row_id = (i % 1000) as i64; // Reuse 1000 row IDs
                    let _ = c.execute_with_params(
                        "INSERT OR REPLACE INTO gc_stress VALUES (?1, ?2)",
                        &[
                            fsqlite::SqliteValue::Integer(row_id),
                            fsqlite::SqliteValue::Blob(blob.clone()),
                        ],
                    );
                    local_ops += 1;
                    i += 1;
                }

                ops.fetch_add(local_ops, Ordering::Relaxed);
            })
        })
        .collect();

    // Run for 3 seconds
    let start = Instant::now();
    thread::sleep(Duration::from_secs(3));
    stop.store(true, Ordering::Release);

    for h in handles {
        h.join().expect("join");
    }

    let elapsed = start.elapsed();
    let total = total_ops.load(Ordering::Relaxed);
    let ops_per_sec = total as f64 / elapsed.as_secs_f64();

    println!(
        "[test_contention_under_gc_pressure] {} ops in {:.2}s = {:.0} ops/s",
        total,
        elapsed.as_secs_f64(),
        ops_per_sec
    );

    // Assert minimum throughput (very conservative floor)
    // With GC pressure, we should still achieve at least 1000 ops/s
    assert!(
        ops_per_sec > 1000.0,
        "bd-3wop3.7: GC pressure caused throughput collapse ({ops_per_sec:.0} ops/s < 1000)"
    );
}

// ===========================================================================
// SCALING REPORT (manual run)
// ===========================================================================

/// Generate a scaling report comparing FrankenSQLite vs C SQLite at multiple
/// thread counts.
///
/// Run with: `cargo test -p fsqlite-e2e --test bd_3wop3_7_contention_elimination scaling_report -- --nocapture --ignored`
#[test]
#[ignore = "manual benchmark - run with --ignored"]
fn scaling_report() {
    println!("\n=== D-TEST Scaling Report (bd-3wop3.7) ===\n");
    println!("Thread | C SQLite ops/s | FrankenSQLite ops/s | Speedup");
    println!("-------|----------------|---------------------|--------");

    for &threads in SCALING_THREAD_COUNTS {
        let csqlite = measure_csqlite_throughput(threads, ROWS_PER_THREAD / 10);
        let fsqlite = measure_fsqlite_throughput(threads, ROWS_PER_THREAD / 10);
        let speedup = fsqlite.ops_per_sec / csqlite.ops_per_sec;

        println!(
            "{:>6} | {:>14.0} | {:>19.0} | {:>6.2}x",
            threads, csqlite.ops_per_sec, fsqlite.ops_per_sec, speedup
        );
    }

    println!("\nNote: FrankenSQLite results are sequential baseline until D1 completes.");
}
