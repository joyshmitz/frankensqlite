//! Benchmark: concurrent write throughput (2/4/8 threads).
//!
//! Bead: bd-3rze
//!
//! Measures aggregate INSERT throughput when multiple threads write
//! concurrently.  C SQLite uses WAL mode with `busy_timeout` for write
//! serialisation; FrankenSQLite runs the equivalent operations sequentially
//! (MVCC concurrent writer path is not yet wired to persistence).
//!
//! Thread counts: 2, 4, 8.  (16 is omitted because in-memory C SQLite
//! doesn't benefit from higher thread counts — the `WAL_WRITE_LOCK`
//! serialises writers regardless.)
//!
//! Each thread inserts into a non-overlapping key range so there is no
//! primary-key contention, only write-lock contention.

use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

const ROWS_PER_THREAD: i64 = 1000;
const RANGE_SIZE: i64 = 100_000;

// ─── PRAGMA helpers ─────────────────────────────────────────────────────

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

const CREATE_TABLE: &str = "CREATE TABLE bench (id INTEGER PRIMARY KEY, name TEXT, score INTEGER);";

fn criterion_config() -> Criterion {
    Criterion::default().configure_from_args()
}

// ─── C SQLite concurrent writers (file-backed WAL) ──────────────────────

fn bench_concurrent_csqlite(c: &mut Criterion, n_threads: usize, label: &str) {
    #[allow(clippy::cast_possible_wrap)]
    let total_rows = n_threads as u64 * ROWS_PER_THREAD as u64;
    let mut group = c.benchmark_group(label);
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    group.throughput(Throughput::Elements(total_rows));

    group.bench_function("csqlite_concurrent", |b| {
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
                    setup.execute_batch(CREATE_TABLE).unwrap();
                }
                (tmp, path)
            },
            |(_tmp, path)| {
                let barrier = Arc::new(Barrier::new(n_threads));
                let handles: Vec<_> = (0..n_threads)
                    .map(|tid| {
                        let p = path.clone();
                        let bar = barrier.clone();
                        thread::spawn(move || {
                            let conn = rusqlite::Connection::open(&p).unwrap();
                            conn.execute_batch(
                                "PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;",
                            )
                            .unwrap();
                            bar.wait();

                            conn.execute_batch("BEGIN").unwrap();
                            #[allow(clippy::cast_possible_wrap)]
                            let base = tid as i64 * RANGE_SIZE;
                            let mut stmt = conn
                                .prepare("INSERT INTO bench VALUES (?1, ('t' || ?1), (?1 * 7))")
                                .unwrap();
                            for i in 0..ROWS_PER_THREAD {
                                stmt.execute(rusqlite::params![base + i]).unwrap();
                            }
                            conn.execute_batch("COMMIT").unwrap();
                        })
                    })
                    .collect();

                for h in handles {
                    h.join().unwrap();
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });

    // FrankenSQLite: sequential equivalent (same total work).
    group.bench_function("frankensqlite_sequential", |b| {
        b.iter_batched(
            || {
                let conn = fsqlite::Connection::open(":memory:").unwrap();
                apply_pragmas_fsqlite(&conn);
                conn.execute(CREATE_TABLE).unwrap();
                conn
            },
            |conn| {
                for tid in 0..n_threads {
                    conn.execute("BEGIN").unwrap();
                    #[allow(clippy::cast_possible_wrap)]
                    let base = tid as i64 * RANGE_SIZE;
                    for i in 0..ROWS_PER_THREAD {
                        let id = base + i;
                        conn.execute(&format!(
                            "INSERT INTO bench VALUES ({id}, 't{id}', {})",
                            id * 7,
                        ))
                        .unwrap();
                    }
                    conn.execute("COMMIT").unwrap();
                }
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_concurrent_2(c: &mut Criterion) {
    bench_concurrent_csqlite(c, 2, "concurrent_write_2t");
}

fn bench_concurrent_4(c: &mut Criterion) {
    bench_concurrent_csqlite(c, 4, "concurrent_write_4t");
}

fn bench_concurrent_8(c: &mut Criterion) {
    bench_concurrent_csqlite(c, 8, "concurrent_write_8t");
}

criterion_group!(
    name = concurrent_write;
    config = criterion_config();
    targets = bench_concurrent_2, bench_concurrent_4, bench_concurrent_8
);
criterion_main!(concurrent_write);
