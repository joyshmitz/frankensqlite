# Troubleshooting & Extension Guide

Common failure modes, debugging steps, and how to extend the E2E harness with
new fixtures and workload presets.

For running commands see [HARNESS.md](HARNESS.md).  For benchmark methodology
see [METHODOLOGY.md](METHODOLOGY.md).  For fixture ingestion details see
[sample_sqlite_db_files/FIXTURES.md](../../sample_sqlite_db_files/FIXTURES.md).

---

## Common Failures

### sqlite3 Missing or Wrong Version

**Symptom:** `corpus scan` fails, or `corpus verify` cannot open golden files.

**Cause:** The system `sqlite3` binary is too old (pre-3.35.0) or missing.
The harness requires a `sqlite3` that supports `.backup` and
`PRAGMA wal_checkpoint(TRUNCATE)`.

**Fix:**

```bash
sqlite3 --version
# Minimum: 3.35.0 (2021-03-12)
```

If outdated, install a newer version:

```bash
# Ubuntu/Debian
sudo apt install sqlite3

# macOS
brew install sqlite
```

The Rust harness itself uses rusqlite (statically linked) and does not depend
on the system `sqlite3` at runtime.  However, the `corpus import` and snapshot
commands shell out to `sqlite3` for the `.backup` API.

### rusqlite Version Mismatch

**Symptom:** Golden copies fail `PRAGMA integrity_check` after a rusqlite
upgrade.

**Cause:** Different rusqlite/libsqlite3-sys versions use different bundled
SQLite libraries.  A golden copy produced by one version may report different
page layouts or freelist counts.

**Fix:**
- Re-verify checksums: `cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus verify`
- If mismatches appear, re-ingest affected fixtures (see "Adding a New Fixture" below)
- Pin rusqlite version in `Cargo.toml` to avoid surprise upgrades

### Huge Databases and Timeouts

**Symptom:** `run` or `bench` hangs or takes unexpectedly long on large
fixtures (>100 MB).

**Cause:** Each iteration copies the golden DB to a temp dir.  Large files
amplify copy time and page cache pressure.

**Mitigations:**
- Reduce `--repeat` count for large fixtures
- Use `--concurrency 1` first to establish a single-threaded baseline
- Increase `cache_size` in `HarnessSettings` if the fixture is much larger
  than the default 2 MB cache
- For benchmarks, prefer medium-sized fixtures (1-50 MB) that fit comfortably
  in the page cache

### WAL vs Rollback Journal Quirks

**Symptom:** `PRAGMA journal_mode=wal` returns `"delete"` or `"memory"`
instead of `"wal"`.

**Cause:** In-memory databases (`:memory:`) always use the `"memory"` journal
mode.  Some golden fixtures may have been captured in rollback journal mode.

**Details:**
- WAL mode requires a file-backed database.  In-memory tests skip WAL.
- The fairness verifier (`fairness.rs`) explicitly allows `"memory"` as a
  valid journal_mode for in-memory connections.
- If a golden fixture was captured with `journal_mode=delete`, setting WAL on
  the working copy may silently fail if the filesystem does not support
  shared-memory files (e.g., some NFS mounts).

**Fix:**
- Use file-backed temp databases for WAL testing (the harness does this by
  default)
- Re-capture fixtures with `journal_mode=wal` if WAL behavior is the test
  target
- Check for filesystem compatibility: WAL requires POSIX shared memory
  (`mmap`, `/dev/shm`)

### SQLITE_BUSY / Lock Contention

**Symptom:** High retry counts, transaction aborts, or `exceeded
max_busy_retries` errors in multi-writer workloads.

**Cause:** Both C SQLite and FrankenSQLite serialize writers through WAL locks.
High concurrency with hot-page workloads can saturate the lock.

**Tuning:**
- The default `max_busy_retries` is 10,000 with exponential backoff (1ms base,
  250ms cap).  Increase via `SqliteExecConfig` for extreme contention.
- Use `--mvcc` with FrankenSQLite to test MVCC concurrent-writer mode, which
  avoids WAL_WRITE_LOCK serialization.
- Use `--concurrency 1,2,4,8` to find the scaling ceiling for each workload.

### Fairness Verification Failures

**Symptom:** `benchmark fairness check failed` with PRAGMA mismatches.

**Cause:** The `fairness.rs` verifier checks that both engines run with
identical PRAGMAs.  If a PRAGMA is set after the fairness check, or if
FrankenSQLite does not implement a particular PRAGMA, mismatches appear.

**Fix:**
- Unimplemented PRAGMAs (returning empty responses) are skipped automatically
- If FrankenSQLite returns text names instead of numeric codes (e.g., `"normal"`
  instead of `"1"` for `synchronous`), the normalizer handles this
- Check `fairness.rs:BENCHMARK_PRAGMAS` for the canonical expected values

### Corruption Injector Refuses to Run

**Symptom:** `CorruptionInjector::new` returns an error about `golden/` path.

**Cause:** The injector has a safety guard that refuses to corrupt files inside
a `golden/` directory.

**Fix:** Always work on copies in `working/` or temp directories.  The harness
does this automatically.  If testing manually:

```bash
cp sample_sqlite_db_files/golden/mydb.db /tmp/test.db
# Now inject corruption into /tmp/test.db
```

---

## Adding a New Fixture

### Step-by-Step

1. **Snapshot the source** using SQLite's backup API:

   ```bash
   sqlite3 "/dp/project/.beads/beads.db" \
     ".backup 'sample_sqlite_db_files/golden/project_name.db'"
   ```

2. **Verify integrity**:

   ```bash
   sqlite3 "sample_sqlite_db_files/golden/project_name.db" \
     "PRAGMA integrity_check;"
   # Must print: ok
   ```

3. **Record the SHA-256 checksum**:

   ```bash
   cd sample_sqlite_db_files/golden
   sha256sum project_name.db | sort -k2 >> ../checksums.sha256
   ```

4. **Generate metadata** (recommended):

   ```bash
   cargo run -p fsqlite-e2e --bin profile-db -- --db project_name
   ```

   This writes `sample_sqlite_db_files/metadata/project_name.json` with
   schema, page stats, and table row counts.

5. **Verify the corpus**:

   ```bash
   cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus verify
   ```

6. **Test against the new fixture**:

   ```bash
   cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
     --engine sqlite3 --db project_name --workload commutative_inserts
   ```

### What Makes a Good Fixture

- **Size:** 1-50 MB for routine benchmarks.  Larger fixtures (100+ MB) are
  useful for stress testing but slow down iteration.
- **Schema complexity:** Multiple tables with indexes, triggers, and views
  exercise more code paths than a single flat table.
- **Realistic data:** Fixtures derived from real databases catch edge cases
  that synthetic schemas miss (mixed collations, unusual type affinities,
  large TEXT/BLOB values).
- **No secrets/PII:** Never ingest databases containing passwords, API keys,
  or personally identifiable information.

### Naming Convention

Use lowercase slugs: `project_name.db`.  The `db_id` used in CLI commands
(e.g., `--db project_name`) is derived by stripping the `.db` extension.

---

## Adding a New Workload Preset

### Where Presets Live

All workload presets are defined in `crates/fsqlite-e2e/src/oplog.rs` as
functions named `preset_*`.  The CLI resolves preset names in
`realdb_e2e.rs:resolve_workload()`.

### Existing Presets

| CLI Name | Function | Description |
|----------|----------|-------------|
| `commutative_inserts` | `preset_commutative_inserts_disjoint_keys` | Each worker inserts into disjoint key ranges |
| `hot_page_contention` | `preset_hot_page_contention` | All workers UPDATE the same rows (lock stress) |
| `mixed_read_write` | `preset_mixed_read_write` | Mix of SELECT, INSERT, UPDATE, DELETE |
| `deterministic_transform` | `preset_deterministic_transform` | Serial CREATE + INSERT + UPDATE sequence |

### Step-by-Step

1. **Define the preset function** in `oplog.rs`:

   ```rust
   pub fn preset_my_workload(
       fixture_id: &str,
       seed: u64,
       worker_count: u16,
       ops_per_worker: u32,
   ) -> OpLog {
       let header = OpLogHeader {
           fixture_id: fixture_id.to_owned(),
           seed,
           rng: RngSpec::default(),
           concurrency: ConcurrencyModel {
               worker_count,
               transaction_size: 50,
               commit_order_policy: "deterministic".to_owned(),
           },
           preset: Some("my_workload".to_owned()),
       };
       let mut records = Vec::new();
       // ... generate OpRecord entries ...
       OpLog { header, records }
   }
   ```

2. **Register it in `resolve_workload()`** in `realdb_e2e.rs`:

   ```rust
   "my_workload" => Ok(oplog::preset_my_workload(
       fixture_id, 42, concurrency, 100,
   )),
   ```

3. **Add tests** in `oplog.rs`:

   ```rust
   #[test]
   fn test_preset_my_workload_deterministic() {
       let a = preset_my_workload("test", 42, 2, 10);
       let b = preset_my_workload("test", 42, 2, 10);
       assert_eq!(a.records.len(), b.records.len());
       for (ra, rb) in a.records.iter().zip(b.records.iter()) {
           assert_eq!(ra.op_id, rb.op_id);
       }
   }
   ```

4. **Run validation**:

   ```bash
   cargo test -p fsqlite-e2e
   cargo clippy -p fsqlite-e2e -- -D warnings
   ```

5. **Smoke test with the CLI**:

   ```bash
   cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
     --engine sqlite3 --db chinook --workload my_workload
   ```

### Workload Design Guidelines

- **Determinism:** Use `StdRng::seed_from_u64(seed)` for all randomness.
  Same seed must produce same OpLog every time.
- **Worker isolation:** If using multiple workers, assign disjoint key ranges
  or use INSERT-only patterns to avoid non-deterministic abort ordering.
- **Transaction boundaries:** Set `transaction_size` to match realistic batch
  sizes (10-100 ops per transaction).
- **Commit ordering:** Use `"deterministic"` commit_order_policy for
  reproducibility.  Switch to `"free"` only for contention stress tests where
  non-determinism is acceptable.
- **Table setup:** Each preset should CREATE its own tables in worker 0's
  first transaction to avoid depending on fixture schema.

---

## Adding a New Criterion Benchmark

1. Create a new bench file `crates/fsqlite-e2e/benches/my_bench.rs`:

   ```rust
   use criterion::{criterion_group, criterion_main, Criterion};

   fn bench_my_workload(c: &mut Criterion) {
       let mut group = c.benchmark_group("my_workload");
       group.bench_function("csqlite", |b| {
           b.iter(|| { /* C SQLite via rusqlite */ });
       });
       group.bench_function("fsqlite", |b| {
           b.iter(|| { /* FrankenSQLite */ });
       });
       group.finish();
   }

   criterion_group!(benches, bench_my_workload);
   criterion_main!(benches);
   ```

2. Register in `Cargo.toml`:

   ```toml
   [[bench]]
   name = "my_bench"
   harness = false
   ```

3. Run: `cargo bench -p fsqlite-e2e --bench my_bench`

---

## Debugging Tips

### Verbose JSON Output

Add `--pretty` to `realdb-e2e run` for readable JSON:

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine sqlite3 --db chinook --workload commutative_inserts --pretty
```

### JSONL Append Log

Use `--output-jsonl` to build a cumulative log across multiple runs:

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine sqlite3 --db chinook --workload commutative_inserts \
  --output-jsonl /tmp/cumulative.jsonl

cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine fsqlite --db chinook --workload commutative_inserts \
  --output-jsonl /tmp/cumulative.jsonl
```

Each line in the JSONL file is a complete `RunRecordV1` with methodology and
environment metadata.

### Structured Logging

The E2E runner supports `--verbose` for TRACE-level structured logging:

```bash
cargo run -p fsqlite-e2e --bin e2e-runner -- run-all --verbose
```

### Inspecting Golden Database Metadata

```bash
cargo run -p fsqlite-e2e --bin profile-db -- --pretty
```

This prints formatted JSON metadata for each golden database.

### Checking a Specific Fixture

```bash
# Direct rusqlite verification
cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus verify

# Profile a single DB
cargo run -p fsqlite-e2e --bin profile-db -- --db chinook --pretty
```
