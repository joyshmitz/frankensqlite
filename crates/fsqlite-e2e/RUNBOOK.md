# Real-DB E2E Demo Runbook

Everything needed to run the FrankenSQLite E2E demo suite — fixtures, perf
benchmarks, correctness gates, and corruption/recovery demos — without tribal
knowledge.

## Document Map

| Doc | Contents |
|-----|----------|
| **This file** | Primary runbook: quick-start, demo walkthroughs |
| [HARNESS.md](HARNESS.md) | Full CLI reference, artifact locations, result interpretation |
| [METHODOLOGY.md](METHODOLOGY.md) | Benchmark methodology (warmup, statistics, PRAGMA fairness) |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Failure modes, adding fixtures/workloads, debugging tips |
| [CORRUPTION_DEMO_RUNBOOK.md](CORRUPTION_DEMO_RUNBOOK.md) | Sidecar encoding, corruption modes, WAL-FEC repair |
| [../../sample_sqlite_db_files/FIXTURES.md](../../sample_sqlite_db_files/FIXTURES.md) | Fixture ingestion policy, safety rules, provenance chain |

---

## Prerequisites

```bash
# Verify Rust nightly toolchain
rustup show active-toolchain
# Expected: nightly-...

# Verify rusqlite links properly
cargo check -p fsqlite-e2e

# Verify golden fixtures exist (if running against real DBs)
ls sample_sqlite_db_files/golden/*.db 2>/dev/null
```

---

## Quick Start: 60-Second Smoke Test

```bash
cargo run -p fsqlite-e2e --bin e2e-runner -- run-smoke
```

Verifies all infrastructure: golden copies, both backends, workload generator,
corruption injector, canonicalization, and logging.  Exit 0 = all green.

---

## Demo 1: Correctness Verification

Proves FrankenSQLite produces logically identical results to C SQLite.

### Step 1 — Run Correctness Tests

```bash
cargo run -p fsqlite-e2e --bin e2e-runner -- run-correctness
```

Runs three checks:
- **Workload replay** — Same oplog replayed on C SQLite produces no errors
- **Canonicalization stability** — VACUUM INTO yields identical SHA-256 on repeat
- **Logical comparison** — Two independently built databases match at logical tier

### Step 2 — Run Both Engines on the Same Workload

```bash
# C SQLite
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine sqlite3 --db chinook --workload commutative_inserts --pretty

# FrankenSQLite
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine fsqlite --db chinook --workload commutative_inserts --pretty
```

### Step 3 — Compare Results

Compare the `correctness` block in each JSON output:

| Field | Meaning |
|-------|---------|
| `integrity_check_ok: true` | B-tree internally consistent |
| `canonical_sha256_match: true` | Byte-identical after canonicalization |
| `logical_sha256` | Compare these across engines for logical equivalence |

**If `integrity_check_ok` is false:** See [HARNESS.md § integrity_check Failure](HARNESS.md#integrity_check-failure).

**If `canonical_sha256` differs:** See [HARNESS.md § SHA Mismatch](HARNESS.md#sha-mismatch).

---

## Demo 2: Performance Matrix

Benchmarks FrankenSQLite vs C SQLite across workloads and concurrency levels.

### Step 1 — Single Baseline Run

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine sqlite3 --db chinook --workload commutative_inserts \
  --concurrency 1 --repeat 5 --output-jsonl /tmp/perf.jsonl

cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine fsqlite --db chinook --workload commutative_inserts \
  --concurrency 1 --repeat 5 --output-jsonl /tmp/perf.jsonl
```

Optional: browse the JSONL output in a TUI viewer:

```bash
cargo run -p fsqlite-e2e --bin e2e-viewer -- /tmp/perf.jsonl
```

### Step 2 — Scaling Sweep

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine sqlite3 --db chinook --workload hot_page_contention \
  --concurrency 1,2,4,8 --repeat 5

cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine fsqlite --db chinook --workload hot_page_contention \
  --concurrency 1,2,4,8 --repeat 5 --mvcc
```

The scaling summary table prints to stderr showing p50 ops/s, p95 ops/s,
median wall time, and median retries per concurrency level.

### Step 3 — Criterion Benchmarks

```bash
cargo bench -p fsqlite-e2e
```

Results in `target/criterion/`.  HTML report at `target/criterion/report/index.html`.

### Reading the Numbers

- **ops_per_sec increasing** with concurrency = good scaling
- **retries climbing** = lock contention overhead
- **ops_per_sec flattening** = throughput ceiling reached
- Compare `--mvcc` (FrankenSQLite) vs default (C SQLite) to see MVCC gains

See [METHODOLOGY.md](METHODOLOGY.md) for warmup, iteration count, and
statistical methodology.

---

## Demo 3: Corruption + Recovery

Demonstrates FrankenSQLite's WAL-FEC resilience vs C SQLite's silent data loss.

### Step 1 — Run Recovery Tests

```bash
cargo run -p fsqlite-e2e --bin e2e-runner -- run-recovery
```

Runs three corruption scenarios: bitflip, page-zero, and header-zero.
Each verifies that corruption is detected (hash changes after injection).

### Step 2 — One-Command Walkthrough (Demo-Friendly)

```bash
cargo run -p fsqlite-e2e --bin corruption-demo
cargo run -p fsqlite-e2e --bin corruption-demo -- --json
```

This runs a guided corruption + recovery narrative across 4 representative
scenarios and exits non-zero on any mismatch.

### Step 3 — Detailed Corruption Demo

See [CORRUPTION_DEMO_RUNBOOK.md](CORRUPTION_DEMO_RUNBOOK.md) for the full
walkthrough covering:
- Sidecar encoding (source symbols, repair symbols, chunk layout)
- How each corruption pattern works (BitFlip, PageZero, HeaderZero, WAL frame, FEC sidecar)
- C SQLite failure mode (silent truncation at bad checksum)
- FrankenSQLite recovery mode (RaptorQ repair from sidecar)
- Verification of repair success

### Step 4 — Manual Corruption Injection (CLI)

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- corrupt \
  --db chinook --strategy bitflip --count 10 --seed 42

cargo run -p fsqlite-e2e --bin realdb-e2e -- corrupt \
  --db chinook --strategy page --page 1 --seed 42
```

The `corrupt` subcommand is copy-only: it refuses to modify anything under
`sample_sqlite_db_files/golden/`.

---

## Demo 4: Full Suite

Run everything in one shot:

```bash
cargo run -p fsqlite-e2e --bin e2e-runner -- run-all --output /tmp/e2e-run
```

Artifacts land in the output directory:

| File | Contents |
|------|----------|
| `results.json` | Machine-readable test results with assertions |
| `report.md` | Human-readable Markdown summary table |

Generate a standalone report from previous results:

```bash
cargo run -p fsqlite-e2e --bin e2e-runner -- report --output /tmp/e2e-run
```

---

## Fixture Management

### Safety Rules

1. **Never touch `/dp/` originals** — use `.backup` for consistent snapshots
2. **Treat `golden/` as immutable** — work on copies only
3. **Never commit `*.db` to git** — only metadata and checksums are tracked
4. **Verify integrity immediately** after capture

### Adding a New Fixture (Quick)

```bash
# 1. Snapshot
sqlite3 "/dp/project/.beads/beads.db" \
  ".backup 'sample_sqlite_db_files/golden/project.db'"

# 2. Verify
sqlite3 "sample_sqlite_db_files/golden/project.db" "PRAGMA integrity_check;"

# 3. Checksum
cd sample_sqlite_db_files/golden && sha256sum project.db >> ../checksums.sha256

# 4. Metadata
cargo run -p fsqlite-e2e --bin profile-db -- --db project

# 5. Verify corpus
cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus verify

# 6. Smoke test
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine sqlite3 --db project --workload commutative_inserts
```

Full details: [FIXTURES.md](../../sample_sqlite_db_files/FIXTURES.md) and
[TROUBLESHOOTING.md § Adding a New Fixture](TROUBLESHOOTING.md#adding-a-new-fixture).

---

## Available Workloads

| Preset Name | Pattern | Best For |
|-------------|---------|----------|
| `commutative_inserts` | Disjoint key INSERTs per worker | Scaling tests, baseline perf |
| `hot_page_contention` | All workers UPDATE same rows | Lock contention, MVCC vs WAL |
| `mixed_read_write` | SELECT + INSERT + UPDATE + DELETE | Realistic OLTP mix |
| `deterministic_transform` | Serial CREATE + INSERT + UPDATE | Correctness verification |

Adding new presets: [TROUBLESHOOTING.md § Adding a New Workload Preset](TROUBLESHOOTING.md#adding-a-new-workload-preset).

---

## Interpreting Failures

| Symptom | Severity | Action |
|---------|----------|--------|
| `integrity_check_ok: false` | Critical | B-tree corruption; file a bug with full JSON report |
| `canonical_sha256_match: false` | Investigate | Check PRAGMA config, workload determinism, canonicalization edge cases |
| `raw_sha256_match: false` | Expected | Page layout differs between implementations; not a bug |
| `dump_match: false` | Investigate | Check for mid-workload errors, high abort counts |
| High retry count | Tuning | Reduce concurrency or use `--mvcc` for FrankenSQLite |
| `exceeded max_busy_retries` | Config | Increase retry limit or reduce contention |

Full details: [HARNESS.md § Interpreting Results](HARNESS.md#interpreting-results).
