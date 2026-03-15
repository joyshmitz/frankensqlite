# Running the RealDB E2E Suite

This document provides exact commands for running every component of the
FrankenSQLite E2E suite: corpus management, deterministic compatibility
checks, concurrency benchmarks, and corruption/recovery demos.

## Prerequisites

```bash
# Nightly Rust toolchain (required for workspace)
rustup default nightly

# Build the E2E binary (release recommended for benchmarks)
cargo build -p fsqlite-e2e --bin realdb-e2e --release

# Verify the binary works
cargo run -p fsqlite-e2e --bin realdb-e2e -- --help
```

## Corpus Safety Rules

1. **Never** run against `/dp/` originals -- always use working copies
2. Ingest fixtures via `sqlite3 ... ".backup ..."` (not `cp`)
3. Treat `sample_sqlite_db_files/golden/` as **immutable**
4. All tests operate on copies in `working/` or temp dirs

See `sample_sqlite_db_files/FIXTURES.md` for the complete ingestion guide.

## 1. Corpus Management

### Scan for Available Databases

```bash
# Discover all SQLite databases under /dp (default root)
cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus scan

# Scan a specific directory with depth limit
cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus scan --root /dp --max-depth 4

# Only show files with valid SQLite magic headers
cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus scan --header-only
```

**Expected output:**

```
[scan] /dp/asupersync/.beads/beads.db
  size: 12,345,678 bytes
  page_size: 4096
  pages: 3015
  ...
```

### Import a Database

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus import \
  --db /dp/asupersync/.beads/beads.db \
  --tag beads
```

This creates a golden copy via `.backup`, computes SHA-256, and updates
`checksums.sha256`.

### Verify Golden Copies

```bash
# Verify all golden copies against recorded checksums
cargo run -p fsqlite-e2e --bin realdb-e2e -- corpus verify
```

**Expected output (success):**

```
[verify] asupersync.db: OK (sha256: 8e91ff61...)
[verify] brenner_bot.db: OK (sha256: 83608b0c...)
...
[verify] 20/20 golden copies verified.
```

**Expected output (failure):**

```
[verify] asupersync.db: MISMATCH
  expected: 8e91ff61...
  actual:   a1b2c3d4...
[verify] ERROR: 1 checksum mismatch(es)
```

## 2. Deterministic Compatibility Hash Checks

These tests verify that FrankenSQLite produces bit-identical results to
C SQLite for the same workload.

### Run a Single Workload

```bash
# Run against C SQLite (baseline)
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine sqlite3 \
  --db asupersync \
  --workload commutative_inserts_disjoint_keys \
  --concurrency 1 \
  --pretty

# Run against FrankenSQLite
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine fsqlite \
  --db asupersync \
  --workload commutative_inserts_disjoint_keys \
  --concurrency 1 \
  --pretty
```

### Available Workload Presets

| Preset | Description | Concurrency |
|--------|-------------|-------------|
| `commutative_inserts_disjoint_keys` | Each worker inserts into disjoint key ranges | Multi-worker safe |
| `hot_page_contention` | All workers target the same table/page range | Conflict-heavy |
| `mixed_read_write` | 70% reads, 30% writes across all workers | Realistic mix |
| `deterministic_transform` | Sequential schema creation + data insertion | Single-worker |

### Run with Multiple Concurrency Levels

```bash
# Test at concurrency 1, 2, 4, 8
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine sqlite3 \
  --db frankensqlite \
  --workload commutative_inserts_disjoint_keys \
  --concurrency 1,2,4,8 \
  --repeat 3 \
  --output-jsonl results/sqlite3_commutative.jsonl
```

### Save Results as JSONL

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine fsqlite \
  --db asupersync \
  --workload hot_page_contention \
  --concurrency 4 \
  --mvcc \
  --output-jsonl results/fsqlite_hot_page.jsonl
```

Each JSONL line contains a `RunRecordV1` with:
- `engine`, `workload`, `fixture_id`, `concurrency`
- `wall_time_ms`, `ops_per_sec`, `retries`, `aborts`
- `correctness.raw_sha256`, `correctness.canonical_sha256`
- `methodology` and `environment` metadata

For profiler-focused runs, use:

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine fsqlite \
  --db asupersync \
  --workload hot_page_contention \
  --concurrency 4 \
  --profile-only
```

or the granular flags:

```bash
cargo run -p fsqlite-e2e --bin realdb-e2e -- run \
  --engine fsqlite \
  --db asupersync \
  --workload hot_page_contention \
  --concurrency 4 \
  --skip-integrity-check \
  --skip-environment-metadata
```

These modes keep the JSONL schema stable. The record still uses
`RunRecordV1`. For profiler-safe runs, `run_mode.profile_only_effective`
becomes `true`, `run_mode.run_integrity_check` becomes `false`,
`run_mode.capture_environment_metadata` becomes `false`,
`environment.capture_mode` becomes `suppressed`, and the post-run
`integrity_check_ok` field is omitted. Default runs remain full correctness
runs with captured environment metadata.

### Interpreting Results

**SHA-256 match (all three tiers agree):**

```json
{
  "correctness": {
    "raw_sha256_match": true,
    "canonical_sha256_match": true,
    "dump_match": true,
    "integrity_check_ok": true
  }
}
```

Both engines produced identical databases. FrankenSQLite is compatible.

**Raw SHA-256 mismatch but canonical match:**

```json
{
  "correctness": {
    "raw_sha256_match": false,
    "canonical_sha256_match": true,
    "dump_match": true
  }
}
```

Page layout differs (e.g., different WAL checkpoint timing), but logical
content is identical.  This is expected for concurrent workloads.

**Logical dump mismatch:**

```json
{
  "correctness": {
    "raw_sha256_match": false,
    "canonical_sha256_match": false,
    "dump_match": false,
    "notes": "divergence detected at row 42"
  }
}
```

Actual data difference between engines.  This indicates a bug that needs
investigation.  Use the mismatch debugger (see below) to produce a minimal
reproduction.

## 3. Concurrency Benchmarks

### Run the Benchmark Matrix

Track A's canonical many-core Beads campaign is pinned in
`sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.json`. That
manifest freezes the fixture source paths and digests, the pinned working-copy
fixtures under `sample_sqlite_db_files/working/beads_bench_20260310/golden/`,
the three benchmark modes (`sqlite_reference`, `fsqlite_mvcc`,
`fsqlite_single_writer`), the fixed seed policy, the retry/backoff contract,
the required placement profiles, the hardware-class identifiers, and the stable
artifact bundle naming template. Future scorecard and verification work should
treat that file as the source of truth instead of inventing ad hoc matrix rows
or artifact names.

```bash
# Run all presets against all golden copies (full matrix)
rch exec -- cargo run -p fsqlite-e2e --bin realdb-e2e --release -- bench

# Run specific database and preset
rch exec -- cargo run -p fsqlite-e2e --bin realdb-e2e --release -- bench \
  --db asupersync \
  --preset commutative_inserts_disjoint_keys

# Control iterations
rch exec -- cargo run -p fsqlite-e2e --bin realdb-e2e --release -- bench \
  --warmup 3 \
  --repeat 10 \
  --output results/bench_report.json
```

### Canonical Beads Campaign

The canonical matrix for `bd-db300.1.2` is tracked in:

```text
sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.json
sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.schema.json
```

That manifest freezes the real-fixture campaign dimensions instead of relying on
ad hoc shell history:

- Fixtures: `frankensqlite`, `frankentui`, `frankensearch`
- Workloads: `commutative_inserts_disjoint_keys`, `hot_page_contention`, `mixed_read_write`
- Modes: `sqlite_reference`, `fsqlite_mvcc`, `fsqlite_single_writer`
- Build profile: manifest ID `release_perf` using Cargo profile `release-perf`
- Seed policy: fixed root seed `42`
- Retry policy: harness-instrumented busy retries (`10_000`, `1ms` base, `250ms` cap)
- Placement profiles: `baseline_unpinned`, `recommended_pinned`, `adversarial_cross_node`
  with taxonomy kinds `baseline`, `recommended_pinned`, `adversarial_topology`
- Hardware classes: `linux_x86_64_any`, `linux_x86_64_many_core_numa`
  with identifier fields `os_family`, `cpu_arch`, `topology_class`

Artifact names for canonical runs must include the source revision and the
`.beads/issues.jsonl` hash so results stay mechanically diffable over time, and
they now also carry the build-profile id and explicit run id. The tracked
templates are:

```text
bundle dir: {row_id}__{workload}__c{concurrency}__{fixture_id}__{mode}__{placement_profile_id}__{build_profile_id}__run_{run_id}__rev_{source_revision}__beads_{beads_hash}
bundle key: {row_id}:{fixture_id}:{workload}:c{concurrency}:{mode}:{placement_profile_id}:{build_profile_id}:run_{run_id}:rev_{source_revision}:beads_{beads_hash}
bundle name: {row_id} {fixture_id} {workload} c{concurrency} {mode} {placement_profile_id} {build_profile_id} run {run_id} rev {source_revision} beads {beads_hash}
files: results.jsonl, summary.md, manifest.json, logs/, profiles/
```

Retention classes are part of the contract as well: `quick_run`,
`full_proof`, `failure_bundle`, and `final_scorecard`.

### Placement Profile Execution Contract

The checked-in campaign manifest now carries an exact
`placement_profiles[].execution_contract` for the three canonical profiles.
Full suites select a profile by matching
`matrix_rows[].placement_variants[].placement_profile_id`. Focused reruns and
topology-bundle capture bind the profile mechanically with:

```text
RUN_ID
ARTIFACT_BUNDLE_DIR
ARTIFACT_BUNDLE_RELPATH
PLACEMENT_PROFILE_ID
HARDWARE_CLASS_ID
MANIFEST_JSON
SOURCE_REVISION
BEADS_HASH
```

Every placement-sensitive run must disclose
`placement_profile_id`, `hardware_class_id`, `hardware_signature`,
`cpu_affinity_mask`, `smt_policy_state`, `memory_policy`,
`helper_lane_cpu_set`, and `numa_balancing_state`. If any run drifts from the
declared placement contract, treat it as `not_comparable` rather than folding
it into a clean comparison.

`baseline_unpinned`

- CPU pinning: scheduler default, with no `taskset` or `numactl` overrides.
- SMT and memory policy: host defaults; record them instead of forcing them.
- Helper lane: disclose where housekeeping work landed, but do not relocate it.
- Fixed knobs: `no_taskset_or_numactl_binding`, `report_host_default_smt_policy`, `report_host_default_memory_policy`, `disclose_helper_lane_policy_without_relocation`.
- Optional knobs: exact scheduler-chosen CPU set and extra profiler capture.
- Mandatory for: portable baseline claims and host-default regression checks.
- Avoid for: transferable many-core win claims and cross-node sensitivity claims.

`recommended_pinned`

- CPU pinning: one thread per physical core, all workers kept in one NUMA/LLC locality domain.
- SMT and memory policy: disable sibling reuse inside the primary worker set and bind memory to the same locality domain.
- Helper lane: keep housekeeping on a non-worker CPU in the same locality domain.
- Fixed knobs: `pin_workers_to_one_thread_per_physical_core`, `keep_workers_inside_one_locality_domain`, `bind_memory_to_worker_locality`, `place_helper_lane_on_housekeeping_cpu_in_same_locality`.
- Optional knobs: exact locality-domain choice, exact worker CPU set, and extra profiler capture.
- Mandatory for: transferable many-core win claims and final scorecard primary claims.
- Avoid for: portable baseline claims.

`adversarial_cross_node`

- CPU pinning: split workers across multiple locality domains to expose remote ownership and cache-line movement penalties.
- SMT and memory policy: avoid sibling reuse inside the primary worker set and use a memory policy that matches the cross-domain split instead of pretending locality.
- Helper lane: place housekeeping outside the primary worker domains.
- Fixed knobs: `split_workers_across_locality_domains`, `avoid_smt_sibling_reuse_inside_primary_worker_set`, `match_memory_policy_to_cross_domain_worker_split`, `place_helper_lane_outside_primary_worker_domains`.
- Optional knobs: exact remote-domain pair, exact cross-domain worker split, and extra profiler capture.
- Mandatory for: cross-node sensitivity claims and placement regression guard claims.
- Avoid for: headline speedup claims and portable baseline claims.

### Capture The Hardware Discovery Bundle

`bd-db300.1.6.1` defines a reusable discovery bundle so canonical benchmark runs
carry explicit hardware identity instead of relying on operator memory. The
capture entrypoint is:

```bash
bash scripts/verify_bd_db300_1_6_1_topology_bundle.sh
```

When you already know the artifact bundle directory and run-identity fields,
bind the capture mechanically to the benchmark session:

```bash
RUN_ID=run-20260315T020400Z \
ARTIFACT_BUNDLE_DIR=artifacts/perf/bd-db300.1.2/example_bundle \
ARTIFACT_BUNDLE_RELPATH=artifacts/perf/bd-db300.1.2/example_bundle \
PLACEMENT_PROFILE_ID=recommended_pinned \
HARDWARE_CLASS_ID=linux_x86_64_many_core_numa \
MANIFEST_JSON=artifacts/perf/bd-db300.1.2/example_bundle/manifest.json \
bash scripts/verify_bd_db300_1_6_1_topology_bundle.sh
```

The script emits:

```text
hardware_discovery_bundle.json
hardware_discovery_summary.md
```

The JSON bundle records CPU model and stepping, microcode or firmware where
available, socket or NUMA layout, SMT sibling sets, LLC sharing domains,
boost/EPP/governor state, THP state, relevant scheduler settings, and a stable
`hardware_signature` hash derived from the captured identity fields. Missing or
partially inferred fields are listed explicitly so later scorecards, logging
contracts, and reports do not overstate how complete the machine fingerprint
really is.

For CPU-heavy canonical runs, offload through `rch` and use the manifest as the
source of truth for dimensions and naming:

```bash
rch exec -- cargo run -p fsqlite-e2e --profile release-perf --bin realdb-e2e -- bench
```

### Run Library Benchmarks (Unit Tests)

```bash
# Run all benchmark-related tests
cargo test -p fsqlite-e2e --lib benchmark -- --nocapture

# Run bench_summary tests
cargo test -p fsqlite-e2e --lib bench_summary -- --nocapture
```

### Interpreting Benchmark Output

The benchmark summary includes latency and throughput statistics:

```json
{
  "benchmark_id": "sqlite3:commutative_inserts:asupersync:c4",
  "warmup_count": 3,
  "measurement_count": 10,
  "latency": {
    "median_ms": 125.0,
    "p95_ms": 142.0,
    "p99_ms": 155.0,
    "stddev_ms": 8.3
  },
  "throughput": {
    "median_ops_per_sec": 8000.0,
    "peak_ops_per_sec": 8500.0
  }
}
```

**Key metrics:**
- `median_ms` — primary statistic (50th percentile wall time)
- `p95_ms` — tail latency statistic (95th percentile)
- `stddev_ms` — measurement stability (lower = more consistent)
- `median_ops_per_sec` — primary throughput metric

See `crates/fsqlite-e2e/METHODOLOGY.md` for the full methodology.

## 4. Corruption and Recovery Demos

### Run All Corruption Scenarios (Library Tests)

```bash
# C SQLite corruption behavior (baseline)
cargo test -p fsqlite-e2e --lib corruption_demo_sqlite -- --nocapture

# FrankenSQLite recovery demos
cargo test -p fsqlite-e2e --lib fsqlite_recovery_demo -- --nocapture

# Full corruption framework unit tests
cargo test -p fsqlite-e2e --lib corruption -- --nocapture
```

### Run the CLI Corruption Demo

```bash
# Inject bitflip corruption
cargo run -p fsqlite-e2e --bin realdb-e2e -- corrupt \
  --db asupersync \
  --strategy bitflip \
  --count 5 \
  --seed 42

# Zero a page
cargo run -p fsqlite-e2e --bin realdb-e2e -- corrupt \
  --db asupersync \
  --strategy page \
  --page 2 \
  --seed 0

# Zero a byte range
cargo run -p fsqlite-e2e --bin realdb-e2e -- corrupt \
  --db asupersync \
  --strategy zero \
  --offset 0 \
  --length 100
```

See `crates/fsqlite-e2e/CORRUPTION_DEMO_RUNBOOK.md` for the full corruption
and repair runbook with expected output for each scenario.

## 5. Quick Smoke Test

Run a fast verification that the infrastructure works:

```bash
# Library smoke tests (fast, <5 seconds)
cargo test -p fsqlite-e2e --lib smoke -- --nocapture

# Full library test suite (includes all modules)
cargo test -p fsqlite-e2e --lib
```

**Expected output:**

```
test result: ok. 254 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## 6. Artifacts and Output Locations

| Artifact | Location | Format |
|----------|----------|--------|
| Run records | `--output-jsonl <PATH>` | JSONL (one record per line) |
| Benchmark report | `--output <PATH>` | JSON |
| Working copies | `sample_sqlite_db_files/working/` | SQLite `.db` (ephemeral) |
| Golden copies | `sample_sqlite_db_files/golden/` | SQLite `.db` (immutable) |
| Metadata | `sample_sqlite_db_files/metadata/` | JSON per DB |
| Checksums | `sample_sqlite_db_files/checksums.sha256` | sha256sum format |

## Troubleshooting

**"No golden copies found"**
- Run `corpus import` first, or check that `sample_sqlite_db_files/golden/` has `.db` files
- Golden files are git-ignored; they must be ingested locally

**"Checksum mismatch"**
- A golden file was modified after ingestion
- Re-ingest from `/dp/` source: `sqlite3 <source> ".backup '<golden>'"`
- Regenerate checksums: `cd golden && sha256sum *.db | sort -k2 > ../checksums.sha256`

**"integrity_check failed"**
- The source database may have been corrupted before ingestion
- Try re-ingesting; if it persists, the source has problems

**"Engine returned different row counts"**
- Investigate with `--pretty` flag to see per-statement results
- Check if the workload uses non-deterministic operations (avoid RANDOM())
- File a bug with the JSONL output for reproduction
