# Extreme Optimization: Performance Gap Closure Program

## 0. Investigation Ground Truth
- [x] Read all of `AGENTS.md`.
- [x] Read all of `README.md`.
- [x] Register with MCP Agent Mail as `WindySalmon`.
- [x] Introduce this session to the other agents in thread `perf-arch-2026-03-18`.
- [x] Compare FrankenSQLite hot paths against vendored legacy SQLite.
- [x] Clone PostgreSQL to `/tmp/postgres-frankensqlite-study-20260319` and study its MVCC and snapshot paths.
- [x] Identify the biggest accidental divergences from SQLite outside the intentional MVCC/RaptorQ differences.
- [x] Identify the biggest structural divergence likely hurting concurrent writers: full-page version-chain MVCC plus heavy SSI bookkeeping.

## 1. Truth Restoration: Stop Lying to Ourselves With Bad Benchmarks
- [x] Audit the existing `fsqlite-e2e` benches for apples-to-oranges API usage.
- [x] Confirm that `crates/fsqlite-e2e/benches/concurrent_write_bench.rs` currently measures SQLite concurrent WAL writers against a FrankenSQLite sequential control.
- [x] Confirm that several benches gave rusqlite prepared-statement reuse while FrankenSQLite used ad hoc `format!()` SQL.
- [x] Replace FrankenSQLite ad hoc SQL in `crates/fsqlite-e2e/benches/e2e_bench.rs` wherever the SQLite side already uses prepared statements.
- [x] Replace FrankenSQLite ad hoc SQL in the in-memory threaded control benchmark at the bottom of `crates/fsqlite-e2e/benches/e2e_bench.rs`.
- [x] Replace FrankenSQLite ad hoc SQL in `crates/fsqlite-e2e/benches/concurrent_write_bench.rs` with prepared execution.
- [x] Rename the FrankenSQLite control in `concurrent_write_bench.rs` so the name says it is sequential and only a control.
- [ ] Add a real persistent concurrent-writer benchmark once the persistent concurrent path is measurable end to end.
- [ ] Add explicit benchmark metadata stating whether each engine path is prepared, cached, ad hoc, file-backed, or in-memory.
- [ ] Add a benchmark gate that fails if SQLite and FrankenSQLite are not using equivalent statement-lifecycle modes.

## 2. Profiling Workflow: Make Instrumentation Unavoidable
- [x] Confirm that hot-path profiling already exists in `fsqlite-core`, `fsqlite-e2e`, and `realdb-e2e`.
- [x] Confirm that `realdb-e2e hot-profile` is already the canonical artifact-producing command.
- [x] Document the canonical `realdb-e2e hot-profile` workflow in `crates/fsqlite-e2e/README.md`.
- [ ] Add a single top-level wrapper command so nobody has to remember the full `hot-profile` invocation.
- [ ] Emit the hot-path summary markdown by default in CI and local perf runs.
- [ ] Add a mandatory perf checklist for every benchmark report: command, seed, scale, workload, concurrency, MVCC mode, fixture id, artifact directory.
- [ ] Add a comparison artifact that puts SQLite and FrankenSQLite component costs side by side for the same workload.

## 3. Immediate Execution-Path Simplification
- [ ] Measure parse, rewrite, compile, execute, B-tree, MVCC, and retry shares for the top three workloads using `realdb-e2e hot-profile`.
- [ ] Rank current hotspots by impact and confidence instead of guessing.
- [ ] Audit every `Connection::execute*` hot path that still does unnecessary background-status, rewrite, canonicalization, or dispatch work in the common case.
- [x] Collapse the MVCC read path's `resolve()` + `get_version()` double arena lookup into a single-pass visible-version helper.
- [x] Reuse the same single-pass helper in MVCC rebase paths so base/head page fetches stop reacquiring the arena separately.
- [x] Add a commit-seq-only MVCC visibility helper so write tracking and scan witnesses stop cloning full page versions when they only need visibility metadata.
- [x] Remove the redundant pre-publish `chain_head()` lookup from `publish_write_set()` since `VersionStore::publish()` already links against the live head internally.
- [ ] Separate the no-fallback common case from compatibility fallback paths in `fsqlite-core::Connection`.
- [ ] Ensure prepared DML and prepared SELECT stay on the precompiled path whenever schema identity is unchanged.
- [ ] Remove avoidable cross-layer state churn from the common path before entering the VDBE.
- [ ] Reduce connection-scoped work that happens even for tiny single-row statements.

## 4. VDBE Runtime Diet
- [x] Track cold per-statement VDBE subsystems explicitly so common-case reuse only clears the maps and vectors that were actually touched.
- [x] Track `OE_REPLACE` secondary-index inserts in rollback bookkeeping so later index-conflict unwinds do not leak provisional replacement entries.
- [ ] Split always-hot VDBE state from rarely used feature state so the common interpreter loop carries less baggage.
- [ ] Move heavy optional subsystems behind cold side structures instead of always living in `VdbeEngine`.
- [ ] Re-check opcode handlers for repeated map clears, vector clears, and state resets that are not necessary for every statement.
- [ ] Measure instruction-cache and branch effects after each VDBE simplification pass.
- [ ] Audit rowid seek, covering-index, and point-lookup opcode generation against SQLite behavior.

## 5. SQLite Emulation Work: Copy the Fast Parts
- [ ] Tighten the prepare/step lifecycle so the common path looks more like legacy SQLite's narrow control flow.
- [ ] Audit rowid equality predicates and ensure they compile to direct seek paths instead of scans.
- [ ] Audit covering-index opportunities and avoid touching the main table when the index already covers the query.
- [ ] Add targeted parity tests for point lookup, narrow range lookup, covering-index lookup, and repeated prepared DML.
- [ ] Make prepared statements the expected benchmark path for steady-state throughput measurements.

## 6. Radical MVCC Redesign Track
- [ ] Stop assuming full-page version chains are the right representation for ordinary row updates.
- [ ] Design an in-memory logical version sidecar keyed by page/slot or page/rowid instead of forcing every logical update through full page cloning.
- [ ] Define which operations stay page-structural: splits, merges, page allocation, free-list mutation, rebalance.
- [ ] Define which operations become logical-row updates: INSERT/UPDATE/DELETE where structure does not change.
- [ ] Prototype a `CellVisibilityLog` / logical delta layer that can answer snapshot visibility without reconstructing whole pages.
- [ ] Keep SQLite-compatible on-disk pages while letting concurrent writers operate through logical deltas in memory.
- [ ] Materialize full page images only when structural changes or persistence boundaries require them.
- [ ] Revisit whether FCW should operate on logical row/slot witnesses for common writes instead of page-wide base drift.

## 7. Snapshot and SSI Cost Reduction
- [x] Stop re-walking the same page chain immediately after successful eager GC in chain-bound backpressure; reuse the known freed-count delta while the page is still write-excluded.
- [ ] Replace expensive exact bookkeeping in the common case with compact reusable structures where possible.
- [ ] Borrow PostgreSQL's habit of reusing snapshot arrays instead of repeated allocation churn.
- [ ] Add approximate visibility horizons for cleanup decisions where exactness is expensive and unnecessary.
- [ ] Make cleanup opportunistic and quick-to-abandon rather than chain-bound blocking with sleeps.
- [ ] Reevaluate the witness/index/history footprint in `ConcurrentRegistry`.
- [ ] Prove which SSI evidence is required for correctness and which current bookkeeping is optional overhead.

## 8. Verification and Safety
- [ ] Run `cargo fmt --check`. Currently blocked by pre-existing formatting drift in `crates/fsqlite-core/src/connection.rs` and `crates/fsqlite-pager/src/pager.rs`.
- [ ] Run `cargo check --workspace --all-targets`. Blocked by pre-existing borrow-check failures in `crates/fsqlite-pager/src/pager.rs`.
- [ ] Run `cargo clippy --workspace --all-targets -- -D warnings`. Blocked by pre-existing clippy failures in `crates/fsqlite-mvcc/src/physical_merge.rs`.
- [x] Run targeted performance harness checks after benchmark edits (`cargo check -p fsqlite-e2e --benches`).
- [x] Run targeted MVCC compile/tests for the visible-version optimization (`cargo check -p fsqlite-mvcc --lib --tests`, `cargo test -p fsqlite-mvcc resolve_visible_version -- --nocapture`, `cargo test -p fsqlite-mvcc chain_head_version -- --nocapture`, `cargo test -p fsqlite-mvcc publish_write_set -- --nocapture`).
- [x] Run targeted MVCC verification for the chain-backpressure reduction (`CARGO_TARGET_DIR=/tmp/frankensqlite-mvcc-verify cargo test -p fsqlite-mvcc test_chain_backpressure_reports_blocked_when_horizon_pinned -- --nocapture`, `CARGO_TARGET_DIR=/tmp/frankensqlite-mvcc-verify cargo test -p fsqlite-mvcc publish_write_set -- --nocapture`, `CARGO_TARGET_DIR=/tmp/frankensqlite-mvcc-verify cargo check -p fsqlite-mvcc --lib --tests`).
- [x] Run targeted VDBE verification for the cold-state reset reduction (`CARGO_TARGET_DIR=/tmp/frankensqlite-vdbe-verify cargo test -p fsqlite-vdbe --lib test_execute_clears_cold_subtype_state_between_statements -- --nocapture`, `CARGO_TARGET_DIR=/tmp/frankensqlite-vdbe-verify cargo test -p fsqlite-vdbe --lib test_reset_for_reuse_keeps_cached_engine_results_clean -- --nocapture`, `CARGO_TARGET_DIR=/tmp/frankensqlite-vdbe-verify cargo check -p fsqlite-vdbe --all-targets`, `CARGO_TARGET_DIR=/tmp/frankensqlite-vdbe-verify cargo clippy -p fsqlite-vdbe --all-targets --no-deps -- -D warnings`).
- [x] Run targeted VDBE verification for the fresh-eyes secondary-index rollback fix (`CARGO_TARGET_DIR=/tmp/frankensqlite-vdbe-verify cargo test -p fsqlite-vdbe --lib test_secondary_index_rollback_removes_tracked_replace_entry -- --nocapture`, `CARGO_TARGET_DIR=/tmp/frankensqlite-vdbe-verify cargo test -p fsqlite-vdbe --lib test_execute_clears_cold_subtype_state_between_statements -- --nocapture`, `CARGO_TARGET_DIR=/tmp/frankensqlite-vdbe-verify cargo check -p fsqlite-vdbe --all-targets`, `CARGO_TARGET_DIR=/tmp/frankensqlite-vdbe-verify cargo clippy -p fsqlite-vdbe --all-targets --no-deps -- -D warnings`).
- [ ] Keep concurrent mode defaults ON in every touched path.
- [ ] Do not touch serialized file-locking behavior that would reintroduce SQLite's writer bottleneck.

## 9. Current Session Ledger
- [x] Reserve the planning files and E2E benchmark files with Agent Mail.
- [x] Avoid touching `crates/fsqlite-e2e/src/bin/realdb_e2e.rs` and `crates/fsqlite-e2e/src/perf_runner.rs` because they are currently reserved by another agent.
- [x] Land the benchmark fairness patch in the two E2E benchmark files I reserved.
- [x] Verify the current patch set with formatting and targeted bench compilation; record the pre-existing workspace blockers.
- [x] Reserve `crates/fsqlite-mvcc/src/invariants.rs` and `crates/fsqlite-mvcc/src/lifecycle.rs` before changing the MVCC read path.
- [x] Add single-pass visible-version and chain-head helpers to `VersionStore`.
- [x] Wire MVCC read/rebase paths to the new helpers to cut repeated arena acquisitions.
- [x] Add `VersionStore` tests covering visible-version resolution and latest chain-head reads.
- [x] Format the touched MVCC files with `cargo fmt -- crates/fsqlite-mvcc/src/invariants.rs crates/fsqlite-mvcc/src/lifecycle.rs`.
- [x] Add `VersionStore::resolve_visible_commit_seq()` and route lifecycle visibility tracking through it.
- [x] Remove the redundant chain-head lookup from `publish_write_set()` and confirm the publish-path regression test still passes.
- [x] Reserve `crates/fsqlite-vdbe/src/engine.rs` and `crates/fsqlite-vdbe/src/lib.rs` before changing the VDBE common path.
- [x] Add VDBE cold-state tracking so prepared/common-case reuse no longer blindly clears unused aggregate, rowset, subtype, bloom, window, sequence, and vtab cursor state.
- [x] Add a repeated-execute regression test proving cold subtype state is cleared between statements without a full engine rebuild.
- [x] Re-read the changed VDBE path with fresh eyes and fix the secondary-index `OE_REPLACE` rollback bookkeeping hole.
- [x] Add a regression test proving rollback removes tracked replacement index entries as well as the provisional table row.
- [x] Reserve `crates/fsqlite-mvcc/src/lifecycle.rs` before changing chain-bound backpressure behavior.
- [x] Remove the redundant post-prune chain-length walk from MVCC chain backpressure by reusing the freed-count delta under page write exclusion.
- [ ] Update the Agent Mail thread with what actually landed in this pass.
