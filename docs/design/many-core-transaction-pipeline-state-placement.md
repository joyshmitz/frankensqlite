# Many-Core Transaction Pipeline State Placement (`bd-db300.5.1.1`)

## Purpose

Map the live FrankenSQLite transaction pipeline into explicit stages, ownership
boundaries, and shared-state touch points so follow-on Track E work can compare
many-core architectures against the actual code instead of the README-level
diagram.

This artifact is the mechanical-sympathy map for:

- `bd-db300.5.1.2` design comparison work
- `bd-db300.5.1.3` decision-record work
- later implementation beads that need a precise answer to "what must stay
  local and what is truly irreducible shared state?"

## Scope

This document describes the **current live path** and the **target placement
contract** for 32-plus-core hardware.

It does not choose the final architecture yet. In particular, it does not
decide:

- whether Track E should prefer share-nothing lanes, tiny-publish shared state,
  or a hierarchical hybrid
- the final durability batching shape
- the final structural-mutation strategy for page 1 and schema-root updates

## Working Definitions

- **Lane-local**: owned by one writer lane for the transaction lifetime. No
  cross-lane mutation on the hot path.
- **NUMA-local**: shared only inside one NUMA or LLC locality domain.
- **Socket-local**: shared across locality domains on the same package when a
  second aggregation tier is required.
- **Global**: shared across every writer touching the same database path.

For Track E, a "lane" should be read as a writer execution slot that can be
affined to a physical core in the recommended pinned benchmark profile.

## Evidence Base

This map is grounded in the live implementation, primarily:

- `crates/fsqlite-core/src/connection.rs`
- `crates/fsqlite-vdbe/src/engine.rs`
- `crates/fsqlite-mvcc/src/core_types.rs`
- `crates/fsqlite-mvcc/src/invariants.rs`

The canonical many-core benchmark matrix lives in
`sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.json`, and the
existing performance/design narrative is in
`STATE_OF_THE_CODEBASE_AND_NEXT_STEPS.md`.

## Current Live Pipeline

### Stage 0: Per-Database Runtime Bootstrap

Live path:

- `Connection::open_with_env` opens the pager backend, looks up the
  per-database `SharedMvccState`, registers a per-connection region, starts the
  write-coordinator service scaffold, and reloads the connection-local
  compatibility image as needed.

Ownership:

- Per-database shared runtime: `SharedMvccState`
- Per-connection runtime: `Connection`

Shared touch points:

- `ConcurrentRegistry`
- `InProcessPageLockTable`
- `CommitIndex`
- `next_commit_seq`
- shared runtime region tree and poison state

Observation:

- The write-coordinator region exists today, but
  `ensure_write_coordinator_service_started()` currently boots lifecycle
  scaffolding rather than a true commit/publication pipeline. Track E should not
  assume a real batching service already exists.

### Stage 1: Connection-Local Front-End Work

Live path:

- SQL text enters `Connection` and is handled by connection-local parse cache,
  compiled cache, schema registry, trigger/view registries, pragma state,
  virtual-table registries, and the connection-local `MemDatabase`
  compatibility image.

Ownership:

- One connection

Shared touch points:

- None on the steady-state front-end fast path

Observation:

- This stage already has the right locality shape. It is large, but it is not
  intrinsically cross-lane shared.

### Stage 2: Transaction Admission and Snapshot Binding

Live path:

- `execute_begin` keeps plain `BEGIN` promoted to concurrent mode when
  `concurrent_mode_default` is true.
- The connection binds to the pager publication plane first, then begins the
  pager transaction, then registers a concurrent session in
  `ConcurrentRegistry`.

Ownership transition:

- Lane-local connection state becomes an active transaction with a bound
  snapshot and optional concurrent session id.

Shared touch points:

- pager publication snapshot
- `ConcurrentRegistry::begin_concurrent`

Observation:

- Snapshot establishment is still cheap enough to treat as admission control,
  but it already crosses from connection-local state into globally visible
  transaction metadata.

### Stage 3: Statement Execution and Local Intent Formation

Live path:

- `SharedTxnPageIo` lets multiple cursors share one pager transaction.
- `ConcurrentContext` captures the stable snapshot high watermark, session id,
  shared handle, lock table, commit index, and busy-timeout budget.
- Repeated writes to already-owned pages stay on a lane-local fast path.

Ownership:

- One lane owns the active statement, transaction handle, and staged page data.

Shared touch points:

- None for reads and already-owned writes

Observation:

- The important distinction is not "read vs write"; it is "already-owned write"
  vs "first-touch or structural write." The former is local. The latter is
  where the architecture is won or lost.

### Stage 4: First-Touch Lock Acquisition and Conflict-Surface Expansion

Live path:

- `write_page_internal` classifies writes into:
  - `Tier0AlreadyOwned`
  - `Tier1FirstTouch`
  - `Tier2CommitSurfaceRare`
- Tier 1 and Tier 2 consult `CommitIndex::latest`, attempt page-lock
  acquisition through `InProcessPageLockTable`, and may park on
  `wait_for_page_lock_holder_change`.
- Tier 2 expands the conflict surface when page-1 tracking is required for
  structural changes.

Ownership transition:

- Lane-local intent becomes a shared claim on page ownership and conflict
  metadata.

Shared touch points:

- `InProcessPageLockTable`
- `CommitIndex`
- page-1 synthetic conflict tracking
- shared concurrent handle metadata

Observation:

- This is the current hot shared boundary. The code already keeps the
  contention path narrow by delaying shared interaction until first touch, but
  the ownership directory and visibility metadata are still global structures.

### Stage 5: Commit Planning, Durable Commit, and Publish

Live path:

- `execute_commit` gathers pending conflict pages, runs
  `plan_concurrent_commit`, commits the pager transaction, advances the commit
  clock, and then finalizes the concurrent commit.
- `plan_concurrent_commit_with_registry` consults registry state, SSI evidence,
  and pending conflict pages before the durable commit step.
- `finalize_concurrent_commit_with_registry` publishes to `CommitIndex`,
  releases locks, records SSI evidence, and recycles the session handle.

Ownership transition:

- Lane-local prepared write set becomes globally visible committed state.

Shared touch points:

- `ConcurrentRegistry`
- `CommitIndex`
- `next_commit_seq`
- lock release and waiter wakeups

Observation:

- This is the correct place for a future tiny-publish window. Everything before
  it should stay as local as possible; everything after it should be
  read-mostly or asynchronous.

### Stage 6: Post-Commit Invalidation, Snapshot Capture, and Reclamation

Live path:

- After commit the connection clears transaction-local state, emits
  invalidations, captures a time-travel snapshot, and maybe runs MVCC GC.
- `VersionStore` owns committed page versions and `gc_tick` prunes superseded
  history subject to the oldest active snapshot horizon.

Ownership:

- Cleanup is lane-local; committed-version management is shared.

Shared touch points:

- `VersionStore`
- `VersionGuardRegistry` through GC
- active-snapshot horizon derived from concurrent sessions

Observation:

- Reclamation is global correctness work, but it is not supposed to be on the
  transaction critical path except for the minimal bookkeeping needed to
  preserve safety.

## State Placement Contract

The table below states where each class of state should live for many-core
work, even when the current implementation still keeps it in a broader shared
container.

| State or responsibility | Current live home | Target placement | Why |
| --- | --- | --- | --- |
| Parse cache, compiled cache, table-execution metadata cache | `Connection` | Lane-local | Reuse is highest within one lane; sharing them globally would create metadata contention for modest upside. |
| Schema/view/trigger registries, pragma state, attached-schema registry | `Connection` | Lane-local with read-mostly refresh from committed catalog state | Planning should be local; catalog publication can remain global and invalidate or refresh local copies. |
| Compatibility `MemDatabase` image and time-travel ring | `Connection` | Lane-local | It is explicitly a compatibility image; it must not become a shared mutable object. |
| Active pager txn handle, savepoints, txn snapshot, read/write sets, staged page images | `Connection` plus concurrent handle | Lane-local | This is the natural private working set for a writer. |
| First-touch ownership directory for hot pages | `InProcessPageLockTable` | NUMA-local | First-touch arbitration is the main shared write hot spot; it should be partitioned by locality rather than remain globally hot. |
| Wait queues and wakeup gates for lock handoff | `InProcessPageLockTable` | NUMA-local | Wake traffic should stay close to the home ownership directory. |
| Commit-index read path | `CommitIndex` | NUMA-local mirrors with tiny global publish authority | Readers need cheap visibility checks; writers should not fight on one global metadata surface more than necessary. |
| Prepared-write aggregation before durable publish | partly implicit inside commit path | NUMA-local first, socket-local if a second tier is needed | This is the natural place to absorb locality and batching without widening the global surface. |
| Structural-mutation service for page-1 and catalog-root work | ad hoc page-1 synthetic conflict surface | Socket-local or global narrow lane | Structural work is rare but globally relevant; isolate it rather than pollute the ordinary page-write path. |
| Durable order counter (`next_commit_seq`) | `SharedMvccState` | Global | Total durable order is irreducibly global. The goal is not to remove it, only to keep it tiny. |
| Canonical committed-version store (`VersionStore`) | shared per database path | Global authoritative store | Snapshot correctness and GC need one canonical committed history, even if read caches are local. |
| SSI evidence, conflict analytics, checkpoint and invalidation publication | mixed shared services | Global asynchronous tail work | These are global concerns, but they should stay off the critical write path whenever possible. |

## Ownership Boundaries That Must Stay Explicit

### Boundary A: Connection/Lane to Shared Admission

The transition from Stage 1 to Stage 2 is where a lane acquires a durable
snapshot contract. The key rule is:

- bind the snapshot before shared commit visibility can advance past it

That rule is already encoded by binding to pager publication before
`begin_concurrent`.

### Boundary B: Local Mutation to First-Touch Arbitration

The transition from Stage 3 to Stage 4 is the key many-core boundary:

- already-owned page mutation stays lane-local
- first-touch mutation may consult shared ownership state
- structural mutation may widen the conflict surface, but only on the rare path

This boundary must stay narrow even if the underlying ownership directory
changes shape.

### Boundary C: Prepared State to Global Publish

The transition from Stage 4 to Stage 5 is where Track E should concentrate its
design effort:

- page images, intent logs, and conflict evidence should already be locally
  prepared before the global publish step
- the global publish step should allocate durable order, publish the committed
  surface, and release ownership
- anything that does not need the global order point should move before or
  after it

### Boundary D: Committed State to Reclamation

The transition from Stage 5 to Stage 6 must preserve:

- committed versions remain visible to every active snapshot that can still see
  them
- GC only prunes versions below the oldest active horizon
- memory reclamation follows the guard/epoch discipline, not convenience timing

## Correctness and Reclamation Invariants

`INV-DB300-E1.1-1` Concurrent-by-default must remain intact:

- `BEGIN` promotion to concurrent mode is a project invariant, not a tuning
  knob

`INV-DB300-E1.1-2` Snapshot validity:

- a writer must never begin from a snapshot newer than the bound pager
  publication plane

`INV-DB300-E1.1-3` First-touch exclusivity:

- no two live concurrent writers may both own the same page write lock at once

`INV-DB300-E1.1-4` Commit-index monotonicity:

- per-page commit visibility must never move backward

`INV-DB300-E1.1-5` Publish-after-durable-order:

- durable order assignment and public visibility publication must stay coupled;
  readers must not observe a half-published commit

`INV-DB300-E1.1-6` Structural conflict explicitness:

- page-1 or catalog-root interaction must remain explicit; hidden widening of
  the conflict surface is not acceptable

`INV-DB300-E1.1-7` Reclamation safety:

- `VersionStore` may prune only versions that are older than the active horizon
  and are protected by the guard registry / GC discipline

`INV-DB300-E1.1-8` Shutdown and region quiescence:

- background coordinator or helper services may stop only after their region
  and waiters are quiescent enough to preserve visibility and wakeup safety

## Current Shared Surfaces That Need Pressure Relief

These are the concrete shared surfaces the comparison bead should score:

- `ConcurrentRegistry` mutex traffic during admission, commit planning, and
  recycle
- `InProcessPageLockTable` shard contention and wakeup traffic
- `CommitIndex` reads on first touch and writes on publish
- page-1 synthetic conflict tracking for structural work
- global publish serialization around `next_commit_seq`

The right question for `bd-db300.5.1.2` is not "can we remove all shared
state?" It is:

- which of these surfaces are truly global
- which should become NUMA-local or socket-local
- which can be moved entirely out of the ordinary writer hot path

## Consequences for `bd-db300.5.1.2` and `.3`

The comparison and decision beads should evaluate candidate architectures
against this contract:

1. Keep Stages 1 through 3 lane-local.
2. Move Stage 4 ownership arbitration as close to NUMA-local as correctness
   allows.
3. Make Stage 5 the only tiny global publish window.
4. Keep Stage 6 asynchronous except for minimal correctness bookkeeping.
5. Treat structural mutation as a separate rarity path, not as the default
   writer architecture.

If a candidate design widens Stage 4 or Stage 5 into a long-lived shared
critical section, it is a regression for Track E even if it looks simpler on
paper.
