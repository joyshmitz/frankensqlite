# ADR-0002: Many-Core Architecture Selection

**Status:** Accepted
**Date:** 2026-03-18
**Bead:** bd-db300.5.1.3

## Context

Track E needs one primary many-core architecture for FrankenSQLite's
transaction pipeline. The preceding design work established:

- the live stage and state-placement map in
  `docs/design/many-core-transaction-pipeline-state-placement.md`
- the direct comparison of the three candidate architectures in
  `docs/design/many-core-architecture-comparison.md`
- the current strongest evidence from
  `STATE_OF_THE_CODEBASE_AND_NEXT_STEPS.md`: the system should avoid shared
  structural mutation whenever possible, and when shared publication is
  unavoidable, that window must be extremely short

The decision must remove architecture ambiguity for the next implementation
beads while still naming the residual global work honestly.

## Options Considered

### 1. Pinned Share-Nothing Lanes

**Pros:**
- strongest ordinary-path locality
- simple one-writer/one-core mental model
- naturally aligns with existing connection-local caches

**Cons:**
- does not honestly remove the shared surfaces already present in this codebase
- performs badly once page-1, allocator, or structural B-tree work enters the
  picture
- tends to express contention as retry storms rather than a small explicit
  publish boundary

### 2. Tiny-Publish Shared State

**Pros:**
- best fit to the live code shape
- matches the strongest current performance interpretation
- keeps most work lane-local while acknowledging one small global publish step
- lets Track E attack the structural problem before building a queue hierarchy

**Cons:**
- requires strict discipline to keep the publish window tiny
- page-1 and allocator surfaces still need explicit isolation
- optimistic validation and retry rules must remain crisp

### 3. Hierarchical NUMA/Socket Hybrid

**Pros:**
- strongest explicit answer for cross-domain ownership and wake penalties
- gives a clear future path for Threadripper/Xeon scale-up
- offers explicit queueing and backpressure tiers

**Cons:**
- highest implementation and debugging complexity
- easy to introduce head-of-line blocking at aggregators
- likely too much machinery before the single-locality-domain path is fixed

## Decision

Adopt **tiny-publish shared state** as the primary Track E architecture.

The other options are retained as follows:

- **Hierarchical NUMA/socket hybrid** stays as the explicit scale-up fallback if
  cross-domain remote-ownership and wake penalties remain dominant after the
  tiny-publish path is implemented and measured.
- **Pinned share-nothing lanes** stays visible as a rejected primary direction,
  but remains useful as a locality discipline and comparison baseline.

## Expected Many-Core Win Mechanism

The expected scale mechanism is:

1. Keep Stage 1 through Stage 3 work lane-local:
   - parse and compile reuse
   - transaction-local planning
   - owned-page mutation
   - local page-image construction
2. Make first-touch arbitration narrow and locality-aware instead of turning it
   into a long-lived shared region.
3. Turn structural mutation into:
   - optimistic local planning first
   - explicit rarity-path publication second
4. Make the final global publish step do only the work that is truly
   irreducible:
   - durable-order assignment
   - final visibility publication
   - lock release and wakeup

This is the smallest architecture change that directly attacks the current
shared-structural convoy problem without pretending that durable order and
committed visibility can be fully sharded away.

## Residual Serialized Region

The residual serialized region is intentionally small and explicit:

- one authoritative durable-order allocator / commit-sequence authority
- final committed-visibility publication into the shared commit surface
- explicit structural publication for page-1 or catalog-root mutations when
  required

Everything else is expected to remain outside that region:

- parse and compile work
- transaction-local page planning
- already-owned page writes
- most retry and validation work
- post-commit asynchronous evidence and observability work

## Tradeoffs

### Durability Semantics

The decision keeps one authoritative committed order for the database path.
Track E does **not** adopt per-domain durability authorities or divergent local
commit orders.

Why:

- correctness and crash-recovery reasoning stay much simpler
- the repo already has shared durable-order and committed-visibility surfaces
- the project's problem is not "too little hierarchy" yet; it is that the
  shared window is too large and structurally polluted

### Latency Protection

Latency protection comes from keeping queueing narrow and explicit:

- no return to broad structural preclaim
- no hidden widening of page-1 or allocator conflict surfaces
- backpressure should appear at first-touch arbitration and the final publish
  boundary, not as broad retry churn across the whole write path

### Implementation Tractability

This is the most tractable choice because it can evolve the live code rather
than replace it wholesale.

It reuses and sharpens existing boundaries around:

- `SharedTxnPageIo`
- first-touch lock acquisition
- commit planning
- `CommitIndex`
- `VersionStore`
- explicit commit publication

The hierarchical hybrid remains a plausible later extension, but it is not the
first architecture to build.

## Verification Obligations

The implementation beads that follow this decision must verify all of the
following:

1. **Publish-window boundedness**
   - instrument Stage 4 and Stage 5 timing so the final shared window can be
     measured directly

2. **Structural-path explicitness**
   - prove that page-1, allocator, and parent-page structural work stay on an
     explicit rarity path rather than silently leaking into the common path

3. **Retry-cause taxonomy**
   - separate hot-page contention, structural amplification, page-1/allocator
     false disjointness, and residual publish serialization in benchmark
     artifacts

4. **Pinned-profile win**
   - demonstrate improvement on the repository's `recommended_pinned`
     many-core profile without regressing low-concurrency or single-writer
     behavior

5. **Adversarial-profile honesty**
   - rerun the `adversarial_cross_node` profile so the project knows whether
     remote ownership remains the next limiter or whether the tiny-publish path
     is sufficient for the current scale target

6. **Concurrency-by-default preservation**
   - preserve `BEGIN` promoting to concurrent mode by default throughout the
     implementation

7. **No accidental hierarchy**
   - if queue tiers or aggregators appear during implementation, they must be
     justified as explicit follow-on work rather than creeping into the primary
     design by accident

## Consequences

- Track E implementation work should optimize for a lane-local pipeline with one
  tiny shared publish window.
- Structural-mutation isolation is now a first-class requirement, not an
  optional optimization.
- Cross-domain hierarchy is deferred until measurements show that the
  tiny-publish design is no longer the dominant limiter.
