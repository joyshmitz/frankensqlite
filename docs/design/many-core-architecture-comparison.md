# Many-Core Architecture Comparison (`bd-db300.5.1.2`)

## Purpose

Compare the three main Track E candidate architectures against the live
FrankenSQLite pipeline and the many-core measurement contract already defined in
this repository:

- pinned share-nothing lanes
- tiny-publish shared state
- hierarchical NUMA/socket hybrid

This artifact builds directly on
`docs/design/many-core-transaction-pipeline-state-placement.md`.

## Scope

The goal is not to write the final decision record yet. The goal is to make the
tradeoffs explicit enough that `bd-db300.5.1.3` can choose a winner without
repeating the same reasoning.

The comparison must keep rejected options visible. A design can lose as the
**primary** Track E direction while still remaining useful as:

- a fallback mode
- a later-stage scale-up path
- a diagnostic or adversarial benchmark configuration

## Hardware and Measurement Assumptions

The comparison is grounded in the repo's own many-core evidence contract:

- `recommended_pinned` means one thread per physical core, workers kept inside
  one NUMA or LLC locality domain, memory bound local, and helper-lane work
  kept in the same locality on a housekeeping CPU.
- `adversarial_cross_node` means workers are deliberately spread across NUMA
  nodes or CCD boundaries to surface remote ownership, wake, and cache-line
  penalties.
- the target hardware class is `linux_x86_64_many_core_numa`, which explicitly
  includes Threadripper- and Xeon-class hosts with NUMA or CCD boundaries.
- the topology bundle requires operator-visible disclosure of sockets,
  cores-per-socket, NUMA state, worker affinity, helper-lane placement, and
  memory policy.

This matters because a design that only works when placement is perfect is a
different proposition from a design that remains understandable and debuggable
when the machine or deployment is less cooperative.

## Candidate A: Pinned Share-Nothing Lanes

### Shape

Each writer lane owns its own:

- connection-local planning state
- transaction-local write working set
- preferred CPU/core affinity
- local allocator and staging behavior as much as possible

Cross-lane interaction is treated as an exception rather than a normal part of
the write path.

### What It Gets Right

- strongest cache locality in the ordinary lane-local path
- clear mental model for one writer on one core
- naturally aligns with the current connection-local parse/compile/cache state

### What It Gets Wrong For This Codebase

- the current engine still has irreducible shared surfaces: page ownership,
  commit visibility, durable order, page-1 structural metadata, and committed
  version history
- "share nothing" becomes misleading once disjoint SQL operations still collide
  on structural B-tree or allocator surfaces
- under structural pressure it tends to express backpressure as retries,
  rework, and convoyed hot lanes rather than as a small explicit publish window

### Failure Signature

- one or a few lanes become retry magnets
- skewed hot pages create starvation or fairness drift
- memory duplication grows while structural conflicts remain

## Candidate B: Tiny-Publish Shared State

### Shape

Writers perform as much work as possible locally:

- plan local page images
- keep owned-page writes lane-local
- delay shared interaction until first-touch arbitration or final publish

The shared portion is deliberately tiny and focused on:

- ownership validation
- durable-order assignment
- final visibility publication
- explicit structural rarity paths

### What It Gets Right

- matches the strongest current interpretation in
  `STATE_OF_THE_CODEBASE_AND_NEXT_STEPS.md`: avoid shared structural mutation
  when possible, and make the unavoidable publish window extremely short
- best fit to the live code, which already distinguishes already-owned writes,
  first-touch arbitration, and commit-time publication
- keeps the global invariants visible instead of pretending they do not exist

### What It Still Risks

- the "tiny" publish window can silently grow over time if more validation,
  bookkeeping, or structural work accretes into it
- page-1 and allocator metadata can still poison the fast path unless isolated
- optimistic validation/retry rules must be extremely crisp to stay correct

### Failure Signature

- global publish becomes a disguised queue
- rare structural work leaks into the common path
- lock-table and commit-index traffic stay hotter than intended

## Candidate C: Hierarchical NUMA/Socket Hybrid

### Shape

Writers prepare locally but shared coordination is tiered:

- lane-local prepare
- NUMA-local ownership and batching
- optional socket-local aggregation
- global durable-order and final visibility authority

This design treats remote-domain traffic as a first-class cost and tries to
keep the hottest coordination inside a locality domain.

### What It Gets Right

- directly addresses Threadripper/Xeon remote-ownership and wakeup penalties
- offers explicit places to apply backpressure before traffic reaches the
  global authority
- provides a plausible scale-up path once one-locality-domain performance is
  good and cross-domain scaling becomes the next limiter

### What It Costs

- highest implementation complexity by far
- more queueing layers and therefore more places for head-of-line blocking
- more operator burden: topology discovery, domain grouping, helper-lane
  placement, memory-policy validation, and debugging of cross-tier drift

### Failure Signature

- queue imbalance between domains
- stalled or overloaded local aggregators
- correctness bugs or observability blind spots at tier boundaries

## Direct Comparison

| Criterion | Pinned share-nothing lanes | Tiny-publish shared state | Hierarchical NUMA/socket hybrid |
| --- | --- | --- | --- |
| Queueing behavior | Queueing is mostly implicit and shows up as retries or hot-lane convoys once shared structural surfaces appear. | Queueing is concentrated into the smallest possible explicit publish or structural window. | Queueing is explicit at multiple levels, which helps control bursts but adds latency and more failure surfaces. |
| Backpressure shape | Weakest. Pressure tends to spill outward as abort/retry churn rather than a small bounded choke point. | Best first-order fit. Backpressure can be attached to first-touch arbitration and publish without inventing a queue tree. | Strongest control surface, but also the easiest way to over-engineer the system and bury latency in staging queues. |
| Memory traffic | Best on the pure local path, worst once shared structure causes cross-lane invalidation or duplicated state. | Best tradeoff: local page preparation plus small shared metadata publication. | Better than global shared structures on remote-domain machines, but pays for mirrors, mergers, and cross-tier synchronization. |
| Structural-mutation handling | Poor as a primary design. Structural work breaks the illusion of independence and causes the hardest pathologies. | Good if structural mutation is isolated into a rarity path with optimistic local planning first. | Good in theory, but only if structural traffic can be cleanly routed through the hierarchy without turning every tier into a convoy point. |
| Operator ergonomics | Medium. Requires disciplined pinning, but the model is conceptually simple. | Best. Works with recommended pinned placement without requiring every operator to understand a hierarchy. | Worst. Depends heavily on accurate topology classification, helper-lane placement, and memory-policy discipline. |
| Fit to the current code | Weak. The live implementation already exposes shared MVCC and publish surfaces that this model cannot honestly erase. | Strongest. The code already has the beginnings of local prepare plus shared publish boundaries. | Moderate. Parts of the current code can evolve this way, but only after significant sharding and queue design work. |
| Risk of building the wrong thing first | High. It underestimates page-1, allocator, and structural shared surfaces. | Lowest. It attacks the most credible current bottleneck without forcing a full architecture tree into existence. | High. It may solve tomorrow's cross-domain problem before today's single-domain structural problem is fixed. |

## Queueing and Backpressure Verdict

If the primary Track E target is the recommended pinned profile, the architecture
should:

- keep Stage 1 through Stage 3 lane-local
- keep Stage 4 narrow and locality-aware
- make Stage 5 the only clearly bounded global publish step

That favors **tiny-publish shared state** over both alternatives.

Why the other two lose on this dimension:

- pinned share-nothing lanes do not provide a good answer once structural
  mutation or page-1 metadata enters the picture; the queue still exists, it is
  just disguised as retries and wasted work
- the hierarchical hybrid gives good control over queueing, but it adds queue
  stages before the repo has proved the single-locality-domain version of the
  problem is solved

## Memory-Traffic Verdict

The many-core measurement contract explicitly cares about remote ownership,
wakeup, and cache-line movement penalties.

That implies:

- share-nothing lanes are excellent only while the system stays truly local and
  physically disjoint
- tiny-publish shared state is the best immediate compromise because it keeps
  most data motion local and limits remote chatter to first-touch and publish
- the hierarchical hybrid becomes attractive only when cross-domain traffic
  remains dominant after the tiny-publish design is already working well inside
  one locality domain

## Failure-Mode Verdict

### Rejected as Primary: Pinned Share-Nothing Lanes

Why it loses:

- it hides the real shared surfaces instead of isolating them
- it is too easy to mistake retry storms for "parallelism"
- it does not honestly answer the structural-conflict geometry already shown by
  the current benchmarks

What remains useful:

- lane-locality remains a design principle
- pinning still matters in the recommended benchmark profile
- this design is useful as a lower-bound thought experiment for "maximum local
  state, minimum honest sharing"

### Conditionally Useful, Not First: Hierarchical NUMA/Socket Hybrid

Why it loses as the first move:

- it adds the most machinery before the repo has verified that the tiny-publish
  approach is insufficient
- it raises the operator and debugging burden substantially
- it risks turning a structural-mutation problem into a queue-topology problem

What remains useful:

- it is the right fallback if adversarial cross-node measurements continue to
  show remote ownership and wake traffic dominating after a tiny-publish design
  is in place
- it provides the right mental model for a future multi-domain scale-up path

### Leading Candidate for `bd-db300.5.1.3`: Tiny-Publish Shared State

Why it leads:

- it matches the current code shape
- it matches the strongest interpretation of the failed structural-preclaim
  experiments
- it keeps global state honest but small
- it gives the project the cleanest path to test "optimistic local plan plus
  tiny publish window" before building a full hierarchy

## Recommended Comparison Outcome

The working ranking for the next bead should be:

1. **Tiny-publish shared state** as the primary Track E architecture candidate.
2. **Hierarchical NUMA/socket hybrid** as the explicit scale-up fallback if
   cross-domain traffic remains dominant after the tiny-publish path is proven.
3. **Pinned share-nothing lanes** kept visible as a rejected primary direction,
   but retained as a useful locality discipline and adversarial comparison
   baseline.

That is not yet the final decision record. It is the comparison result that the
decision record should formalize unless new evidence overturns it.
