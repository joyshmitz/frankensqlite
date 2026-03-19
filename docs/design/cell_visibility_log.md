# CellVisibilityLog Design Document

> **Bead**: bd-l9k8e.1 (C1: Design the CellVisibilityLog data structure)
> **Status**: DRAFT
> **Author**: Agent (Claude Opus 4.5)
> **Date**: 2026-03-19

## Executive Summary

This document specifies the `CellVisibilityLog` data structure that enables cell-level MVCC for FrankenSQLite. The core insight is that **most database mutations are logical row operations** (INSERT/UPDATE/DELETE on individual rows) that should not require cloning entire 4KB pages.

By separating **logical operations** (cell-level) from **structural operations** (page-level), we enable concurrent writers to modify different rows on the same page without conflict.

---

## Problem Statement

### Current Architecture

The existing MVCC design (`crates/fsqlite-mvcc/src/invariants.rs`) uses `PageVersion`:

```rust
pub struct PageVersion {
    pub pgno: PageNumber,
    pub commit_seq: CommitSeq,
    pub created_by: TxnToken,
    pub data: PageData,      // FULL 4KB page clone!
    pub prev: Option<VersionPointer>,
}
```

**Cost per row operation:**
- 4KB page clone (allocation + copy)
- Version chain insertion
- Chain walk on every read (O(chain_length))
- GC pressure for short-lived versions
- SSI witness overhead

### The Problem

If transaction A inserts row 1000 and transaction B inserts row 1001, both on page 47:
- **Current**: PAGE-LEVEL CONFLICT. One wins, one retries.
- **Desired**: NO CONFLICT. Both rows are on different cells of the same page.

---

## Design Goals

1. **Cell-level conflict detection**: Only conflict when two transactions modify the *same cell*
2. **O(1) visibility resolution**: Snapshot lookup should be constant-time, not O(chain_length)
3. **Bounded memory**: Cell deltas must fit within a memory budget (~10% of page cache)
4. **SQLite-compatible persistence**: On-disk pages remain SQLite format; cell versioning is memory-only
5. **Correct SSI**: Write-skew detection still works at the granularity that matters

---

## CellKey Design

The hardest design decision: how to uniquely identify a cell within a page.

### Page Types in SQLite

| Page Type | Flag | Key Structure | Cell Content |
|-----------|------|---------------|--------------|
| LeafTable (0x0D) | Leaf | RowId (i64) | Record payload |
| LeafIndex (0x0A) | Leaf | Encoded key + RowId suffix | (none) |
| InteriorTable (0x05) | Interior | RowId | Child page pointer |
| InteriorIndex (0x02) | Interior | Encoded key | Child page pointer |

### Decision: CellKey Enum

```rust
/// Unique identifier for a cell within a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellKey {
    /// Table leaf page: cells are keyed by RowId (simple, unique, efficient).
    TableRow(RowId),

    /// Index leaf page: cells are keyed by hash of encoded key + RowId suffix.
    /// The hash provides fast equality checking; RowId disambiguates collisions.
    IndexEntry {
        /// XXH3 hash of the full encoded index key (excluding RowId suffix).
        key_hash: u64,
        /// RowId suffix from the index entry (for collision disambiguation).
        rowid: RowId,
    },
}
```

**Rationale:**
- **TableRow**: RowId is the natural primary key. Unique, monotonic for AUTOINCREMENT.
- **IndexEntry**: Index keys are variable-length byte sequences. Hashing gives O(1) lookup. RowId suffix handles hash collisions (each index entry includes the table's RowId).

**Not included:**
- **Interior pages**: Always use page-level MVCC. They contain structural information (child pointers) that affects tree traversal. Cell-level versioning would complicate cursor navigation.

---

## CellDelta Design

Each cell modification is recorded as a delta, not a full cell snapshot.

```rust
/// A single cell modification operation.
#[derive(Debug, Clone)]
pub enum CellOp {
    /// Cell inserted with this content.
    Insert(CellContent),

    /// Cell updated to this content.
    /// (We store full content, not a diff, for simplicity. Diffs add complexity
    /// with minimal gain since cells are typically small.)
    Update(CellContent),

    /// Cell deleted (tombstone).
    Delete,
}

/// Cell content: the payload bytes without the cell header.
#[derive(Debug, Clone)]
pub struct CellContent {
    /// Raw cell payload (record data for tables, key data for indexes).
    /// Does NOT include cell header (payload size, rowid varint, etc.).
    /// Typical size: 20-200 bytes for most rows.
    pub payload: SmallVec<[u8; 64]>,

    /// For overflow cells: first overflow page number.
    /// None if the entire payload fits in the leaf page.
    pub overflow_page: Option<PageNumber>,
}

/// A versioned cell delta with commit metadata.
#[derive(Debug, Clone)]
pub struct CellDelta {
    /// The operation performed.
    pub op: CellOp,

    /// Commit sequence (0 = uncommitted/in write-set).
    pub commit_seq: CommitSeq,

    /// Transaction that created this delta.
    pub created_by: TxnToken,

    /// Timestamp for age-based GC prioritization.
    pub created_at_ms: u64,
}
```

---

## CellVisibilityLog Structure

The main data structure holding all cell-level version information.

```rust
/// The cell-level visibility log for MVCC.
///
/// This is the core innovation: logical row operations are tracked at cell
/// granularity, not page granularity. Structural operations (splits, merges)
/// still use the page-level VersionStore.
pub struct CellVisibilityLog {
    /// Primary index: (PageNumber, CellKey) -> version chain.
    /// Uses a sharded concurrent map for scalability.
    cells: ShardedCellMap,

    /// Memory budget tracking.
    budget: CellMemoryBudget,

    /// GC state for opportunistic pruning.
    gc_state: CellGcState,

    /// Per-page cell count for split detection.
    page_cell_counts: DashMap<PageNumber, u16>,
}

/// Sharded map for concurrent cell access.
/// 64 shards = good balance for typical concurrency.
struct ShardedCellMap {
    shards: [RwLock<HashMap<(PageNumber, CellKey), CellVersionChain>>; 64],
}

/// Version chain for a single cell.
/// Ordered by commit_seq descending (newest first).
struct CellVersionChain {
    /// The deltas, newest first.
    deltas: SmallVec<[CellDelta; 4]>,

    /// The "base" page version's commit_seq at which this chain was started.
    /// Used to fall back to page-level resolution for old snapshots.
    base_commit_seq: CommitSeq,
}
```

---

## Visibility Resolution

### Algorithm

Given snapshot `S` and key `(page, cell_key)`:

```rust
fn resolve_cell_visibility(
    &self,
    page: PageNumber,
    cell_key: CellKey,
    snapshot: &Snapshot,
) -> CellVisibility {
    // 1. Find the cell's version chain (if any).
    let chain = match self.cells.get((page, cell_key)) {
        Some(chain) => chain,
        None => return CellVisibility::NotInLog, // Fall back to page-level
    };

    // 2. If snapshot is older than the chain's base, fall back to page-level.
    if snapshot.high < chain.base_commit_seq {
        return CellVisibility::FallbackToPage;
    }

    // 3. Find the newest delta visible to this snapshot.
    for delta in &chain.deltas {
        if delta.commit_seq.get() > 0 && delta.commit_seq <= snapshot.high {
            return match &delta.op {
                CellOp::Insert(content) => CellVisibility::Exists(content.clone()),
                CellOp::Update(content) => CellVisibility::Exists(content.clone()),
                CellOp::Delete => CellVisibility::Deleted,
            };
        }
    }

    // 4. No visible delta: cell doesn't exist for this snapshot.
    CellVisibility::NotVisible
}

enum CellVisibility {
    /// Cell exists with this content.
    Exists(CellContent),
    /// Cell was deleted.
    Deleted,
    /// Cell not in the log; fall back to page-level resolution.
    NotInLog,
    /// Snapshot too old; fall back to page-level resolution.
    FallbackToPage,
    /// No visible version (cell doesn't exist in this snapshot).
    NotVisible,
}
```

**Complexity**: O(1) hash lookup + O(chain_length) scan. But chains are short (pruned by GC) and SmallVec is cache-friendly.

---

## Memory Budget Model

### Overhead Per Delta

| Component | Size | Notes |
|-----------|------|-------|
| CellKey | 16 bytes | Enum discriminant + RowId (8) + key_hash (8) |
| CellDelta header | 32 bytes | op discriminant + commit_seq + created_by + timestamp |
| CellContent | 64+ bytes | SmallVec inline + overflow page |
| HashMap entry overhead | ~40 bytes | Bucket pointer, hash, metadata |
| **Total per delta** | **~152 bytes** | Conservative estimate |

### Budget Calculation

For a 256MB page cache:
- Cell delta budget: 10% = ~25MB
- At 152 bytes/delta: **~164K outstanding deltas**
- At 4 deltas/cell average: ~41K unique cells tracked

This is sufficient for workloads with high write throughput. When the budget is exceeded:
1. Trigger eager materialization of oldest cell chains
2. Merge cell deltas back into page versions
3. Prune GC'd chains from the log

### Enforcement

```rust
struct CellMemoryBudget {
    /// Current estimated memory usage in bytes.
    current_bytes: AtomicU64,

    /// Maximum allowed bytes.
    max_bytes: u64,

    /// Threshold for eager materialization (e.g., 80% of max).
    eager_threshold: u64,
}
```

---

## Interaction with Page-Level MVCC

The CellVisibilityLog **complements** the existing VersionStore, not replaces it.

### Dual-Path Resolution

```
┌─────────────────────────────────────────────────────────────┐
│                     Read Path                                │
├─────────────────────────────────────────────────────────────┤
│  1. Check CellVisibilityLog for (page, cell_key)            │
│     ├── Exists(content) → Return cell content               │
│     ├── Deleted → Return "cell not found"                   │
│     ├── NotInLog → Fall through to page-level               │
│     └── FallbackToPage → Fall through to page-level         │
│                                                              │
│  2. Page-level fallback: VersionStore.resolve(page, snap)   │
│     └── Extract cell from resolved page image               │
└─────────────────────────────────────────────────────────────┘
```

### Materialization (Cell → Page)

At persistence boundaries (checkpoint, commit with structural changes):

1. For each page with cell deltas:
   a. Resolve base page from VersionStore
   b. Apply cell deltas in commit_seq order
   c. Serialize the updated page
   d. Publish as a new PageVersion
   e. Clear the cell chains (they're now in the page)

---

## GC Strategy

### Opportunistic Pruning (Postgres-style)

- On read access to a cell chain, prune deltas older than the oldest active snapshot
- Batch pruning when memory pressure triggers materialization
- No blocking GC — all pruning is lazy and incremental

### Chain Length Limit

- Maximum 8 deltas per chain before forcing materialization
- Prevents unbounded chain growth for hot cells

---

## Postgres Reference Implementation

| Postgres Concept | FrankenSQLite Equivalent |
|------------------|--------------------------|
| `HeapTupleHeaderData.t_xmin/t_xmax` | `CellDelta.created_by` + `CellDelta.commit_seq` |
| `SnapshotData` (htup.h) | `Snapshot` (unchanged) |
| `HeapTupleSatisfiesSnapshot` | `resolve_cell_visibility()` |
| `heap_page_prune()` | Opportunistic chain pruning |
| `HEAP_XMIN_COMMITTED` hint bit | `commit_seq > 0` check |

---

## Deliverables

1. **This design document** (docs/design/cell_visibility_log.md)
2. **Stub types** in `crates/fsqlite-mvcc/src/cell_visibility.rs`
3. **CellKey enum** with explicit documentation
4. **Memory budget model** with per-delta overhead calculation

---

## Open Questions

1. **Overflow cells**: Should we track overflow page chains in the CellVisibilityLog, or always fall back to page-level for overflow cells?
   - **Recommendation**: Fall back to page-level. Overflow cells are rare and complex.

2. **Index key hashing**: XXH3 vs FxHash vs other?
   - **Recommendation**: XXH3. It's already a dependency and has excellent distribution.

3. **Chain ordering**: Newest-first vs oldest-first?
   - **Recommendation**: Newest-first. Most reads want the latest visible version.

---

## Next Steps

- **C2**: Define the structural vs logical operation boundary
- **C3**: Implement CellVisibilityLog and cell-level snapshot resolution
