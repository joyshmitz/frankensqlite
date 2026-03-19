# Structural vs Logical Operation Boundary (C2)

> **Bead**: bd-l9k8e.2 (C2: Define structural vs logical operation boundary)
> **Status**: DRAFT
> **Author**: Agent (Claude Opus 4.5)
> **Date**: 2026-03-19

## Executive Summary

This document defines the boundary between **structural** and **logical** B-tree
operations for FrankenSQLite's cell-level MVCC system. The key insight:

> **Logical operations** modify cell content within existing page boundaries.
> **Structural operations** change the B-tree topology (page allocation, splits,
> merges, depth changes).

Cell-level MVCC (CellVisibilityLog) handles logical operations. Page-level MVCC
(VersionStore) handles structural operations. This separation enables concurrent
writers to modify different rows on the same page without conflict.

---

## Classification Table

### Structural Operations (Page-Level MVCC)

| Operation | Code Location | Trigger Condition | Pages Affected |
|-----------|---------------|-------------------|----------------|
| **balance_deeper** | `balance.rs:96` | Root page overflow | Root + new child |
| **balance_quick** | `balance.rs:201` | Rightmost leaf append, page full | Parent + new sibling |
| **balance_nonroot** | `balance.rs:389` | Page overflow/underflow | Up to 3 siblings + parent |
| **balance_table_leaf_local_split** | `balance.rs:1104` | Table leaf overflow | Leaf + new sibling + parent |
| **balance_shallower** | `balance.rs:1818` | Root has single child | Root absorbs child |
| **balance_for_insert** | `cursor.rs:1805` | `try_insert_on_leaf` returns false | Propagates up tree |
| **balance_for_delete** | `cursor.rs:1966` | Page becomes empty (cell_count=0) | Propagates up tree |
| **Overflow chain create** | `cursor.rs` (encode_*_cell) | Payload > local max | Leaf + overflow pages |
| **Overflow chain free** | `cursor.rs:remove_cell_from_leaf` | Cell with overflow deleted | Overflow pages freed |
| **Freelist allocate** | `PageWriter::allocate_page` | Any page needed | Freelist trunk/leaf |
| **Freelist free** | `PageWriter::free_page` | Page no longer needed | Freelist trunk/leaf |
| **Interior cell replace** | `cursor.rs:replace_interior_cell` | Index delete (successor promotion) | Interior page |

### Logical Operations (Cell-Level MVCC)

| Operation | Code Location | Trigger Condition | Effect |
|-----------|---------------|-------------------|--------|
| **Cell insert (fits)** | `cursor.rs:try_insert_on_leaf` | Cell fits in free space | Single page cell array + content |
| **Cell delete (page non-empty)** | `cursor.rs:remove_cell_from_leaf` | `new_count > 0` | Single page cell array |
| **Cell update (same size)** | N/A (currently rewrites) | Payload size unchanged | Single page content area |

---

## Decision Boundary

The fundamental question: **Does this operation change only cell content, or does
it change page structure/allocation?**

```
                    ┌─────────────────────────────────────────┐
                    │           B-tree Mutation               │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │  Does cell fit in existing page space?  │
                    └─────────────────┬───────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                    ┌────▼────┐              ┌─────▼─────┐
                    │   YES   │              │    NO     │
                    └────┬────┘              └─────┬─────┘
                         │                         │
              ┌──────────▼──────────┐    ┌─────────▼─────────┐
              │  LOGICAL OPERATION  │    │ STRUCTURAL (split)│
              │  → CellVisibilityLog│    │ → VersionStore    │
              └─────────────────────┘    └───────────────────┘

                    ┌─────────────────────────────────────────┐
                    │             Cell Deletion               │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │    Does page become empty after delete? │
                    └─────────────────┬───────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                    ┌────▼────┐              ┌─────▼─────┐
                    │   NO    │              │    YES    │
                    └────┬────┘              └─────┬─────┘
                         │                         │
              ┌──────────▼──────────┐    ┌─────────▼─────────┐
              │  LOGICAL OPERATION  │    │ STRUCTURAL (merge)│
              │  → CellVisibilityLog│    │ → VersionStore    │
              └─────────────────────┘    └───────────────────┘
```

---

## Predicate Function

```rust
/// Determines whether a B-tree mutation is structural (requiring page-level
/// MVCC) or logical (eligible for cell-level MVCC).
///
/// This is called AFTER the mutation attempt to classify what happened.
/// The caller must provide the outcome of the mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationOutcome {
    /// Cell was inserted/updated in-place without page restructuring.
    CellFit,
    /// Cell was deleted and page still has other cells.
    CellRemovedPageNonEmpty,
    /// Insert triggered page split (balance_for_insert was called).
    PageSplit,
    /// Delete triggered page merge (balance_for_delete was called).
    PageMerge,
    /// Overflow chain was created or modified.
    OverflowChainModified,
    /// Interior page cell was replaced (index delete with successor promotion).
    InteriorCellReplaced,
}

impl MutationOutcome {
    /// Returns true if this mutation is structural (page-level MVCC required).
    #[inline]
    #[must_use]
    pub const fn is_structural(self) -> bool {
        match self {
            // Logical operations: only affect cell content within one page.
            Self::CellFit | Self::CellRemovedPageNonEmpty => false,

            // Structural operations: change B-tree topology.
            Self::PageSplit
            | Self::PageMerge
            | Self::OverflowChainModified
            | Self::InteriorCellReplaced => true,
        }
    }

    /// Returns true if this mutation can use cell-level MVCC.
    #[inline]
    #[must_use]
    pub const fn is_logical(self) -> bool {
        !self.is_structural()
    }
}

/// Pre-mutation check: can this insertion potentially be logical?
///
/// Returns false if structural operation is guaranteed (e.g., payload requires
/// overflow). Returns true if the operation MIGHT be logical (cell might fit).
/// Actual outcome depends on page free space at mutation time.
#[must_use]
pub fn can_be_logical_insert(
    payload_size: usize,
    local_max: usize,
    page_free_space: usize,
    cell_overhead: usize,
) -> bool {
    // If payload requires overflow, it's structural.
    if payload_size > local_max {
        return false;
    }

    // If cell (with overhead) fits in current free space, it's logical.
    let total_cell_size = payload_size + cell_overhead;
    total_cell_size <= page_free_space
}

/// Pre-mutation check: will this deletion be logical?
///
/// Returns true if the page will have cells remaining after deletion.
pub fn will_be_logical_delete(current_cell_count: u16) -> bool {
    current_cell_count > 1
}
```

---

## Edge Cases and Rationale

### 1. Overflow Cells → Always Structural

**Rationale**: Overflow chains span multiple pages. Even if the leaf cell "fits",
the overflow pages are separate allocations. Tracking cell-level deltas for the
leaf cell while the overflow pages use page-level versioning creates consistency
hazards.

**Decision**: Any mutation that creates, extends, or frees overflow pages is
classified as structural.

### 2. Interior Page Modifications → Always Structural

**Rationale**: Interior pages contain separator keys and child pointers that
affect tree traversal. Cell-level versioning of interior pages would require
cursor operations to understand version chains, significantly complicating
navigation.

**Decision**: All interior page modifications (including the "replace separator
with successor" step of index deletion) are structural.

### 3. Index Leaf Cells → Logical When No Overflow

**Rationale**: Index leaf cells contain encoded keys. Most index keys are short
enough to fit locally. When they do fit, cell-level MVCC applies just like
table leaf cells.

**Decision**: Index leaf INSERT/DELETE is logical when the cell fits without
overflow.

### 4. Page Defragmentation → Depends

**Rationale**: SQLite defragments pages (compacts cell content area) during
normal operations. This doesn't change cell content, only cell pointer offsets.

**Decision**: Defragmentation that occurs during a logical operation is still
logical. The CellVisibilityLog tracks cells by key (RowId/IndexKey), not by
page offset.

### 5. UPDATE Operations → Currently All Structural

**Rationale**: FrankenSQLite currently implements UPDATE as DELETE + INSERT.
There's no in-place cell replacement path.

**Decision**: For now, all UPDATEs are structural. Future optimization could
add an in-place update path for same-size payloads, which would be logical.

### 6. Root Page with Single Cell → Depends

**Rationale**: A root leaf page can have any number of cells (including 1)
without triggering balance. Only when the root becomes an interior page with
zero cells pointing to a single child does `balance_shallower` trigger.

**Decision**: Single-cell operations on root leaf pages are logical. Root
interior page collapse is structural.

---

## Integration Points

### 1. BtCursor Integration

The `BtCursor` already tracks:
- `note_split_event()` — called when balance_for_insert triggers
- `note_merge_event()` — called when balance_for_delete triggers

These can be extended to return `MutationOutcome` for MVCC classification.

### 2. Connection Layer Integration

At the connection layer (where SQL mutations are executed):

```rust
// Pseudocode for INSERT path
let outcome = cursor.table_insert(cx, rowid, data)?;
if outcome.is_logical() {
    cell_visibility_log.record_insert(page, CellKey::TableRow(rowid), delta);
} else {
    // Page-level MVCC already handled by pager
}
```

### 3. VersionStore Coordination

When a structural operation occurs:
1. The pager creates a new `PageVersion` as usual
2. Any pending cell deltas for affected pages must be **materialized** into the
   new page version
3. The `CellVisibilityLog` chains for those pages are cleared

This ensures consistency: after a split, the new pages contain the correct
cell content, and future reads don't see stale deltas from pre-split state.

---

## Next Steps

- **C3**: Implement `MutationOutcome` reporting in `BtCursor`
- **C4**: Add `record_logical_*` methods to `CellVisibilityLog`
- **C5**: Implement materialization on structural operations
