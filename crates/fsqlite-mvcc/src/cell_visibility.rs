//! Cell-Level MVCC Visibility Log (C1: bd-l9k8e.1, C3: bd-l9k8e.3)
//!
//! This module implements cell-level MVCC for ordinary row operations, replacing
//! full-page version chains with lightweight cell deltas for INSERT/UPDATE/DELETE
//! operations that don't trigger structural B-tree changes.
//!
//! # Architecture Overview
//!
//! ## The Problem (Why Page-Level Is Wrong)
//!
//! Current page-level MVCC (see [`crate::invariants::VersionStore`]) stores full 4KB
//! page copies in version chains. If txn A inserts row 1000 and txn B inserts row 1001,
//! both on page 47, this causes a **page-level conflict** even though the rows are
//! independent. One transaction wins, the other retries with `SQLITE_BUSY_SNAPSHOT`.
//!
//! This is the wrong granularity. A row INSERT that touches 50 bytes of a 4KB page
//! should NOT create a 4KB page copy, link it into a version chain, walk that chain
//! on every read, and then GC it later.
//!
//! ## The Solution (Cell-Level Deltas)
//!
//! Split operations into two classes:
//!
//! 1. **LOGICAL (cell-level, cheap path):** INSERT/UPDATE/DELETE that fit in existing
//!    page without structural change. Recorded in [`CellVisibilityLog`], not as full
//!    page versions.
//!
//! 2. **STRUCTURAL (page-level, existing path):** Page splits, merges, freelist changes,
//!    overflow chains. Uses existing page-level versioning.
//!
//! ## Key Design Decisions
//!
//! ### CellKey Strategy
//!
//! We reuse [`fsqlite_types::SemanticKeyRef`] which provides:
//! - `btree: BtreeRef` (table vs index identity)
//! - `kind: SemanticKeyKind` (TableRow vs IndexEntry)
//! - `key_digest: [u8; 16]` (BLAKE3-based stable identity)
//!
//! For **table leaf pages**, the canonical key bytes are the rowid (i64 varint).
//! For **index leaf pages**, the canonical key bytes are the full encoded index key.
//!
//! ### Visibility Resolution
//!
//! Given snapshot S and page P, to find the visible version of cell C:
//!
//! 1. Look up `(P, C.key_digest)` in [`CellVisibilityLog`]
//! 2. Walk the cell delta chain (newest to oldest) until finding a delta where
//!    `delta.commit_seq <= S.high` and `delta.commit_seq != 0`
//! 3. Apply the delta to reconstruct cell content
//!
//! Complexity: O(log N) where N is the number of deltas for that cell.
//!
//! ### Memory Budget
//!
//! The log operates within a bounded memory budget:
//! - Per-delta overhead: ~64 bytes (key + metadata) + cell_data (variable)
//! - Target: <10% of page cache memory for cell deltas
//! - For a 256MB page cache: cell delta budget = ~25MB
//! - At 64 bytes per delta: ~400K outstanding deltas before eager materialization
//!
//! When the budget is exceeded, the oldest deltas are materialized into full page
//! versions and evicted from the log.
//!
//! ### Interior Pages
//!
//! Interior (non-leaf) pages ALWAYS use page-level MVCC. They contain child pointers
//! and are structural by definition. Only leaf pages participate in cell-level MVCC.
//!
//! ## PostgreSQL Reference
//!
//! This design is inspired by PostgreSQL's tuple-level MVCC:
//! - `htup_details.h:122` — tuple header with xmin/xmax
//! - `snapshot.h:138` — compact Snapshot struct
//! - `heapam.c:522` — batched page visibility
//! - `pruneheap.c:199` — opportunistic pruning
//!
//! # Module Status
//!
//! C1 design complete. C3 implementation complete with:
//! - Transaction write set tracking (bulk commit/rollback)
//! - Cell-level conflict detection
//! - Garbage collection below GC horizon
//! - Full tracing instrumentation
//! - Memory budget enforcement with materialization callback

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use fsqlite_types::sync_primitives::{Mutex, RwLock};
use fsqlite_types::{BtreeRef, CommitSeq, PageNumber, SemanticKeyKind, SemanticKeyRef, TxnToken};
use smallvec::SmallVec;
use tracing::{debug, trace};

use crate::cache_aligned::CacheAligned;

// ---------------------------------------------------------------------------
// CellKey — Stable cell identity (§C1.1)
// ---------------------------------------------------------------------------

/// Stable identity for a cell within a B-tree leaf page.
///
/// This wraps [`SemanticKeyRef`] with additional context about whether the cell
/// has been deleted. The `key_digest` provides O(1) hashing and comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellKey {
    /// The B-tree containing this cell (table or index).
    pub btree: BtreeRef,
    /// Whether this is a table row or index entry.
    pub kind: SemanticKeyKind,
    /// BLAKE3-truncated digest of the canonical key bytes.
    /// For tables: rowid as varint bytes.
    /// For indexes: full encoded index key bytes.
    pub key_digest: [u8; 16],
}

impl CellKey {
    /// Create a cell key from a semantic key reference.
    #[must_use]
    pub fn from_semantic_ref(skr: &SemanticKeyRef) -> Self {
        Self {
            btree: skr.btree,
            kind: skr.kind,
            key_digest: skr.key_digest,
        }
    }

    /// Create a table row cell key from a rowid.
    ///
    /// The canonical key bytes are the rowid encoded as a varint.
    #[must_use]
    pub fn table_row(btree: BtreeRef, rowid: i64) -> Self {
        let mut key_bytes = [0u8; 10]; // Max varint length
        let len = encode_varint_i64(rowid, &mut key_bytes);
        Self::from_semantic_ref(&SemanticKeyRef::new(
            btree,
            SemanticKeyKind::TableRow,
            &key_bytes[..len],
        ))
    }

    /// Create an index entry cell key from the encoded index key bytes.
    #[must_use]
    pub fn index_entry(btree: BtreeRef, index_key_bytes: &[u8]) -> Self {
        Self::from_semantic_ref(&SemanticKeyRef::new(
            btree,
            SemanticKeyKind::IndexEntry,
            index_key_bytes,
        ))
    }
}

// ---------------------------------------------------------------------------
// CellDeltaKind — What happened to the cell (§C1.2)
// ---------------------------------------------------------------------------

/// The kind of change applied to a cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellDeltaKind {
    /// A new cell was inserted.
    Insert,
    /// An existing cell was deleted.
    Delete,
    /// An existing cell was updated (replaced with new content).
    Update,
}

// ---------------------------------------------------------------------------
// CellDelta — A single versioned change to a cell (§C1.3)
// ---------------------------------------------------------------------------

/// A single versioned change to a cell.
///
/// Cell deltas form a chain similar to page versions, but with much lower
/// overhead since they store only the cell content, not the full page.
///
/// # Memory Layout (targeting ~64 bytes + cell_data)
///
/// - `commit_seq`: 8 bytes
/// - `created_by`: 16 bytes (TxnToken = TxnId + TxnEpoch)
/// - `kind`: 1 byte (+ padding)
/// - `prev_idx`: 8 bytes (Option<CellDeltaIdx>)
/// - `cell_data`: Variable (typically 50-200 bytes for a row)
/// - `page_number`: 4 bytes (for materialization)
/// - Estimated total: ~40 bytes fixed + cell_data
#[derive(Debug, Clone)]
pub struct CellDelta {
    /// The commit sequence when this delta became visible.
    /// `CommitSeq(0)` means uncommitted (part of an active transaction's write set).
    pub commit_seq: CommitSeq,

    /// The transaction that created this delta (for debugging/audit only).
    pub created_by: TxnToken,

    /// What kind of change this represents.
    pub kind: CellDeltaKind,

    /// The page number where this cell lives (needed for materialization).
    pub page_number: PageNumber,

    /// The cell content after this delta is applied.
    /// For `Delete`, this is empty (the cell no longer exists).
    /// For `Insert`/`Update`, this contains the full cell bytes.
    pub cell_data: Vec<u8>,

    /// Index of the previous delta in the chain (older version).
    /// `None` if this is the oldest known version.
    pub prev_idx: Option<CellDeltaIdx>,
}

impl CellDelta {
    /// Memory footprint of this delta (fixed overhead + cell data).
    #[must_use]
    pub fn memory_size(&self) -> usize {
        // Fixed overhead (commit_seq, created_by, kind, page_number, prev_idx)
        const FIXED_OVERHEAD: usize = 8 + 16 + 1 + 4 + 8;
        // Vec overhead (ptr, len, cap) + actual data
        const VEC_OVERHEAD: usize = 24;
        FIXED_OVERHEAD + VEC_OVERHEAD + self.cell_data.len()
    }

    /// Whether this delta is visible to the given snapshot.
    #[must_use]
    pub fn is_visible_to(&self, snapshot_high: CommitSeq) -> bool {
        self.commit_seq.get() != 0 && self.commit_seq <= snapshot_high
    }
}

// ---------------------------------------------------------------------------
// CellDeltaIdx — Index into the delta arena (§C1.4)
// ---------------------------------------------------------------------------

/// Index into the [`CellDeltaArena`].
///
/// Uses the same chunk+offset+generation pattern as [`crate::core_types::VersionIdx`]
/// for ABA protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellDeltaIdx {
    chunk: u32,
    offset: u32,
    generation: u32,
}

impl CellDeltaIdx {
    #[inline]
    const fn new(chunk: u32, offset: u32, generation: u32) -> Self {
        Self {
            chunk,
            offset,
            generation,
        }
    }

    #[inline]
    #[must_use]
    pub fn chunk(&self) -> u32 {
        self.chunk
    }

    #[inline]
    #[must_use]
    pub fn offset(&self) -> u32 {
        self.offset
    }

    #[inline]
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

// ---------------------------------------------------------------------------
// CellDeltaArena — Bump-allocated storage for deltas (§C1.5)
// ---------------------------------------------------------------------------

/// Number of deltas per arena chunk.
const DELTA_ARENA_CHUNK: usize = 4096;

struct DeltaSlot {
    generation: u32,
    delta: Option<CellDelta>,
}

/// Bump-allocated arena for [`CellDelta`] objects.
///
/// Similar to [`crate::core_types::VersionArena`], but for cell deltas.
/// Includes generation counting to detect use-after-free/ABA bugs.
pub struct CellDeltaArena {
    chunks: Vec<Vec<DeltaSlot>>,
    free_list: Vec<CellDeltaIdx>,
    /// Total memory consumed by cell data (excludes fixed overhead).
    cell_data_bytes: AtomicU64,
    high_water: u64,
}

impl CellDeltaArena {
    /// Create an empty arena.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunks: vec![Vec::with_capacity(DELTA_ARENA_CHUNK)],
            free_list: Vec::new(),
            cell_data_bytes: AtomicU64::new(0),
            high_water: 0,
        }
    }

    /// Allocate a slot for `delta`, returning its index.
    pub fn alloc(&mut self, delta: CellDelta) -> CellDeltaIdx {
        let data_size = delta.cell_data.len() as u64;
        self.cell_data_bytes.fetch_add(data_size, Ordering::Relaxed);

        if let Some(idx) = self.free_list.pop() {
            let slot = &mut self.chunks[idx.chunk as usize][idx.offset as usize];
            slot.delta = Some(delta);
            return CellDeltaIdx::new(idx.chunk, idx.offset, slot.generation);
        }

        let last_chunk = self.chunks.len() - 1;
        if self.chunks[last_chunk].len() >= DELTA_ARENA_CHUNK {
            self.chunks.push(Vec::with_capacity(DELTA_ARENA_CHUNK));
        }

        let chunk_idx = self.chunks.len() - 1;
        let offset = self.chunks[chunk_idx].len();
        self.chunks[chunk_idx].push(DeltaSlot {
            generation: 0,
            delta: Some(delta),
        });
        self.high_water += 1;

        let chunk_u32 = u32::try_from(chunk_idx).unwrap_or(u32::MAX);
        let offset_u32 = u32::try_from(offset).unwrap_or(u32::MAX);
        CellDeltaIdx::new(chunk_u32, offset_u32, 0)
    }

    /// Free the slot at `idx`, making it available for reuse.
    pub fn free(&mut self, idx: CellDeltaIdx) -> Option<CellDelta> {
        let slot = self
            .chunks
            .get_mut(idx.chunk as usize)?
            .get_mut(idx.offset as usize)?;

        if slot.generation != idx.generation {
            return None; // Stale pointer
        }

        let delta = slot.delta.take()?;
        let data_size = delta.cell_data.len() as u64;
        self.cell_data_bytes.fetch_sub(data_size, Ordering::Relaxed);

        // Increment generation on free
        let mut next_gen = slot.generation.wrapping_add(1);
        if next_gen == u32::MAX {
            next_gen = 0;
        }
        slot.generation = next_gen;

        self.free_list.push(idx);
        Some(delta)
    }

    /// Look up a delta by index.
    #[must_use]
    pub fn get(&self, idx: CellDeltaIdx) -> Option<&CellDelta> {
        let slot = self
            .chunks
            .get(idx.chunk as usize)?
            .get(idx.offset as usize)?;

        if slot.generation != idx.generation {
            return None;
        }
        slot.delta.as_ref()
    }

    /// Total cell data bytes currently in the arena.
    #[must_use]
    pub fn cell_data_bytes(&self) -> u64 {
        self.cell_data_bytes.load(Ordering::Relaxed)
    }

    /// High water mark (total deltas ever allocated).
    #[must_use]
    pub fn high_water(&self) -> u64 {
        self.high_water
    }
}

impl Default for CellDeltaArena {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CellVisibilityLog — The main cell-level MVCC structure (§C1.6)
// ---------------------------------------------------------------------------

/// Number of shards in the cell visibility log.
const CELL_LOG_SHARDS: usize = 64;

/// Entry in the cell head table mapping CellKey -> head delta.
#[derive(Debug, Clone, Copy)]
struct CellHeadEntry {
    head_idx: CellDeltaIdx,
}

/// A single shard of the cell head table.
struct CellLogShard {
    /// Maps (PageNumber, key_digest) -> head delta index.
    /// Uses default hasher since the key is a composite (PageNumber, [u8; 16]).
    heads: RwLock<HashMap<(PageNumber, [u8; 16]), CellHeadEntry>>,
}

impl CellLogShard {
    fn new() -> Self {
        Self {
            heads: RwLock::new(HashMap::new()),
        }
    }
}

// ---------------------------------------------------------------------------
// Transaction Delta Tracking (§C3.1)
// ---------------------------------------------------------------------------

/// Tracks deltas created by each active transaction.
///
/// This enables bulk commit (update all deltas for a txn) and rollback
/// (remove all deltas for a txn) operations.
struct TxnDeltaTracker {
    /// Maps TxnToken -> list of delta indices created by that transaction.
    /// Uses SmallVec because most transactions create <16 deltas.
    txn_deltas: HashMap<TxnToken, SmallVec<[CellDeltaIdx; 16]>>,
    /// Per-transaction byte budget tracking.
    txn_bytes: HashMap<TxnToken, u64>,
}

impl TxnDeltaTracker {
    fn new() -> Self {
        Self {
            txn_deltas: HashMap::new(),
            txn_bytes: HashMap::new(),
        }
    }

    fn record(&mut self, txn: TxnToken, idx: CellDeltaIdx, bytes: u64) {
        self.txn_deltas.entry(txn).or_default().push(idx);
        *self.txn_bytes.entry(txn).or_insert(0) += bytes;
    }

    /// Get all delta indices for a transaction (used during abort/rollback).
    #[allow(dead_code)]
    fn get_deltas(&self, txn: TxnToken) -> Option<&SmallVec<[CellDeltaIdx; 16]>> {
        self.txn_deltas.get(&txn)
    }

    /// Remove a transaction's tracking data (used during abort/rollback/commit finalization).
    #[allow(dead_code)]
    fn remove_txn(&mut self, txn: TxnToken) -> Option<SmallVec<[CellDeltaIdx; 16]>> {
        self.txn_bytes.remove(&txn);
        self.txn_deltas.remove(&txn)
    }

    fn txn_bytes(&self, txn: TxnToken) -> u64 {
        self.txn_bytes.get(&txn).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Conflict Detection (§C3.2)
// ---------------------------------------------------------------------------

/// Result of a cell-level conflict check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellConflict {
    /// No conflict detected.
    None,
    /// Conflict detected with the given transaction.
    Conflict { with_txn: TxnToken },
}

// ---------------------------------------------------------------------------
// GC Statistics (§C3.3)
// ---------------------------------------------------------------------------

/// Statistics from a GC run.
#[derive(Debug, Clone, Default)]
pub struct CellGcStats {
    /// Number of deltas examined.
    pub examined: u64,
    /// Number of deltas reclaimed.
    pub reclaimed: u64,
    /// Bytes freed.
    pub bytes_freed: u64,
}

// ---------------------------------------------------------------------------
// Materialization Callback (§C3.4)
// ---------------------------------------------------------------------------

/// Callback invoked when eager materialization is needed.
///
/// The callback receives the page number and a list of cells that should be
/// materialized into a full page version.
pub type MaterializationCallback = Box<dyn Fn(PageNumber, &[CellKey]) + Send + Sync>;

/// The cell-level MVCC visibility log.
///
/// This is the central data structure for cell-level MVCC. It maintains:
///
/// 1. A sharded hash table mapping (PageNumber, CellKey) -> head of delta chain
/// 2. A delta arena storing all CellDelta objects
/// 3. Memory budget tracking for eager materialization
/// 4. Per-transaction delta tracking for bulk commit/rollback
/// 5. Conflict detection for concurrent writers
///
/// # Thread Safety
///
/// - Reads are lock-free (RwLock read + atomic arena access)
/// - Writes acquire a write lock on the target shard only
/// - The arena is protected by a Mutex for allocation (single-writer)
///
/// # Usage
///
/// ```ignore
/// let log = CellVisibilityLog::new(budget_bytes);
///
/// // Record a cell insert
/// log.record_insert(cell_key, page_number, cell_bytes, txn_token);
///
/// // Resolve visible cell for a snapshot
/// let cell_data = log.resolve(page_number, cell_key, snapshot);
///
/// // Commit all deltas for a transaction
/// log.commit_txn(txn_token, commit_seq);
///
/// // Or rollback
/// log.rollback_txn(txn_token);
/// ```
pub struct CellVisibilityLog {
    /// Sharded cell head table.
    shards: Box<[CacheAligned<CellLogShard>; CELL_LOG_SHARDS]>,
    /// Delta arena (protected by Mutex for writes).
    arena: Mutex<CellDeltaArena>,
    /// Transaction delta tracker (protected by Mutex).
    txn_tracker: Mutex<TxnDeltaTracker>,
    /// Memory budget for cell data (bytes).
    budget_bytes: u64,
    /// Per-transaction byte budget (bytes).
    per_txn_budget_bytes: u64,
    /// Total deltas currently stored.
    delta_count: AtomicU64,
    /// Optional materialization callback for budget enforcement.
    materialization_cb: Option<MaterializationCallback>,
}

/// Default per-transaction budget: 4MB
const DEFAULT_PER_TXN_BUDGET: u64 = 4 * 1024 * 1024;

impl CellVisibilityLog {
    /// Create a new cell visibility log with the given memory budget.
    ///
    /// The budget controls when eager materialization kicks in to prevent
    /// unbounded memory growth.
    #[must_use]
    pub fn new(budget_bytes: u64) -> Self {
        Self::with_per_txn_budget(budget_bytes, DEFAULT_PER_TXN_BUDGET)
    }

    /// Create a new cell visibility log with custom per-transaction budget.
    #[must_use]
    pub fn with_per_txn_budget(budget_bytes: u64, per_txn_budget_bytes: u64) -> Self {
        trace!(
            budget_bytes,
            per_txn_budget_bytes, "cell_visibility_log_created"
        );
        Self {
            shards: Box::new(std::array::from_fn(|_| {
                CacheAligned::new(CellLogShard::new())
            })),
            arena: Mutex::new(CellDeltaArena::new()),
            txn_tracker: Mutex::new(TxnDeltaTracker::new()),
            budget_bytes,
            per_txn_budget_bytes,
            delta_count: AtomicU64::new(0),
            materialization_cb: None,
        }
    }

    /// Set the materialization callback for budget enforcement.
    pub fn set_materialization_callback(&mut self, cb: MaterializationCallback) {
        self.materialization_cb = Some(cb);
    }

    /// Compute the shard index for a page number.
    #[inline]
    fn shard_index(pgno: PageNumber) -> usize {
        (pgno.get() as usize) & (CELL_LOG_SHARDS - 1)
    }

    /// Record a cell insert operation.
    ///
    /// # Arguments
    ///
    /// * `cell_key` - The stable identity of the cell
    /// * `page_number` - The page containing this cell
    /// * `cell_data` - The cell content bytes
    /// * `created_by` - The transaction creating this delta
    ///
    /// Returns the index of the new delta, or `None` if per-txn budget exceeded.
    pub fn record_insert(
        &self,
        cell_key: CellKey,
        page_number: PageNumber,
        cell_data: Vec<u8>,
        created_by: TxnToken,
    ) -> Option<CellDeltaIdx> {
        self.record_delta(
            cell_key,
            page_number,
            CellDeltaKind::Insert,
            cell_data,
            created_by,
        )
    }

    /// Record a cell delete operation.
    ///
    /// Returns the index of the new delta, or `None` if per-txn budget exceeded.
    pub fn record_delete(
        &self,
        cell_key: CellKey,
        page_number: PageNumber,
        created_by: TxnToken,
    ) -> Option<CellDeltaIdx> {
        self.record_delta(
            cell_key,
            page_number,
            CellDeltaKind::Delete,
            Vec::new(),
            created_by,
        )
    }

    /// Record a cell update operation.
    ///
    /// Returns the index of the new delta, or `None` if per-txn budget exceeded.
    pub fn record_update(
        &self,
        cell_key: CellKey,
        page_number: PageNumber,
        new_cell_data: Vec<u8>,
        created_by: TxnToken,
    ) -> Option<CellDeltaIdx> {
        self.record_delta(
            cell_key,
            page_number,
            CellDeltaKind::Update,
            new_cell_data,
            created_by,
        )
    }

    /// Internal: record a delta of any kind.
    ///
    /// Returns `None` if the per-transaction budget would be exceeded.
    fn record_delta(
        &self,
        cell_key: CellKey,
        page_number: PageNumber,
        kind: CellDeltaKind,
        cell_data: Vec<u8>,
        created_by: TxnToken,
    ) -> Option<CellDeltaIdx> {
        let shard_idx = Self::shard_index(page_number);
        let shard = &self.shards[shard_idx];
        let data_len = cell_data.len() as u64;

        // Check per-transaction budget
        {
            let tracker = self.txn_tracker.lock();
            let current_txn_bytes = tracker.txn_bytes(created_by);
            if current_txn_bytes + data_len > self.per_txn_budget_bytes {
                debug!(
                    txn_id = created_by.id.get(),
                    current_bytes = current_txn_bytes,
                    requested_bytes = data_len,
                    budget = self.per_txn_budget_bytes,
                    "cell_delta_txn_budget_exceeded"
                );
                return None;
            }
        }

        // Look up existing head
        let lookup_key = (page_number, cell_key.key_digest);
        let prev_idx = {
            let heads = shard.heads.read();
            heads.get(&lookup_key).map(|e| e.head_idx)
        };

        // Create new delta
        let delta = CellDelta {
            commit_seq: CommitSeq::new(0), // Uncommitted
            created_by,
            kind: kind.clone(),
            page_number,
            cell_data,
            prev_idx,
        };

        let delta_memory = delta.memory_size() as u64;

        // Allocate in arena
        let new_idx = {
            let mut arena = self.arena.lock();
            arena.alloc(delta)
        };

        // Update head
        {
            let mut heads = shard.heads.write();
            heads.insert(lookup_key, CellHeadEntry { head_idx: new_idx });
        }

        // Track this delta for the transaction
        {
            let mut tracker = self.txn_tracker.lock();
            tracker.record(created_by, new_idx, delta_memory);
        }

        self.delta_count.fetch_add(1, Ordering::Relaxed);

        trace!(
            pgno = page_number.get(),
            key_digest = ?&cell_key.key_digest[..4],
            txn_id = created_by.id.get(),
            op = ?kind,
            delta_idx_chunk = new_idx.chunk(),
            delta_idx_offset = new_idx.offset(),
            "cell_delta_recorded"
        );

        // Check global budget and trigger materialization if needed
        self.maybe_trigger_materialization();

        Some(new_idx)
    }

    /// Check if global budget is exceeded and trigger materialization.
    fn maybe_trigger_materialization(&self) {
        let arena = self.arena.lock();
        let current_bytes = arena.cell_data_bytes();
        drop(arena);

        if current_bytes > self.budget_bytes {
            if let Some(ref cb) = self.materialization_cb {
                // Find pages with highest delta counts for materialization
                let pages_to_materialize = self.find_high_delta_pages(10);
                for (pgno, cell_keys) in pages_to_materialize {
                    debug!(
                        pgno = pgno.get(),
                        cell_count = cell_keys.len(),
                        "cell_budget_exceeded_materializing"
                    );
                    cb(pgno, &cell_keys);
                }
            }
        }
    }

    /// Find pages with the highest number of deltas.
    fn find_high_delta_pages(&self, max_pages: usize) -> Vec<(PageNumber, Vec<CellKey>)> {
        let mut page_counts: HashMap<PageNumber, Vec<CellKey>> = HashMap::new();

        for shard in self.shards.iter() {
            let heads = shard.heads.read();
            for ((pgno, key_digest), _) in heads.iter() {
                let cell_key = CellKey {
                    btree: BtreeRef::Table(fsqlite_types::TableId::new(0)),
                    kind: SemanticKeyKind::TableRow,
                    key_digest: *key_digest,
                };
                page_counts.entry(*pgno).or_default().push(cell_key);
            }
        }

        let mut pages: Vec<_> = page_counts.into_iter().collect();
        pages.sort_by_key(|p| std::cmp::Reverse(p.1.len()));
        pages.truncate(max_pages);
        pages
    }

    /// Commit a delta (assign a commit sequence).
    ///
    /// Called at transaction commit time to make the delta visible.
    pub fn commit_delta(&self, idx: CellDeltaIdx, commit_seq: CommitSeq) {
        let mut arena = self.arena.lock();
        if let Some(slot) = arena
            .chunks
            .get_mut(idx.chunk as usize)
            .and_then(|c| c.get_mut(idx.offset as usize))
        {
            if slot.generation == idx.generation {
                if let Some(ref mut delta) = slot.delta {
                    delta.commit_seq = commit_seq;
                }
            }
        }
    }

    /// Resolve the visible cell data for a snapshot.
    ///
    /// Returns `None` if:
    /// - The cell has no deltas in the log
    /// - The cell was deleted in the visible version
    /// - No delta is visible to this snapshot
    #[must_use]
    pub fn resolve(
        &self,
        page_number: PageNumber,
        cell_key: &CellKey,
        snapshot_high: CommitSeq,
    ) -> Option<Vec<u8>> {
        let shard_idx = Self::shard_index(page_number);
        let shard = &self.shards[shard_idx];
        let lookup_key = (page_number, cell_key.key_digest);

        let head_idx = {
            let heads = shard.heads.read();
            match heads.get(&lookup_key) {
                Some(entry) => entry.head_idx,
                None => {
                    trace!(
                        pgno = page_number.get(),
                        key_digest = ?&cell_key.key_digest[..4],
                        snapshot_high = snapshot_high.get(),
                        result = "not_tracked",
                        "cell_resolved"
                    );
                    return None;
                }
            }
        };

        let arena = self.arena.lock();
        let mut current_idx = Some(head_idx);

        // Walk chain to find visible delta
        while let Some(idx) = current_idx {
            if let Some(delta) = arena.get(idx) {
                if delta.is_visible_to(snapshot_high) {
                    let result = match delta.kind {
                        CellDeltaKind::Delete => {
                            trace!(
                                pgno = page_number.get(),
                                key_digest = ?&cell_key.key_digest[..4],
                                snapshot_high = snapshot_high.get(),
                                result = "deleted",
                                "cell_resolved"
                            );
                            None
                        }
                        CellDeltaKind::Insert | CellDeltaKind::Update => {
                            trace!(
                                pgno = page_number.get(),
                                key_digest = ?&cell_key.key_digest[..4],
                                snapshot_high = snapshot_high.get(),
                                result = "visible",
                                data_len = delta.cell_data.len(),
                                "cell_resolved"
                            );
                            Some(delta.cell_data.clone())
                        }
                    };
                    return result;
                }
                current_idx = delta.prev_idx;
            } else {
                break;
            }
        }

        trace!(
            pgno = page_number.get(),
            key_digest = ?&cell_key.key_digest[..4],
            snapshot_high = snapshot_high.get(),
            result = "no_visible_version",
            "cell_resolved"
        );
        None
    }

    /// Check if the memory budget is exceeded.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        let arena = self.arena.lock();
        arena.cell_data_bytes() > self.budget_bytes
    }

    /// Total number of deltas currently stored.
    #[must_use]
    pub fn delta_count(&self) -> u64 {
        self.delta_count.load(Ordering::Relaxed)
    }

    /// Current cell data memory usage.
    #[must_use]
    pub fn cell_data_bytes(&self) -> u64 {
        let arena = self.arena.lock();
        arena.cell_data_bytes()
    }

    /// Memory budget.
    #[must_use]
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Per-transaction memory budget.
    #[must_use]
    pub fn per_txn_budget_bytes(&self) -> u64 {
        self.per_txn_budget_bytes
    }

    /// Commit all deltas for a transaction atomically.
    pub fn commit_txn(&self, txn: TxnToken, commit_seq: CommitSeq) {
        let mut tracker = self.txn_tracker.lock();
        let delta_indices = match tracker.get_deltas(txn) {
            Some(indices) => indices.clone(),
            None => {
                trace!(txn_id = txn.id.get(), "cell_txn_commit_no_deltas");
                return;
            }
        };

        let mut arena = self.arena.lock();
        let mut committed_count = 0u64;

        for idx in &delta_indices {
            if let Some(slot) = arena
                .chunks
                .get_mut(idx.chunk as usize)
                .and_then(|c| c.get_mut(idx.offset as usize))
            {
                if slot.generation == idx.generation {
                    if let Some(ref mut delta) = slot.delta {
                        delta.commit_seq = commit_seq;
                        committed_count += 1;
                    }
                }
            }
        }

        drop(arena);
        tracker.remove_txn(txn);

        debug!(
            txn_id = txn.id.get(),
            commit_seq = commit_seq.get(),
            delta_count = committed_count,
            "cell_txn_committed"
        );
    }

    /// Rollback all deltas for a transaction.
    pub fn rollback_txn(&self, txn: TxnToken) -> u64 {
        let mut tracker = self.txn_tracker.lock();
        let delta_indices = match tracker.remove_txn(txn) {
            Some(indices) => indices,
            None => {
                trace!(txn_id = txn.id.get(), "cell_txn_rollback_no_deltas");
                return 0;
            }
        };

        let mut arena = self.arena.lock();
        let mut removed_count = 0u64;
        let mut bytes_freed = 0u64;

        for idx in &delta_indices {
            if let Some(delta) = arena.free(*idx) {
                bytes_freed += delta.memory_size() as u64;
                removed_count += 1;

                let shard_idx = Self::shard_index(delta.page_number);
                let shard = &self.shards[shard_idx];
                let mut heads = shard.heads.write();

                for ((pgno, _), entry) in heads.iter_mut() {
                    if *pgno == delta.page_number && entry.head_idx == *idx {
                        if let Some(prev) = delta.prev_idx {
                            entry.head_idx = prev;
                        }
                    }
                }
            }
        }

        self.delta_count.fetch_sub(removed_count, Ordering::Relaxed);

        debug!(
            txn_id = txn.id.get(),
            delta_count = removed_count,
            bytes_freed,
            "cell_txn_rolled_back"
        );

        removed_count
    }

    /// Check for cell-level conflict between two transactions.
    #[must_use]
    pub fn check_conflict(&self, txn: TxnToken, other_txn: TxnToken) -> CellConflict {
        if txn == other_txn {
            return CellConflict::None;
        }

        let tracker = self.txn_tracker.lock();

        let our_deltas = match tracker.get_deltas(txn) {
            Some(d) => d,
            None => return CellConflict::None,
        };

        let their_deltas = match tracker.get_deltas(other_txn) {
            Some(d) => d,
            None => return CellConflict::None,
        };

        let arena = self.arena.lock();

        let mut our_cells: std::collections::HashSet<(PageNumber, [u8; 16])> =
            std::collections::HashSet::new();

        for idx in our_deltas {
            if let Some(delta) = arena.get(*idx) {
                let shard_idx = Self::shard_index(delta.page_number);
                let shard = &self.shards[shard_idx];
                let heads = shard.heads.read();

                for ((pgno, key_digest), entry) in heads.iter() {
                    if *pgno == delta.page_number {
                        let mut check_idx = Some(entry.head_idx);
                        while let Some(cidx) = check_idx {
                            if cidx == *idx {
                                our_cells.insert((*pgno, *key_digest));
                                break;
                            }
                            if let Some(d) = arena.get(cidx) {
                                check_idx = d.prev_idx;
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }

        for idx in their_deltas {
            if let Some(delta) = arena.get(*idx) {
                let shard_idx = Self::shard_index(delta.page_number);
                let shard = &self.shards[shard_idx];
                let heads = shard.heads.read();

                for ((pgno, key_digest), entry) in heads.iter() {
                    if *pgno == delta.page_number {
                        let mut check_idx = Some(entry.head_idx);
                        while let Some(cidx) = check_idx {
                            if cidx == *idx && our_cells.contains(&(*pgno, *key_digest)) {
                                debug!(
                                    txn_id = txn.id.get(),
                                    other_txn_id = other_txn.id.get(),
                                    pgno = pgno.get(),
                                    "cell_conflict_detected"
                                );
                                return CellConflict::Conflict {
                                    with_txn: other_txn,
                                };
                            }
                            if let Some(d) = arena.get(cidx) {
                                check_idx = d.prev_idx;
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }

        CellConflict::None
    }

    /// Garbage collect deltas below the GC horizon.
    pub fn gc(&self, gc_horizon: CommitSeq) -> CellGcStats {
        let mut stats = CellGcStats::default();
        let mut to_free: Vec<CellDeltaIdx> = Vec::new();

        {
            let arena = self.arena.lock();

            for shard in self.shards.iter() {
                let heads = shard.heads.read();

                for (_, entry) in heads.iter() {
                    let mut current_idx = Some(entry.head_idx);
                    let mut found_visible_below_horizon = false;

                    while let Some(idx) = current_idx {
                        stats.examined += 1;

                        if let Some(delta) = arena.get(idx) {
                            if delta.commit_seq.get() != 0 && delta.commit_seq <= gc_horizon {
                                if found_visible_below_horizon {
                                    to_free.push(idx);
                                } else {
                                    found_visible_below_horizon = true;
                                }
                            }
                            current_idx = delta.prev_idx;
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        if !to_free.is_empty() {
            let mut arena = self.arena.lock();

            for idx in to_free {
                if let Some(delta) = arena.free(idx) {
                    stats.reclaimed += 1;
                    stats.bytes_freed += delta.memory_size() as u64;
                }
            }
        }

        self.delta_count
            .fetch_sub(stats.reclaimed, Ordering::Relaxed);

        debug!(
            gc_horizon = gc_horizon.get(),
            examined = stats.examined,
            reclaimed = stats.reclaimed,
            bytes_freed = stats.bytes_freed,
            "cell_gc_completed"
        );

        stats
    }

    /// Get the number of bytes used by a transaction's deltas.
    #[must_use]
    pub fn txn_bytes(&self, txn: TxnToken) -> u64 {
        let tracker = self.txn_tracker.lock();
        tracker.txn_bytes(txn)
    }

    /// Get the number of active transactions with deltas.
    #[must_use]
    pub fn active_txn_count(&self) -> usize {
        let tracker = self.txn_tracker.lock();
        tracker.txn_deltas.len()
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for CellVisibilityLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CellVisibilityLog")
            .field("budget_bytes", &self.budget_bytes)
            .field("delta_count", &self.delta_count.load(Ordering::Relaxed))
            .field("cell_data_bytes", &self.cell_data_bytes())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// MutationOutcome — Classification of B-tree mutations (§C2)
// ---------------------------------------------------------------------------

/// Outcome of a B-tree mutation for MVCC classification.
///
/// This enum classifies the result of an INSERT, UPDATE, or DELETE operation
/// to determine whether cell-level MVCC (logical) or page-level MVCC (structural)
/// should be used.
///
/// # Design Rationale (C2: bd-l9k8e.2)
///
/// The fundamental question: does this operation change only cell content, or
/// does it change page structure/allocation?
///
/// - **Logical operations** modify cell content within existing page boundaries.
/// - **Structural operations** change the B-tree topology (page allocation, splits,
///   merges, depth changes).
///
/// Cell-level MVCC (CellVisibilityLog) handles logical operations.
/// Page-level MVCC (VersionStore) handles structural operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationOutcome {
    /// Cell was inserted/updated in-place without page restructuring.
    /// The cell fit in the existing page free space.
    CellFit,

    /// Cell was deleted and page still has other cells remaining.
    /// No structural changes needed.
    CellRemovedPageNonEmpty,

    /// Insert triggered page split (`balance_for_insert` was called).
    /// One or more pages were allocated, cells redistributed.
    PageSplit,

    /// Delete triggered page merge (`balance_for_delete` was called).
    /// Pages may have been freed or tree depth reduced.
    PageMerge,

    /// Overflow chain was created, extended, or freed.
    /// Affects pages beyond the leaf page.
    OverflowChainModified,

    /// Interior page cell was replaced (index delete with successor promotion).
    /// Structural because it affects tree navigation keys.
    InteriorCellReplaced,
}

impl MutationOutcome {
    /// Returns true if this mutation is structural (page-level MVCC required).
    ///
    /// Structural operations change B-tree topology: page allocation, splits,
    /// merges, overflow chains, or interior page modifications.
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
    ///
    /// Logical operations only affect cell content within a single page and
    /// are candidates for the `CellVisibilityLog`.
    #[inline]
    #[must_use]
    pub const fn is_logical(self) -> bool {
        !self.is_structural()
    }
}

/// Pre-mutation check: can this insertion potentially be logical?
///
/// Returns `false` if a structural operation is guaranteed (e.g., payload requires
/// overflow). Returns `true` if the operation MIGHT be logical (cell might fit).
/// The actual outcome depends on page free space at mutation time.
///
/// # Arguments
///
/// * `payload_size` - Size of the cell payload in bytes
/// * `local_max` - Maximum local payload for this page type
/// * `page_free_space` - Current free space in the target page
/// * `cell_overhead` - Fixed overhead per cell (header, cell pointer, etc.)
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
/// Returns `true` if the page will have cells remaining after deletion.
/// When a page becomes empty, `balance_for_delete` is triggered, making
/// the operation structural.
#[inline]
#[must_use]
pub const fn will_be_logical_delete(current_cell_count: u16) -> bool {
    current_cell_count > 1
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Encode an i64 as a SQLite-style varint.
///
/// Returns the number of bytes written.
fn encode_varint_i64(value: i64, buf: &mut [u8; 10]) -> usize {
    // SQLite uses unsigned varints, so cast to u64
    let uval = value as u64;
    encode_varint_u64(uval, buf)
}

/// Encode a u64 as a SQLite-style varint.
fn encode_varint_u64(mut value: u64, buf: &mut [u8; 10]) -> usize {
    let mut i = 0;
    loop {
        if value <= 0x7f {
            buf[i] = value as u8;
            return i + 1;
        }
        buf[i] = ((value & 0x7f) | 0x80) as u8;
        value >>= 7;
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fsqlite_types::{TableId, TxnEpoch, TxnId};

    fn test_txn_token() -> TxnToken {
        TxnToken::new(TxnId::new(1).unwrap(), TxnEpoch::new(1))
    }

    #[test]
    fn cell_key_from_rowid() {
        let btree = BtreeRef::Table(TableId::new(1));
        let key1 = CellKey::table_row(btree, 100);
        let key2 = CellKey::table_row(btree, 100);
        let key3 = CellKey::table_row(btree, 101);

        assert_eq!(key1.key_digest, key2.key_digest, "Same rowid = same digest");
        assert_ne!(
            key1.key_digest, key3.key_digest,
            "Different rowid = different digest"
        );
    }

    #[test]
    fn cell_delta_arena_alloc_free() {
        let mut arena = CellDeltaArena::new();

        let delta = CellDelta {
            commit_seq: CommitSeq::new(1),
            created_by: test_txn_token(),
            kind: CellDeltaKind::Insert,
            page_number: PageNumber::new(42).unwrap(),
            cell_data: vec![1, 2, 3, 4],
            prev_idx: None,
        };

        let idx = arena.alloc(delta);
        assert!(arena.get(idx).is_some());

        let freed = arena.free(idx);
        assert!(freed.is_some());

        // After free, same idx should be invalid (generation changed)
        assert!(arena.get(idx).is_none());
    }

    #[test]
    fn cell_visibility_log_basic() {
        let log = CellVisibilityLog::new(1024 * 1024); // 1MB budget
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();
        let token = test_txn_token();

        // Record an insert (uncommitted)
        let idx = log
            .record_insert(cell_key, page_number, vec![1, 2, 3, 4], token)
            .expect("insert should succeed");

        // Should not be visible yet (commit_seq = 0)
        let snapshot = CommitSeq::new(10);
        assert!(log.resolve(page_number, &cell_key, snapshot).is_none());

        // Commit the delta
        log.commit_delta(idx, CommitSeq::new(5));

        // Now should be visible
        let result = log.resolve(page_number, &cell_key, snapshot);
        assert_eq!(result, Some(vec![1, 2, 3, 4]));

        // Older snapshot should not see it
        let old_snapshot = CommitSeq::new(4);
        assert!(log.resolve(page_number, &cell_key, old_snapshot).is_none());
    }

    #[test]
    fn cell_visibility_log_delete() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();
        let token = test_txn_token();

        // Insert then delete
        let idx1 = log
            .record_insert(cell_key, page_number, vec![1, 2, 3], token)
            .expect("insert should succeed");
        log.commit_delta(idx1, CommitSeq::new(5));

        let idx2 = log
            .record_delete(cell_key, page_number, token)
            .expect("delete should succeed");
        log.commit_delta(idx2, CommitSeq::new(10));

        // Snapshot after delete: cell not visible
        assert!(
            log.resolve(page_number, &cell_key, CommitSeq::new(15))
                .is_none()
        );

        // Snapshot before delete: cell visible
        let result = log.resolve(page_number, &cell_key, CommitSeq::new(7));
        assert_eq!(result, Some(vec![1, 2, 3]));
    }

    #[test]
    fn mutation_outcome_classification() {
        // Logical operations
        assert!(MutationOutcome::CellFit.is_logical());
        assert!(MutationOutcome::CellRemovedPageNonEmpty.is_logical());
        assert!(!MutationOutcome::CellFit.is_structural());
        assert!(!MutationOutcome::CellRemovedPageNonEmpty.is_structural());

        // Structural operations
        assert!(MutationOutcome::PageSplit.is_structural());
        assert!(MutationOutcome::PageMerge.is_structural());
        assert!(MutationOutcome::OverflowChainModified.is_structural());
        assert!(MutationOutcome::InteriorCellReplaced.is_structural());
        assert!(!MutationOutcome::PageSplit.is_logical());
        assert!(!MutationOutcome::PageMerge.is_logical());
    }

    #[test]
    fn can_be_logical_insert_checks() {
        // Fits: 50 byte payload, 200 byte local max, 100 byte free, 10 byte overhead
        assert!(can_be_logical_insert(50, 200, 100, 10)); // 50 + 10 = 60 < 100

        // Doesn't fit: not enough free space
        assert!(!can_be_logical_insert(50, 200, 50, 10)); // 50 + 10 = 60 > 50

        // Requires overflow: payload > local max
        assert!(!can_be_logical_insert(250, 200, 1000, 10)); // 250 > 200
    }

    #[test]
    fn will_be_logical_delete_checks() {
        assert!(will_be_logical_delete(2)); // 2 cells -> 1 cell: logical
        assert!(will_be_logical_delete(10)); // 10 cells -> 9 cells: logical
        assert!(!will_be_logical_delete(1)); // 1 cell -> 0 cells: structural (merge)
        assert!(!will_be_logical_delete(0)); // Already empty: shouldn't happen, but structural
    }

    // -----------------------------------------------------------------------
    // C3 Required Tests
    // -----------------------------------------------------------------------

    fn txn_token_n(n: u32) -> TxnToken {
        TxnToken::new(TxnId::new(u64::from(n)).unwrap(), TxnEpoch::new(1))
    }

    #[test]
    fn test_uncommitted_invisible() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();

        // Insert but don't commit
        log.record_insert(cell_key, page_number, vec![1, 2, 3, 4], txn_token_n(1))
            .expect("insert should succeed");

        // Uncommitted delta should not be visible at any snapshot
        assert!(
            log.resolve(page_number, &cell_key, CommitSeq::new(100))
                .is_none()
        );
        assert!(
            log.resolve(page_number, &cell_key, CommitSeq::new(1))
                .is_none()
        );
    }

    #[test]
    fn test_rollback_removes() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();
        let token = txn_token_n(1);

        // Insert, then rollback
        log.record_insert(cell_key, page_number, vec![1, 2, 3, 4], token)
            .expect("insert should succeed");

        assert_eq!(log.delta_count(), 1);
        let removed = log.rollback_txn(token);
        assert_eq!(removed, 1);
        assert_eq!(log.delta_count(), 0);

        // After rollback, cell should not be resolvable
        assert!(
            log.resolve(page_number, &cell_key, CommitSeq::new(100))
                .is_none()
        );
    }

    #[test]
    fn test_update_versioning() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();
        let token = txn_token_n(1);

        // Insert, commit at seq 5
        let idx1 = log
            .record_insert(cell_key, page_number, vec![1, 1, 1], token)
            .expect("insert should succeed");
        log.commit_delta(idx1, CommitSeq::new(5));

        // Update, commit at seq 10
        let idx2 = log
            .record_update(cell_key, page_number, vec![2, 2, 2], token)
            .expect("update should succeed");
        log.commit_delta(idx2, CommitSeq::new(10));

        // Snapshot at 7 should see insert
        assert_eq!(
            log.resolve(page_number, &cell_key, CommitSeq::new(7)),
            Some(vec![1, 1, 1])
        );

        // Snapshot at 12 should see update
        assert_eq!(
            log.resolve(page_number, &cell_key, CommitSeq::new(12)),
            Some(vec![2, 2, 2])
        );
    }

    #[test]
    fn test_multi_txn_ordering() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();

        // Txn 1 inserts at seq 5
        let idx1 = log
            .record_insert(cell_key, page_number, vec![1], txn_token_n(1))
            .expect("insert should succeed");
        log.commit_delta(idx1, CommitSeq::new(5));

        // Txn 2 updates at seq 10
        let idx2 = log
            .record_update(cell_key, page_number, vec![2], txn_token_n(2))
            .expect("update should succeed");
        log.commit_delta(idx2, CommitSeq::new(10));

        // Txn 3 updates at seq 15
        let idx3 = log
            .record_update(cell_key, page_number, vec![3], txn_token_n(3))
            .expect("update should succeed");
        log.commit_delta(idx3, CommitSeq::new(15));

        // Verify each snapshot sees the correct version
        assert_eq!(
            log.resolve(page_number, &cell_key, CommitSeq::new(7)),
            Some(vec![1])
        );
        assert_eq!(
            log.resolve(page_number, &cell_key, CommitSeq::new(12)),
            Some(vec![2])
        );
        assert_eq!(
            log.resolve(page_number, &cell_key, CommitSeq::new(20)),
            Some(vec![3])
        );
    }

    #[test]
    fn test_different_cells_no_conflict() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let page_number = PageNumber::new(42).unwrap();

        let cell_key1 = CellKey::table_row(btree, 100);
        let cell_key2 = CellKey::table_row(btree, 101);
        let token1 = txn_token_n(1);
        let token2 = txn_token_n(2);

        // Txn 1 touches cell 1
        log.record_insert(cell_key1, page_number, vec![1], token1)
            .expect("insert should succeed");

        // Txn 2 touches cell 2
        log.record_insert(cell_key2, page_number, vec![2], token2)
            .expect("insert should succeed");

        // No conflict because different cells
        assert_eq!(log.check_conflict(token1, token2), CellConflict::None);
        assert_eq!(log.check_conflict(token2, token1), CellConflict::None);
    }

    #[test]
    fn test_same_cell_conflict() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();
        let token1 = txn_token_n(1);
        let token2 = txn_token_n(2);

        // Txn 1 inserts
        log.record_insert(cell_key, page_number, vec![1], token1)
            .expect("insert should succeed");

        // Txn 2 also inserts same cell
        log.record_insert(cell_key, page_number, vec![2], token2)
            .expect("insert should succeed");

        // Conflict detected
        assert!(matches!(
            log.check_conflict(token1, token2),
            CellConflict::Conflict { .. }
        ));
    }

    #[test]
    fn test_gc_reclaims_old() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();
        let token = txn_token_n(1);

        // Insert at seq 5
        let idx1 = log
            .record_insert(cell_key, page_number, vec![1, 2, 3], token)
            .expect("insert should succeed");
        log.commit_delta(idx1, CommitSeq::new(5));

        // Update at seq 10
        let idx2 = log
            .record_update(cell_key, page_number, vec![4, 5, 6], token)
            .expect("update should succeed");
        log.commit_delta(idx2, CommitSeq::new(10));

        assert_eq!(log.delta_count(), 2);

        // GC with horizon at 15 should reclaim the older delta
        let stats = log.gc(CommitSeq::new(15));
        assert_eq!(stats.reclaimed, 1);
        assert_eq!(log.delta_count(), 1);

        // Newer version still visible
        assert_eq!(
            log.resolve(page_number, &cell_key, CommitSeq::new(15)),
            Some(vec![4, 5, 6])
        );
    }

    #[test]
    fn test_gc_preserves_visible() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let cell_key = CellKey::table_row(btree, 100);
        let page_number = PageNumber::new(42).unwrap();
        let token = txn_token_n(1);

        // Insert at seq 10
        let idx = log
            .record_insert(cell_key, page_number, vec![1, 2, 3], token)
            .expect("insert should succeed");
        log.commit_delta(idx, CommitSeq::new(10));

        assert_eq!(log.delta_count(), 1);

        // GC with horizon at 5 should NOT reclaim (delta is above horizon)
        let stats = log.gc(CommitSeq::new(5));
        assert_eq!(stats.reclaimed, 0);
        assert_eq!(log.delta_count(), 1);

        // Still visible
        assert_eq!(
            log.resolve(page_number, &cell_key, CommitSeq::new(15)),
            Some(vec![1, 2, 3])
        );
    }

    #[test]
    fn test_shard_distribution() {
        // Verify that different pages use different shards
        // Pages should be distributed across all 64 shards
        let mut shard_used = [false; CELL_LOG_SHARDS];

        for pgno in 1..=256 {
            let page_number = PageNumber::new(pgno).unwrap();
            let shard_idx = CellVisibilityLog::shard_index(page_number);
            shard_used[shard_idx] = true;
        }

        // At least half the shards should be used with 256 pages
        let used_count = shard_used.iter().filter(|&&x| x).count();
        assert!(used_count >= CELL_LOG_SHARDS / 2);
    }

    #[test]
    fn test_memory_tracking_accurate() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let page_number = PageNumber::new(42).unwrap();
        let token = txn_token_n(1);

        // Insert several cells with known sizes
        for i in 0..10 {
            let cell_key = CellKey::table_row(btree, i);
            log.record_insert(cell_key, page_number, vec![0; 100], token)
                .expect("insert should succeed");
        }

        // Each cell has 100 bytes of data
        let expected_min_bytes = 10 * 100;
        let actual_bytes = log.cell_data_bytes();
        assert!(
            actual_bytes >= expected_min_bytes as u64,
            "Expected at least {} bytes, got {}",
            expected_min_bytes,
            actual_bytes
        );
    }

    #[test]
    fn test_commit_txn_bulk() {
        let log = CellVisibilityLog::new(1024 * 1024);
        let btree = BtreeRef::Table(TableId::new(1));
        let page_number = PageNumber::new(42).unwrap();
        let token = txn_token_n(1);

        // Insert several cells
        for i in 0..5 {
            let cell_key = CellKey::table_row(btree, i);
            log.record_insert(cell_key, page_number, vec![i as u8], token)
                .expect("insert should succeed");
        }

        // All should be uncommitted
        for i in 0..5 {
            let cell_key = CellKey::table_row(btree, i);
            assert!(
                log.resolve(page_number, &cell_key, CommitSeq::new(100))
                    .is_none()
            );
        }

        // Bulk commit
        log.commit_txn(token, CommitSeq::new(10));

        // All should now be visible
        for i in 0..5 {
            let cell_key = CellKey::table_row(btree, i);
            assert_eq!(
                log.resolve(page_number, &cell_key, CommitSeq::new(15)),
                Some(vec![i as u8])
            );
        }
    }

    #[test]
    fn test_per_txn_budget_exceeded() {
        // Create log with tiny per-txn budget
        let log = CellVisibilityLog::with_per_txn_budget(1024 * 1024, 100);
        let btree = BtreeRef::Table(TableId::new(1));
        let page_number = PageNumber::new(42).unwrap();
        let token = txn_token_n(1);

        // First insert should succeed
        let cell_key1 = CellKey::table_row(btree, 1);
        assert!(
            log.record_insert(cell_key1, page_number, vec![0; 50], token)
                .is_some()
        );

        // Second insert should fail (budget exceeded)
        let cell_key2 = CellKey::table_row(btree, 2);
        assert!(
            log.record_insert(cell_key2, page_number, vec![0; 100], token)
                .is_none()
        );
    }

    #[test]
    fn test_budget_exceeded_triggers_materialization() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

        let materialization_triggered = Arc::new(AtomicBool::new(false));
        let trigger_clone = Arc::clone(&materialization_triggered);

        // Create log with tiny global budget (200 bytes)
        let mut log = CellVisibilityLog::with_per_txn_budget(200, 1024 * 1024);
        log.set_materialization_callback(Box::new(move |_pgno, _cells| {
            trigger_clone.store(true, AtomicOrdering::SeqCst);
        }));

        let btree = BtreeRef::Table(TableId::new(1));
        let page_number = PageNumber::new(42).unwrap();
        let token = txn_token_n(1);

        // Insert cells until budget is exceeded
        for i in 0..5 {
            let cell_key = CellKey::table_row(btree, i);
            log.record_insert(cell_key, page_number, vec![0; 100], token);
        }

        // Materialization callback should have been triggered
        assert!(
            materialization_triggered.load(AtomicOrdering::SeqCst),
            "Materialization callback should be triggered when global budget exceeded"
        );
    }
}
