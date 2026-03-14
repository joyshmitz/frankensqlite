//! BEGIN CONCURRENT transaction protocol (§12.10).
//!
//! Implements MVCC concurrent-writer mode where multiple transactions can
//! write simultaneously to different pages.  Page-level conflict detection
//! uses first-committer-wins: if two CONCURRENT transactions modify the same
//! page, the second committer receives `SQLITE_BUSY_SNAPSHOT`.
//!
//! # Protocol
//!
//! 1. `BEGIN CONCURRENT` establishes a read snapshot without acquiring the
//!    global write mutex.
//! 2. Reads resolve through MVCC: `resolve(page, snapshot)` returns the
//!    newest committed version with `commit_seq <= snapshot.high`.
//! 3. Writes acquire per-page locks (not a global mutex).
//! 4. At commit time, the write set is validated against the commit index:
//!    any page modified by a concurrent transaction since the snapshot was taken
//!    triggers `BusySnapshot`.
//! 5. Savepoints within concurrent transactions work normally; `ROLLBACK TO`
//!    reverts write-set state but preserves page locks and the snapshot.

use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use fsqlite_types::{
    CommitSeq, PageData, PageNumber, Snapshot, TxnEpoch, TxnId, TxnToken, WitnessKey,
};
use parking_lot::{Mutex, MutexGuard};

use crate::core_types::{CommitIndex, InProcessPageLockTable, TransactionMode, TransactionState};
use crate::lifecycle::MvccError;
use crate::ssi_validation::{
    ActiveTxnView, CommittedReaderInfo, CommittedWriterInfo, DiscoveredEdge, SsiAbortReason,
    discover_incoming_edges, discover_outgoing_edges, evaluate_t3_dro, witness_key_page,
};

/// Maximum number of concurrent writers that can be active simultaneously.
///
/// This is a soft limit enforced at `begin_concurrent` time to prevent
/// unbounded resource consumption.
pub const MAX_CONCURRENT_WRITERS: usize = 128;

/// Stable shared handle for one active concurrent transaction.
pub type SharedConcurrentHandle = Arc<Mutex<ConcurrentHandle>>;

/// Result of first-committer-wins validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FcwResult {
    /// No conflicts: the write set is clean relative to the snapshot.
    Clean,
    /// One or more pages were modified by a concurrent transaction after
    /// the snapshot was established.
    Conflict {
        /// Pages that conflict (modified by another committer since snapshot).
        conflicting_pages: Vec<PageNumber>,
        /// The authoritative commit sequence that caused the conflict.
        conflicting_commit_seq: CommitSeq,
    },
    /// SSI abort due to anomalous access patterns.
    Abort {
        reason: crate::ssi_validation::SsiAbortReason,
    },
}

/// Lightweight handle representing one active concurrent session.
///
/// A `ConcurrentHandle` tracks the write set, page locks, snapshot, and SSI
/// state for a single `BEGIN CONCURRENT` transaction.
#[derive(Debug)]
pub struct ConcurrentHandle {
    /// Read snapshot established at `BEGIN CONCURRENT` time.
    snapshot: Snapshot,
    /// Per-page transactional state for staged writes, frees, synthetic
    /// conflict tracking, and held page locks.
    page_states: HashMap<PageNumber, PageTxnState>,
    /// Transaction state (Active / Committed / Aborted).
    state: TransactionState,
    /// Pages read by this transaction (for SSI rw-antidependency detection).
    read_set: HashSet<PageNumber>,
    /// Granular read witnesses bucketed by page for O(1) page-level lookup.
    /// Uses SmallVec to avoid allocations for the common case of 1-4 witnesses per page.
    read_index: HashMap<PageNumber, smallvec::SmallVec<[WitnessKey; 4]>>,
    /// Witnesses that are not bound to a specific page (global or custom).
    global_read_witnesses: Vec<WitnessKey>,
    /// Granular write witnesses bucketed by page.
    write_index: HashMap<PageNumber, smallvec::SmallVec<[WitnessKey; 4]>>,
    /// Global write witnesses.
    global_write_witnesses: Vec<WitnessKey>,
    /// Transaction token for SSI tracking.
    txn_token: TxnToken,
    /// Whether this transaction has an incoming rw-antidependency edge (SSI).
    has_in_rw: Cell<bool>,
    /// Whether this transaction has an outgoing rw-antidependency edge (SSI).
    has_out_rw: Cell<bool>,
    /// Whether this transaction was marked for abort by another committer.
    marked_for_abort: Cell<bool>,
}

#[derive(Debug, Clone, Default)]
struct PageTxnState {
    staged_data: Option<PageData>,
    is_freed: bool,
    is_conflict_only: bool,
    held_lock: bool,
}

impl PageTxnState {
    #[must_use]
    fn tracks_write_conflict(&self) -> bool {
        self.staged_data.is_some() || self.is_freed || self.is_conflict_only
    }

    #[must_use]
    fn is_empty(&self) -> bool {
        !self.held_lock && !self.tracks_write_conflict()
    }
}

pub struct WriteSetView<'a> {
    page_states: &'a HashMap<PageNumber, PageTxnState>,
}

impl WriteSetView<'_> {
    #[must_use]
    pub fn contains_key(&self, page: &PageNumber) -> bool {
        self.page_states
            .get(page)
            .is_some_and(|state| state.staged_data.is_some())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.page_states
            .values()
            .all(|state| state.staged_data.is_none())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.page_states
            .values()
            .filter(|state| state.staged_data.is_some())
            .count()
    }
}

pub struct HeldLocksView<'a> {
    page_states: &'a HashMap<PageNumber, PageTxnState>,
}

impl HeldLocksView<'_> {
    #[must_use]
    pub fn contains(&self, page: &PageNumber) -> bool {
        self.page_states
            .get(page)
            .is_some_and(|state| state.held_lock)
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.page_states.values().all(|state| !state.held_lock)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.page_states
            .values()
            .filter(|state| state.held_lock)
            .count()
    }
}

/// Snapshot of one page's local concurrent tracking state.
///
/// This is used to restore the in-memory MVCC bookkeeping if the underlying
/// pager rejects a write/free after we already updated the concurrent handle.
/// SSI witnesses are intentionally not rolled back; that remains a safe
/// overapproximation just like savepoint rollback.
#[derive(Debug, Clone)]
pub struct ConcurrentPageState {
    page: PageNumber,
    staged_data: Option<PageData>,
    was_freed: bool,
    was_conflict_only: bool,
    held_lock: bool,
}

impl ConcurrentPageState {
    #[must_use]
    pub fn is_synthetic_conflict_only(&self) -> bool {
        self.was_conflict_only && self.staged_data.is_none() && !self.was_freed
    }
}

#[derive(Debug, Clone, Default)]
struct SavepointPageState {
    staged_data: Option<PageData>,
    is_freed: bool,
    is_conflict_only: bool,
}

impl ConcurrentHandle {
    /// Create a new concurrent handle with the given snapshot and token.
    #[must_use]
    pub fn new(snapshot: Snapshot, txn_token: TxnToken) -> Self {
        Self {
            snapshot,
            page_states: HashMap::new(),
            state: TransactionState::Active,
            read_set: HashSet::new(),
            read_index: HashMap::new(),
            global_read_witnesses: Vec::new(),
            write_index: HashMap::new(),
            global_write_witnesses: Vec::new(),
            txn_token,
            has_in_rw: Cell::new(false),
            has_out_rw: Cell::new(false),
            marked_for_abort: Cell::new(false),
        }
    }

    /// Returns the read snapshot for this concurrent transaction.
    #[must_use]
    pub const fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Returns the current transaction state.
    #[must_use]
    pub const fn state(&self) -> TransactionState {
        self.state
    }

    /// Returns the set of pages in the write set.
    #[must_use]
    pub fn write_set_pages(&self) -> Vec<PageNumber> {
        let mut pages = self
            .page_states
            .iter()
            .filter_map(|(&page, state)| state.tracks_write_conflict().then_some(page))
            .collect::<Vec<_>>();
        pages.sort_unstable();
        pages
    }

    /// Returns the number of pages in the write set.
    #[must_use]
    pub fn write_set_len(&self) -> usize {
        self.page_states
            .values()
            .filter(|state| state.staged_data.is_some())
            .count()
    }

    /// Access the raw write set (for `is_dirty` checks).
    #[must_use]
    pub fn write_set(&self) -> WriteSetView<'_> {
        WriteSetView {
            page_states: &self.page_states,
        }
    }

    #[must_use]
    pub fn is_page_freed(&self, page: PageNumber) -> bool {
        self.page_states
            .get(&page)
            .is_some_and(|state| state.is_freed)
    }

    #[must_use]
    pub fn tracks_write_conflict_page(&self, page: PageNumber) -> bool {
        self.page_states
            .get(&page)
            .is_some_and(PageTxnState::tracks_write_conflict)
    }

    /// Returns the set of page locks held.
    #[must_use]
    pub fn held_locks(&self) -> HeldLocksView<'_> {
        HeldLocksView {
            page_states: &self.page_states,
        }
    }

    #[must_use]
    pub fn holds_page_lock(&self, page: PageNumber) -> bool {
        self.page_states
            .get(&page)
            .is_some_and(|state| state.held_lock)
    }

    /// Check whether this transaction is still active.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self.state, TransactionState::Active)
    }

    /// Return the transaction token used by SSI and commit tracking.
    #[must_use]
    pub const fn txn_token(&self) -> TxnToken {
        self.txn_token
    }

    /// Mark the transaction as committed.
    pub fn mark_committed(&mut self) {
        self.state = TransactionState::Committed;
    }

    /// Mark the transaction as aborted.
    pub fn mark_aborted(&mut self) {
        self.state = TransactionState::Aborted;
    }

    /// Record a page read (for SSI rw-antidependency detection).
    pub fn record_read(&mut self, page: PageNumber) {
        if self.read_set.insert(page) {
            self.read_index
                .entry(page)
                .or_default()
                .push(WitnessKey::Page(page));
        }
    }

    /// Record a granular read witness for fine-grained SSI.
    pub fn record_read_witness(&mut self, key: WitnessKey) {
        if let Some(p) = witness_key_page(&key) {
            self.read_set.insert(p);
            self.read_index.entry(p).or_default().push(key);
        } else {
            self.global_read_witnesses.push(key);
        }
    }

    /// Record a granular write witness for fine-grained SSI.
    pub fn record_write_witness(&mut self, key: WitnessKey) {
        if let Some(p) = witness_key_page(&key) {
            self.ensure_page_state(p).held_lock = true;
            let witnesses = self.write_index.entry(p).or_default();
            if !witnesses.iter().any(|existing| existing == &key) {
                witnesses.push(key);
            }
        } else if !self
            .global_write_witnesses
            .iter()
            .any(|existing| existing == &key)
        {
            self.global_write_witnesses.push(key);
        }
    }

    /// Returns the set of pages that were read.
    #[must_use]
    pub fn read_set(&self) -> &HashSet<PageNumber> {
        &self.read_set
    }

    /// Returns the number of pages in the read set.
    #[must_use]
    pub fn read_set_len(&self) -> usize {
        self.read_set.len()
    }

    /// Returns witness keys for all read pages (for SSI validation).
    #[must_use]
    pub fn read_witness_keys(&self) -> Vec<WitnessKey> {
        let mut keys: Vec<_> = self.read_index.values().flatten().cloned().collect();
        keys.extend(self.global_read_witnesses.iter().cloned());
        keys
    }

    /// Returns witness keys for all written pages (for SSI validation).
    #[must_use]
    pub fn write_witness_keys(&self) -> Vec<WitnessKey> {
        let mut keys: Vec<_> = self.write_index.values().flatten().cloned().collect();
        keys.extend(self.global_write_witnesses.iter().cloned());
        keys
    }

    #[must_use]
    pub fn has_global_read_witnesses(&self) -> bool {
        !self.global_read_witnesses.is_empty()
    }

    #[must_use]
    pub fn has_global_write_witnesses(&self) -> bool {
        !self.global_write_witnesses.is_empty()
    }

    /// Whether this handle has incoming rw-antidependency (SSI).
    #[must_use]
    pub fn has_in_rw(&self) -> bool {
        self.has_in_rw.get()
    }

    /// Whether this handle has outgoing rw-antidependency (SSI).
    #[must_use]
    pub fn has_out_rw(&self) -> bool {
        self.has_out_rw.get()
    }

    /// Whether this transaction was marked for abort by another committer.
    #[must_use]
    pub fn is_marked_for_abort(&self) -> bool {
        self.marked_for_abort.get()
    }

    fn page_state(&self, page: PageNumber) -> Option<&PageTxnState> {
        self.page_states.get(&page)
    }

    fn ensure_page_state(&mut self, page: PageNumber) -> &mut PageTxnState {
        self.page_states.entry(page).or_default()
    }

    fn remove_page_state_if_empty(&mut self, page: PageNumber) {
        if self
            .page_states
            .get(&page)
            .is_some_and(PageTxnState::is_empty)
        {
            self.page_states.remove(&page);
        }
    }
}

/// Implement ActiveTxnView for ConcurrentHandle to enable SSI validation.
impl ActiveTxnView for ConcurrentHandle {
    fn token(&self) -> TxnToken {
        self.txn_token
    }

    fn begin_seq(&self) -> CommitSeq {
        self.snapshot.high
    }

    fn is_active(&self) -> bool {
        matches!(self.state, TransactionState::Active)
    }

    fn read_keys(&self) -> &[WitnessKey] {
        // NOTE: We can't return a slice directly since indices are non-contiguous.
        &[]
    }

    fn write_keys(&self) -> &[WitnessKey] {
        &[]
    }

    fn check_read_overlap(&self, key: &WitnessKey) -> bool {
        // First check global witnesses.
        if self
            .global_read_witnesses
            .iter()
            .any(|w| crate::witness_plane::witness_keys_overlap(w, key))
        {
            return true;
        }

        // Level 0: Fast reject if the page isn't in our read set at all.
        let page = match witness_key_page(key) {
            Some(p) => p,
            None => return !self.read_set.is_empty() || !self.global_read_witnesses.is_empty(),
        };

        if !self.read_set.contains(&page) {
            return false;
        }

        // Level 1: Refined check against witnesses for THIS page only.
        if let Some(witnesses) = self.read_index.get(&page) {
            return witnesses
                .iter()
                .any(|w| crate::witness_plane::witness_keys_overlap(w, key));
        }

        // Fallback: If no granular index but page is in read_set, assume overlap.
        true
    }

    fn check_write_overlap(&self, key: &WitnessKey) -> bool {
        if self
            .global_write_witnesses
            .iter()
            .any(|w| crate::witness_plane::witness_keys_overlap(w, key))
        {
            return true;
        }

        let page = match witness_key_page(key) {
            Some(p) => p,
            None => {
                return self
                    .page_states
                    .values()
                    .any(PageTxnState::tracks_write_conflict)
                    || !self.global_write_witnesses.is_empty();
            }
        };

        if !self.tracks_write_conflict_page(page) {
            return false;
        }

        if let Some(witnesses) = self.write_index.get(&page) {
            return witnesses
                .iter()
                .any(|w| crate::witness_plane::witness_keys_overlap(w, key));
        }

        true
    }

    fn has_in_rw(&self) -> bool {
        self.has_in_rw.get()
    }

    fn has_out_rw(&self) -> bool {
        self.has_out_rw.get()
    }

    fn set_has_out_rw(&self, val: bool) {
        self.has_out_rw.set(val);
    }

    fn set_has_in_rw(&self, val: bool) {
        self.has_in_rw.set(val);
    }

    fn set_marked_for_abort(&self, val: bool) {
        self.marked_for_abort.set(val);
    }
}

/// Savepoint within a concurrent transaction.
///
/// Per spec §5.4: page locks are NOT released on `ROLLBACK TO`.
/// SSI witnesses are NOT rolled back (safe overapproximation).
#[derive(Debug, Clone)]
pub struct ConcurrentSavepoint {
    /// Savepoint name.
    pub name: String,
    /// Snapshot of per-page tracking state at savepoint creation time.
    page_states_snapshot: HashMap<PageNumber, SavepointPageState>,
    /// Number of pages in write_set at savepoint creation.
    write_set_len: usize,
}

impl ConcurrentSavepoint {
    /// Returns the number of pages captured in this savepoint.
    #[must_use]
    pub fn captured_len(&self) -> usize {
        self.write_set_len
    }
}

/// Registry tracking all active concurrent writers for a database.
///
/// Enforces the soft limit on concurrent writers and provides the shared
/// state needed for first-committer-wins and SSI validation.
#[derive(Debug)]
pub struct ConcurrentRegistry {
    /// Active concurrent handles, keyed by an opaque session id.
    active: HashMap<u64, SharedConcurrentHandle>,
    /// Committed-reader history (RCRI-like) for SSI edge discovery.
    committed_readers: Vec<CommittedReaderInfo>,
    /// Page-local index into committed reader history.
    committed_readers_by_page: HashMap<PageNumber, Vec<usize>>,
    /// Committed reader entries that include at least one global witness.
    committed_readers_with_global_keys: Vec<usize>,
    /// Committed-writer history (commit-log-like) for SSI edge discovery.
    committed_writers: Vec<CommittedWriterInfo>,
    /// Page-local index into committed writer history.
    committed_writers_by_page: HashMap<PageNumber, Vec<usize>>,
    /// Committed writer entries that include at least one global witness.
    committed_writers_with_global_keys: Vec<usize>,
    /// Next session id to assign.
    next_session_id: u64,
    /// Epoch counter for TxnToken generation (increments on each session).
    epoch_counter: u32,
}

impl ConcurrentRegistry {
    /// Create a new, empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active: HashMap::new(),
            committed_readers: Vec::new(),
            committed_readers_by_page: HashMap::new(),
            committed_readers_with_global_keys: Vec::new(),
            committed_writers: Vec::new(),
            committed_writers_by_page: HashMap::new(),
            committed_writers_with_global_keys: Vec::new(),
            next_session_id: 1,
            epoch_counter: 0,
        }
    }

    /// Number of currently active concurrent writers.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Begin a new concurrent transaction.
    ///
    /// Establishes a read snapshot and registers a new concurrent handle.
    /// Returns the session id and handle, or an error if the soft limit
    /// is reached.
    pub fn begin_concurrent(&mut self, snapshot: Snapshot) -> Result<u64, MvccError> {
        if self.active.len() >= MAX_CONCURRENT_WRITERS {
            return Err(MvccError::Busy);
        }
        let session_id = self.next_session_id;
        self.next_session_id = self.next_session_id.wrapping_add(1);
        self.epoch_counter = self.epoch_counter.wrapping_add(1);

        // Create TxnToken for SSI tracking.
        let txn_id = TxnId::new(session_id).ok_or(MvccError::InvalidState)?;
        let txn_token = TxnToken::new(txn_id, TxnEpoch::new(self.epoch_counter));

        let handle = Arc::new(Mutex::new(ConcurrentHandle::new(snapshot, txn_token)));
        self.active.insert(session_id, handle);
        Ok(session_id)
    }

    /// Returns an iterator over all active handles (for SSI validation).
    pub fn iter_active(&self) -> impl Iterator<Item = (u64, SharedConcurrentHandle)> + '_ {
        self.active
            .iter()
            .map(|(&id, handle)| (id, Arc::clone(handle)))
    }

    /// Look up a shared concurrent handle by session id.
    #[must_use]
    pub fn handle(&self, session_id: u64) -> Option<SharedConcurrentHandle> {
        self.active.get(&session_id).map(Arc::clone)
    }

    /// Lock a concurrent handle by session id.
    #[must_use]
    pub fn get(&self, session_id: u64) -> Option<MutexGuard<'_, ConcurrentHandle>> {
        self.active.get(&session_id).map(|handle| handle.lock())
    }

    /// Lock a concurrent handle by session id for mutation.
    pub fn get_mut(&self, session_id: u64) -> Option<MutexGuard<'_, ConcurrentHandle>> {
        self.active.get(&session_id).map(|handle| handle.lock())
    }

    /// Remove a session (after commit or abort).
    pub fn remove(&mut self, session_id: u64) -> Option<SharedConcurrentHandle> {
        self.active.remove(&session_id)
    }

    /// Prune committed SSI history that cannot overlap any active transaction.
    fn prune_committed_conflict_history(&mut self) {
        let Some(min_active_begin) = self.history_retention_horizon() else {
            self.committed_readers.clear();
            self.committed_readers_by_page.clear();
            self.committed_readers_with_global_keys.clear();
            self.committed_writers.clear();
            self.committed_writers_by_page.clear();
            self.committed_writers_with_global_keys.clear();
            return;
        };
        self.committed_readers
            .retain(|reader| reader.commit_seq > min_active_begin);
        self.committed_writers
            .retain(|writer| writer.commit_seq > min_active_begin);

        // Safety bound: prevent unbounded memory growth if a long-running
        // transaction pins the horizon.
        const MAX_HISTORY_ENTRIES: usize = 16384;
        while self.committed_readers.len() + self.committed_writers.len() > MAX_HISTORY_ENTRIES {
            // Find the oldest active transaction and mark it for abort to unpin the horizon.
            let mut oldest_id = None;
            let mut oldest_seq = CommitSeq::new(u64::MAX);
            for (&id, handle) in &self.active {
                let handle = handle.lock();
                if handle.is_active()
                    && !handle.is_marked_for_abort()
                    && handle.snapshot.high < oldest_seq
                {
                    oldest_seq = handle.snapshot.high;
                    oldest_id = Some(id);
                }
            }

            if let Some(id) = oldest_id {
                tracing::warn!(
                    session_id = id,
                    snapshot_high = oldest_seq.get(),
                    "prune_committed_conflict_history: marking long-running transaction for abort due to SSI history limit"
                );
                if let Some(handle) = self.active.get_mut(&id) {
                    let handle = handle.lock();
                    handle.set_marked_for_abort(true);
                }

                // Recompute the retained-history horizon and prune again now
                // that the oldest transaction is doomed to abort.
                let new_horizon = self
                    .history_retention_horizon()
                    .unwrap_or(CommitSeq::new(u64::MAX));
                self.committed_readers
                    .retain(|reader| reader.commit_seq > new_horizon);
                self.committed_writers
                    .retain(|writer| writer.commit_seq > new_horizon);
            } else {
                // If we couldn't find an active transaction to abort (should be impossible if
                // history is bounded by active transactions), just clear everything.
                tracing::warn!("prune_committed_conflict_history: forced to clear SSI history");
                self.committed_readers.clear();
                self.committed_readers_by_page.clear();
                self.committed_readers_with_global_keys.clear();
                self.committed_writers.clear();
                self.committed_writers_by_page.clear();
                self.committed_writers_with_global_keys.clear();
                break;
            }
        }

        self.rebuild_committed_history_indexes();
    }

    fn rebuild_committed_history_indexes(&mut self) {
        self.committed_readers_by_page.clear();
        self.committed_readers_with_global_keys.clear();
        for (idx, reader) in self.committed_readers.iter().enumerate() {
            let summary = summarize_witness_keys(&reader.keys);
            if summary.has_global_keys {
                self.committed_readers_with_global_keys.push(idx);
            }
            for page in summary.pages {
                self.committed_readers_by_page
                    .entry(page)
                    .or_default()
                    .push(idx);
            }
        }

        self.committed_writers_by_page.clear();
        self.committed_writers_with_global_keys.clear();
        for (idx, writer) in self.committed_writers.iter().enumerate() {
            let summary = summarize_witness_keys(&writer.keys);
            if summary.has_global_keys {
                self.committed_writers_with_global_keys.push(idx);
            }
            for page in summary.pages {
                self.committed_writers_by_page
                    .entry(page)
                    .or_default()
                    .push(idx);
            }
        }
    }

    fn index_committed_reader(&mut self, entry_idx: usize) {
        let Some(reader) = self.committed_readers.get(entry_idx) else {
            return;
        };
        let summary = summarize_witness_keys(&reader.keys);
        if summary.has_global_keys {
            self.committed_readers_with_global_keys.push(entry_idx);
        }
        for page in summary.pages {
            self.committed_readers_by_page
                .entry(page)
                .or_default()
                .push(entry_idx);
        }
    }

    fn index_committed_writer(&mut self, entry_idx: usize) {
        let Some(writer) = self.committed_writers.get(entry_idx) else {
            return;
        };
        let summary = summarize_witness_keys(&writer.keys);
        if summary.has_global_keys {
            self.committed_writers_with_global_keys.push(entry_idx);
        }
        for page in summary.pages {
            self.committed_writers_by_page
                .entry(page)
                .or_default()
                .push(entry_idx);
        }
    }

    fn committed_reader_candidates(
        &self,
        committing_txn: TxnToken,
        committing_begin_seq: CommitSeq,
        committing_commit_seq: CommitSeq,
        write_key_summary: &WitnessKeySummary,
    ) -> Vec<CommittedReaderInfo> {
        if self.committed_readers.is_empty() {
            return Vec::new();
        }
        let candidate_indexes = if write_key_summary.has_global_keys {
            (0..self.committed_readers.len()).collect::<Vec<_>>()
        } else {
            collect_indexed_candidates(
                &self.committed_readers_with_global_keys,
                &self.committed_readers_by_page,
                &write_key_summary.pages,
            )
        };
        let committing_begin = committing_begin_seq.get();
        let committing_end = committing_commit_seq.get();
        candidate_indexes
            .into_iter()
            .filter_map(|idx| self.committed_readers.get(idx))
            .filter(|reader| {
                reader.token != committing_txn
                    && committing_begin < reader.commit_seq.get()
                    && reader.begin_seq.get() < committing_end
            })
            .cloned()
            .collect()
    }

    fn committed_writer_candidates(
        &self,
        committing_txn: TxnToken,
        committing_begin_seq: CommitSeq,
        _committing_commit_seq: CommitSeq,
        read_key_summary: &WitnessKeySummary,
    ) -> Vec<CommittedWriterInfo> {
        if self.committed_writers.is_empty() {
            return Vec::new();
        }
        let candidate_indexes = if read_key_summary.has_global_keys {
            (0..self.committed_writers.len()).collect::<Vec<_>>()
        } else {
            collect_indexed_candidates(
                &self.committed_writers_with_global_keys,
                &self.committed_writers_by_page,
                &read_key_summary.pages,
            )
        };
        let committing_begin = committing_begin_seq.get();
        candidate_indexes
            .into_iter()
            .filter_map(|idx| self.committed_writers.get(idx))
            .filter(|writer| {
                writer.token != committing_txn && committing_begin < writer.commit_seq.get()
            })
            .cloned()
            .collect()
    }

    /// Compute the GC horizon: the minimum `snapshot.high` across all active
    /// concurrent transactions.
    ///
    /// Versions with `commit_seq <= horizon` that are superseded by a newer
    /// version are safe to reclaim (no active snapshot can see them).
    ///
    /// Returns `None` if no active transactions exist (caller should use the
    /// current commit_seq as the horizon, meaning all old versions can be pruned
    /// except the latest).
    #[must_use]
    pub fn gc_horizon(&self) -> Option<CommitSeq> {
        self.active
            .values()
            .filter_map(|handle| {
                let handle = handle.lock();
                handle.is_active().then_some(handle.snapshot.high)
            })
            .min()
    }

    /// Compute the retention horizon for committed SSI history.
    ///
    /// Unlike page-version GC, committed conflict history only needs to be
    /// retained for active transactions that may still commit. Transactions
    /// already marked for abort still pin MVCC visibility, but they are
    /// excluded here so history can be bounded once they are guaranteed to
    /// fail at commit time.
    #[must_use]
    fn history_retention_horizon(&self) -> Option<CommitSeq> {
        self.active
            .values()
            .filter_map(|handle| {
                let handle = handle.lock();
                (handle.is_active() && !handle.is_marked_for_abort())
                    .then_some(handle.snapshot.high)
            })
            .min()
    }
}

impl Default for ConcurrentRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Write a page within a concurrent transaction.
///
/// Acquires a page-level lock if not already held, then records the page
/// data in the write set.  Returns an error if the lock is held by another
/// concurrent transaction.
pub fn concurrent_prepare_write_page(
    handle: &mut ConcurrentHandle,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    page: PageNumber,
) -> Result<(), MvccError> {
    if !handle.is_active() {
        return Err(MvccError::InvalidState);
    }
    let (holds_lock, was_freed, was_conflict_only, has_staged_data) = handle
        .page_state(page)
        .map_or((false, false, false, false), |state| {
            (
                state.held_lock,
                state.is_freed,
                state.is_conflict_only,
                state.staged_data.is_some(),
            )
        });

    if has_staged_data {
        debug_assert!(holds_lock, "staged write pages must retain their page lock");
        debug_assert!(!was_freed, "staged write pages cannot also be marked freed");
        debug_assert!(
            !was_conflict_only,
            "staged write pages cannot also be conflict-only"
        );
        return Ok(());
    }

    if holds_lock && (was_freed || was_conflict_only) {
        let state = handle.ensure_page_state(page);
        state.is_freed = false;
        state.is_conflict_only = false;
        handle.remove_page_state_if_empty(page);
        return Ok(());
    }

    let already_tracked = handle.tracks_write_conflict_page(page);
    if !holds_lock {
        let txn_id = TxnId::new(session_id).ok_or(MvccError::InvalidState)?;
        if lock_table.try_acquire(page, txn_id).is_err() {
            handle.remove_page_state_if_empty(page);
            return Err(MvccError::Busy);
        }
        handle.ensure_page_state(page).held_lock = true;
    }
    let state = handle.ensure_page_state(page);
    state.is_freed = false;
    state.is_conflict_only = false;
    if !already_tracked {
        handle.record_write_witness(fsqlite_types::WitnessKey::Page(page));
    }
    handle.remove_page_state_if_empty(page);
    Ok(())
}

/// Stage payload bytes for a page that is already prepared for writing.
pub fn concurrent_stage_prepared_write_page(
    handle: &mut ConcurrentHandle,
    page: PageNumber,
    data: PageData,
) -> Result<(), MvccError> {
    if !handle.is_active() {
        return Err(MvccError::InvalidState);
    }

    debug_assert!(
        handle.holds_page_lock(page),
        "prepared page writes must retain their page lock before staging bytes"
    );
    debug_assert!(
        !handle
            .page_state(page)
            .is_some_and(|state| state.is_conflict_only),
        "prepared write staging must clear conflict-only tracking first"
    );
    debug_assert!(
        !handle.page_state(page).is_some_and(|state| state.is_freed),
        "prepared write staging must clear freed-page tracking first"
    );

    handle.ensure_page_state(page).staged_data = Some(data);
    Ok(())
}

/// Write a page within a concurrent transaction.
///
/// Acquires a page-level lock if not already held, then records the page
/// data in the write set. Returns an error if the lock is held by another
/// concurrent transaction.
pub fn concurrent_write_page(
    handle: &mut ConcurrentHandle,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    page: PageNumber,
    data: PageData,
) -> Result<(), MvccError> {
    concurrent_prepare_write_page(handle, lock_table, session_id, page)?;
    concurrent_stage_prepared_write_page(handle, page, data)
}

/// Capture the local concurrent tracking state for a page.
#[must_use]
pub fn concurrent_page_state(handle: &ConcurrentHandle, page: PageNumber) -> ConcurrentPageState {
    let state = handle.page_state(page);
    ConcurrentPageState {
        page,
        staged_data: state.and_then(|state| state.staged_data.clone()),
        was_freed: state.is_some_and(|state| state.is_freed),
        was_conflict_only: state.is_some_and(|state| state.is_conflict_only),
        held_lock: state.is_some_and(|state| state.held_lock),
    }
}

/// Restore a page's local concurrent tracking state after a failed pager write.
pub fn concurrent_restore_page_state(
    handle: &mut ConcurrentHandle,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    state: &ConcurrentPageState,
) -> Result<(), MvccError> {
    if !handle.is_active() {
        return Err(MvccError::InvalidState);
    }
    let txn_id = TxnId::new(session_id).ok_or(MvccError::InvalidState)?;

    {
        let restored = handle.ensure_page_state(state.page);
        restored.staged_data.clone_from(&state.staged_data);
        restored.is_freed = state.was_freed;
        restored.is_conflict_only = state.was_conflict_only;
        restored.held_lock = state.held_lock;
    }

    if !state.held_lock {
        lock_table.release(state.page, txn_id);
    }
    handle.remove_page_state_if_empty(state.page);
    Ok(())
}

/// Clear all local concurrent tracking for a page.
///
/// Upper layers use this when a synthetic conflict surface is no longer part
/// of the transaction's pending commit state. SSI witnesses are intentionally
/// retained as a safe overapproximation.
pub fn concurrent_clear_page_state(
    handle: &mut ConcurrentHandle,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    page: PageNumber,
) -> Result<(), MvccError> {
    if !handle.is_active() {
        return Err(MvccError::InvalidState);
    }
    let txn_id = TxnId::new(session_id).ok_or(MvccError::InvalidState)?;

    if let Some(state) = handle.page_state(page)
        && state.held_lock
    {
        lock_table.release(page, txn_id);
    }

    handle.page_states.remove(&page);
    Ok(())
}

/// Track a page in the write-conflict surface without staging payload bytes.
pub fn concurrent_track_write_conflict_page(
    handle: &mut ConcurrentHandle,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    page: PageNumber,
) -> Result<(), MvccError> {
    if !handle.is_active() {
        return Err(MvccError::InvalidState);
    }
    let already_tracked = handle.tracks_write_conflict_page(page);
    let holds_lock = handle.holds_page_lock(page);
    if already_tracked && holds_lock {
        debug_assert!(
            holds_lock,
            "tracked conflict pages must retain their page lock"
        );
        return Ok(());
    }
    if holds_lock {
        let state = handle.ensure_page_state(page);
        if state.staged_data.is_none() && !state.is_freed {
            state.is_conflict_only = true;
        }
        if !already_tracked {
            handle.record_write_witness(fsqlite_types::WitnessKey::Page(page));
        }
        return Ok(());
    }

    let txn_id = TxnId::new(session_id).ok_or(MvccError::InvalidState)?;
    if lock_table.try_acquire(page, txn_id).is_err() {
        handle.remove_page_state_if_empty(page);
        return Err(MvccError::Busy);
    }
    let state = handle.ensure_page_state(page);
    state.held_lock = true;
    if state.staged_data.is_none() && !state.is_freed {
        state.is_conflict_only = true;
    }
    if !already_tracked {
        handle.record_write_witness(fsqlite_types::WitnessKey::Page(page));
    }
    Ok(())
}

/// Record that a page was freed within a concurrent transaction.
///
/// The page remains part of the write-conflict surface even though it no
/// longer has staged payload bytes.
pub fn concurrent_free_page(
    handle: &mut ConcurrentHandle,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    page: PageNumber,
) -> Result<(), MvccError> {
    if !handle.is_active() {
        return Err(MvccError::InvalidState);
    }
    if handle.is_page_freed(page) {
        debug_assert!(
            handle.holds_page_lock(page),
            "freed pages must retain their page lock"
        );
        debug_assert!(
            handle
                .page_state(page)
                .is_none_or(|state| state.staged_data.is_none()),
            "freed pages cannot retain staged page bytes"
        );
        return Ok(());
    }
    if handle.holds_page_lock(page) {
        let already_tracked = handle.tracks_write_conflict_page(page);
        let state = handle.ensure_page_state(page);
        state.staged_data = None;
        state.is_conflict_only = false;
        state.is_freed = true;
        if !already_tracked {
            handle.record_write_witness(fsqlite_types::WitnessKey::Page(page));
        }
        return Ok(());
    }

    let txn_id = TxnId::new(session_id).ok_or(MvccError::InvalidState)?;
    let already_tracked = handle.tracks_write_conflict_page(page);
    if lock_table.try_acquire(page, txn_id).is_err() {
        handle.remove_page_state_if_empty(page);
        return Err(MvccError::Busy);
    }
    let state = handle.ensure_page_state(page);
    state.held_lock = true;
    state.staged_data = None;
    state.is_conflict_only = false;
    state.is_freed = true;
    if !already_tracked {
        handle.record_write_witness(fsqlite_types::WitnessKey::Page(page));
    }
    Ok(())
}

/// Read a page within a concurrent transaction.
///
/// Returns the page from the local write set if it was modified by this
/// transaction, otherwise returns `None` (caller should resolve via MVCC
/// version store using the handle's snapshot).
#[must_use]
pub fn concurrent_read_page(handle: &ConcurrentHandle, page: PageNumber) -> Option<&PageData> {
    handle
        .page_state(page)
        .and_then(|state| state.staged_data.as_ref())
}

/// Whether the page has been freed by this concurrent transaction.
#[must_use]
pub fn concurrent_page_is_freed(handle: &ConcurrentHandle, page: PageNumber) -> bool {
    handle.is_page_freed(page)
}

/// Validate the write set against the commit index using first-committer-wins.
///
/// For each page in the write set, checks whether any other transaction
/// committed a newer version since the snapshot was established.  If so,
/// the conflicting pages and the authoritative commit sequence are returned.
pub fn validate_first_committer_wins(
    handle: &ConcurrentHandle,
    commit_index: &CommitIndex,
) -> FcwResult {
    let snapshot_seq = handle.snapshot.high;
    let mut conflicting_pages = Vec::new();
    let mut max_conflicting_seq = CommitSeq::ZERO;

    for page in handle.write_set_pages() {
        if let Some(committed_seq) = commit_index.latest(page) {
            if committed_seq > snapshot_seq {
                conflicting_pages.push(page);
                if committed_seq > max_conflicting_seq {
                    max_conflicting_seq = committed_seq;
                }
            }
        }
    }

    if conflicting_pages.is_empty() {
        tracing::debug!(
            write_set_size = handle.write_set_len(),
            snapshot_seq = snapshot_seq.get(),
            "fcw_validation: clean (no base drift)"
        );
        FcwResult::Clean
    } else {
        // Sort for deterministic output.
        conflicting_pages.sort();
        tracing::warn!(
            conflicting_page_count = conflicting_pages.len(),
            max_conflicting_seq = max_conflicting_seq.get(),
            snapshot_seq = snapshot_seq.get(),
            "fcw_validation: base drift detected"
        );
        FcwResult::Conflict {
            conflicting_pages,
            conflicting_commit_seq: max_conflicting_seq,
        }
    }
}

/// SSI validation result for concurrent commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SsiResult {
    /// SSI validation passed.
    Clean,
    /// SSI validation failed — dangerous structure detected.
    Abort { reason: SsiAbortReason },
}

/// Prepared concurrent commit plan produced by SSI validation.
///
/// This captures the pre-commit validation result without publishing commit
/// side effects. Callers should run physical pager commit first, then pass
/// this plan to [`finalize_prepared_concurrent_commit_with_ssi`] to publish
/// conflict history and edge propagation. `planned_commit_seq` is an
/// optimistic planning frontier, not necessarily the final published commit
/// sequence if another commit slips in between prepare and finalize.
#[derive(Debug, Clone)]
pub struct PreparedConcurrentCommit {
    session_id: u64,
    planned_commit_seq: CommitSeq,
    txn_token: TxnToken,
    begin_seq: CommitSeq,
    read_keys: Vec<WitnessKey>,
    read_key_summary: WitnessKeySummary,
    write_keys: Vec<WitnessKey>,
    write_key_summary: WitnessKeySummary,
    write_set_pages: Vec<PageNumber>,
    has_in_rw: bool,
    has_out_rw: bool,
    incoming_edges: Vec<DiscoveredEdge>,
    outgoing_edges: Vec<DiscoveredEdge>,
    dro_t3_decision: Option<crate::ssi_abort_policy::DroHotPathDecision>,
}

impl PreparedConcurrentCommit {
    #[must_use]
    pub const fn session_id(&self) -> u64 {
        self.session_id
    }

    #[must_use]
    pub const fn planned_commit_seq(&self) -> CommitSeq {
        self.planned_commit_seq
    }

    #[must_use]
    pub const fn txn_token(&self) -> TxnToken {
        self.txn_token
    }

    #[must_use]
    pub const fn begin_seq(&self) -> CommitSeq {
        self.begin_seq
    }

    #[must_use]
    pub const fn has_in_rw(&self) -> bool {
        self.has_in_rw
    }

    #[must_use]
    pub const fn has_out_rw(&self) -> bool {
        self.has_out_rw
    }

    #[must_use]
    pub fn read_keys(&self) -> &[WitnessKey] {
        &self.read_keys
    }

    #[must_use]
    pub fn write_keys(&self) -> &[WitnessKey] {
        &self.write_keys
    }

    #[must_use]
    pub fn read_pages(&self) -> Vec<PageNumber> {
        self.read_key_summary.pages.clone()
    }

    #[must_use]
    pub fn write_set_pages(&self) -> &[PageNumber] {
        &self.write_set_pages
    }

    #[must_use]
    pub fn conflicting_txns(&self) -> Vec<TxnToken> {
        let mut txns = self
            .incoming_edges
            .iter()
            .map(|edge| edge.from)
            .chain(self.outgoing_edges.iter().map(|edge| edge.to))
            .collect::<Vec<_>>();
        txns.sort_by(|left, right| {
            left.id
                .get()
                .cmp(&right.id.get())
                .then_with(|| left.epoch.get().cmp(&right.epoch.get()))
        });
        txns.dedup();
        txns
    }

    #[must_use]
    pub fn conflict_pages(&self) -> Vec<PageNumber> {
        let mut pages = self
            .incoming_edges
            .iter()
            .chain(self.outgoing_edges.iter())
            .filter_map(|edge| witness_key_page(&edge.overlap_key))
            .collect::<Vec<_>>();
        if pages.is_empty() {
            pages.extend(self.write_set_pages.iter().copied());
        }
        pages.sort_unstable();
        pages.dedup();
        pages
    }

    #[must_use]
    pub const fn dro_t3_decision(&self) -> Option<crate::ssi_abort_policy::DroHotPathDecision> {
        self.dro_t3_decision
    }
}

/// Snapshot of one active transaction used during SSI edge discovery.
struct HandleView {
    token: TxnToken,
    begin_seq: CommitSeq,
    is_active: bool,
    read_pages: HashSet<PageNumber>,
    tracked_write_pages: HashSet<PageNumber>,
    read_keys: Vec<WitnessKey>,
    write_keys: Vec<WitnessKey>,
    has_global_read_witnesses: bool,
    has_global_write_witnesses: bool,
    has_in_rw: Cell<bool>,
    has_out_rw: Cell<bool>,
}

impl HandleView {
    fn new(handle: &ConcurrentHandle) -> Self {
        Self {
            token: handle.token(),
            begin_seq: handle.begin_seq(),
            is_active: handle.is_active(),
            read_pages: handle.read_set().clone(),
            tracked_write_pages: handle.write_set_pages().into_iter().collect(),
            read_keys: handle.read_witness_keys(),
            write_keys: handle.write_witness_keys(),
            has_global_read_witnesses: handle.has_global_read_witnesses(),
            has_global_write_witnesses: handle.has_global_write_witnesses(),
            has_in_rw: Cell::new(handle.has_in_rw()),
            has_out_rw: Cell::new(handle.has_out_rw()),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct WitnessKeySummary {
    pages: Vec<PageNumber>,
    has_global_keys: bool,
}

fn summarize_witness_keys(keys: &[WitnessKey]) -> WitnessKeySummary {
    let mut pages = Vec::new();
    let mut has_global_keys = false;
    for key in keys {
        if let Some(page) = witness_key_page(key) {
            pages.push(page);
        } else {
            has_global_keys = true;
        }
    }
    pages.sort_unstable();
    pages.dedup();
    WitnessKeySummary {
        pages,
        has_global_keys,
    }
}

fn collect_indexed_candidates(
    global_indexes: &[usize],
    indexes_by_page: &HashMap<PageNumber, Vec<usize>>,
    pages: &[PageNumber],
) -> Vec<usize> {
    let mut candidate_indexes = global_indexes.to_vec();
    for &page in pages {
        if let Some(indexes) = indexes_by_page.get(&page) {
            candidate_indexes.extend(indexes.iter().copied());
        }
    }
    candidate_indexes.sort_unstable();
    candidate_indexes.dedup();
    candidate_indexes
}

#[derive(Debug, Default)]
struct ActiveEdgeDiscoveryIndex {
    all_readers: Vec<usize>,
    all_writers: Vec<usize>,
    readers_with_global_keys: Vec<usize>,
    writers_with_global_keys: Vec<usize>,
    readers_by_page: HashMap<PageNumber, Vec<usize>>,
    writers_by_page: HashMap<PageNumber, Vec<usize>>,
}

impl ActiveEdgeDiscoveryIndex {
    fn build(views: &[HandleView]) -> Self {
        let mut index = Self::default();
        for (idx, view) in views.iter().enumerate() {
            if !view.read_keys.is_empty() {
                index.all_readers.push(idx);
                if view.has_global_read_witnesses {
                    index.readers_with_global_keys.push(idx);
                }
                for &page in &view.read_pages {
                    index.readers_by_page.entry(page).or_default().push(idx);
                }
            }
            if !view.write_keys.is_empty() {
                index.all_writers.push(idx);
                if view.has_global_write_witnesses {
                    index.writers_with_global_keys.push(idx);
                }
                for &page in &view.tracked_write_pages {
                    index.writers_by_page.entry(page).or_default().push(idx);
                }
            }
        }
        index
    }

    fn incoming_candidate_refs<'a>(
        &'a self,
        views: &'a [HandleView],
        committing_txn: TxnToken,
        _committing_begin_seq: CommitSeq,
        committing_commit_seq: CommitSeq,
        write_key_summary: &WitnessKeySummary,
    ) -> Vec<&'a dyn ActiveTxnView> {
        let candidate_indexes = if write_key_summary.has_global_keys {
            self.all_readers.clone()
        } else {
            collect_indexed_candidates(
                &self.readers_with_global_keys,
                &self.readers_by_page,
                &write_key_summary.pages,
            )
        };
        let committing_end = committing_commit_seq.get();
        candidate_indexes
            .into_iter()
            .filter_map(|idx| views.get(idx))
            .filter(|view| {
                view.token != committing_txn
                    && view.is_active
                    && view.begin_seq.get() < committing_end
            })
            .map(|view| view as &dyn ActiveTxnView)
            .collect()
    }

    fn outgoing_candidate_refs<'a>(
        &'a self,
        views: &'a [HandleView],
        committing_txn: TxnToken,
        _committing_begin_seq: CommitSeq,
        committing_commit_seq: CommitSeq,
        read_key_summary: &WitnessKeySummary,
    ) -> Vec<&'a dyn ActiveTxnView> {
        let candidate_indexes = if read_key_summary.has_global_keys {
            self.all_writers.clone()
        } else {
            collect_indexed_candidates(
                &self.writers_with_global_keys,
                &self.writers_by_page,
                &read_key_summary.pages,
            )
        };
        let committing_end = committing_commit_seq.get();
        candidate_indexes
            .into_iter()
            .filter_map(|idx| views.get(idx))
            .filter(|view| {
                view.token != committing_txn
                    && view.is_active
                    && view.begin_seq.get() < committing_end
            })
            .map(|view| view as &dyn ActiveTxnView)
            .collect()
    }
}

impl ActiveTxnView for HandleView {
    fn token(&self) -> TxnToken {
        self.token
    }

    fn begin_seq(&self) -> CommitSeq {
        self.begin_seq
    }

    fn is_active(&self) -> bool {
        self.is_active
    }

    fn read_keys(&self) -> &[WitnessKey] {
        &self.read_keys
    }

    fn write_keys(&self) -> &[WitnessKey] {
        &self.write_keys
    }

    fn check_read_overlap(&self, key: &WitnessKey) -> bool {
        match key {
            WitnessKey::Page(p)
            | WitnessKey::Cell { btree_root: p, .. }
            | WitnessKey::ByteRange { page: p, .. }
            | WitnessKey::KeyRange { btree_root: p, .. } => self.read_pages.contains(p),
            WitnessKey::Custom { .. } => {
                !self.read_pages.is_empty() || self.has_global_read_witnesses
            }
        }
    }

    fn check_write_overlap(&self, key: &WitnessKey) -> bool {
        match key {
            WitnessKey::Page(p)
            | WitnessKey::Cell { btree_root: p, .. }
            | WitnessKey::ByteRange { page: p, .. }
            | WitnessKey::KeyRange { btree_root: p, .. } => self.tracked_write_pages.contains(p),
            WitnessKey::Custom { .. } => {
                !self.tracked_write_pages.is_empty() || self.has_global_write_witnesses
            }
        }
    }

    fn has_in_rw(&self) -> bool {
        self.has_in_rw.get()
    }

    fn has_out_rw(&self) -> bool {
        self.has_out_rw.get()
    }

    fn set_has_out_rw(&self, val: bool) {
        self.has_out_rw.set(val);
    }

    fn set_has_in_rw(&self, val: bool) {
        self.has_in_rw.set(val);
    }

    fn set_marked_for_abort(&self, _val: bool) {}
}

fn evaluate_prepare_t3_dro(
    txn: TxnToken,
    incoming_edges: &[DiscoveredEdge],
    outgoing_edges: &[DiscoveredEdge],
    active_txn_count: usize,
    committed_reader_count: usize,
    committed_writer_count: usize,
) -> Option<crate::ssi_abort_policy::DroHotPathDecision> {
    if incoming_edges.is_empty() && outgoing_edges.is_empty() {
        return None;
    }

    let active_other_txns = active_txn_count.saturating_sub(1);
    let active_reader_population = active_other_txns
        .max(incoming_edges.len())
        .saturating_add(committed_reader_count);
    let active_writer_population = active_other_txns
        .max(outgoing_edges.len())
        .saturating_add(committed_writer_count);

    Some(evaluate_t3_dro(
        txn,
        active_reader_population,
        active_writer_population,
    ))
}

/// Commit a concurrent transaction.
///
/// Validates with first-committer-wins, then SSI (Serializable Snapshot
/// Isolation), then either commits (returning the assigned sequence) or
/// returns `BusySnapshot` on conflict.
pub fn concurrent_commit(
    handle: &mut ConcurrentHandle,
    commit_index: &CommitIndex,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    assign_commit_seq: CommitSeq,
) -> Result<CommitSeq, (MvccError, FcwResult)> {
    if !handle.is_active() {
        return Err((MvccError::InvalidState, FcwResult::Clean));
    }
    let txn_id = TxnId::new(session_id).ok_or((MvccError::InvalidState, FcwResult::Clean))?;

    // Step 1: First-committer-wins validation.
    let fcw_result = validate_first_committer_wins(handle, commit_index);
    match &fcw_result {
        FcwResult::Clean => {
            // FCW passed. Now run SSI validation.
            // NOTE: For now, we use a simplified SSI check that only looks at
            // the local handle's has_in_rw/has_out_rw flags and marked_for_abort.
            // Full SSI with cross-transaction edge discovery requires the
            // registry to be passed in (done in concurrent_commit_with_ssi).

            // Check if marked for abort by another committer.
            if handle.is_marked_for_abort() {
                tracing::warn!(
                    txn = %txn_id,
                    "concurrent_commit: SSI marked_for_abort"
                );
                lock_table.release_all(txn_id);
                handle.mark_aborted();
                return Err((MvccError::BusySnapshot, FcwResult::Clean));
            }

            // Check for dangerous structure (both in + out rw edges).
            if handle.has_in_rw() && handle.has_out_rw() {
                tracing::warn!(
                    txn = %txn_id,
                    "concurrent_commit: SSI pivot (in+out rw edges)"
                );
                lock_table.release_all(txn_id);
                handle.mark_aborted();
                return Err((MvccError::BusySnapshot, FcwResult::Clean));
            }

            // Commit: update commit index for every tracked write-conflict page,
            // including structural frees that no longer have staged bytes.
            for page in handle.write_set_pages() {
                commit_index.update(page, assign_commit_seq);
            }
            // Release all page locks.
            lock_table.release_all(txn_id);
            handle.mark_committed();
            Ok(assign_commit_seq)
        }
        FcwResult::Conflict { .. } | FcwResult::Abort { .. } => {
            // Release all page locks on conflict.
            lock_table.release_all(txn_id);
            handle.mark_aborted();
            Err((MvccError::BusySnapshot, fcw_result))
        }
    }
}

/// Commit a concurrent transaction with full SSI validation.
///
/// This version takes the registry mutably to perform cross-transaction SSI
/// edge discovery. It handles getting the handle internally.
#[allow(clippy::too_many_lines)]
pub fn prepare_concurrent_commit_with_ssi(
    registry: &mut ConcurrentRegistry,
    commit_index: &CommitIndex,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    planned_commit_seq: CommitSeq,
) -> Result<PreparedConcurrentCommit, (MvccError, FcwResult)> {
    let txn_id = TxnId::new(session_id).ok_or((MvccError::InvalidState, FcwResult::Clean))?;

    let commit_view = {
        let handle = registry
            .get(session_id)
            .ok_or((MvccError::InvalidState, FcwResult::Clean))?;
        if !handle.is_active() {
            return Err((MvccError::InvalidState, FcwResult::Clean));
        }

        // Step 1: First-committer-wins validation.
        let fcw_result = validate_first_committer_wins(&handle, commit_index);
        if !matches!(fcw_result, FcwResult::Clean) {
            Err(fcw_result)
        } else {
            let mut read_keys = handle.read_witness_keys();
            read_keys.sort_unstable();
            let mut write_keys = handle.write_witness_keys();
            write_keys.sort_unstable();
            Ok((
                handle.token(),
                handle.begin_seq(),
                read_keys,
                write_keys,
                handle.write_set_pages(),
                handle.is_marked_for_abort(),
            ))
        }
    };
    let (txn, begin_seq, sorted_read_keys, sorted_write_keys, write_set_pages, marked_for_abort) =
        match commit_view {
            Ok(view) => view,
            Err(fcw_result) => {
                lock_table.release_all(txn_id);
                if let Some(mut handle) = registry.get_mut(session_id) {
                    handle.mark_aborted();
                }
                return Err((MvccError::BusySnapshot, fcw_result));
            }
        };

    if marked_for_abort {
        tracing::warn!(
            txn = %txn_id,
            "prepare_concurrent_commit_with_ssi: marked_for_abort"
        );
        lock_table.release_all(txn_id);
        if let Some(mut handle) = registry.get_mut(session_id) {
            handle.mark_aborted();
        }
        return Err((MvccError::BusySnapshot, FcwResult::Clean));
    }

    // Step 2: Discover SSI edges without publishing side effects yet.
    let views = registry
        .iter_active()
        .filter_map(|(_, handle)| {
            let guard = handle.lock();
            guard.is_active().then_some(HandleView::new(&guard))
        })
        .collect::<Vec<_>>();
    let active_index = ActiveEdgeDiscoveryIndex::build(&views);
    let read_key_summary = summarize_witness_keys(&sorted_read_keys);
    let write_key_summary = summarize_witness_keys(&sorted_write_keys);
    let active_reader_candidates = active_index.incoming_candidate_refs(
        &views,
        txn,
        begin_seq,
        planned_commit_seq,
        &write_key_summary,
    );
    let committed_reader_candidates = registry.committed_reader_candidates(
        txn,
        begin_seq,
        planned_commit_seq,
        &write_key_summary,
    );
    let active_writer_candidates = active_index.outgoing_candidate_refs(
        &views,
        txn,
        begin_seq,
        planned_commit_seq,
        &read_key_summary,
    );
    let committed_writer_candidates =
        registry.committed_writer_candidates(txn, begin_seq, planned_commit_seq, &read_key_summary);

    let incoming_edges = discover_incoming_edges(
        txn,
        begin_seq,
        planned_commit_seq,
        &sorted_write_keys,
        &active_reader_candidates,
        &committed_reader_candidates,
    );
    let outgoing_edges = discover_outgoing_edges(
        txn,
        begin_seq,
        planned_commit_seq,
        &sorted_read_keys,
        &active_writer_candidates,
        &committed_writer_candidates,
    );

    let has_in_rw = !incoming_edges.is_empty();
    let has_out_rw = !outgoing_edges.is_empty();
    let dro_t3_decision = evaluate_prepare_t3_dro(
        txn,
        &incoming_edges,
        &outgoing_edges,
        registry.active.len(),
        registry.committed_readers.len(),
        registry.committed_writers.len(),
    );

    if has_in_rw && has_out_rw {
        let reason = SsiAbortReason::Pivot;
        tracing::warn!(
            ?txn,
            ?reason,
            dro_penalty = dro_t3_decision.map_or(0.0, |decision| decision.cvar_penalty),
            dro_threshold = dro_t3_decision.map_or(0.0, |decision| decision.threshold),
            "SSI validation aborted"
        );
        return Err((MvccError::BusySnapshot, FcwResult::Abort { reason }));
    }

    Ok(PreparedConcurrentCommit {
        session_id,
        planned_commit_seq,
        txn_token: txn,
        begin_seq,
        read_keys: sorted_read_keys,
        read_key_summary,
        write_keys: sorted_write_keys,
        write_key_summary,
        write_set_pages,
        has_in_rw,
        has_out_rw,
        incoming_edges,
        outgoing_edges,
        dro_t3_decision,
    })
}

/// Publish a previously prepared concurrent commit after physical pager commit.
///
/// Applies SSI side effects (T3 propagation), records committed conflict
/// history, updates the commit index, and releases page locks.
#[allow(clippy::too_many_lines)]
pub fn finalize_prepared_concurrent_commit_with_ssi(
    registry: &mut ConcurrentRegistry,
    commit_index: &CommitIndex,
    lock_table: &InProcessPageLockTable,
    prepared: &PreparedConcurrentCommit,
    committed_seq: CommitSeq,
) {
    debug_assert!(
        committed_seq >= prepared.planned_commit_seq,
        "final commit sequence must not move backwards from the planning frontier"
    );

    let Some(txn_id) = TxnId::new(prepared.session_id) else {
        return;
    };

    // Re-scan against current active state to capture overlap edges that may
    // appear after prepare but before finalize. This keeps committed pivot
    // decisions deterministic.
    let active_views: Vec<HandleView> = registry
        .active
        .values()
        .map(|handle| {
            let guard = handle.lock();
            HandleView::new(&guard)
        })
        .collect();
    let active_index = ActiveEdgeDiscoveryIndex::build(&active_views);
    let active_reader_candidates = active_index.incoming_candidate_refs(
        &active_views,
        prepared.txn_token,
        prepared.begin_seq,
        committed_seq,
        &prepared.write_key_summary,
    );
    let committed_reader_candidates = registry.committed_reader_candidates(
        prepared.txn_token,
        prepared.begin_seq,
        committed_seq,
        &prepared.write_key_summary,
    );
    let active_writer_candidates = active_index.outgoing_candidate_refs(
        &active_views,
        prepared.txn_token,
        prepared.begin_seq,
        committed_seq,
        &prepared.read_key_summary,
    );
    let committed_writer_candidates = registry.committed_writer_candidates(
        prepared.txn_token,
        prepared.begin_seq,
        committed_seq,
        &prepared.read_key_summary,
    );

    let mut incoming_edges = prepared.incoming_edges.clone();
    for edge in discover_incoming_edges(
        prepared.txn_token,
        prepared.begin_seq,
        committed_seq,
        &prepared.write_keys,
        &active_reader_candidates,
        &[],
    ) {
        if incoming_edges
            .iter()
            .all(|existing| existing.from != edge.from)
        {
            incoming_edges.push(edge);
        }
    }
    for edge in discover_incoming_edges(
        prepared.txn_token,
        prepared.begin_seq,
        committed_seq,
        &prepared.write_keys,
        &[],
        &committed_reader_candidates,
    ) {
        if incoming_edges
            .iter()
            .all(|existing| existing.from != edge.from)
        {
            incoming_edges.push(edge);
        }
    }

    let mut outgoing_edges = prepared.outgoing_edges.clone();
    for edge in discover_outgoing_edges(
        prepared.txn_token,
        prepared.begin_seq,
        committed_seq,
        &prepared.read_keys,
        &active_writer_candidates,
        &[],
    ) {
        if outgoing_edges.iter().all(|existing| existing.to != edge.to) {
            outgoing_edges.push(edge);
        }
    }
    for edge in discover_outgoing_edges(
        prepared.txn_token,
        prepared.begin_seq,
        committed_seq,
        &prepared.read_keys,
        &[],
        &committed_writer_candidates,
    ) {
        if outgoing_edges.iter().all(|existing| existing.to != edge.to) {
            outgoing_edges.push(edge);
        }
    }

    let has_in_rw = !incoming_edges.is_empty();
    let has_out_rw = !outgoing_edges.is_empty();
    let dro_t3_decision = evaluate_prepare_t3_dro(
        prepared.txn_token,
        &incoming_edges,
        &outgoing_edges,
        registry.active.len(),
        registry.committed_readers.len(),
        registry.committed_writers.len(),
    );
    let should_abort_active_pivot = dro_t3_decision.is_none_or(|decision| decision.should_abort());

    // T3 propagation for active readers on incoming edges.
    for edge in &incoming_edges {
        if !edge.source_is_active {
            continue;
        }
        for reader in registry.active.values() {
            let reader = reader.lock();
            if !reader.is_active() || reader.token() != edge.from {
                continue;
            }
            reader.set_has_out_rw(true);
            if reader.has_in_rw() {
                if should_abort_active_pivot {
                    tracing::debug!(
                        pivot = ?edge.from,
                        dro_penalty = dro_t3_decision.map_or(0.0, |decision| decision.cvar_penalty),
                        dro_threshold = dro_t3_decision.map_or(0.0, |decision| decision.threshold),
                        "prepare/finalize T3 rule: active reader is pivot, marking for abort"
                    );
                    reader.set_marked_for_abort(true);
                } else {
                    tracing::debug!(
                        pivot = ?edge.from,
                        dro_penalty = dro_t3_decision.map_or(0.0, |decision| decision.cvar_penalty),
                        dro_threshold = dro_t3_decision.map_or(0.0, |decision| decision.threshold),
                        "prepare/finalize T3 rule: active reader is pivot, DRO allows it to continue"
                    );
                }
            }
            break;
        }
    }

    // T3 propagation for active writers on outgoing edges.
    for edge in &outgoing_edges {
        if !edge.source_is_active {
            continue;
        }
        for writer in registry.active.values() {
            let writer = writer.lock();
            if !writer.is_active() || writer.token() != edge.to {
                continue;
            }
            writer.set_has_in_rw(true);
            if writer.has_out_rw() {
                if should_abort_active_pivot {
                    tracing::debug!(
                        pivot = ?edge.to,
                        dro_penalty = dro_t3_decision.map_or(0.0, |decision| decision.cvar_penalty),
                        dro_threshold = dro_t3_decision.map_or(0.0, |decision| decision.threshold),
                        "prepare/finalize T3 rule: active writer is pivot, marking for abort"
                    );
                    writer.set_marked_for_abort(true);
                } else {
                    tracing::debug!(
                        pivot = ?edge.to,
                        dro_penalty = dro_t3_decision.map_or(0.0, |decision| decision.cvar_penalty),
                        dro_threshold = dro_t3_decision.map_or(0.0, |decision| decision.threshold),
                        "prepare/finalize T3 rule: active writer is pivot, DRO allows it to continue"
                    );
                }
            }
            break;
        }
    }

    let mut mark_committed = false;
    if let Some(handle) = registry.get_mut(prepared.session_id) {
        if handle.is_active() {
            handle.has_in_rw.set(has_in_rw);
            handle.has_out_rw.set(has_out_rw);
            mark_committed = true;
        } else {
            tracing::warn!(
                session_id = prepared.session_id,
                "finalize_prepared_concurrent_commit_with_ssi: session inactive during finalize; applying commit-index/lock-table side effects"
            );
        }
    } else {
        tracing::warn!(
            session_id = prepared.session_id,
            "finalize_prepared_concurrent_commit_with_ssi: session missing during finalize; applying commit-index/lock-table side effects"
        );
    }

    for &page in &prepared.write_set_pages {
        commit_index.update(page, committed_seq);
    }
    lock_table.release_all(txn_id);
    if mark_committed {
        if let Some(mut handle) = registry.get_mut(prepared.session_id) {
            if handle.is_active() {
                handle.mark_committed();
            }
        }
    }

    if !prepared.read_keys.is_empty() {
        registry.committed_readers.push(CommittedReaderInfo {
            token: prepared.txn_token,
            begin_seq: prepared.begin_seq,
            commit_seq: committed_seq,
            had_in_rw: has_in_rw,
            keys: prepared.read_keys.clone(),
        });
        registry.index_committed_reader(registry.committed_readers.len() - 1);
    }
    if !prepared.write_keys.is_empty() {
        registry.committed_writers.push(CommittedWriterInfo {
            token: prepared.txn_token,
            commit_seq: committed_seq,
            had_out_rw: has_out_rw,
            keys: prepared.write_keys.clone(),
        });
        registry.index_committed_writer(registry.committed_writers.len() - 1);
    }
    registry.prune_committed_conflict_history();
}

#[allow(clippy::too_many_lines)]
pub fn concurrent_commit_with_ssi(
    registry: &mut ConcurrentRegistry,
    commit_index: &CommitIndex,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    assign_commit_seq: CommitSeq,
) -> Result<CommitSeq, (MvccError, FcwResult)> {
    let prepared = match prepare_concurrent_commit_with_ssi(
        registry,
        commit_index,
        lock_table,
        session_id,
        assign_commit_seq,
    ) {
        Ok(p) => p,
        Err(e) => {
            if let Some(handle) = registry.remove(session_id) {
                let mut handle = handle.lock();
                concurrent_abort(&mut handle, lock_table, session_id);
            }
            return Err(e);
        }
    };
    finalize_prepared_concurrent_commit_with_ssi(
        registry,
        commit_index,
        lock_table,
        &prepared,
        assign_commit_seq,
    );
    Ok(assign_commit_seq)
}

/// Abort a concurrent transaction, releasing all page locks.
pub fn concurrent_abort(
    handle: &mut ConcurrentHandle,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
) {
    if let Some(txn_id) = TxnId::new(session_id) {
        lock_table.release_all(txn_id);
    }
    handle.mark_aborted();
}

/// Create a savepoint within a concurrent transaction.
///
/// Captures the current write set state so it can be restored on
/// `ROLLBACK TO`.  Page locks are NOT captured (they persist across
/// rollback).
pub fn concurrent_savepoint(
    handle: &ConcurrentHandle,
    name: &str,
) -> Result<ConcurrentSavepoint, MvccError> {
    if !handle.is_active() {
        return Err(MvccError::InvalidState);
    }
    let page_states_snapshot = handle
        .page_states
        .iter()
        .filter_map(|(&page, state)| {
            state.tracks_write_conflict().then_some((
                page,
                SavepointPageState {
                    staged_data: state.staged_data.clone(),
                    is_freed: state.is_freed,
                    is_conflict_only: state.is_conflict_only,
                },
            ))
        })
        .collect();
    Ok(ConcurrentSavepoint {
        name: name.to_owned(),
        page_states_snapshot,
        write_set_len: handle.write_set_len(),
    })
}

/// Rollback to a savepoint within a concurrent transaction.
///
/// Restores the write set to the state captured by the savepoint.
/// Page locks are NOT released (per spec §5.4).
/// The snapshot remains active for continued operations.
pub fn concurrent_rollback_to_savepoint(
    handle: &mut ConcurrentHandle,
    lock_table: &InProcessPageLockTable,
    session_id: u64,
    savepoint: &ConcurrentSavepoint,
) -> Result<(), MvccError> {
    if !handle.is_active() {
        return Err(MvccError::InvalidState);
    }
    let txn_id = TxnId::new(session_id).ok_or(MvccError::InvalidState)?;
    let mut reacquired_pages = Vec::new();
    for &page in savepoint.page_states_snapshot.keys() {
        if !handle.page_state(page).is_some_and(|state| state.held_lock) {
            if lock_table.try_acquire(page, txn_id).is_err() {
                for reacquired in reacquired_pages {
                    lock_table.release(reacquired, txn_id);
                }
                return Err(MvccError::Busy);
            }
            reacquired_pages.push(page);
        }
    }

    let mut restored = HashMap::with_capacity(
        handle
            .page_states
            .len()
            .max(savepoint.page_states_snapshot.len()),
    );

    for (&page, snapshot_state) in &savepoint.page_states_snapshot {
        restored.insert(
            page,
            PageTxnState {
                staged_data: snapshot_state.staged_data.clone(),
                is_freed: snapshot_state.is_freed,
                is_conflict_only: snapshot_state.is_conflict_only,
                held_lock: true,
            },
        );
    }

    for (&page, state) in &handle.page_states {
        if state.held_lock {
            restored.entry(page).or_insert_with(|| PageTxnState {
                held_lock: true,
                ..PageTxnState::default()
            });
        }
    }

    handle.page_states = restored;
    Ok(())
}

/// Check whether a transaction mode supports concurrent writers.
#[must_use]
pub const fn is_concurrent_mode(mode: TransactionMode) -> bool {
    matches!(mode, TransactionMode::Concurrent)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fsqlite_types::{
        CommitSeq, PageData, PageNumber, PageSize, SchemaEpoch, Snapshot, TxnEpoch, TxnId,
        TxnToken, WitnessKey,
    };

    use crate::core_types::{CommitIndex, InProcessPageLockTable};
    use crate::lifecycle::MvccError;
    use crate::ssi_validation::ActiveTxnView;

    use super::{
        CommittedReaderInfo, CommittedWriterInfo, ConcurrentHandle, ConcurrentRegistry, FcwResult,
        HandleView, MAX_CONCURRENT_WRITERS, concurrent_abort, concurrent_clear_page_state,
        concurrent_commit, concurrent_commit_with_ssi, concurrent_free_page,
        concurrent_page_is_freed, concurrent_page_state, concurrent_read_page,
        concurrent_restore_page_state, concurrent_rollback_to_savepoint, concurrent_savepoint,
        concurrent_track_write_conflict_page, concurrent_write_page,
        finalize_prepared_concurrent_commit_with_ssi, prepare_concurrent_commit_with_ssi,
        summarize_witness_keys, validate_first_committer_wins,
    };

    fn test_snapshot(high: u64) -> Snapshot {
        Snapshot {
            high: CommitSeq::new(high),
            schema_epoch: SchemaEpoch::ZERO,
        }
    }

    fn test_page(n: u32) -> PageNumber {
        PageNumber::new(n).expect("page number must be nonzero")
    }

    fn test_data() -> PageData {
        PageData::zeroed(PageSize::DEFAULT)
    }

    fn test_token(id: u64) -> TxnToken {
        TxnToken::new(
            TxnId::new(id).expect("test transaction id"),
            TxnEpoch::new(id as u32 + 1),
        )
    }

    // -----------------------------------------------------------------------
    // Test 1: Two connections both BEGIN CONCURRENT; insert into different
    //         pages; both commit successfully.
    // -----------------------------------------------------------------------
    #[test]
    fn test_begin_concurrent_multiple_writers() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        // Two concurrent sessions with the same snapshot.
        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 1");
        let s2 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 2");

        // Session 1 writes page 5.
        let mut h1 = registry.get_mut(s1).expect("handle 1");
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data())
            .expect("write page 5");

        // Session 2 writes page 10 (different page => no conflict).
        let mut h2 = registry.get_mut(s2).expect("handle 2");
        concurrent_write_page(&mut h2, &lock_table, s2, test_page(10), test_data())
            .expect("write page 10");

        // Both commit successfully.
        let mut h1 = registry.get_mut(s1).expect("handle 1");
        let seq1 = concurrent_commit(&mut h1, &commit_index, &lock_table, s1, CommitSeq::new(11))
            .expect("commit 1");
        assert_eq!(seq1, CommitSeq::new(11));

        let mut h2 = registry.get_mut(s2).expect("handle 2");
        let seq2 = concurrent_commit(&mut h2, &commit_index, &lock_table, s2, CommitSeq::new(12))
            .expect("commit 2");
        assert_eq!(seq2, CommitSeq::new(12));
    }

    // -----------------------------------------------------------------------
    // Test 2: Page conflict triggers SQLITE_BUSY_SNAPSHOT.
    // -----------------------------------------------------------------------
    #[test]
    fn test_begin_concurrent_page_conflict_busy_snapshot() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 1");
        let s2 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 2");

        // Both write to page 5, but lock contention prevents s2 from
        // acquiring the same lock.  In our model, each session uses its
        // session_id as the lock holder.
        let mut h1 = registry.get_mut(s1).expect("handle 1");
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data())
            .expect("s1 write page 5");

        // s1 commits first (first-committer-wins).
        let mut h1 = registry.get_mut(s1).expect("handle 1");
        concurrent_commit(&mut h1, &commit_index, &lock_table, s1, CommitSeq::new(11))
            .expect("s1 commits first");

        // Now s2 tries to write and commit the same page.  The lock was
        // released by s1's commit, so s2 can acquire it.
        let mut h2 = registry.get_mut(s2).expect("handle 2");
        concurrent_write_page(&mut h2, &lock_table, s2, test_page(5), test_data())
            .expect("s2 write page 5");

        let mut h2 = registry.get_mut(s2).expect("handle 2");
        let result = concurrent_commit(&mut h2, &commit_index, &lock_table, s2, CommitSeq::new(12));
        assert!(result.is_err());
        let (err, fcw) = result.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);
        assert!(matches!(fcw, FcwResult::Conflict { .. }));
    }

    // -----------------------------------------------------------------------
    // Test 3: First-committer-wins with three concurrent transactions.
    // -----------------------------------------------------------------------
    #[test]
    fn test_begin_concurrent_first_committer_wins() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 1");
        let s2 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 2");
        let s3 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 3");

        // s1 writes page 5, s3 writes page 10 (no overlap).
        let mut h1 = registry.get_mut(s1).expect("h1");
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data()).unwrap();

        let mut h3 = registry.get_mut(s3).expect("h3");
        concurrent_write_page(&mut h3, &lock_table, s3, test_page(10), test_data()).unwrap();

        // s1 commits first on page 5.
        let mut h1 = registry.get_mut(s1).expect("h1");
        concurrent_commit(&mut h1, &commit_index, &lock_table, s1, CommitSeq::new(11))
            .expect("s1 commits");

        // s2 now tries page 5 (same as s1, but s1 already committed).
        let mut h2 = registry.get_mut(s2).expect("h2");
        concurrent_write_page(&mut h2, &lock_table, s2, test_page(5), test_data()).unwrap();

        let mut h2 = registry.get_mut(s2).expect("h2");
        let result = concurrent_commit(&mut h2, &commit_index, &lock_table, s2, CommitSeq::new(12));
        assert!(result.is_err());
        let (err, _) = result.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);

        // s3 commits on page 10 (no conflict with s1's page 5).
        let mut h3 = registry.get_mut(s3).expect("h3");
        let seq3 = concurrent_commit(&mut h3, &commit_index, &lock_table, s3, CommitSeq::new(13))
            .expect("s3 commits");
        assert_eq!(seq3, CommitSeq::new(13));
    }

    // -----------------------------------------------------------------------
    // Test 4: Savepoint within a concurrent transaction.
    // -----------------------------------------------------------------------
    #[test]
    fn test_savepoint_within_concurrent() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session");

        // Write page 1 (INSERT A).
        {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_write_page(&mut handle, &lock_table, s1, test_page(1), test_data()).unwrap();
        }

        // Create savepoint.
        let sp = {
            let handle = registry.get(s1).expect("handle");
            concurrent_savepoint(&handle, "sp1").unwrap()
        };
        assert_eq!(sp.captured_len(), 1);

        // Write page 2 (INSERT B).
        {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_write_page(&mut handle, &lock_table, s1, test_page(2), test_data()).unwrap();
            assert_eq!(handle.write_set_len(), 2);
        }

        // Rollback to savepoint: page 2 should be removed from write set,
        // but its lock should still be held.
        {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_rollback_to_savepoint(&mut handle, &lock_table, s1, &sp).unwrap();
            assert_eq!(handle.write_set_len(), 1);
            assert!(handle.held_locks().contains(&test_page(2))); // Lock preserved.
        }

        // Write page 3 (INSERT C).
        {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_write_page(&mut handle, &lock_table, s1, test_page(3), test_data()).unwrap();
        }

        // Commit: pages 1 and 3 are in the write set (not page 2).
        {
            let handle = registry.get_mut(s1).expect("handle");
            let mut pages = handle.write_set_pages();
            pages.sort();
            assert_eq!(pages, vec![test_page(1), test_page(3)]);
        }

        {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_commit(
                &mut handle,
                &commit_index,
                &lock_table,
                s1,
                CommitSeq::new(11),
            )
            .expect("commit succeeds");
        }
    }

    #[test]
    fn test_savepoint_within_concurrent_restores_freed_pages() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_write_page(&mut handle, &lock_table, s1, test_page(1), test_data()).unwrap();
        }

        let sp = {
            let handle = registry.get(s1).expect("handle");
            concurrent_savepoint(&handle, "sp1").unwrap()
        };

        {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_free_page(&mut handle, &lock_table, s1, test_page(1)).unwrap();
            assert!(concurrent_page_is_freed(&handle, test_page(1)));

            concurrent_rollback_to_savepoint(&mut handle, &lock_table, s1, &sp).unwrap();
            assert!(!concurrent_page_is_freed(&handle, test_page(1)));
            assert!(concurrent_read_page(&handle, test_page(1)).is_some());
            assert_eq!(handle.write_set_pages(), vec![test_page(1)]);
        }
    }

    // -----------------------------------------------------------------------
    // Test 5: Read from local write set vs MVCC fallback.
    // -----------------------------------------------------------------------
    #[test]
    fn test_concurrent_read_local_vs_mvcc() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session");

        // Before writing, local read returns None (would fall through to MVCC).
        let handle = registry.get(s1).expect("handle");
        assert!(concurrent_read_page(&handle, test_page(5)).is_none());

        // After writing, local read returns the written data.
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_write_page(&mut handle, &lock_table, s1, test_page(5), test_data()).unwrap();

        let handle = registry.get(s1).expect("handle");
        assert!(concurrent_read_page(&handle, test_page(5)).is_some());
        assert!(concurrent_read_page(&handle, test_page(6)).is_none());
    }

    #[test]
    fn test_concurrent_free_page_removes_local_read_and_still_tracks_conflict_page() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_write_page(&mut handle, &lock_table, s1, test_page(5), test_data()).unwrap();
        concurrent_free_page(&mut handle, &lock_table, s1, test_page(5)).unwrap();

        assert!(concurrent_page_is_freed(&handle, test_page(5)));
        assert!(concurrent_read_page(&handle, test_page(5)).is_none());
        assert_eq!(handle.write_set_len(), 0);
        assert_eq!(handle.write_set_pages(), vec![test_page(5)]);
    }

    #[test]
    fn test_validate_first_committer_wins_considers_freed_pages() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_free_page(&mut handle, &lock_table, s1, test_page(5)).unwrap();

        commit_index.update(test_page(5), CommitSeq::new(11));
        assert_eq!(
            validate_first_committer_wins(&handle, &commit_index),
            FcwResult::Conflict {
                conflicting_pages: vec![test_page(5)],
                conflicting_commit_seq: CommitSeq::new(11),
            }
        );
    }

    #[test]
    fn test_restore_page_state_releases_new_lock_and_clears_failed_free() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let mut handle = registry.get_mut(s1).expect("handle");
        let saved = concurrent_page_state(&handle, test_page(8));

        concurrent_free_page(&mut handle, &lock_table, s1, test_page(8)).unwrap();
        assert!(concurrent_page_is_freed(&handle, test_page(8)));

        concurrent_restore_page_state(&mut handle, &lock_table, s1, &saved).unwrap();
        assert!(!concurrent_page_is_freed(&handle, test_page(8)));
        assert!(concurrent_read_page(&handle, test_page(8)).is_none());
        assert!(!handle.held_locks().contains(&test_page(8)));

        let other_txn = fsqlite_types::TxnId::new(999).unwrap();
        assert!(
            lock_table.try_acquire(test_page(8), other_txn).is_ok(),
            "restoring clean state must release the transient page lock"
        );
        assert!(lock_table.release(test_page(8), other_txn));
    }

    #[test]
    fn test_clear_page_state_releases_lock_and_conflict_tracking() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_track_write_conflict_page(&mut handle, &lock_table, s1, PageNumber::ONE)
            .unwrap();

        assert!(handle.tracks_write_conflict_page(PageNumber::ONE));
        assert!(handle.held_locks().contains(&PageNumber::ONE));

        concurrent_clear_page_state(&mut handle, &lock_table, s1, PageNumber::ONE).unwrap();

        assert!(
            !handle.tracks_write_conflict_page(PageNumber::ONE),
            "clearing the page state must remove the synthetic conflict surface"
        );
        assert!(
            !handle.held_locks().contains(&PageNumber::ONE),
            "clearing the page state must release the page lock"
        );

        let other_txn = fsqlite_types::TxnId::new(999).unwrap();
        assert!(
            lock_table.try_acquire(PageNumber::ONE, other_txn).is_ok(),
            "clearing the page state must make the page lock available to other writers"
        );
        assert!(lock_table.release(PageNumber::ONE, other_txn));
    }

    #[test]
    fn test_rollback_to_savepoint_reacquires_cleared_page_state_lock() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let savepoint = {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_track_write_conflict_page(&mut handle, &lock_table, s1, PageNumber::ONE)
                .unwrap();
            concurrent_savepoint(&handle, "sp1").unwrap()
        };

        {
            let mut handle = registry.get_mut(s1).expect("handle");
            concurrent_clear_page_state(&mut handle, &lock_table, s1, PageNumber::ONE).unwrap();
            assert!(!handle.tracks_write_conflict_page(PageNumber::ONE));
            assert!(!handle.holds_page_lock(PageNumber::ONE));

            concurrent_rollback_to_savepoint(&mut handle, &lock_table, s1, &savepoint).unwrap();
            assert!(handle.tracks_write_conflict_page(PageNumber::ONE));
            assert!(handle.holds_page_lock(PageNumber::ONE));
        }

        let other_txn = fsqlite_types::TxnId::new(999).unwrap();
        assert!(
            lock_table.try_acquire(PageNumber::ONE, other_txn).is_err(),
            "rollback-to-savepoint must reacquire the cleared page lock"
        );
    }

    #[test]
    fn test_concurrent_commit_updates_commit_index_for_freed_pages() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_free_page(&mut handle, &lock_table, s1, test_page(11)).unwrap();

        concurrent_commit(
            &mut handle,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("commit should succeed");

        assert_eq!(commit_index.latest(test_page(11)), Some(CommitSeq::new(11)));
    }

    #[test]
    fn test_concurrent_commit_updates_commit_index_for_conflict_only_pages() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_track_write_conflict_page(&mut handle, &lock_table, s1, PageNumber::ONE)
            .unwrap();

        concurrent_commit(
            &mut handle,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("commit should succeed");

        assert_eq!(
            commit_index.latest(PageNumber::ONE),
            Some(CommitSeq::new(11))
        );
    }

    #[test]
    fn test_concurrent_write_page_fast_path_reuses_owned_page_without_duplicate_witnesses() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let page = test_page(5);
        let updated = PageData::from_vec(vec![0x7A; PageSize::DEFAULT.as_usize()]);

        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_write_page(&mut handle, &lock_table, s1, page, test_data()).unwrap();
        concurrent_write_page(&mut handle, &lock_table, s1, page, updated.clone()).unwrap();

        assert_eq!(concurrent_read_page(&handle, page), Some(&updated));
        assert_eq!(handle.held_locks().len(), 1);
        assert_eq!(
            handle
                .write_witness_keys()
                .iter()
                .filter(
                    |key| matches!(key, WitnessKey::Page(witness_page) if *witness_page == page)
                )
                .count(),
            1,
            "rewriting an already-owned page should not duplicate page witnesses"
        );
    }

    #[test]
    fn test_concurrent_track_write_conflict_page_fast_path_reuses_tracked_page_without_duplicate_witnesses()
     {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let page = PageNumber::ONE;

        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_track_write_conflict_page(&mut handle, &lock_table, s1, page).unwrap();
        concurrent_track_write_conflict_page(&mut handle, &lock_table, s1, page).unwrap();

        assert!(handle.tracks_write_conflict_page(page));
        assert_eq!(handle.held_locks().len(), 1);
        assert_eq!(
            handle
                .write_witness_keys()
                .iter()
                .filter(
                    |key| matches!(key, WitnessKey::Page(witness_page) if *witness_page == page)
                )
                .count(),
            1,
            "retracking an already-owned conflict-only page should not duplicate page witnesses"
        );
    }

    #[test]
    fn test_concurrent_write_page_reuses_savepoint_preserved_lock_without_duplicate_witnesses() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let page = test_page(11);
        let expected = test_data();

        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_write_page(&mut handle, &lock_table, s1, page, expected.clone()).unwrap();
        let savepoint = concurrent_savepoint(&handle, "sp1").unwrap();
        concurrent_write_page(&mut handle, &lock_table, s1, test_page(12), test_data()).unwrap();
        concurrent_rollback_to_savepoint(&mut handle, &lock_table, s1, &savepoint).unwrap();
        assert!(handle.held_locks().contains(&test_page(12)));
        assert!(!handle.tracks_write_conflict_page(test_page(12)));

        concurrent_write_page(
            &mut handle,
            &lock_table,
            s1,
            test_page(12),
            expected.clone(),
        )
        .unwrap();
        assert_eq!(
            concurrent_read_page(&handle, test_page(12)),
            Some(&expected)
        );
        assert_eq!(
            handle
                .write_witness_keys()
                .iter()
                .filter(
                    |key| matches!(key, WitnessKey::Page(witness_page) if *witness_page == test_page(12))
                )
                .count(),
            1,
            "reusing a savepoint-preserved lock should not duplicate page witnesses"
        );
    }

    #[test]
    fn test_concurrent_write_page_promotes_conflict_only_page_without_duplicate_witnesses() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let page = PageNumber::ONE;
        let expected = test_data();

        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_track_write_conflict_page(&mut handle, &lock_table, s1, page).unwrap();
        concurrent_write_page(&mut handle, &lock_table, s1, page, expected.clone()).unwrap();

        assert_eq!(concurrent_read_page(&handle, page), Some(&expected));
        assert!(
            !handle
                .page_state(page)
                .is_some_and(|state| state.is_conflict_only)
        );
        assert_eq!(
            handle
                .write_witness_keys()
                .iter()
                .filter(
                    |key| matches!(key, WitnessKey::Page(witness_page) if *witness_page == page)
                )
                .count(),
            1,
            "promoting a conflict-only page into the write set should not duplicate page witnesses"
        );
    }

    #[test]
    fn test_concurrent_write_page_rewrites_freed_page_without_duplicate_witnesses() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let page = test_page(13);
        let expected = test_data();

        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_write_page(&mut handle, &lock_table, s1, page, expected.clone()).unwrap();
        concurrent_free_page(&mut handle, &lock_table, s1, page).unwrap();
        concurrent_write_page(&mut handle, &lock_table, s1, page, expected.clone()).unwrap();

        assert_eq!(concurrent_read_page(&handle, page), Some(&expected));
        assert!(!concurrent_page_is_freed(&handle, page));
        assert_eq!(
            handle
                .write_witness_keys()
                .iter()
                .filter(
                    |key| matches!(key, WitnessKey::Page(witness_page) if *witness_page == page)
                )
                .count(),
            1,
            "rewriting a previously freed page should not duplicate page witnesses"
        );
    }

    #[test]
    fn test_concurrent_registry_remove_keeps_shared_handle_alive_for_existing_clones() {
        let mut registry = ConcurrentRegistry::new();
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let shared = registry.handle(s1).expect("shared handle");
        let removed = registry.remove(s1).expect("removed handle");

        assert!(Arc::ptr_eq(&shared, &removed));
        assert_eq!(registry.active_count(), 0);
        assert!(removed.lock().is_active());
    }

    // -----------------------------------------------------------------------
    // Test 6: Abort releases all page locks.
    // -----------------------------------------------------------------------
    #[test]
    fn test_concurrent_abort_releases_locks() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session");

        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_write_page(&mut handle, &lock_table, s1, test_page(5), test_data()).unwrap();
        concurrent_write_page(&mut handle, &lock_table, s1, test_page(6), test_data()).unwrap();
        assert_eq!(handle.held_locks().len(), 2);
        drop(handle);

        // Abort: locks released.
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_abort(&mut handle, &lock_table, s1);
        assert!(!handle.is_active());
        drop(handle);

        // Another session can now acquire the same locks.
        let s2 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 2");
        let mut handle2 = registry.get_mut(s2).expect("handle 2");
        concurrent_write_page(&mut handle2, &lock_table, s2, test_page(5), test_data())
            .expect("lock should be available after abort");
    }

    // -----------------------------------------------------------------------
    // Test 7: Registry enforces max concurrent writers.
    // -----------------------------------------------------------------------
    #[test]
    fn test_registry_max_concurrent_writers() {
        let mut registry = ConcurrentRegistry::new();
        for _ in 0..MAX_CONCURRENT_WRITERS {
            registry
                .begin_concurrent(test_snapshot(1))
                .expect("should succeed");
        }
        let result = registry.begin_concurrent(test_snapshot(1));
        assert_eq!(result.unwrap_err(), MvccError::Busy);
    }

    // -----------------------------------------------------------------------
    // Test 8: FCW validation with clean write set.
    // -----------------------------------------------------------------------
    #[test]
    fn test_fcw_validation_clean() {
        let commit_index = CommitIndex::new();
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session");
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_write_page(&mut handle, &lock_table, s1, test_page(5), test_data()).unwrap();

        let handle = registry.get(s1).expect("handle");
        assert_eq!(
            validate_first_committer_wins(&handle, &commit_index),
            FcwResult::Clean
        );
    }

    // -----------------------------------------------------------------------
    // Test 9: FCW validation detects conflicts.
    // -----------------------------------------------------------------------
    #[test]
    fn test_fcw_validation_conflict() {
        let commit_index = CommitIndex::new();
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        // Pre-populate commit index: page 5 was committed at seq 15.
        commit_index.update(test_page(5), CommitSeq::new(15));

        // Session with snapshot at seq 10 writes page 5.
        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session");
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_write_page(&mut handle, &lock_table, s1, test_page(5), test_data()).unwrap();

        let handle = registry.get(s1).expect("handle");
        let result = validate_first_committer_wins(&handle, &commit_index);
        match result {
            FcwResult::Conflict {
                conflicting_pages,
                conflicting_commit_seq,
            } => {
                assert_eq!(conflicting_pages, vec![test_page(5)]);
                assert_eq!(conflicting_commit_seq, CommitSeq::new(15));
            }
            FcwResult::Clean => panic!("expected conflict"),
            FcwResult::Abort { .. } => panic!("expected conflict, got abort"),
        }
    }

    // -----------------------------------------------------------------------
    // Test 10: BUSY_SNAPSHOT is distinguishable from BUSY.
    // -----------------------------------------------------------------------
    #[test]
    fn test_busy_snapshot_vs_busy() {
        // BusySnapshot (stale snapshot) vs Busy (lock contention) are different
        // error codes for application retry logic.
        assert_ne!(MvccError::BusySnapshot, MvccError::Busy);

        // Display representations are distinct.
        assert_eq!(
            format!("{}", MvccError::BusySnapshot),
            "SQLITE_BUSY_SNAPSHOT"
        );
        assert_eq!(format!("{}", MvccError::Busy), "SQLITE_BUSY");
    }

    // -----------------------------------------------------------------------
    // Test 11: Concurrent session lifecycle.
    // -----------------------------------------------------------------------
    #[test]
    fn test_concurrent_session_lifecycle() {
        let mut registry = ConcurrentRegistry::new();
        assert_eq!(registry.active_count(), 0);

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session");
        assert_eq!(registry.active_count(), 1);

        let handle = registry.get(s1).expect("handle");
        assert!(handle.is_active());
        drop(handle);

        let removed = registry.remove(s1);
        assert!(removed.is_some());
        assert_eq!(registry.active_count(), 0);
    }

    #[test]
    fn test_marked_for_abort_txn_still_pins_gc_but_not_history_retention() {
        let mut registry = ConcurrentRegistry::new();
        let doomed = registry
            .begin_concurrent(test_snapshot(10))
            .expect("doomed session");
        let survivor = registry
            .begin_concurrent(test_snapshot(20))
            .expect("survivor session");

        registry
            .get_mut(doomed)
            .expect("doomed handle")
            .set_marked_for_abort(true);
        let survivor_token = registry.get(survivor).expect("survivor handle").txn_token();
        registry
            .committed_readers
            .push(crate::ssi_validation::CommittedReaderInfo {
                token: survivor_token,
                begin_seq: CommitSeq::new(20),
                commit_seq: CommitSeq::new(15),
                had_in_rw: false,
                keys: vec![WitnessKey::Page(test_page(7))],
            });

        assert_eq!(
            registry.gc_horizon(),
            Some(CommitSeq::new(10)),
            "marked-for-abort transactions remain active until they actually abort"
        );
        assert_eq!(
            registry.history_retention_horizon(),
            Some(CommitSeq::new(20)),
            "committed SSI history may ignore transactions already doomed to abort"
        );

        registry.prune_committed_conflict_history();
        assert!(
            registry.committed_readers.is_empty(),
            "history older than the surviving retention horizon should be pruned"
        );
    }

    #[test]
    fn test_finalize_releases_locks_and_updates_commit_index_when_session_missing() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session 1");
        {
            let mut h1 = registry.get_mut(s1).expect("handle 1");
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data())
                .expect("session 1 writes page 5");
        }

        let prepared = prepare_concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("prepare should succeed");

        let removed = registry.remove(s1);
        assert!(
            removed.is_some(),
            "session should be removable to simulate handle disappearance"
        );

        finalize_prepared_concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            &prepared,
            CommitSeq::new(11),
        );

        assert_eq!(
            commit_index.latest(test_page(5)),
            Some(CommitSeq::new(11)),
            "finalize must update commit index even if handle is missing"
        );

        let s2 = registry
            .begin_concurrent(test_snapshot(11))
            .expect("session 2");
        let mut h2 = registry.get_mut(s2).expect("handle 2");
        concurrent_write_page(&mut h2, &lock_table, s2, test_page(5), test_data())
            .expect("page lock should be released during finalize");
    }

    #[test]
    fn test_finalize_rechecks_committed_writers_that_commit_after_prepare() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(7));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(9), test_data()).unwrap();
        }

        let prepared = prepare_concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("prepare should succeed before the late writer commits");

        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(7), test_data()).unwrap();
        }
        concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(11),
        )
        .expect("late writer should commit before the prepared txn finalizes");

        finalize_prepared_concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            &prepared,
            CommitSeq::new(12),
        );

        let committed = registry
            .get(s1)
            .expect("prepared txn handle should remain present until explicit removal");
        assert!(
            committed.has_out_rw(),
            "finalize must discover committed writers that were not present during prepare"
        );
    }

    #[test]
    fn test_finalize_rechecks_committed_readers_that_commit_after_prepare() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(3));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(9), test_data()).unwrap();
        }

        let prepared = prepare_concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("prepare should succeed before the late reader commits");

        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(9));
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(11), test_data()).unwrap();
        }
        concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(11),
        )
        .expect("late reader should commit before the prepared txn finalizes");

        finalize_prepared_concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            &prepared,
            CommitSeq::new(12),
        );

        let committed = registry
            .get(s1)
            .expect("prepared txn handle should remain present until explicit removal");
        assert!(
            committed.has_in_rw(),
            "finalize must discover committed readers that were not present during prepare"
        );
    }

    #[test]
    fn test_committed_reader_candidates_include_matching_page_and_global_entries_only() {
        let mut registry = ConcurrentRegistry::new();
        let relevant = test_token(101);
        let global = test_token(102);
        let unrelated = test_token(103);

        registry.committed_readers.push(CommittedReaderInfo {
            token: unrelated,
            begin_seq: CommitSeq::new(5),
            commit_seq: CommitSeq::new(12),
            had_in_rw: false,
            keys: vec![WitnessKey::Page(test_page(99))],
        });
        registry.index_committed_reader(0);
        registry.committed_readers.push(CommittedReaderInfo {
            token: relevant,
            begin_seq: CommitSeq::new(5),
            commit_seq: CommitSeq::new(12),
            had_in_rw: false,
            keys: vec![WitnessKey::Page(test_page(7))],
        });
        registry.index_committed_reader(1);
        registry.committed_readers.push(CommittedReaderInfo {
            token: global,
            begin_seq: CommitSeq::new(5),
            commit_seq: CommitSeq::new(12),
            had_in_rw: false,
            keys: vec![WitnessKey::Custom {
                namespace: 7,
                bytes: b"global-reader".to_vec(),
            }],
        });
        registry.index_committed_reader(2);

        let summary = summarize_witness_keys(&[WitnessKey::Page(test_page(7))]);
        let mut candidates = registry
            .committed_reader_candidates(
                test_token(999),
                CommitSeq::new(10),
                CommitSeq::new(20),
                &summary,
            )
            .into_iter()
            .map(|reader| reader.token)
            .collect::<Vec<_>>();
        candidates.sort_by_key(|token| token.id.get());

        assert_eq!(candidates, vec![relevant, global]);
    }

    #[test]
    fn test_committed_writer_candidates_include_matching_page_and_global_entries_only() {
        let mut registry = ConcurrentRegistry::new();
        let relevant = test_token(201);
        let global = test_token(202);
        let unrelated = test_token(203);

        registry.committed_writers.push(CommittedWriterInfo {
            token: unrelated,
            commit_seq: CommitSeq::new(12),
            had_out_rw: false,
            keys: vec![WitnessKey::Page(test_page(88))],
        });
        registry.index_committed_writer(0);
        registry.committed_writers.push(CommittedWriterInfo {
            token: relevant,
            commit_seq: CommitSeq::new(12),
            had_out_rw: false,
            keys: vec![WitnessKey::Page(test_page(9))],
        });
        registry.index_committed_writer(1);
        registry.committed_writers.push(CommittedWriterInfo {
            token: global,
            commit_seq: CommitSeq::new(12),
            had_out_rw: false,
            keys: vec![WitnessKey::Custom {
                namespace: 9,
                bytes: b"global-writer".to_vec(),
            }],
        });
        registry.index_committed_writer(2);

        let summary = summarize_witness_keys(&[WitnessKey::Page(test_page(9))]);
        let mut candidates = registry
            .committed_writer_candidates(
                test_token(999),
                CommitSeq::new(10),
                CommitSeq::new(20),
                &summary,
            )
            .into_iter()
            .map(|writer| writer.token)
            .collect::<Vec<_>>();
        candidates.sort_by_key(|token| token.id.get());

        assert_eq!(candidates, vec![relevant, global]);
    }

    #[test]
    fn test_handle_view_custom_overlap_respects_global_witnesses() {
        let mut handle = ConcurrentHandle::new(test_snapshot(10), test_token(301));
        handle.record_read_witness(WitnessKey::Custom {
            namespace: 3,
            bytes: b"read-global".to_vec(),
        });
        handle.record_write_witness(WitnessKey::Custom {
            namespace: 4,
            bytes: b"write-global".to_vec(),
        });
        let view = HandleView::new(&handle);

        assert!(view.check_read_overlap(&WitnessKey::Custom {
            namespace: 5,
            bytes: b"candidate".to_vec(),
        }));
        assert!(view.check_write_overlap(&WitnessKey::Custom {
            namespace: 6,
            bytes: b"candidate".to_vec(),
        }));
    }

    #[test]
    fn test_prepare_materializes_dro_decision_for_edgeful_commit() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(20));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(10), test_data()).unwrap();
        }
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(30));
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(20), test_data()).unwrap();
        }

        let prepared = prepare_concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("prepare should succeed");

        let decision = prepared
            .dro_t3_decision()
            .expect("edgeful prepare should materialize a DRO decision");
        assert_eq!(decision.active_readers, 1);
        assert_eq!(decision.active_writers, 1);
        assert!(decision.threshold >= 0.0);
    }

    #[test]
    fn test_prepare_skips_dro_decision_for_edge_free_commit() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data()).unwrap();
        }

        let prepared = prepare_concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("prepare should succeed");

        assert!(
            prepared.dro_t3_decision().is_none(),
            "edge-free prepare should not emit a DRO decision"
        );
    }

    // -----------------------------------------------------------------------
    // Test 12: Operations on non-active handle return InvalidState.
    // -----------------------------------------------------------------------
    #[test]
    fn test_operations_on_inactive_handle() {
        let lock_table = InProcessPageLockTable::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session");

        // Abort the handle.
        let mut handle = registry.get_mut(s1).expect("handle");
        concurrent_abort(&mut handle, &lock_table, s1);

        // Write should fail on aborted handle.
        let mut handle = registry.get_mut(s1).expect("handle");
        let result = concurrent_write_page(&mut handle, &lock_table, s1, test_page(1), test_data());
        assert_eq!(result.unwrap_err(), MvccError::InvalidState);

        // Savepoint should fail on aborted handle.
        let handle = registry.get(s1).expect("handle");
        let result = concurrent_savepoint(&handle, "sp1");
        assert_eq!(result.unwrap_err(), MvccError::InvalidState);
    }

    // -----------------------------------------------------------------------
    // SSI Tests (bd-1xo1)
    // -----------------------------------------------------------------------

    // Test 13: SSI - read tracking records pages.
    #[test]
    fn test_ssi_read_tracking() {
        let mut registry = ConcurrentRegistry::new();
        let s1 = registry
            .begin_concurrent(test_snapshot(10))
            .expect("session");

        let mut handle = registry.get_mut(s1).expect("handle");
        assert_eq!(handle.read_set_len(), 0);

        handle.record_read(test_page(5));
        handle.record_read(test_page(10));
        handle.record_read(test_page(5)); // Duplicate, should be deduplicated.

        assert_eq!(handle.read_set_len(), 2);
        assert!(handle.read_set().contains(&test_page(5)));
        assert!(handle.read_set().contains(&test_page(10)));
    }

    // Test 14: SSI - no conflict when reads/writes don't overlap.
    #[test]
    fn test_ssi_no_conflict_disjoint() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        // T1: reads page 5, writes page 10.
        // T2: reads page 20, writes page 30.
        // No overlap → both commit.
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        // T1 reads and writes.
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(5));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(10), test_data()).unwrap();
            // B
        }

        // T2 reads and writes.
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(20));
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(30), test_data()).unwrap();
        }

        // Both should commit successfully.
        let seq1 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("T1 commits");
        assert_eq!(seq1, CommitSeq::new(11));

        let seq2 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(12),
        )
        .expect("T2 commits");
        assert_eq!(seq2, CommitSeq::new(12));
    }

    // Test 14b: committed SSI history is pruned on transaction completion.
    #[test]
    fn test_ssi_committed_history_pruned_on_completion() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(5));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(10), test_data()).unwrap();
        }
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(10));
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(20), test_data()).unwrap();
        }

        concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        )
        .expect("first txn commits while second is still active");

        assert_eq!(
            registry.committed_readers.len(),
            1,
            "reader history retained while overlapping txn is active"
        );
        assert_eq!(
            registry.committed_writers.len(),
            1,
            "writer history retained while overlapping txn is active"
        );

        concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(11),
        )
        .expect("second txn commits");

        assert!(
            registry.committed_readers.is_empty(),
            "reader history pruned once no active transactions remain"
        );
        assert!(
            registry.committed_writers.is_empty(),
            "writer history pruned once no active transactions remain"
        );
    }

    // Test 15: SSI - pivot detection aborts transaction with both in/out edges.
    #[test]
    fn test_ssi_pivot_abort() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        // Classic write skew setup:
        // T1: reads A, writes B.
        // T2: reads B, writes A.
        //
        // When T1 tries to commit:
        // - Incoming: T2 read B, T1 writes B → incoming edge from T2 to T1
        // - Outgoing: T1 read A, T2 writes A → outgoing edge to T2
        // T1 has BOTH → T1 must abort.

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        // T1: reads page 5 (A), writes page 10 (B).
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(5)); // A
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(10), test_data()).unwrap();
            // B
        }

        // T2: reads page 10 (B), writes page 5 (A).
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(10)); // B
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(5), test_data()).unwrap();
            // A
        }

        // T1 tries to commit first: it has both incoming and outgoing edges.
        // Incoming: T2 read B (page 10), T1 writes B → edge from T2 to T1
        // Outgoing: T1 read A (page 5), T2 writes A → edge from T1 to T2
        // T1 is pivot and must abort.
        let result1 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        );
        assert!(
            result1.is_err(),
            "T1 should abort as pivot (both in and out edges)"
        );
        let (err, _) = result1.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);

        // After T1 aborted, T2 can now commit (T1 is no longer active, so no edges).
        let result2 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(11),
        );
        assert!(result2.is_ok(), "T2 should commit after T1 aborted");
    }

    // Test 16: SSI - transaction marked for abort fails.
    #[test]
    fn test_ssi_marked_for_abort() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let mut h1 = registry.get_mut(s1).unwrap();
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data()).unwrap();

        // Manually mark for abort.
        h1.marked_for_abort.set(true);

        // Commit should fail.
        let result = concurrent_commit(&mut h1, &commit_index, &lock_table, s1, CommitSeq::new(11));
        assert!(result.is_err());
        let (err, _) = result.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);
    }

    // Test 17: SSI - only incoming edge allows commit.
    #[test]
    fn test_ssi_only_incoming_edge_commits() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let mut h1 = registry.get_mut(s1).unwrap();
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data()).unwrap();

        // Set only incoming edge (no outgoing).
        h1.has_in_rw.set(true);
        h1.has_out_rw.set(false);

        // Commit should succeed (not a pivot).
        let result = concurrent_commit(&mut h1, &commit_index, &lock_table, s1, CommitSeq::new(11));
        assert!(result.is_ok(), "only incoming edge should allow commit");
    }

    // Test 18: SSI - only outgoing edge allows commit.
    #[test]
    fn test_ssi_only_outgoing_edge_commits() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let mut h1 = registry.get_mut(s1).unwrap();
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data()).unwrap();

        // Set only outgoing edge (no incoming).
        h1.has_in_rw.set(false);
        h1.has_out_rw.set(true);

        // Commit should succeed (not a pivot).
        let result = concurrent_commit(&mut h1, &commit_index, &lock_table, s1, CommitSeq::new(11));
        assert!(result.is_ok(), "only outgoing edge should allow commit");
    }

    // Test 19: SSI - both edges trigger abort.
    #[test]
    fn test_ssi_both_edges_aborts() {
        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let mut h1 = registry.get_mut(s1).unwrap();
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(5), test_data()).unwrap();

        // Set both edges → dangerous structure.
        h1.has_in_rw.set(true);
        h1.has_out_rw.set(true);

        // Commit should fail (pivot).
        let result = concurrent_commit(&mut h1, &commit_index, &lock_table, s1, CommitSeq::new(11));
        assert!(result.is_err());
        let (err, _) = result.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);
    }

    // Test 20: SSI - witness keys generated correctly.
    #[test]
    fn test_ssi_witness_keys() {
        let mut registry = ConcurrentRegistry::new();
        let lock_table = InProcessPageLockTable::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let mut h1 = registry.get_mut(s1).unwrap();
        h1.record_read(test_page(5));
        h1.record_read(test_page(10));
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(15), test_data()).unwrap();
        concurrent_write_page(&mut h1, &lock_table, s1, test_page(20), test_data()).unwrap();

        let read_keys = h1.read_witness_keys();
        let write_keys = h1.write_witness_keys();

        assert_eq!(read_keys.len(), 2);
        assert_eq!(write_keys.len(), 2);
    }

    // -----------------------------------------------------------------------
    // bd-mblr.6.7: Critical-path no-mock unit assertions
    //
    // The tests below exercise critical SSI invariants using ONLY real
    // components (ConcurrentHandle, ConcurrentRegistry, InProcessPageLockTable,
    // CommitIndex) — no MockActiveTxn, no manual flag setting.
    // -----------------------------------------------------------------------

    // Test 21 (bd-mblr.6.7): Three-transaction SSI marked-for-abort
    // propagation via real edge detection through concurrent_commit_with_ssi.
    //
    // Scenario:
    //   T1: reads D, writes C      (T1 will commit second)
    //   T2: reads C, writes D      (T2 will get marked_for_abort)
    //   T3: reads A, writes B      (T3 commits first, no overlap)
    //
    // When T3 commits first (disjoint pages → clean):
    //   - No edges, T3 commits cleanly.
    //
    // When T1 commits second:
    //   - Scans T2: T2 reads C, T1 writes C → incoming edge for T1.
    //     T2.has_out_rw = true.
    //   - Scans T2: T2 writes D, T1 reads D → outgoing edge for T1.
    //     T2.has_in_rw = true.
    //   - T1 has both in+out → T1 is pivot → T1 aborts.
    //   - T2 has both flags set, but was NOT marked_for_abort because
    //     has_in_rw was set in the outgoing-check AFTER the incoming-check
    //     tested it (false at that point).
    //
    // After T1 aborts, T2 can now commit because the pivot (T1) is gone.
    //
    // This test exercises the full real-component SSI path without any
    // mock objects or manual flag manipulation.
    #[test]
    fn test_ssi_three_txn_pivot_abort_real_components() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        // T1: reads pages {10, 20}, writes page 30.
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        // T2: reads page 50, writes page 10.
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        // T3: reads page 30, writes page 40.
        let s3 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        // T1 operations.
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(10));
            h1.record_read(test_page(20));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(30), test_data()).unwrap();
        }
        // T2 operations.
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(50));
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(10), test_data()).unwrap();
        }
        // T3 operations.
        {
            let mut h3 = registry.get_mut(s3).unwrap();
            h3.record_read(test_page(30));
            concurrent_write_page(&mut h3, &lock_table, s3, test_page(40), test_data()).unwrap();
        }

        // Step 1: T3 commits. T1 writes page 30, T3 reads page 30
        // (outgoing check: T1 wrote what T3 read). T1.has_in_rw = true.
        let result3 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s3,
            CommitSeq::new(11),
        );
        assert!(result3.is_ok(), "T3 commits (only outgoing edge)");

        // Verify T1's has_in_rw was set by T3's outgoing edge detection.
        {
            let h1 = registry.get(s1).unwrap();
            assert!(
                h1.has_in_rw(),
                "T1 must have has_in_rw: T1 writes 30, T3 reads 30"
            );
            assert!(!h1.has_out_rw(), "T1 must NOT have has_out_rw yet");
            assert!(
                !h1.is_marked_for_abort(),
                "T1 must NOT be marked_for_abort yet"
            );
        }

        // Step 2: T2 commits. T2 writes page 10, T1 reads page 10
        // (incoming check: T1 read what T2 writes). T1.has_out_rw = true.
        // T1 keeps running because the DRO hot-path decision stays below the
        // abort threshold at this contention level.
        let result2 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(12),
        );
        assert!(
            result2.is_ok(),
            "T2 commits (only incoming edge, not pivot)"
        );

        // Verify T1 is still live even though it now carries both SSI flags.
        {
            let h1 = registry.get(s1).unwrap();
            assert!(h1.has_in_rw(), "T1 still has has_in_rw (from T3's commit)");
            assert!(
                h1.has_out_rw(),
                "T1 now has has_out_rw (T2's incoming edge scan set it)"
            );
            assert!(
                !h1.is_marked_for_abort(),
                "low-contention DRO should defer the active-pivot abort mark"
            );
        }

        // Step 3: T1 tries to commit → fails with BusySnapshot
        // (actual pivot detected during T1's own commit-time scan).
        let result1 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(13),
        );
        assert!(
            result1.is_err(),
            "T1 must still abort when its own commit observes the full pivot"
        );
        let (err, _) = result1.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);
    }

    // Test 22 (bd-mblr.6.7): SSI edge propagation through real edge detection
    // with the DRO gate left open under low contention.
    //
    // The marked_for_abort path fires when the INCOMING edge check finds
    // that the other handle already has has_in_rw set. The incoming check
    // runs before the outgoing check, so has_in_rw must be set by a PRIOR
    // commit's scan.
    //
    // T1: reads pages {10, 20}, writes page 30
    // T2: reads page 50, writes page 10
    // T3: reads page 30, writes page 40
    //
    // Step 1: T3 commits. Scans T1:
    //   Outgoing check: T1 writes 30, T3 reads 30. Match!
    //   T1.has_in_rw = true. T3 commits (only outgoing, safe).
    //
    // Step 2: T2 commits. Scans T1:
    //   Incoming check: T2 writes 10, T1 reads {10,20}. Match on 10!
    //   T1.has_out_rw = true. T1 has both flags now, but the default
    //   low-contention DRO matrix keeps the active pivot running.
    //   T2 has only incoming edge => T2 commits.
    //
    // Step 3: T1 tries to commit → still fails with BusySnapshot
    // (actual pivot detected during T1's own commit-time scan).
    #[test]
    fn test_ssi_low_contention_dro_defers_marked_for_abort() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        // T1: reads pages 10+20, writes page 30.
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        // T2: reads page 50, writes page 10.
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        // T3: reads page 30, writes page 40.
        let s3 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        // T1 operations.
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(10));
            h1.record_read(test_page(20));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(30), test_data()).unwrap();
        }
        // T2 operations.
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(50));
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(10), test_data()).unwrap();
        }
        // T3 operations.
        {
            let mut h3 = registry.get_mut(s3).unwrap();
            h3.record_read(test_page(30));
            concurrent_write_page(&mut h3, &lock_table, s3, test_page(40), test_data()).unwrap();
        }

        // Step 1: T3 commits. T1 writes page 30, T3 reads page 30
        // (outgoing check: T1 wrote what T3 read). T1.has_in_rw = true.
        let result3 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s3,
            CommitSeq::new(11),
        );
        assert!(result3.is_ok(), "T3 commits (only outgoing edge)");

        // Verify T1's has_in_rw was set by T3's outgoing edge detection.
        {
            let h1 = registry.get(s1).unwrap();
            assert!(
                h1.has_in_rw(),
                "T1 must have has_in_rw: T1 writes 30, T3 reads 30"
            );
            assert!(!h1.has_out_rw(), "T1 must NOT have has_out_rw yet");
            assert!(
                !h1.is_marked_for_abort(),
                "T1 must NOT be marked_for_abort yet"
            );
        }

        // Step 2: T2 commits. T2 writes page 10, T1 reads page 10
        // (incoming check: T1 read what T2 writes). T1.has_out_rw = true.
        // T1 keeps running because the DRO hot-path decision stays below the
        // abort threshold at this contention level.
        let result2 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(12),
        );
        assert!(
            result2.is_ok(),
            "T2 commits (only incoming edge, not pivot)"
        );

        // Verify T1 is still live even though it now carries both SSI flags.
        {
            let h1 = registry.get(s1).unwrap();
            assert!(h1.has_in_rw(), "T1 still has has_in_rw (from T3's commit)");
            assert!(
                h1.has_out_rw(),
                "T1 now has has_out_rw (T2's incoming edge scan set it)"
            );
            assert!(
                !h1.is_marked_for_abort(),
                "low-contention DRO should defer the active-pivot abort mark"
            );
        }

        // Step 3: T1 tries to commit → still fails with BusySnapshot
        // (actual pivot detected during T1's own commit-time scan).
        let result1 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(13),
        );
        assert!(
            result1.is_err(),
            "T1 must still abort when its own commit observes the full pivot"
        );
        let (err, _) = result1.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);
    }

    // Test 23 (bd-mblr.6.7): SSI edge propagation — verify that a
    // commit's edge scan correctly sets other transactions' SSI flags
    // without any manual flag manipulation.
    #[test]
    fn test_ssi_edge_propagation_sets_flags_automatically() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        // T1: reads page 100, writes page 200.
        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        // T2: reads page 200, writes page 300.
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(100));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(200), test_data()).unwrap();
        }
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(200));
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(300), test_data()).unwrap();
        }

        // Before any commit: both T1 and T2 have no SSI flags.
        {
            let h1 = registry.get(s1).unwrap();
            let h2 = registry.get(s2).unwrap();
            assert!(!h1.has_in_rw());
            assert!(!h1.has_out_rw());
            assert!(!h2.has_in_rw());
            assert!(!h2.has_out_rw());
        }

        // T1 commits: T2 reads page 200, T1 writes page 200 →
        // incoming edge from T2→T1. T2.has_out_rw = true.
        // No outgoing edge (T1 reads 100, T2 writes 300 → no overlap).
        let result1 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        );
        assert!(result1.is_ok(), "T1 commits (only incoming edge)");

        // Verify T2's flags were set by the edge scan.
        {
            let h2 = registry.get(s2).unwrap();
            assert!(
                h2.has_out_rw(),
                "T2.has_out_rw must be set: T2 read page 200 that T1 wrote"
            );
            assert!(
                !h2.has_in_rw(),
                "T2.has_in_rw must NOT be set: no outgoing edge from T1 to T2"
            );
        }

        // T2 can commit (only has outgoing edge, not pivot).
        let result2 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(12),
        );
        assert!(
            result2.is_ok(),
            "T2 commits (only outgoing edge, not pivot)"
        );
    }

    // Test 24 (bd-2kkn4.5): FCW conflict detection with real CommitIndex
    // — verifies first-committer-wins uses real commit_index state.
    //
    // T1 and T2 both start with snapshot at seq 10.
    // T1 writes page 42, commits at seq 11 → commit_index[page42] = 11.
    // After T1 commits, the page lock is released.
    // T2 then writes page 42 (lock now available), and tries to commit.
    // FCW detects: commit_index[page42] = 11 > snapshot high = 10 → conflict.
    #[test]
    fn test_fcw_real_commit_index_conflict() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        // T1 writes page 42 and commits first.
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(42), test_data()).unwrap();
        }
        let result1 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        );
        assert!(result1.is_ok(), "T1 first-committer wins");

        // After T1 committed, the page lock on page 42 is released.
        // T2 now writes the same page.
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(42), test_data()).unwrap();
        }

        // T2 commits second — FCW detects page 42 was modified after
        // T2's snapshot (commit_seq 11 > snapshot high 10).
        let result2 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s2,
            CommitSeq::new(11),
        );
        assert!(result2.is_err(), "T2 must fail: FCW conflict on page 42");
        let (err, fcw) = result2.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);
        assert!(
            matches!(fcw, FcwResult::Conflict { .. }),
            "FCW must report conflict"
        );
    }

    // Test 25 (bd-2kkn4.5): Deterministic FCW tiebreaker policy.
    //
    // Simultaneous commit window (same commit_seq) is resolved deterministically
    // by ordering commit attempts on txn_id. Lower txn_id commits first and
    // therefore wins; higher txn_id sees BusySnapshot.
    #[test]
    fn test_fcw_deterministic_tiebreak_lower_txn_id_wins() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        let token1 = registry.get(s1).unwrap().txn_token();
        let token2 = registry.get(s2).unwrap().txn_token();
        let (winner_session, loser_session, winner_token) = if token1.id <= token2.id {
            (s1, s2, token1)
        } else {
            (s2, s1, token2)
        };

        {
            let mut winner_handle = registry.get_mut(winner_session).unwrap();
            concurrent_write_page(
                &mut winner_handle,
                &lock_table,
                winner_session,
                test_page(77),
                test_data(),
            )
            .unwrap();
        }

        // Same commit sequence for both contenders (simultaneous window).
        let winner = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            winner_session,
            CommitSeq::new(11),
        );
        assert!(
            winner.is_ok(),
            "lower txn_id should deterministically win tie window"
        );

        {
            let mut loser_handle = registry.get_mut(loser_session).unwrap();
            concurrent_write_page(
                &mut loser_handle,
                &lock_table,
                loser_session,
                test_page(77),
                test_data(),
            )
            .unwrap();
        }

        let loser = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            loser_session,
            CommitSeq::new(11),
        );
        assert!(
            loser.is_err(),
            "higher txn_id should lose deterministic tie"
        );
        let (err, fcw) = loser.unwrap_err();
        assert_eq!(err, MvccError::BusySnapshot);
        assert!(matches!(fcw, FcwResult::Conflict { .. }));

        let committed = registry
            .committed_writers
            .iter()
            .find(|entry| entry.token.id.get() == winner_session)
            .map(|entry| entry.token)
            .expect("winning writer should be recorded");
        assert_eq!(committed, winner_token);
    }

    // Test 26: committed writer pivot forces abort in prepare path.
    //
    // Scenario:
    // - T1 reads B, writes A.
    // - T2 reads C, writes B (active while T1 commits), so T1 has outgoing rw.
    // - T3 reads A, writes D from an old snapshot.
    //
    // T1 commits and should carry had_out_rw=true in committed writer history.
    // T3 then discovers outgoing edge T3 -> T1. Because T1 was already a
    // committed writer pivot source, T3 must abort.
    #[test]
    fn test_prepare_aborts_on_committed_writer_pivot() {
        use super::concurrent_commit_with_ssi;

        let lock_table = InProcessPageLockTable::new();
        let commit_index = CommitIndex::new();
        let mut registry = ConcurrentRegistry::new();

        let s1 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let s2 = registry.begin_concurrent(test_snapshot(10)).unwrap();
        let s3 = registry.begin_concurrent(test_snapshot(10)).unwrap();

        // T1: reads B (20), writes A (10)
        {
            let mut h1 = registry.get_mut(s1).unwrap();
            h1.record_read(test_page(20));
            concurrent_write_page(&mut h1, &lock_table, s1, test_page(10), test_data()).unwrap();
        }

        // T2: reads C (30), writes B (20)
        {
            let mut h2 = registry.get_mut(s2).unwrap();
            h2.record_read(test_page(30));
            concurrent_write_page(&mut h2, &lock_table, s2, test_page(20), test_data()).unwrap();
        }

        // T1 commits and should carry had_out_rw=true due to T2 writing B.
        let result1 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s1,
            CommitSeq::new(11),
        );
        assert!(
            result1.is_ok(),
            "T1 should commit with only outgoing rw edge"
        );
        let t1_writer = registry
            .committed_writers
            .iter()
            .find(|entry| entry.token.id.get() == s1)
            .expect("T1 writer history should be present");
        assert!(
            t1_writer.had_out_rw,
            "T1 should be recorded with had_out_rw"
        );

        // T3 now performs its workload using the earlier snapshot.
        {
            let mut h3 = registry.get_mut(s3).unwrap();
            h3.record_read(test_page(10));
            concurrent_write_page(&mut h3, &lock_table, s3, test_page(40), test_data()).unwrap();
        }

        // T3 must abort due to outgoing edge to committed writer pivot T1.
        let result3 = concurrent_commit_with_ssi(
            &mut registry,
            &commit_index,
            &lock_table,
            s3,
            CommitSeq::new(12),
        );
        assert!(
            result3.is_err(),
            "T3 must abort when it depends on committed writer pivot T1"
        );
        let (err3, _) = result3.unwrap_err();
        assert_eq!(err3, MvccError::BusySnapshot);
    }
}
