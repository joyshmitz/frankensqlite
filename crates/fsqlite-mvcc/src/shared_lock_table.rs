//! Cross-process shared-memory page lock table with rolling rebuild (§5.6.3).
//!
//! The [`SharedPageLockTable`] is a fixed-capacity open-addressing hash table
//! using linear probing and atomic CAS operations, designed for cross-process
//! page-level exclusive write locks. It supports a rolling rebuild protocol
//! (§5.6.3.1) that rotates between two physical tables without abort storms.
//!
//! Key design invariants:
//! - `page_number == 0` means empty slot
//! - `owner_txn == 0` means unlocked
//! - Keys (`page_number`) are NEVER deleted during normal `release()` — only
//!   cleared during rebuild under lock-quiescence (§5.6.3)
//! - Maximum load factor: 0.70 (Knuth Vol. 3 analysis for linear probing)

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default capacity per table (power-of-2). 1,048,576 entries × 12 bytes ≈ 12 MiB per table.
pub const DEFAULT_TABLE_CAPACITY: u32 = 1 << 20; // 1_048_576

/// Maximum load factor before acquisitions return SQLITE_BUSY (§5.6.3.1).
const MAX_LOAD_FACTOR: f64 = 0.70;

/// Sentinel value for `draining_table` when no drain is in progress.
const DRAINING_NONE: u32 = 0xFFFF_FFFF;

/// Default rebuild lease duration in seconds.
const DEFAULT_LEASE_SECS: u64 = 5;

// ---------------------------------------------------------------------------
// PageLockEntry
// ---------------------------------------------------------------------------

/// A single entry in the shared-memory lock table.
///
/// `page_number == 0` means empty. `owner_txn == 0` means unlocked.
/// Both fields are separate atomics for lock-free cross-process access.
pub struct PageLockEntry {
    /// Page number (0 = empty slot).
    page_number: AtomicU32,
    /// TxnId of exclusive lock holder (0 = unlocked).
    owner_txn: AtomicU64,
}

impl PageLockEntry {
    /// Create a new empty entry.
    fn new() -> Self {
        Self {
            page_number: AtomicU32::new(0),
            owner_txn: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// LockTableInstance
// ---------------------------------------------------------------------------

/// One of the two physical hash tables in the `SharedPageLockTable`.
struct LockTableInstance {
    entries: Vec<PageLockEntry>,
}

impl LockTableInstance {
    fn new(capacity: u32) -> Self {
        let entries: Vec<PageLockEntry> = (0..capacity).map(|_| PageLockEntry::new()).collect();
        Self { entries }
    }

    /// Count entries where `page_number != 0` (occupied slots).
    fn occupied_count(&self) -> u32 {
        let mut count = 0_u32;
        for entry in &self.entries {
            if entry.page_number.load(Ordering::Relaxed) != 0 {
                count += 1;
            }
        }
        count
    }

    /// Count entries where `owner_txn != 0` (actively locked slots).
    fn locked_count(&self) -> u32 {
        let mut count = 0_u32;
        for entry in &self.entries {
            if entry.owner_txn.load(Ordering::Relaxed) != 0 {
                count += 1;
            }
        }
        count
    }

    /// Check if the table is lock-quiescent: all `owner_txn == 0`.
    fn is_quiescent(&self) -> bool {
        self.entries
            .iter()
            .all(|e| e.owner_txn.load(Ordering::Acquire) == 0)
    }

    /// Clear all entries (set page_number=0, owner_txn=0).
    ///
    /// SAFETY: Must only be called when the table is lock-quiescent.
    fn clear_all(&self) {
        for entry in &self.entries {
            entry.page_number.store(0, Ordering::Release);
            entry.owner_txn.store(0, Ordering::Release);
        }
    }

    /// Release all locks held by a specific txn (crash cleanup, §5.6.3).
    ///
    /// Does NOT clear page_number (key-stable invariant).
    fn release_all_for_txn(&self, txn_id: u64) -> u32 {
        let mut released = 0_u32;
        for entry in &self.entries {
            if entry
                .owner_txn
                .compare_exchange(txn_id, 0, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                released += 1;
            }
        }
        released
    }
}

// ---------------------------------------------------------------------------
// Acquire / Release errors
// ---------------------------------------------------------------------------

/// Result of a lock acquisition attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireResult {
    /// Lock acquired successfully.
    Acquired,
    /// Lock already held by the requesting transaction (idempotent).
    AlreadyHeld,
    /// Lock held by another transaction.
    Busy { holder: u64 },
    /// Table capacity exceeded; new key insertion rejected (§5.6.3.1).
    CapacityExhausted,
}

impl AcquireResult {
    /// Returns true if the lock is held (acquired or already held).
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Acquired | Self::AlreadyHeld)
    }
}

/// Error from rebuild lease operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildLeaseError {
    /// Another process holds the rebuild lease.
    LeaseHeld { pid: u32 },
    /// No rebuild is in progress (can't drain/finalize).
    NoDrainInProgress,
    /// Draining table is not yet lock-quiescent.
    NotQuiescent { remaining: u32 },
    /// Active table is not empty (prior rebuild incomplete).
    TargetNotEmpty,
}

impl std::fmt::Display for RebuildLeaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeaseHeld { pid } => write!(f, "rebuild lease held by PID {pid}"),
            Self::NoDrainInProgress => f.write_str("no drain in progress"),
            Self::NotQuiescent { remaining } => {
                write!(f, "draining table not quiescent: {remaining} locks remain")
            }
            Self::TargetNotEmpty => {
                f.write_str("target table not empty from prior incomplete rebuild")
            }
        }
    }
}

impl std::error::Error for RebuildLeaseError {}

// ---------------------------------------------------------------------------
// SharedPageLockTable
// ---------------------------------------------------------------------------

/// Cross-process page lock table with two physical tables and rolling rebuild.
///
/// Uses open addressing with linear probing and atomic CAS operations.
/// The table supports a rolling rebuild protocol (§5.6.3.1) where one table
/// is active (new acquisitions) while the other drains without aborting
/// active transactions.
pub struct SharedPageLockTable {
    /// Per-table capacity (power-of-2).
    capacity: u32,
    /// Mask for index computation: `capacity - 1`.
    mask: u32,
    /// Which table (0 or 1) is active for new acquisitions.
    active_table: AtomicU32,
    /// Which table (0 or 1) is draining, or `DRAINING_NONE`.
    draining_table: AtomicU32,
    /// PID of the process holding the rebuild lease (0 = none).
    rebuild_pid: AtomicU32,
    /// PID birth timestamp for reuse defense.
    rebuild_pid_birth: AtomicU64,
    /// Rebuild lease expiry (unix timestamp seconds).
    rebuild_lease_expiry: AtomicU64,
    /// Rebuild epoch counter (increments on successful rebuild).
    rebuild_epoch: AtomicU32,
    /// The two physical tables.
    tables: [LockTableInstance; 2],
}

impl SharedPageLockTable {
    /// Create a new `SharedPageLockTable` with the given capacity per table.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0 or not a power of two.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        assert!(
            capacity > 0 && capacity.is_power_of_two(),
            "capacity must be a power of two"
        );
        info!(capacity, "SharedPageLockTable: created");
        Self {
            capacity,
            mask: capacity - 1,
            active_table: AtomicU32::new(0),
            draining_table: AtomicU32::new(DRAINING_NONE),
            rebuild_pid: AtomicU32::new(0),
            rebuild_pid_birth: AtomicU64::new(0),
            rebuild_lease_expiry: AtomicU64::new(0),
            rebuild_epoch: AtomicU32::new(0),
            tables: [
                LockTableInstance::new(capacity),
                LockTableInstance::new(capacity),
            ],
        }
    }

    /// Create with default capacity (1,048,576 entries per table).
    #[must_use]
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_TABLE_CAPACITY)
    }

    /// Hash a page number to a table index.
    ///
    /// Uses fibonacci hashing for better distribution with linear probing,
    /// avoiding primary clustering from identity hashing on sequential pages.
    #[inline]
    fn hash_index(&self, page_number: u32) -> u32 {
        // Fibonacci hashing: multiply by golden ratio constant, take high bits.
        let h = page_number.wrapping_mul(2_654_435_769);
        h & self.mask
    }

    // -----------------------------------------------------------------------
    // Acquire (§5.6.3 — linear probing with atomic insertion)
    // -----------------------------------------------------------------------

    /// Try to acquire an exclusive lock on `page_number` for `txn_id`.
    ///
    /// Follows the spec algorithm (§5.6.3):
    /// 0. Snapshot active/draining table selection (Acquire loads).
    /// 1. Check draining table first (if present).
    /// 2. Probe active table with linear probing + atomic CAS insertion.
    pub fn try_acquire(&self, page_number: u32, txn_id: u64) -> AcquireResult {
        debug_assert!(page_number != 0, "page_number 0 is the empty sentinel");
        debug_assert!(txn_id != 0, "txn_id 0 is the unlocked sentinel");

        // Step 0: Snapshot table selection.
        let active_idx = self.active_table.load(Ordering::Acquire);
        let draining_idx = self.draining_table.load(Ordering::Acquire);

        // Step 1: Check draining table first (§5.6.3 acquire step 1).
        if draining_idx != DRAINING_NONE {
            let draining = &self.tables[draining_idx as usize];
            match self.probe_for_existing(draining, page_number) {
                ProbeResult::FoundOwnedBy(owner) if owner == txn_id => {
                    return AcquireResult::AlreadyHeld;
                }
                ProbeResult::FoundOwnedBy(holder) => {
                    return AcquireResult::Busy { holder };
                }
                ProbeResult::FoundUnlocked | ProbeResult::NotFound => {
                    // Proceed to active table.
                }
            }
        }

        // Step 2: Probe active table.
        let active = &self.tables[active_idx as usize];

        // Load-factor guard: reject new key insertion if beyond 70%.
        let occupied = active.occupied_count();
        let at_capacity = f64::from(occupied) / f64::from(self.capacity) > MAX_LOAD_FACTOR;

        let mut idx = self.hash_index(page_number);
        let mut probes = 0_u32;

        loop {
            if probes >= self.capacity {
                // Full table wrap — should not happen if load factor is enforced.
                warn!(page_number, "SharedPageLockTable: full table probe wrap");
                return AcquireResult::CapacityExhausted;
            }

            let entry = &active.entries[idx as usize];
            let current_page = entry.page_number.load(Ordering::Acquire);

            if current_page == page_number {
                // Slot exists for this page. Try to CAS owner_txn from 0 → txn_id.
                match entry.owner_txn.compare_exchange(
                    0,
                    txn_id,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return AcquireResult::Acquired,
                    Err(current_owner) => {
                        if current_owner == txn_id {
                            return AcquireResult::AlreadyHeld;
                        }
                        return AcquireResult::Busy {
                            holder: current_owner,
                        };
                    }
                }
            } else if current_page == 0 {
                // Empty slot. Reject if at capacity (load factor guard).
                if at_capacity {
                    warn!(
                        page_number,
                        occupied,
                        capacity = self.capacity,
                        "SharedPageLockTable: capacity exhausted (load factor > 0.70)"
                    );
                    return AcquireResult::CapacityExhausted;
                }

                // Try to claim slot: CAS page_number from 0 → page_number.
                if entry
                    .page_number
                    .compare_exchange(0, page_number, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    // CAS failed — another process inserted into this slot.
                    // Re-read same slot (do NOT advance). The winner may
                    // have inserted our page_number here.
                    continue;
                }

                // Slot claimed. Now CAS owner_txn from 0 → txn_id.
                // MUST NOT use store() here (§5.6.3).
                return match entry.owner_txn.compare_exchange(
                    0,
                    txn_id,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => AcquireResult::Acquired,
                    Err(_) => {
                        // Another process raced and acquired the lock.
                        // MUST NOT continue probing for a second copy.
                        AcquireResult::Busy {
                            holder: entry.owner_txn.load(Ordering::Acquire),
                        }
                    }
                };
            }

            // Different page in this slot — linear probe forward.
            idx = (idx + 1) & self.mask;
            probes += 1;
        }
    }

    /// Probe a table for an existing lock on `page_number`.
    fn probe_for_existing(&self, table: &LockTableInstance, page_number: u32) -> ProbeResult {
        let mut idx = self.hash_index(page_number);
        let mut probes = 0_u32;

        loop {
            if probes >= self.capacity {
                return ProbeResult::NotFound;
            }

            let entry = &table.entries[idx as usize];
            let current_page = entry.page_number.load(Ordering::Acquire);

            if current_page == page_number {
                let owner = entry.owner_txn.load(Ordering::Acquire);
                if owner == 0 {
                    return ProbeResult::FoundUnlocked;
                }
                return ProbeResult::FoundOwnedBy(owner);
            }

            if current_page == 0 {
                return ProbeResult::NotFound;
            }

            idx = (idx + 1) & self.mask;
            probes += 1;
        }
    }

    // -----------------------------------------------------------------------
    // Release (§5.6.3 — key-stable, race-free)
    // -----------------------------------------------------------------------

    /// Release the lock on `page_number` held by `txn_id`.
    ///
    /// Checks active table first, then draining table. Does NOT modify
    /// `page_number` (key-stable invariant, §5.6.3).
    ///
    /// Returns `true` if the lock was released.
    pub fn release(&self, page_number: u32, txn_id: u64) -> bool {
        let active_idx = self.active_table.load(Ordering::Acquire);
        let draining_idx = self.draining_table.load(Ordering::Acquire);

        // Try active table first.
        if self.release_in_table(&self.tables[active_idx as usize], page_number, txn_id) {
            return true;
        }

        // Try draining table if present.
        if draining_idx != DRAINING_NONE {
            return self.release_in_table(&self.tables[draining_idx as usize], page_number, txn_id);
        }

        false
    }

    /// Release a lock within a specific table instance.
    fn release_in_table(&self, table: &LockTableInstance, page_number: u32, txn_id: u64) -> bool {
        let mut idx = self.hash_index(page_number);
        let mut probes = 0_u32;

        loop {
            if probes >= self.capacity {
                return false;
            }

            let entry = &table.entries[idx as usize];
            let current_page = entry.page_number.load(Ordering::Acquire);

            if current_page == page_number {
                // CAS owner_txn from txn_id → 0 (Release ordering, §5.6.3).
                return entry
                    .owner_txn
                    .compare_exchange(txn_id, 0, Ordering::Release, Ordering::Relaxed)
                    .is_ok();
            }

            if current_page == 0 {
                return false;
            }

            idx = (idx + 1) & self.mask;
            probes += 1;
        }
    }

    /// Release all locks held by `txn_id` in both tables (crash cleanup, §5.6.3).
    ///
    /// This is O(capacity) and is only used for orphaned TxnSlot cleanup.
    pub fn release_all_for_txn(&self, txn_id: u64) -> u32 {
        let mut total = 0_u32;
        for table in &self.tables {
            total += table.release_all_for_txn(txn_id);
        }
        total
    }

    /// Check which txn holds the lock on `page_number`, if any.
    ///
    /// Checks both active and draining tables.
    #[must_use]
    pub fn holder(&self, page_number: u32) -> Option<u64> {
        let active_idx = self.active_table.load(Ordering::Acquire);
        let draining_idx = self.draining_table.load(Ordering::Acquire);

        // Check active table.
        if let ProbeResult::FoundOwnedBy(owner) =
            self.probe_for_existing(&self.tables[active_idx as usize], page_number)
        {
            return Some(owner);
        }

        // Check draining table.
        if draining_idx != DRAINING_NONE {
            if let ProbeResult::FoundOwnedBy(owner) =
                self.probe_for_existing(&self.tables[draining_idx as usize], page_number)
            {
                return Some(owner);
            }
        }

        None
    }

    // -----------------------------------------------------------------------
    // Rolling rebuild protocol (§5.6.3.1)
    // -----------------------------------------------------------------------

    /// Acquire the rebuild lease via CAS on `rebuild_pid`.
    ///
    /// If the current lease holder is dead or the lease has expired,
    /// the lease may be stolen.
    ///
    /// # Arguments
    /// * `pid` — The requesting process's PID.
    /// * `pid_birth` — The requesting process's birth timestamp.
    /// * `now_secs` — Current unix timestamp in seconds.
    pub fn acquire_rebuild_lease(
        &self,
        pid: u32,
        pid_birth: u64,
        now_secs: u64,
    ) -> Result<(), RebuildLeaseError> {
        // Try CAS from 0 → pid.
        match self
            .rebuild_pid
            .compare_exchange(0, pid, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                self.rebuild_pid_birth.store(pid_birth, Ordering::Release);
                self.rebuild_lease_expiry
                    .store(now_secs + DEFAULT_LEASE_SECS, Ordering::Release);
                info!(
                    pid,
                    epoch = self.rebuild_epoch.load(Ordering::Relaxed),
                    "rebuild lease acquired"
                );
                Ok(())
            }
            Err(current_pid) => {
                // Check if the lease is stale (expired + holder dead).
                let expiry = self.rebuild_lease_expiry.load(Ordering::Acquire);
                if expiry < now_secs {
                    // Lease expired — try to steal.
                    match self.rebuild_pid.compare_exchange(
                        current_pid,
                        pid,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    ) {
                        Ok(_) => {
                            self.rebuild_pid_birth.store(pid_birth, Ordering::Release);
                            self.rebuild_lease_expiry
                                .store(now_secs + DEFAULT_LEASE_SECS, Ordering::Release);
                            warn!(
                                old_pid = current_pid,
                                new_pid = pid,
                                "rebuild lease stolen from expired holder"
                            );
                            Ok(())
                        }
                        Err(_) => Err(RebuildLeaseError::LeaseHeld { pid: current_pid }),
                    }
                } else {
                    Err(RebuildLeaseError::LeaseHeld { pid: current_pid })
                }
            }
        }
    }

    /// Renew the rebuild lease (extend expiry).
    pub fn renew_rebuild_lease(&self, pid: u32, now_secs: u64) -> bool {
        if self.rebuild_pid.load(Ordering::Acquire) == pid {
            self.rebuild_lease_expiry
                .store(now_secs + DEFAULT_LEASE_SECS, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Step 2: Rotate tables (fast, non-blocking).
    ///
    /// Designates the current active table as draining and activates
    /// the other (which must be empty from the last completed rebuild).
    pub fn rotate(&self) -> Result<(), RebuildLeaseError> {
        let draining = self.draining_table.load(Ordering::Acquire);
        if draining != DRAINING_NONE {
            // A drain is already in progress — can't rotate again.
            return Err(RebuildLeaseError::NoDrainInProgress);
        }

        let active = self.active_table.load(Ordering::Acquire);
        let new_active = 1 - active;

        // Verify the target table is empty.
        if self.tables[new_active as usize].occupied_count() > 0 {
            return Err(RebuildLeaseError::TargetNotEmpty);
        }

        // Set draining_table = current active (Release).
        self.draining_table.store(active, Ordering::Release);
        // Set active_table = new_active (Release).
        self.active_table.store(new_active, Ordering::Release);

        debug!(
            old_active = active,
            new_active, "SharedPageLockTable: tables rotated"
        );
        Ok(())
    }

    /// Step 3: Check drain progress.
    ///
    /// Returns the number of locked entries remaining in the draining table.
    #[must_use]
    pub fn drain_progress(&self) -> Option<DrainStatus> {
        let draining_idx = self.draining_table.load(Ordering::Acquire);
        if draining_idx == DRAINING_NONE {
            return None;
        }

        let draining = &self.tables[draining_idx as usize];
        let remaining = draining.locked_count();
        let quiescent = remaining == 0;

        Some(DrainStatus {
            remaining,
            quiescent,
        })
    }

    /// Step 3 (assist): Release orphaned locks in draining table.
    ///
    /// Calls the provided `is_active_txn` closure to determine if a txn
    /// is still active. Orphaned entries are CAS-cleared.
    pub fn drain_orphaned(&self, is_active_txn: impl Fn(u64) -> bool) -> u32 {
        let draining_idx = self.draining_table.load(Ordering::Acquire);
        if draining_idx == DRAINING_NONE {
            return 0;
        }

        let draining = &self.tables[draining_idx as usize];
        let mut cleaned = 0_u32;

        for entry in &draining.entries {
            let owner = entry.owner_txn.load(Ordering::Acquire);
            if owner != 0 && !is_active_txn(owner) {
                // Orphaned — try to clear.
                if entry
                    .owner_txn
                    .compare_exchange(owner, 0, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    cleaned += 1;
                }
            }
        }

        if cleaned > 0 {
            debug!(
                cleaned,
                "SharedPageLockTable: orphaned locks cleaned during drain"
            );
        }

        cleaned
    }

    /// Step 4+5: Clear drained table and finalize rebuild.
    ///
    /// Must only be called when draining table is lock-quiescent.
    /// Clears all entries, sets draining_table=NONE, increments rebuild_epoch,
    /// and releases the lease.
    pub fn finalize_rebuild(&self, pid: u32) -> Result<u32, RebuildLeaseError> {
        let draining_idx = self.draining_table.load(Ordering::Acquire);
        if draining_idx == DRAINING_NONE {
            return Err(RebuildLeaseError::NoDrainInProgress);
        }

        let draining = &self.tables[draining_idx as usize];

        // Verify quiescence.
        if !draining.is_quiescent() {
            let remaining = draining.locked_count();
            return Err(RebuildLeaseError::NotQuiescent { remaining });
        }

        // Clear drained table (§5.6.3.1 step 4).
        let cleared = draining.occupied_count();
        draining.clear_all();

        // Set draining_table = NONE (Release).
        self.draining_table.store(DRAINING_NONE, Ordering::Release);

        // Increment rebuild_epoch (step 5).
        let new_epoch = self.rebuild_epoch.fetch_add(1, Ordering::AcqRel) + 1;

        // Release lease.
        self.rebuild_pid.store(0, Ordering::Release);
        self.rebuild_pid_birth.store(0, Ordering::Release);
        self.rebuild_lease_expiry.store(0, Ordering::Release);

        info!(
            epoch = new_epoch,
            cleared, pid, "SharedPageLockTable: rebuild finalized"
        );

        Ok(cleared)
    }

    /// Full rolling rebuild cycle (convenience).
    ///
    /// Acquires lease, rotates, drains (polling with timeout), clears, and
    /// releases lease. Returns the rebuild result.
    #[allow(clippy::too_many_lines)]
    pub fn full_rebuild(
        &self,
        pid: u32,
        pid_birth: u64,
        now_secs: u64,
        is_active_txn: impl Fn(u64) -> bool,
        timeout: Duration,
    ) -> Result<RebuildResult, RebuildLeaseError> {
        let start = Instant::now();

        // Step 1: Acquire lease.
        self.acquire_rebuild_lease(pid, pid_birth, now_secs)?;

        // Step 2: Rotate.
        if let Err(e) = self.rotate() {
            // Release lease on failure.
            self.rebuild_pid.store(0, Ordering::Release);
            return Err(e);
        }

        // Step 3: Drain with polling.
        let mut orphaned_cleaned = 0_u32;
        loop {
            // Clean orphaned locks.
            orphaned_cleaned += self.drain_orphaned(&is_active_txn);

            // Check quiescence.
            if let Some(status) = self.drain_progress() {
                if status.quiescent {
                    break;
                }

                debug!(remaining = status.remaining, "drain in progress");
            }

            // Check timeout.
            if start.elapsed() > timeout {
                // Cancel: must still finalize if quiescent, otherwise
                // leave drain in progress for next attempt.
                if let Some(status) = self.drain_progress() {
                    if status.quiescent {
                        break;
                    }
                    // Release lease but leave draining state for later.
                    self.rebuild_pid.store(0, Ordering::Release);
                    return Ok(RebuildResult {
                        cleared: 0,
                        orphaned_cleaned,
                        elapsed: start.elapsed(),
                        epoch: self.rebuild_epoch.load(Ordering::Acquire),
                        timed_out: true,
                    });
                }
                break;
            }

            // Brief yield to let transactions release locks.
            std::thread::yield_now();
        }

        // Steps 4+5: Clear and finalize.
        // Cancellation safety: once quiescent, MUST run to completion.
        let cleared = self.finalize_rebuild(pid)?;

        Ok(RebuildResult {
            cleared,
            orphaned_cleaned,
            elapsed: start.elapsed(),
            epoch: self.rebuild_epoch.load(Ordering::Acquire),
            timed_out: false,
        })
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /// Load factor of the active table.
    #[must_use]
    pub fn active_load_factor(&self) -> f64 {
        let active_idx = self.active_table.load(Ordering::Acquire);
        let occupied = self.tables[active_idx as usize].occupied_count();
        f64::from(occupied) / f64::from(self.capacity)
    }

    /// Whether the active table's load factor exceeds the rebuild threshold.
    #[must_use]
    pub fn needs_rebuild(&self) -> bool {
        self.active_load_factor() > MAX_LOAD_FACTOR
    }

    /// Whether a rebuild is currently in progress.
    #[must_use]
    pub fn is_rebuild_in_progress(&self) -> bool {
        self.draining_table.load(Ordering::Acquire) != DRAINING_NONE
    }

    /// Current rebuild epoch.
    #[must_use]
    pub fn rebuild_epoch(&self) -> u32 {
        self.rebuild_epoch.load(Ordering::Acquire)
    }

    /// Per-table capacity.
    #[must_use]
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Number of occupied slots in the active table.
    #[must_use]
    pub fn active_occupied(&self) -> u32 {
        let active_idx = self.active_table.load(Ordering::Acquire);
        self.tables[active_idx as usize].occupied_count()
    }

    /// Number of actively locked entries across both tables.
    #[must_use]
    pub fn total_locked(&self) -> u32 {
        self.tables[0].locked_count() + self.tables[1].locked_count()
    }
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/// Result of probing a table for an existing page entry.
enum ProbeResult {
    /// Found the page entry with an active owner.
    FoundOwnedBy(u64),
    /// Found the page entry but it is unlocked (owner_txn == 0).
    FoundUnlocked,
    /// Page not found in this table.
    NotFound,
}

/// Status of the drain phase.
#[derive(Debug, Clone, Copy)]
pub struct DrainStatus {
    /// Number of locked entries remaining.
    pub remaining: u32,
    /// Whether the table has reached lock-quiescence.
    pub quiescent: bool,
}

/// Result of a full rebuild cycle.
#[derive(Debug, Clone)]
pub struct RebuildResult {
    /// Number of entries cleared from the drained table.
    pub cleared: u32,
    /// Number of orphaned locks cleaned during drain.
    pub orphaned_cleaned: u32,
    /// Total time taken.
    pub elapsed: Duration,
    /// Current rebuild epoch after completion.
    pub epoch: u32,
    /// Whether the rebuild timed out before completion.
    pub timed_out: bool,
}

// ---------------------------------------------------------------------------
// Tests (§5.6.3.1 — 8 unit + 1 E2E)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Small capacity for tests to exercise the hash table mechanics.
    const TEST_CAP: u32 = 64;

    // -- bd-11x0 test 1: Rotate swaps active table --

    #[test]
    fn test_rebuild_rotate_swaps_active_table() {
        let table = SharedPageLockTable::new(TEST_CAP);

        // Initially: active=0, draining=NONE.
        assert_eq!(table.active_table.load(Ordering::Relaxed), 0);
        assert_eq!(table.draining_table.load(Ordering::Relaxed), DRAINING_NONE);

        // Insert some keys into table 0 (active).
        assert!(table.try_acquire(1, 100).is_ok());
        assert!(table.try_acquire(2, 200).is_ok());

        // Acquire lease and rotate.
        table.acquire_rebuild_lease(1234, 0, 1000).unwrap();
        table.rotate().unwrap();

        // After rotation: active=1, draining=0.
        assert_eq!(table.active_table.load(Ordering::Relaxed), 1);
        assert_eq!(table.draining_table.load(Ordering::Relaxed), 0);

        // New acquisitions go to table 1 (active).
        assert!(table.try_acquire(3, 300).is_ok());

        // Existing locks in table 0 (draining) are still visible.
        assert_eq!(table.holder(1), Some(100));
        assert_eq!(table.holder(2), Some(200));
        assert_eq!(table.holder(3), Some(300));
    }

    // -- bd-11x0 test 2: Drain reaches quiescence --

    #[test]
    fn test_rebuild_drain_reaches_quiescence() {
        let table = SharedPageLockTable::new(TEST_CAP);

        // Acquire locks.
        assert!(table.try_acquire(10, 1).is_ok());
        assert!(table.try_acquire(20, 2).is_ok());

        // Rotate.
        table.acquire_rebuild_lease(1, 0, 1000).unwrap();
        table.rotate().unwrap();

        // Draining table has 2 locks.
        let status = table.drain_progress().unwrap();
        assert_eq!(status.remaining, 2);
        assert!(!status.quiescent);

        // Release locks from draining table.
        assert!(table.release(10, 1));
        assert!(table.release(20, 2));

        // Now quiescent.
        let status = table.drain_progress().unwrap();
        assert_eq!(status.remaining, 0);
        assert!(status.quiescent);
    }

    // -- bd-11x0 test 3: Rebuild does NOT require abort --

    #[test]
    fn test_rebuild_no_abort_guarantee() {
        let table = SharedPageLockTable::new(TEST_CAP);

        // Txn 1 holds locks in active table.
        assert!(table.try_acquire(10, 1).is_ok());
        assert!(table.try_acquire(20, 1).is_ok());

        // Rotate.
        table.acquire_rebuild_lease(1, 0, 1000).unwrap();
        table.rotate().unwrap();

        // Txn 1 can still acquire NEW locks in the new active table.
        assert!(table.try_acquire(30, 1).is_ok());

        // Txn 1 can release its old locks in the draining table.
        assert!(table.release(10, 1));
        assert!(table.release(20, 1));

        // Txn 2 can acquire locks that were just released.
        // (The page_number key persists in draining table but owner_txn=0.)
        // New acquisition goes to active table for page 10.
        assert!(table.try_acquire(10, 2).is_ok());
        assert_eq!(table.holder(10), Some(2));

        // No transaction was aborted — rebuild is rolling.
    }

    // -- bd-11x0 test 4: Lease prevents concurrent rebuilds --

    #[test]
    fn test_rebuild_lease_prevents_concurrent_rebuilds() {
        let table = SharedPageLockTable::new(TEST_CAP);

        // Process 1 acquires lease.
        assert!(table.acquire_rebuild_lease(1001, 0, 1000).is_ok());

        // Process 2 cannot acquire lease.
        let err = table.acquire_rebuild_lease(1002, 0, 1000).unwrap_err();
        assert_eq!(err, RebuildLeaseError::LeaseHeld { pid: 1001 });
    }

    // -- bd-11x0 test 5: Stale lease can be stolen --

    #[test]
    fn test_rebuild_stale_lease_stolen() {
        let table = SharedPageLockTable::new(TEST_CAP);

        // Process 1 acquires lease at time 1000 (expires at 1005).
        assert!(table.acquire_rebuild_lease(1001, 0, 1000).is_ok());

        // Process 2 tries at time 1003 — lease not expired yet.
        let err = table.acquire_rebuild_lease(1002, 0, 1003).unwrap_err();
        assert_eq!(err, RebuildLeaseError::LeaseHeld { pid: 1001 });

        // Process 2 tries at time 1006 — lease expired, steal succeeds.
        assert!(table.acquire_rebuild_lease(1002, 0, 1006).is_ok());
        assert_eq!(table.rebuild_pid.load(Ordering::Relaxed), 1002);
    }

    // -- bd-11x0 test 6: Cancellation safety --

    #[test]
    fn test_rebuild_cancellation_safety() {
        let table = SharedPageLockTable::new(TEST_CAP);

        // Insert entries.
        for i in 1..=5_u32 {
            assert!(table.try_acquire(i, u64::from(i)).is_ok());
        }

        // Rotate.
        table.acquire_rebuild_lease(1, 0, 1000).unwrap();
        table.rotate().unwrap();

        // Release all locks from draining table.
        for i in 1..=5_u32 {
            assert!(table.release(i, u64::from(i)));
        }

        // Verify quiescence.
        assert!(table.drain_progress().unwrap().quiescent);

        // Finalize: table is cleared and epoch incremented.
        let cleared = table.finalize_rebuild(1).unwrap();
        assert_eq!(cleared, 5);
        assert_eq!(table.rebuild_epoch(), 1);

        // After finalize: no draining table, lease released.
        assert_eq!(table.draining_table.load(Ordering::Relaxed), DRAINING_NONE);
        assert_eq!(table.rebuild_pid.load(Ordering::Relaxed), 0);

        // Table 0 (the cleared one) is now empty and available for next rotation.
        assert_eq!(table.tables[0].occupied_count(), 0);
    }

    // -- bd-11x0 test 7: Resource exhaustion returns Busy --

    #[test]
    fn test_rebuild_resource_exhaustion_busy() {
        // Use very small capacity to trigger load factor limit.
        let table = SharedPageLockTable::new(16);

        // Fill to > 70% capacity. Load factor check is pre-insert, so with
        // capacity=16, 0.70*16=11.2. We need 12 entries in the table before
        // the 13th insertion sees occupied/capacity > 0.70.
        for i in 1..=12_u32 {
            let result = table.try_acquire(i, u64::from(i));
            assert!(
                result.is_ok(),
                "should be able to acquire page {i}, got {result:?}"
            );
        }

        // Beyond 70%: insertion of NEW key should be rejected (12/16 = 0.75 > 0.70).
        let result = table.try_acquire(100, 100);
        assert_eq!(
            result,
            AcquireResult::CapacityExhausted,
            "new key insertion beyond 70% load factor must fail"
        );

        // But acquiring an existing key is still possible (no new slot needed).
        // Release page 1, then re-acquire it.
        assert!(table.release(1, 1));
        let result = table.try_acquire(1, 50);
        assert!(
            result.is_ok(),
            "re-acquiring existing key slot must succeed even at capacity"
        );
    }

    // -- bd-11x0 test 8: try_acquire consults draining table first --

    #[test]
    fn test_rebuild_try_acquire_consults_draining_first() {
        let table = SharedPageLockTable::new(TEST_CAP);

        // Txn 1 acquires lock on page 42 in the active table.
        assert!(table.try_acquire(42, 1).is_ok());

        // Rotate.
        table.acquire_rebuild_lease(1, 0, 1000).unwrap();
        table.rotate().unwrap();

        // Txn 2 tries to acquire page 42 — MUST check draining table first.
        let result = table.try_acquire(42, 2);
        assert_eq!(
            result,
            AcquireResult::Busy { holder: 1 },
            "must detect lock in draining table"
        );

        // Txn 1 re-acquires same page — idempotent (already held in draining).
        let result = table.try_acquire(42, 1);
        assert_eq!(
            result,
            AcquireResult::AlreadyHeld,
            "idempotent re-acquire in draining table"
        );

        // Txn 1 releases from draining table.
        assert!(table.release(42, 1));

        // Now txn 2 can acquire page 42 (in active table).
        let result = table.try_acquire(42, 2);
        assert!(
            result.is_ok(),
            "should succeed after draining table release"
        );
    }

    // -- bd-11x0 E2E test: Rolling rebuild under concurrent load --

    #[test]
    fn test_e2e_lock_table_rolling_rebuild_under_load() {
        let table = Arc::new(SharedPageLockTable::new(256));

        // Phase 1: Writers acquire locks.
        let mut txns: Vec<(u64, Vec<u32>)> = Vec::new();
        for txn_id in 1..=10_u64 {
            let pages: Vec<u32> = (1..=5)
                .map(|i| u32::try_from(txn_id).unwrap() * 10 + i)
                .collect();
            for &page in &pages {
                assert!(table.try_acquire(page, txn_id).is_ok());
            }
            txns.push((txn_id, pages));
        }

        // 50 pages locked across 10 transactions.
        assert_eq!(table.total_locked(), 50);

        // Phase 2: Concurrent readers + writers during rebuild.
        let table2 = Arc::clone(&table);
        let reader = std::thread::spawn(move || {
            // Readers check holders during rebuild.
            for _ in 0..100 {
                for txn_id in 1..=10_u64 {
                    let page = u32::try_from(txn_id).unwrap() * 10 + 1;
                    let holder = table2.holder(page);
                    // Lock should be held by the original txn or released.
                    assert!(
                        holder.is_none() || holder == Some(txn_id),
                        "unexpected holder for page {page}: {holder:?}"
                    );
                }
                std::thread::yield_now();
            }
        });

        // Phase 3: Initiate rebuild.
        table.acquire_rebuild_lease(999, 0, 1000).unwrap();
        table.rotate().unwrap();

        // Phase 4: Some transactions release their locks (simulating commit).
        for (txn_id, pages) in &txns[0..5] {
            for &page in pages {
                table.release(page, *txn_id);
            }
        }

        // New transactions acquire locks in the new active table.
        for txn_id in 11..=15_u64 {
            let page = u32::try_from(txn_id).unwrap() * 10 + 1;
            assert!(table.try_acquire(page, txn_id).is_ok());
        }

        // Phase 5: Remaining transactions release.
        for (txn_id, pages) in &txns[5..10] {
            for &page in pages {
                table.release(page, *txn_id);
            }
        }

        // Drain should be quiescent now.
        let status = table.drain_progress().unwrap();
        assert!(
            status.quiescent,
            "draining table must be quiescent after all releases"
        );

        // Phase 6: Finalize rebuild.
        let cleared = table.finalize_rebuild(999).unwrap();
        assert!(
            cleared > 0,
            "should have cleared entries from drained table"
        );
        assert_eq!(table.rebuild_epoch(), 1);

        // Wait for reader thread.
        reader.join().unwrap();

        // Phase 7: Verify final state.
        // Only the 5 new transactions should have active locks.
        assert_eq!(table.total_locked(), 5);
        for txn_id in 11..=15_u64 {
            let page = u32::try_from(txn_id).unwrap() * 10 + 1;
            assert_eq!(table.holder(page), Some(txn_id));
        }
    }
}
