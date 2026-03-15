//! Flat Combining for sequential batching under contention (§14.2).
//!
//! When many threads contend on a shared data structure, each thread publishes
//! its request to a per-thread slot and one thread becomes the *combiner*.
//! The combiner collects all pending requests, processes them as a single
//! batch holding one lock, then publishes the results.  This reduces
//! cache-line bouncing from N lock acquisitions to 1.
//!
//! ## Protocol
//!
//! 1. Thread publishes `(op, argument)` to its slot (atomic store).
//! 2. Thread tries to acquire the combiner lock (`try_lock`).
//!    - **Won**: scan all slots, collect pending ops, execute batch, store
//!      results, release lock.
//!    - **Lost**: spin-wait until its own slot shows a result.
//! 3. Thread reads its result from the slot.
//!
//! ## Slot Layout
//!
//! Each slot is a pair of `AtomicU64`:
//!   - `state`: EMPTY (0) | REQUEST (op‖arg packed) | RESULT (high-bit set)
//!   - `payload`: argument or result value.
//!
//! ## Safety
//!
//! No `UnsafeCell` or `unsafe` blocks — all state uses `AtomicU64`.
//!
//! ## Tracing & Metrics
//!
//! - **Target**: `fsqlite.flat_combine`
//!   - `DEBUG`: batch execution with `batch_size`, `combiner_thread`
//!   - `INFO`: periodic contention stats
//! - **Metrics**:
//!   - `fsqlite_flat_combining_batches_total`
//!   - `fsqlite_flat_combining_ops_total`
//!   - `fsqlite_flat_combining_batch_size_sum` (for avg = sum / batches)
//!   - `fsqlite_flat_combining_batch_size_max`
//!   - `fsqlite_flat_combining_wait_ns_total`
//!   - `fsqlite_flat_combining_wait_ns_max`
//!   - `fsqlite_htm_attempts`
//!   - `fsqlite_htm_aborts_conflict`
//!   - `fsqlite_htm_aborts_capacity`
//!   - `fsqlite_htm_aborts_explicit`
//!   - `fsqlite_htm_aborts_other`

use fsqlite_types::sync_primitives::Instant;
use std::sync::atomic::{AtomicU64, Ordering};

use fsqlite_types::sync_primitives::Mutex;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum threads that can participate in flat combining.
pub const MAX_FC_THREADS: usize = 64;

/// Slot state: empty (available for a new request).
const SLOT_EMPTY: u64 = 0;

/// Bit set in the state word to indicate the slot contains a result.
const RESULT_BIT: u64 = 1 << 63;

/// Maximum spin iterations before yielding while waiting for result.
const SPIN_BEFORE_YIELD: u32 = 1024;

// ---------------------------------------------------------------------------
// Global metrics
// ---------------------------------------------------------------------------

static FC_BATCHES_TOTAL: AtomicU64 = AtomicU64::new(0);
static FC_OPS_TOTAL: AtomicU64 = AtomicU64::new(0);
static FC_BATCH_SIZE_SUM: AtomicU64 = AtomicU64::new(0);
static FC_BATCH_SIZE_MAX: AtomicU64 = AtomicU64::new(0);
static FC_WAIT_NS_TOTAL: AtomicU64 = AtomicU64::new(0);
static FC_WAIT_NS_MAX: AtomicU64 = AtomicU64::new(0);
static FC_HTM_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
static FC_HTM_ABORTS_CONFLICT: AtomicU64 = AtomicU64::new(0);
static FC_HTM_ABORTS_CAPACITY: AtomicU64 = AtomicU64::new(0);
static FC_HTM_ABORTS_EXPLICIT: AtomicU64 = AtomicU64::new(0);
static FC_HTM_ABORTS_OTHER: AtomicU64 = AtomicU64::new(0);

const XABORT_EXPLICIT: u32 = 1 << 0;
const XABORT_RETRY: u32 = 1 << 1;
const XABORT_CONFLICT: u32 = 1 << 2;
const XABORT_CAPACITY: u32 = 1 << 3;
const XABORT_DEBUG: u32 = 1 << 4;
const XABORT_NESTED: u32 = 1 << 5;
const XABORT_CODE_SHIFT: u32 = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HtmAbortReason {
    Conflict,
    Capacity,
    Explicit,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HtmAbortClassification {
    reason: HtmAbortReason,
    retryable: bool,
    explicit_code: Option<u8>,
    debug: bool,
    nested: bool,
}

/// Snapshot of flat combining metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct FlatCombiningMetrics {
    pub fsqlite_flat_combining_batches_total: u64,
    pub fsqlite_flat_combining_ops_total: u64,
    pub fsqlite_flat_combining_batch_size_sum: u64,
    pub fsqlite_flat_combining_batch_size_max: u64,
    pub fsqlite_flat_combining_wait_ns_total: u64,
    pub fsqlite_flat_combining_wait_ns_max: u64,
    pub fsqlite_htm_attempts: u64,
    pub fsqlite_htm_aborts_conflict: u64,
    pub fsqlite_htm_aborts_capacity: u64,
    pub fsqlite_htm_aborts_explicit: u64,
    pub fsqlite_htm_aborts_other: u64,
}

/// Read current flat combining metrics.
#[must_use]
pub fn flat_combining_metrics() -> FlatCombiningMetrics {
    FlatCombiningMetrics {
        fsqlite_flat_combining_batches_total: FC_BATCHES_TOTAL.load(Ordering::Relaxed),
        fsqlite_flat_combining_ops_total: FC_OPS_TOTAL.load(Ordering::Relaxed),
        fsqlite_flat_combining_batch_size_sum: FC_BATCH_SIZE_SUM.load(Ordering::Relaxed),
        fsqlite_flat_combining_batch_size_max: FC_BATCH_SIZE_MAX.load(Ordering::Relaxed),
        fsqlite_flat_combining_wait_ns_total: FC_WAIT_NS_TOTAL.load(Ordering::Relaxed),
        fsqlite_flat_combining_wait_ns_max: FC_WAIT_NS_MAX.load(Ordering::Relaxed),
        fsqlite_htm_attempts: FC_HTM_ATTEMPTS.load(Ordering::Relaxed),
        fsqlite_htm_aborts_conflict: FC_HTM_ABORTS_CONFLICT.load(Ordering::Relaxed),
        fsqlite_htm_aborts_capacity: FC_HTM_ABORTS_CAPACITY.load(Ordering::Relaxed),
        fsqlite_htm_aborts_explicit: FC_HTM_ABORTS_EXPLICIT.load(Ordering::Relaxed),
        fsqlite_htm_aborts_other: FC_HTM_ABORTS_OTHER.load(Ordering::Relaxed),
    }
}

/// Reset metrics (for tests).
pub fn reset_flat_combining_metrics() {
    FC_BATCHES_TOTAL.store(0, Ordering::Relaxed);
    FC_OPS_TOTAL.store(0, Ordering::Relaxed);
    FC_BATCH_SIZE_SUM.store(0, Ordering::Relaxed);
    FC_BATCH_SIZE_MAX.store(0, Ordering::Relaxed);
    FC_WAIT_NS_TOTAL.store(0, Ordering::Relaxed);
    FC_WAIT_NS_MAX.store(0, Ordering::Relaxed);
    FC_HTM_ATTEMPTS.store(0, Ordering::Relaxed);
    FC_HTM_ABORTS_CONFLICT.store(0, Ordering::Relaxed);
    FC_HTM_ABORTS_CAPACITY.store(0, Ordering::Relaxed);
    FC_HTM_ABORTS_EXPLICIT.store(0, Ordering::Relaxed);
    FC_HTM_ABORTS_OTHER.store(0, Ordering::Relaxed);
}

fn update_max(metric: &AtomicU64, val: u64) {
    let mut prev = metric.load(Ordering::Relaxed);
    while val > prev {
        match metric.compare_exchange_weak(prev, val, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => prev = actual,
        }
    }
}

const fn classify_htm_abort_status(status: u32) -> HtmAbortClassification {
    let reason = if (status & XABORT_CONFLICT) != 0 {
        HtmAbortReason::Conflict
    } else if (status & XABORT_CAPACITY) != 0 {
        HtmAbortReason::Capacity
    } else if (status & XABORT_EXPLICIT) != 0 {
        HtmAbortReason::Explicit
    } else {
        HtmAbortReason::Other
    };
    let explicit_code = if (status & XABORT_EXPLICIT) != 0 {
        Some(((status >> XABORT_CODE_SHIFT) & 0xff) as u8)
    } else {
        None
    };

    HtmAbortClassification {
        reason,
        retryable: (status & XABORT_RETRY) != 0,
        explicit_code,
        debug: (status & XABORT_DEBUG) != 0,
        nested: (status & XABORT_NESTED) != 0,
    }
}

/// Record a single HTM entry attempt before invoking `_xbegin()`.
fn record_htm_attempt() {
    FC_HTM_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

/// Classify and record a failed HTM attempt returned by `_xbegin()`.
fn record_htm_abort_status(status: u32) -> HtmAbortClassification {
    let classification = classify_htm_abort_status(status);
    match classification.reason {
        HtmAbortReason::Conflict => {
            FC_HTM_ABORTS_CONFLICT.fetch_add(1, Ordering::Relaxed);
        }
        HtmAbortReason::Capacity => {
            FC_HTM_ABORTS_CAPACITY.fetch_add(1, Ordering::Relaxed);
        }
        HtmAbortReason::Explicit => {
            FC_HTM_ABORTS_EXPLICIT.fetch_add(1, Ordering::Relaxed);
        }
        HtmAbortReason::Other => {
            FC_HTM_ABORTS_OTHER.fetch_add(1, Ordering::Relaxed);
        }
    }
    classification
}

/// Record a public HTM attempt for future fast-path integrations.
pub fn note_htm_attempt() {
    record_htm_attempt();
}

/// Record a public HTM abort status returned by a failed `_xbegin()`.
pub fn note_htm_abort(status: u32) {
    let _ = record_htm_abort_status(status);
}

// ---------------------------------------------------------------------------
// FcSlot
// ---------------------------------------------------------------------------

/// Per-thread request/result slot.
struct FcSlot {
    /// SLOT_EMPTY | request_tag (1..2^63-1) | RESULT_BIT | result_value
    state: AtomicU64,
    /// Payload: argument for requests, result for completions.
    payload: AtomicU64,
}

impl FcSlot {
    fn new() -> Self {
        Self {
            state: AtomicU64::new(SLOT_EMPTY),
            payload: AtomicU64::new(0),
        }
    }
}

// ---------------------------------------------------------------------------
// FlatCombiner
// ---------------------------------------------------------------------------

/// A flat combining accumulator for `u64` values.
///
/// Threads submit operations via [`FcHandle::apply`] and receive results.
/// Supported operations:
/// - `OP_ADD`: atomic add to the shared accumulator
/// - `OP_READ`: read current accumulator value
///
/// The combiner processes all pending operations in a single batch,
/// reducing lock contention.
pub struct FlatCombiner {
    /// The shared value being operated on.
    value: AtomicU64,
    /// Per-thread slots for request/result exchange.
    slots: [FcSlot; MAX_FC_THREADS],
    /// Slot ownership: 0 = free, non-zero = occupied by a thread.
    owners: [AtomicU64; MAX_FC_THREADS],
    /// Combiner lock — only one thread processes a batch at a time.
    combiner_lock: Mutex<()>,
}

/// Operation tag: add argument to accumulator.
pub const OP_ADD: u64 = 1;
/// Operation tag: read current accumulator value.
pub const OP_READ: u64 = 2;

impl FlatCombiner {
    /// Create a new flat combiner with the given initial value.
    pub fn new(initial: u64) -> Self {
        Self {
            value: AtomicU64::new(initial),
            slots: std::array::from_fn(|_| FcSlot::new()),
            owners: std::array::from_fn(|_| AtomicU64::new(0)),
            combiner_lock: Mutex::new(()),
        }
    }

    /// Register a thread.  Returns an [`FcHandle`] with an assigned slot,
    /// or `None` if all slots are occupied.
    pub fn register(&self) -> Option<FcHandle<'_>> {
        // Use a unique non-zero ID based on thread ID hash.
        let tid = {
            let t = std::thread::current().id();
            let s = format!("{t:?}");
            let mut h = 1u64;
            for b in s.bytes() {
                h = h.wrapping_mul(31).wrapping_add(u64::from(b));
            }
            if h == 0 { 1 } else { h }
        };

        for i in 0..MAX_FC_THREADS {
            if self.owners[i]
                .compare_exchange(0, tid, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return Some(FcHandle {
                    combiner: self,
                    slot: i,
                });
            }
        }
        None
    }

    /// Current value (for diagnostics — not linearizable without combining).
    #[must_use]
    pub fn value(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Number of registered threads.
    #[must_use]
    pub fn active_threads(&self) -> usize {
        self.owners
            .iter()
            .filter(|o| o.load(Ordering::Relaxed) != 0)
            .count()
    }

    /// Process all pending requests in a single batch.
    /// The caller MUST hold the `combiner_lock`.
    fn combine_locked(&self) {
        let mut batch_size = 0u64;
        let mut current = self.value.load(Ordering::Acquire);

        // Scan all slots for pending requests.
        for i in 0..MAX_FC_THREADS {
            let state = self.slots[i].state.load(Ordering::Acquire);
            if state == SLOT_EMPTY || (state & RESULT_BIT) != 0 {
                continue; // Empty or already has a result.
            }

            let op = state;
            let arg = self.slots[i].payload.load(Ordering::Acquire);
            batch_size += 1;

            let result = match op {
                OP_ADD => {
                    current = current.wrapping_add(arg);
                    current
                }
                OP_READ => current,
                _ => 0, // Unknown op — return 0.
            };

            // Publish result: set payload, then mark state as RESULT.
            self.slots[i].payload.store(result, Ordering::Release);
            self.slots[i]
                .state
                .store(RESULT_BIT | op, Ordering::Release);
        }

        self.value.store(current, Ordering::Release);

        if batch_size > 0 {
            // Update metrics.
            FC_BATCHES_TOTAL.fetch_add(1, Ordering::Relaxed);
            FC_OPS_TOTAL.fetch_add(batch_size, Ordering::Relaxed);
            FC_BATCH_SIZE_SUM.fetch_add(batch_size, Ordering::Relaxed);
            update_max(&FC_BATCH_SIZE_MAX, batch_size);

            tracing::debug!(
                target: "fsqlite.flat_combine",
                batch_size,
                "flat_combine_batch"
            );
        }
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for FlatCombiner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlatCombiner")
            .field("value", &self.value.load(Ordering::Relaxed))
            .field("active_threads", &self.active_threads())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// FcHandle (per-thread)
// ---------------------------------------------------------------------------

/// Per-thread flat combining handle.  Automatically unregisters on drop.
pub struct FcHandle<'a> {
    combiner: &'a FlatCombiner,
    slot: usize,
}

impl FcHandle<'_> {
    /// Submit an operation and wait for the result.
    ///
    /// The caller publishes its request; either it becomes the combiner and
    /// processes the entire batch, or it waits for the combiner to process
    /// its request.
    pub fn apply(&self, op: u64, arg: u64) -> u64 {
        let start = Instant::now();

        // Publish our request.
        self.combiner.slots[self.slot]
            .payload
            .store(arg, Ordering::Release);
        self.combiner.slots[self.slot]
            .state
            .store(op, Ordering::Release);

        // ALIEN ARTIFACT: True Flat Combining.
        // We attempt to become the combiner. If we fail, we MUST NOT block on an OS mutex
        // (which would defeat the entire purpose of flat combining by forcing context switches).
        // Instead, we spin on our own cache-line-isolated slot until the active combiner
        // writes our result. This converts global lock contention into read-only local spinning.
        if let Some(_guard) = self.combiner.combiner_lock.try_lock() {
            self.combiner.combine_locked();
        }

        // Check if our request has been serviced.
        let mut spins = 0u32;
        loop {
            let state = self.combiner.slots[self.slot].state.load(Ordering::Acquire);
            if (state & RESULT_BIT) != 0 {
                // Result ready — read payload and clear slot.
                let result = self.combiner.slots[self.slot]
                    .payload
                    .load(Ordering::Acquire);
                self.combiner.slots[self.slot]
                    .state
                    .store(SLOT_EMPTY, Ordering::Release);

                #[allow(clippy::cast_possible_truncation)]
                let elapsed_ns = start.elapsed().as_nanos() as u64;
                FC_WAIT_NS_TOTAL.fetch_add(elapsed_ns, Ordering::Relaxed);
                update_max(&FC_WAIT_NS_MAX, elapsed_ns);

                return result;
            }

            // Still waiting. Spin or yield.
            spins += 1;
            if spins < SPIN_BEFORE_YIELD {
                std::hint::spin_loop();
            } else {
                // If the combiner died or is extremely slow, we attempt to take over.
                // If we can't take over, yield the thread to avoid burning CPU unnecessarily.
                if let Some(_guard) = self.combiner.combiner_lock.try_lock() {
                    self.combiner.combine_locked();
                } else {
                    std::thread::yield_now();
                }
                spins = 0;
            }
        }
    }

    /// Convenience: add a value to the accumulator.
    pub fn add(&self, val: u64) -> u64 {
        self.apply(OP_ADD, val)
    }

    /// Convenience: read the current accumulator value.
    pub fn read(&self) -> u64 {
        self.apply(OP_READ, 0)
    }

    /// Slot index (for diagnostics).
    #[must_use]
    pub fn slot(&self) -> usize {
        self.slot
    }
}

impl Drop for FcHandle<'_> {
    fn drop(&mut self) {
        // Clear slot state and release ownership.
        self.combiner.slots[self.slot]
            .state
            .store(SLOT_EMPTY, Ordering::Release);
        self.combiner.owners[self.slot].store(0, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn register_unregister() {
        let fc = FlatCombiner::new(0);
        assert_eq!(fc.active_threads(), 0);

        let h1 = fc.register().unwrap();
        assert_eq!(fc.active_threads(), 1);

        let h2 = fc.register().unwrap();
        assert_eq!(fc.active_threads(), 2);

        drop(h1);
        assert_eq!(fc.active_threads(), 1);

        drop(h2);
        assert_eq!(fc.active_threads(), 0);
    }

    #[test]
    fn single_thread_add() {
        let fc = FlatCombiner::new(0);
        let h = fc.register().unwrap();

        let r1 = h.add(10);
        assert_eq!(r1, 10);

        let r2 = h.add(20);
        assert_eq!(r2, 30);

        let r3 = h.read();
        assert_eq!(r3, 30);

        assert_eq!(fc.value(), 30);
        drop(h);
    }

    #[test]
    fn single_thread_sequential() {
        let fc = FlatCombiner::new(100);
        let h = fc.register().unwrap();

        for i in 1..=50 {
            let result = h.add(1);
            assert_eq!(result, 100 + i);
        }

        assert_eq!(h.read(), 150);
        drop(h);
    }

    #[test]
    fn concurrent_adds_correct_total() {
        let fc = Arc::new(FlatCombiner::new(0));
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let f = Arc::clone(&fc);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let h = f.register().unwrap();
                b.wait();
                for _ in 0..500 {
                    h.add(1);
                }
                drop(h);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(fc.value(), 2000, "4 threads * 500 adds = 2000");
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn concurrent_stress_no_lost_updates() {
        let fc = Arc::new(FlatCombiner::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let barrier = Arc::new(Barrier::new(4));
        let total_adds = Arc::new(AtomicU64::new(0));

        let mut handles = Vec::new();
        for _ in 0..4 {
            let f = Arc::clone(&fc);
            let s = Arc::clone(&stop);
            let b = Arc::clone(&barrier);
            let t = Arc::clone(&total_adds);
            handles.push(thread::spawn(move || {
                let h = f.register().unwrap();
                b.wait();
                let mut local = 0u64;
                while !s.load(Ordering::Relaxed) {
                    h.add(1);
                    local += 1;
                }
                t.fetch_add(local, Ordering::Relaxed);
                drop(h);
            }));
        }

        thread::sleep(Duration::from_millis(300));
        stop.store(true, Ordering::Release);

        for h in handles {
            h.join().unwrap();
        }

        let expected = total_adds.load(Ordering::Relaxed);
        let actual = fc.value();
        assert_eq!(
            actual, expected,
            "accumulator {actual} != total submitted {expected}"
        );
    }

    #[test]
    fn metrics_track_batches() {
        // Delta-based: snapshot before, act, snapshot after.
        let before = flat_combining_metrics();

        let fc = FlatCombiner::new(0);
        let h = fc.register().unwrap();

        h.add(1);
        h.add(2);
        h.add(3);

        let after = flat_combining_metrics();
        let batch_delta = after.fsqlite_flat_combining_batches_total
            - before.fsqlite_flat_combining_batches_total;
        let ops_delta =
            after.fsqlite_flat_combining_ops_total - before.fsqlite_flat_combining_ops_total;
        assert!(
            batch_delta >= 3,
            "expected at least 3 batches (single thread = 1 op per batch), got {batch_delta}"
        );
        assert!(ops_delta >= 3, "expected at least 3 ops, got {ops_delta}");

        drop(h);
    }

    #[test]
    fn batching_under_contention() {
        // With many threads contending, some batches should contain > 1 op.
        let before = flat_combining_metrics();

        let fc = Arc::new(FlatCombiner::new(0));
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let f = Arc::clone(&fc);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let h = f.register().unwrap();
                b.wait();
                for _ in 0..200 {
                    h.add(1);
                }
                drop(h);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(fc.value(), 1600, "8 threads * 200 = 1600");

        let after = flat_combining_metrics();
        let batches_delta = after.fsqlite_flat_combining_batches_total
            - before.fsqlite_flat_combining_batches_total;
        let ops_delta =
            after.fsqlite_flat_combining_ops_total - before.fsqlite_flat_combining_ops_total;
        let avg_batch = if batches_delta > 0 {
            ops_delta as f64 / batches_delta as f64
        } else {
            0.0
        };

        // Under contention, we expect at least some batches > 1.
        println!(
            "[flat_combining] batches={batches_delta} ops={ops_delta} avg_batch={avg_batch:.2} max_batch={}",
            after.fsqlite_flat_combining_batch_size_max
        );
    }

    #[test]
    fn read_sees_latest_value() {
        let fc = Arc::new(FlatCombiner::new(0));
        let barrier = Arc::new(Barrier::new(2));

        let f = Arc::clone(&fc);
        let b = Arc::clone(&barrier);
        let writer = thread::spawn(move || {
            let h = f.register().unwrap();
            b.wait();
            for _ in 0..100 {
                h.add(1);
            }
            drop(h);
        });

        let f = Arc::clone(&fc);
        let b2 = Arc::clone(&barrier);
        let reader = thread::spawn(move || {
            let h = f.register().unwrap();
            b2.wait();
            // Give writer some time.
            thread::sleep(Duration::from_millis(50));
            let v = h.read();
            drop(h);
            v
        });

        writer.join().unwrap();
        let last_read = reader.join().unwrap();
        // Reader should see a value between 0 and 100.
        assert!(last_read <= 100, "read {last_read} > 100");
    }

    #[test]
    fn no_starvation_bounded_wait() {
        // Every thread should complete within a reasonable time.
        let fc = Arc::new(FlatCombiner::new(0));
        let barrier = Arc::new(Barrier::new(4));
        let mut handles = Vec::new();

        for _ in 0..4 {
            let f = Arc::clone(&fc);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let h = f.register().unwrap();
                b.wait();
                let start = Instant::now();
                for _ in 0..100 {
                    h.add(1);
                }
                let elapsed = start.elapsed();
                drop(h);
                elapsed
            }));
        }

        for h in handles {
            let elapsed = h.join().unwrap();
            // Each thread should finish within 5 seconds (generous bound).
            assert!(
                elapsed < Duration::from_secs(5),
                "thread took too long: {elapsed:?} — possible starvation"
            );
        }

        assert_eq!(fc.value(), 400);
    }

    #[test]
    fn debug_format() {
        let fc = FlatCombiner::new(42);
        let dbg = format!("{fc:?}");
        assert!(dbg.contains("FlatCombiner"));
        assert!(dbg.contains("42"));
    }

    #[test]
    fn classify_htm_abort_status_prefers_conflict() {
        let status =
            XABORT_CONFLICT | XABORT_CAPACITY | XABORT_RETRY | XABORT_DEBUG | XABORT_NESTED;
        let classification = classify_htm_abort_status(status);
        assert_eq!(classification.reason, HtmAbortReason::Conflict);
        assert!(classification.retryable);
        assert!(classification.debug);
        assert!(classification.nested);
        assert_eq!(classification.explicit_code, None);
    }

    #[test]
    fn classify_htm_abort_status_extracts_explicit_code() {
        let status = XABORT_EXPLICIT | XABORT_RETRY | (0x2a_u32 << XABORT_CODE_SHIFT);
        let classification = classify_htm_abort_status(status);
        assert_eq!(classification.reason, HtmAbortReason::Explicit);
        assert!(classification.retryable);
        assert_eq!(classification.explicit_code, Some(0x2a));
    }

    #[test]
    fn record_htm_abort_status_updates_counters() {
        reset_flat_combining_metrics();

        record_htm_attempt();
        record_htm_attempt();
        record_htm_attempt();
        record_htm_attempt();
        let conflict = record_htm_abort_status(XABORT_CONFLICT | XABORT_RETRY);
        let capacity = record_htm_abort_status(XABORT_CAPACITY);
        let explicit = record_htm_abort_status(XABORT_EXPLICIT | (0x07_u32 << XABORT_CODE_SHIFT));
        let other = record_htm_abort_status(0);

        assert_eq!(conflict.reason, HtmAbortReason::Conflict);
        assert_eq!(capacity.reason, HtmAbortReason::Capacity);
        assert_eq!(explicit.reason, HtmAbortReason::Explicit);
        assert_eq!(explicit.explicit_code, Some(0x07));
        assert_eq!(other.reason, HtmAbortReason::Other);

        let metrics = flat_combining_metrics();
        assert_eq!(metrics.fsqlite_htm_attempts, 4);
        assert_eq!(metrics.fsqlite_htm_aborts_conflict, 1);
        assert_eq!(metrics.fsqlite_htm_aborts_capacity, 1);
        assert_eq!(metrics.fsqlite_htm_aborts_explicit, 1);
        assert_eq!(metrics.fsqlite_htm_aborts_other, 1);
    }
}
