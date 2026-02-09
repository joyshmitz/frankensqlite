//! Epoch clock, key derivation, and epoch transition barrier (§4.18, bd-3go.12).
//!
//! `ecs_epoch` is a monotone `u64` stored durably in [`RootManifest::ecs_epoch`]
//! and mirrored in `SharedMemoryLayout.ecs_epoch`. Epochs MUST NOT be reused.
//!
//! The [`EpochClock`] provides the in-process monotone counter. The
//! [`EpochBarrier`] coordinates all-or-nothing epoch transitions across
//! participants (WriteCoordinator, SymbolStore, Replicator, CheckpointGc).

use std::sync::atomic::{AtomicU64, Ordering};

use fsqlite_error::{FrankenError, Result};
use fsqlite_types::EpochId;
use tracing::{debug, error, info, warn};

// ── Domain separation constants (§4.18.2) ──────────────────────────────

/// Domain separator for deriving the symbol auth master key from a DEK.
///
/// `master_key = BLAKE3_KEYED(DEK, "fsqlite:symbol-auth-master:v1")`
const MASTER_KEY_DOMAIN: &[u8] = b"fsqlite:symbol-auth-master:v1";

/// Domain separator for deriving per-epoch auth keys.
///
/// `K_epoch = BLAKE3_KEYED(master_key, "fsqlite:symbol-auth:epoch:v1" || le_u64(ecs_epoch))`
const EPOCH_KEY_DOMAIN: &[u8] = b"fsqlite:symbol-auth:epoch:v1";

// ── EpochClock ─────────────────────────────────────────────────────────

/// In-process monotone epoch counter (§4.18).
///
/// Wraps an `AtomicU64` for lock-free reads. Increments are serialized by
/// the coordinator (only one caller should call [`increment`] at a time).
#[derive(Debug)]
pub struct EpochClock {
    current: AtomicU64,
}

impl EpochClock {
    /// Create a new clock initialised at the given epoch.
    #[must_use]
    pub fn new(initial: EpochId) -> Self {
        Self {
            current: AtomicU64::new(initial.get()),
        }
    }

    /// Read the current epoch (Acquire ordering).
    #[must_use]
    pub fn current(&self) -> EpochId {
        EpochId::new(self.current.load(Ordering::Acquire))
    }

    /// Atomically increment the epoch counter by one.
    ///
    /// Returns the *new* epoch on success, or an error if the counter
    /// would overflow (saturated at `u64::MAX`).
    ///
    /// # Errors
    ///
    /// Returns [`FrankenError::OutOfRange`] if the epoch has reached `u64::MAX`.
    pub fn increment(&self) -> Result<EpochId> {
        loop {
            let old = self.current.load(Ordering::Acquire);
            let new = old.checked_add(1).ok_or_else(|| {
                error!(
                    bead_id = "bd-3go.12",
                    old_epoch = old,
                    "epoch counter overflow — cannot increment past u64::MAX"
                );
                FrankenError::OutOfRange {
                    what: "ecs_epoch".to_owned(),
                    value: old.to_string(),
                }
            })?;
            if self
                .current
                .compare_exchange_weak(old, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                info!(
                    bead_id = "bd-3go.12",
                    old_epoch = old,
                    new_epoch = new,
                    "epoch incremented"
                );
                return Ok(EpochId::new(new));
            }
        }
    }

    /// Store a specific epoch value (Release ordering).
    ///
    /// Used during bootstrap or recovery to set the epoch from a persisted
    /// `RootManifest.ecs_epoch`.
    pub fn store(&self, epoch: EpochId) {
        self.current.store(epoch.get(), Ordering::Release);
    }
}

// ── Epoch-scoped key derivation (§4.18.2) ──────────────────────────────

/// A 32-byte epoch-scoped symbol authentication key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochAuthKey([u8; 32]);

impl EpochAuthKey {
    /// View the raw key bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Derive the symbol auth master key from a DEK with domain separation (§4.18.2).
///
/// `master_key = BLAKE3_KEYED(DEK, "fsqlite:symbol-auth-master:v1")`
///
/// The DEK must be exactly 32 bytes (XChaCha20-Poly1305 key size).
///
/// # Errors
///
/// Returns [`FrankenError::TypeMismatch`] if `dek` is not 32 bytes.
pub fn derive_master_key_from_dek(dek: &[u8; 32]) -> [u8; 32] {
    let keyed_hasher = blake3::Hasher::new_keyed(dek);
    let mut hasher = keyed_hasher;
    hasher.update(MASTER_KEY_DOMAIN);
    let hash = hasher.finalize();
    debug!(
        bead_id = "bd-3go.12",
        domain = std::str::from_utf8(MASTER_KEY_DOMAIN).unwrap_or("<invalid>"),
        "derived master key from DEK with domain separation"
    );
    *hash.as_bytes()
}

/// Derive an epoch-scoped authentication key from a master key (§4.18.2).
///
/// `K_epoch = BLAKE3_KEYED(master_key, "fsqlite:symbol-auth:epoch:v1" || le_u64(ecs_epoch))`
///
/// Deterministic: same `(master_key, epoch)` always produces the same key.
#[must_use]
pub fn derive_epoch_auth_key(master_key: &[u8; 32], epoch: EpochId) -> EpochAuthKey {
    let mut hasher = blake3::Hasher::new_keyed(master_key);
    hasher.update(EPOCH_KEY_DOMAIN);
    hasher.update(&epoch.get().to_le_bytes());
    let hash = hasher.finalize();
    debug!(
        bead_id = "bd-3go.12",
        epoch = epoch.get(),
        domain = std::str::from_utf8(EPOCH_KEY_DOMAIN).unwrap_or("<invalid>"),
        "derived epoch auth key (NOT logging key material)"
    );
    EpochAuthKey(*hash.as_bytes())
}

// ── EpochBarrier (§4.18.4) ──────────────────────────────────────────────

/// Outcome of an epoch barrier attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierOutcome {
    /// All participants arrived; epoch was incremented.
    AllArrived {
        /// The new epoch after increment.
        new_epoch: EpochId,
    },
    /// The barrier timed out before all participants arrived.
    Timeout {
        /// How many participants arrived before timeout.
        arrived: usize,
        /// Total expected participants.
        expected: usize,
    },
    /// The barrier was explicitly cancelled.
    Cancelled,
}

/// Epoch transition barrier (§4.18.4).
///
/// Coordinates quiescence across all correctness-critical services before
/// incrementing the epoch. The barrier is all-or-nothing: either all
/// participants arrive, or the epoch does not advance.
///
/// Participants: WriteCoordinator, SymbolStore, Replicator, CheckpointGc.
#[derive(Debug)]
pub struct EpochBarrier {
    /// The epoch being transitioned *from*.
    current_epoch: EpochId,
    /// Total expected participants.
    expected: usize,
    /// Number of participants that have arrived.
    arrived: AtomicU64,
    /// Whether the barrier has been cancelled.
    cancelled: std::sync::atomic::AtomicBool,
}

impl EpochBarrier {
    /// Create a new epoch barrier.
    ///
    /// `current_epoch` is the epoch being transitioned from.
    /// `participants` is the number of services that must drain and arrive.
    #[must_use]
    pub fn new(current_epoch: EpochId, participants: usize) -> Self {
        info!(
            bead_id = "bd-3go.12",
            epoch = current_epoch.get(),
            participants,
            "epoch barrier created"
        );
        Self {
            current_epoch,
            expected: participants,
            arrived: AtomicU64::new(0),
            cancelled: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// The epoch being transitioned from.
    #[must_use]
    pub fn epoch(&self) -> EpochId {
        self.current_epoch
    }

    /// Number of participants that have arrived so far.
    #[must_use]
    pub fn arrived_count(&self) -> usize {
        let val = self.arrived.load(Ordering::Acquire);
        usize::try_from(val).unwrap_or(usize::MAX)
    }

    /// Total expected participants.
    #[must_use]
    pub fn expected_count(&self) -> usize {
        self.expected
    }

    /// Register that a participant has drained in-flight work and arrived.
    ///
    /// Returns `true` if this was the last participant (barrier is complete).
    pub fn arrive(&self, participant_name: &str) -> bool {
        if self.cancelled.load(Ordering::Acquire) {
            warn!(
                bead_id = "bd-3go.12",
                participant = participant_name,
                "participant arrived at cancelled barrier — ignoring"
            );
            return false;
        }
        let prev = self.arrived.fetch_add(1, Ordering::AcqRel);
        let new_count = usize::try_from(prev.saturating_add(1)).unwrap_or(usize::MAX);
        debug!(
            bead_id = "bd-3go.12",
            participant = participant_name,
            arrived = new_count,
            expected = self.expected,
            "barrier participant arrived"
        );
        new_count >= self.expected
    }

    /// Whether all participants have arrived.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.arrived_count() >= self.expected
    }

    /// Cancel the barrier. The epoch will NOT advance.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        warn!(
            bead_id = "bd-3go.12",
            epoch = self.current_epoch.get(),
            arrived = self.arrived_count(),
            expected = self.expected,
            "epoch barrier cancelled — epoch will NOT advance"
        );
    }

    /// Whether the barrier has been cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Resolve the barrier after a timeout or cancellation check.
    ///
    /// If all participants arrived and the barrier is not cancelled,
    /// the epoch clock is incremented and the new epoch is returned.
    ///
    /// # Errors
    ///
    /// Returns [`FrankenError::OutOfRange`] if the epoch clock overflows.
    pub fn resolve(&self, clock: &EpochClock) -> Result<BarrierOutcome> {
        if self.is_cancelled() {
            return Ok(BarrierOutcome::Cancelled);
        }
        if !self.is_complete() {
            return Ok(BarrierOutcome::Timeout {
                arrived: self.arrived_count(),
                expected: self.expected,
            });
        }
        let new_epoch = clock.increment()?;
        info!(
            bead_id = "bd-3go.12",
            old_epoch = self.current_epoch.get(),
            new_epoch = new_epoch.get(),
            participants = self.expected,
            "epoch transition completed — all participants arrived"
        );
        Ok(BarrierOutcome::AllArrived { new_epoch })
    }
}

// ── Validation helpers ──────────────────────────────────────────────────

/// Validate a symbol's epoch against the current validity window (§4.18.1).
///
/// Fail-closed: rejects symbols with `epoch_id > current_epoch`.
///
/// # Errors
///
/// Returns [`FrankenError::CorruptData`] if the symbol epoch is outside
/// the validity window.
pub fn validate_symbol_epoch(
    symbol_epoch: EpochId,
    window: &fsqlite_types::SymbolValidityWindow,
) -> Result<()> {
    if window.contains(symbol_epoch) {
        Ok(())
    } else {
        error!(
            bead_id = "bd-3go.12",
            symbol_epoch = symbol_epoch.get(),
            window_from = window.from_epoch.get(),
            window_to = window.to_epoch.get(),
            "symbol epoch outside validity window — fail-closed rejection"
        );
        Err(FrankenError::DatabaseCorrupt {
            detail: format!(
                "symbol epoch {} outside validity window [{}, {}]",
                symbol_epoch.get(),
                window.from_epoch.get(),
                window.to_epoch.get(),
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use fsqlite_types::SymbolValidityWindow;

    use super::*;

    const BEAD_ID: &str = "bd-3go.12";

    // ── test_epoch_id_monotone ──────────────────────────────────────────

    #[test]
    fn test_epoch_id_monotone() {
        let clock = EpochClock::new(EpochId::ZERO);
        let mut prev = clock.current();
        for i in 0..100 {
            let next = clock.increment().unwrap_or_else(|e| {
                panic!("bead_id={BEAD_ID} case=epoch_monotone_increment_{i} err={e}")
            });
            assert!(
                next > prev,
                "bead_id={BEAD_ID} case=epoch_monotone prev={} next={}",
                prev.get(),
                next.get()
            );
            prev = next;
        }
        assert_eq!(
            clock.current().get(),
            100,
            "bead_id={BEAD_ID} case=epoch_monotone_final"
        );
    }

    // ── test_symbol_validity_window_rejects_future ──────────────────────

    #[test]
    fn test_symbol_validity_window_rejects_future() {
        let current = EpochId::new(5);
        let window = SymbolValidityWindow::default_window(current);
        let future = EpochId::new(6);
        assert!(
            !window.contains(future),
            "bead_id={BEAD_ID} case=validity_window_rejects_future"
        );
        let result = validate_symbol_epoch(future, &window);
        assert!(
            result.is_err(),
            "bead_id={BEAD_ID} case=validity_window_future_epoch_error"
        );
    }

    // ── test_symbol_validity_window_accepts_past ────────────────────────

    #[test]
    fn test_symbol_validity_window_accepts_past() {
        let current = EpochId::new(10);
        let window = SymbolValidityWindow::default_window(current);
        for past in [0, 1, 5, 9, 10] {
            let epoch = EpochId::new(past);
            assert!(
                window.contains(epoch),
                "bead_id={BEAD_ID} case=validity_window_accepts_past epoch={past}"
            );
            let result = validate_symbol_epoch(epoch, &window);
            assert!(
                result.is_ok(),
                "bead_id={BEAD_ID} case=validity_window_past_epoch_ok epoch={past}"
            );
        }
    }

    // ── test_epoch_scoped_key_derivation ────────────────────────────────

    #[test]
    fn test_epoch_scoped_key_derivation() {
        let master_key = [0xAB_u8; 32];
        let key_5 = derive_epoch_auth_key(&master_key, EpochId::new(5));
        let key_6 = derive_epoch_auth_key(&master_key, EpochId::new(6));
        assert_ne!(
            key_5, key_6,
            "bead_id={BEAD_ID} case=epoch_keys_differ_across_epochs"
        );
        // Deterministic: same inputs produce the same key.
        let key_5_again = derive_epoch_auth_key(&master_key, EpochId::new(5));
        assert_eq!(
            key_5, key_5_again,
            "bead_id={BEAD_ID} case=epoch_key_deterministic"
        );
    }

    // ── test_epoch_key_derivation_domain_separation ─────────────────────

    #[test]
    fn test_epoch_key_derivation_domain_separation() {
        let dek = [0x42_u8; 32];
        let master_key = derive_master_key_from_dek(&dek);
        // Master key MUST differ from raw DEK (domain separation).
        assert_ne!(
            master_key, dek,
            "bead_id={BEAD_ID} case=master_key_differs_from_dek"
        );
        // Auth key for epoch 0 MUST differ from master key.
        let auth_key = derive_epoch_auth_key(&master_key, EpochId::ZERO);
        assert_ne!(
            auth_key.as_bytes(),
            &master_key,
            "bead_id={BEAD_ID} case=auth_key_differs_from_master"
        );
    }

    // ── test_epoch_transition_barrier_all_arrive ────────────────────────

    #[test]
    fn test_epoch_transition_barrier_all_arrive() {
        let clock = EpochClock::new(EpochId::new(5));
        let barrier = EpochBarrier::new(EpochId::new(5), 4);

        assert!(!barrier.arrive("WriteCoordinator"));
        assert!(!barrier.arrive("SymbolStore"));
        assert!(!barrier.arrive("Replicator"));
        assert!(barrier.arrive("CheckpointGc"));

        assert!(
            barrier.is_complete(),
            "bead_id={BEAD_ID} case=barrier_complete"
        );

        let outcome = barrier.resolve(&clock).expect("resolve must succeed");
        assert_eq!(
            outcome,
            BarrierOutcome::AllArrived {
                new_epoch: EpochId::new(6),
            },
            "bead_id={BEAD_ID} case=barrier_all_arrived_epoch_incremented"
        );
        assert_eq!(
            clock.current().get(),
            6,
            "bead_id={BEAD_ID} case=clock_advanced_after_barrier"
        );
    }

    // ── test_epoch_transition_barrier_timeout ───────────────────────────

    #[test]
    fn test_epoch_transition_barrier_timeout() {
        let clock = EpochClock::new(EpochId::new(5));
        let barrier = EpochBarrier::new(EpochId::new(5), 4);

        barrier.arrive("WriteCoordinator");
        barrier.arrive("SymbolStore");
        barrier.arrive("Replicator");
        // CheckpointGc does NOT arrive.

        let outcome = barrier.resolve(&clock).expect("resolve must succeed");
        assert_eq!(
            outcome,
            BarrierOutcome::Timeout {
                arrived: 3,
                expected: 4,
            },
            "bead_id={BEAD_ID} case=barrier_timeout_epoch_unchanged"
        );
        assert_eq!(
            clock.current().get(),
            5,
            "bead_id={BEAD_ID} case=clock_unchanged_after_timeout"
        );
    }

    // ── test_epoch_bootstrap_from_ecs_root ──────────────────────────────

    #[test]
    fn test_epoch_bootstrap_from_ecs_root() {
        // Before RootManifest is decoded, use EcsRootPointer.ecs_epoch as
        // provisional upper bound. Symbols with epoch_id > root_epoch must
        // be rejected.
        let root_epoch = EpochId::new(7);
        let window = SymbolValidityWindow::default_window(root_epoch);

        // Epoch 8 (future) → rejected.
        assert!(
            !window.contains(EpochId::new(8)),
            "bead_id={BEAD_ID} case=bootstrap_rejects_future"
        );
        // Epoch 7 (current) → accepted.
        assert!(
            window.contains(EpochId::new(7)),
            "bead_id={BEAD_ID} case=bootstrap_accepts_current"
        );
        // Epoch 0 (past) → accepted.
        assert!(
            window.contains(EpochId::ZERO),
            "bead_id={BEAD_ID} case=bootstrap_accepts_zero"
        );
    }

    // ── test_barrier_cancelled ──────────────────────────────────────────

    #[test]
    fn test_barrier_cancelled() {
        let clock = EpochClock::new(EpochId::new(3));
        let barrier = EpochBarrier::new(EpochId::new(3), 2);

        barrier.arrive("WriteCoordinator");
        barrier.cancel();

        let outcome = barrier.resolve(&clock).expect("resolve must succeed");
        assert_eq!(
            outcome,
            BarrierOutcome::Cancelled,
            "bead_id={BEAD_ID} case=barrier_cancelled_epoch_unchanged"
        );
        assert_eq!(
            clock.current().get(),
            3,
            "bead_id={BEAD_ID} case=clock_unchanged_after_cancel"
        );
    }

    // ── test_epoch_clock_store_and_recover ──────────────────────────────

    #[test]
    fn test_epoch_clock_store_and_recover() {
        let clock = EpochClock::new(EpochId::ZERO);
        clock.store(EpochId::new(42));
        assert_eq!(
            clock.current().get(),
            42,
            "bead_id={BEAD_ID} case=clock_store_recovery"
        );
        let next = clock.increment().expect("increment after store");
        assert_eq!(
            next.get(),
            43,
            "bead_id={BEAD_ID} case=clock_increment_after_store"
        );
    }

    // ── test_validity_window_boundary ───────────────────────────────────

    #[test]
    fn test_validity_window_boundary() {
        let window = SymbolValidityWindow::new(EpochId::new(3), EpochId::new(7));
        assert!(
            !window.contains(EpochId::new(2)),
            "bead_id={BEAD_ID} case=window_below_lower_bound"
        );
        assert!(
            window.contains(EpochId::new(3)),
            "bead_id={BEAD_ID} case=window_at_lower_bound"
        );
        assert!(
            window.contains(EpochId::new(5)),
            "bead_id={BEAD_ID} case=window_within_bounds"
        );
        assert!(
            window.contains(EpochId::new(7)),
            "bead_id={BEAD_ID} case=window_at_upper_bound"
        );
        assert!(
            !window.contains(EpochId::new(8)),
            "bead_id={BEAD_ID} case=window_above_upper_bound"
        );
    }
}
