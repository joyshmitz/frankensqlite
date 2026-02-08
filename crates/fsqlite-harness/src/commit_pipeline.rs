//! Two-phase MPSC commit-pipeline scaffolding for §4.5 (`bd-3go.5`).
//!
//! This module wraps the asupersync bounded MPSC channel in FrankenSQLite-specific
//! commit types and adds deterministic batch-size control helpers for harness tests.

use std::collections::VecDeque;

use asupersync::channel::mpsc;

/// Default bounded capacity derived from Little's Law in §4.5.
pub const DEFAULT_COMMIT_CHANNEL_CAPACITY: usize = 16;

/// Commit payload sent to the write coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitRequest {
    /// Transaction identifier for correlation and ordering assertions.
    pub txn_id: u64,
    /// Monotonic reserve order used by FIFO verification tests.
    pub reserve_order: u64,
    /// Opaque write-set payload.
    pub payload: Vec<u8>,
}

impl CommitRequest {
    /// Create a commit request.
    #[must_use]
    pub fn new(txn_id: u64, reserve_order: u64, payload: Vec<u8>) -> Self {
        Self {
            txn_id,
            reserve_order,
            payload,
        }
    }
}

/// Two-phase commit channel endpoint pair.
///
/// Writers call `sender().reserve(cx).await` (phase 1), then `permit.send(req)`
/// or `permit.abort()` (phase 2).
pub struct CommitPipeline {
    sender: mpsc::Sender<CommitRequest>,
    capacity: usize,
}

impl CommitPipeline {
    /// Create a bounded commit pipeline and its coordinator receiver.
    #[must_use]
    pub fn new(capacity: usize) -> (Self, mpsc::Receiver<CommitRequest>) {
        let normalized_capacity = capacity.max(1);
        let (sender, receiver) = mpsc::channel::<CommitRequest>(normalized_capacity);
        (
            Self {
                sender,
                capacity: normalized_capacity,
            },
            receiver,
        )
    }

    /// Create with the spec default capacity (16).
    #[must_use]
    pub fn with_default_capacity() -> (Self, mpsc::Receiver<CommitRequest>) {
        Self::new(DEFAULT_COMMIT_CHANNEL_CAPACITY)
    }

    /// Create from optional PRAGMA override.
    #[must_use]
    pub fn from_pragma(pragma_capacity: Option<usize>) -> (Self, mpsc::Receiver<CommitRequest>) {
        Self::new(resolve_commit_channel_capacity(pragma_capacity))
    }

    /// Configured bounded capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sender endpoint exposing two-phase reserve/send semantics.
    #[must_use]
    pub const fn sender(&self) -> &mpsc::Sender<CommitRequest> {
        &self.sender
    }
}

/// Resolve commit channel capacity from PRAGMA value.
#[must_use]
pub const fn resolve_commit_channel_capacity(pragma_capacity: Option<usize>) -> usize {
    match pragma_capacity {
        Some(capacity) if capacity > 0 => capacity,
        _ => DEFAULT_COMMIT_CHANNEL_CAPACITY,
    }
}

/// Little's Law capacity approximation used by §4.5 tuning notes.
///
/// `capacity ≈ ceil(lambda * t_commit * burst * jitter)` where:
/// - `lambda_per_second` is arrival rate in commits/sec
/// - `commit_latency_micros` is average end-to-end commit latency
/// - `burst_multiplier` captures peak burst factor (e.g. 4x)
/// - `jitter_percent` captures safety margin (e.g. 250 for 2.5x)
#[must_use]
pub fn little_law_capacity(
    lambda_per_second: u64,
    commit_latency_micros: u64,
    burst_multiplier: u32,
    jitter_percent: u32,
) -> usize {
    if lambda_per_second == 0 || commit_latency_micros == 0 {
        return 1;
    }

    let steady = (u128::from(lambda_per_second) * u128::from(commit_latency_micros))
        .div_ceil(1_000_000_u128);
    let with_burst = steady * u128::from(burst_multiplier.max(1));
    let with_jitter = (with_burst * u128::from(jitter_percent.max(100))).div_ceil(100_u128);

    usize::try_from(with_jitter.max(1)).unwrap_or(usize::MAX)
}

/// Conformal-inspired batch-size controller for group commit.
#[derive(Debug, Clone)]
pub struct ConformalBatchController {
    capacity: usize,
    fsync_samples_micros: VecDeque<u64>,
    validate_samples_micros: VecDeque<u64>,
    max_samples: usize,
    quantile_numerator: usize,
    quantile_denominator: usize,
    regime_shift_factor_percent: u32,
    baseline_fsync_micros: Option<u64>,
    regime_shift_resets: u64,
}

impl ConformalBatchController {
    /// Create a controller with default quantile and regime-shift settings.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            fsync_samples_micros: VecDeque::new(),
            validate_samples_micros: VecDeque::new(),
            max_samples: 128,
            quantile_numerator: 9,
            quantile_denominator: 10,
            regime_shift_factor_percent: 250,
            baseline_fsync_micros: None,
            regime_shift_resets: 0,
        }
    }

    /// Number of BOCPD-style regime-reset events observed.
    #[must_use]
    pub const fn regime_shift_resets(&self) -> u64 {
        self.regime_shift_resets
    }

    /// Observe one commit cycle's fsync + validation latency samples.
    pub fn observe_samples(&mut self, fsync_micros: u64, validate_micros: u64) {
        if fsync_micros == 0 || validate_micros == 0 {
            return;
        }

        if let Some(baseline) = self.baseline_fsync_micros {
            let lhs = u128::from(fsync_micros) * 100_u128;
            let rhs = u128::from(baseline) * u128::from(self.regime_shift_factor_percent);
            if lhs > rhs {
                self.fsync_samples_micros.clear();
                self.validate_samples_micros.clear();
                self.regime_shift_resets = self.regime_shift_resets.saturating_add(1);
            }
        }

        self.baseline_fsync_micros = Some(match self.baseline_fsync_micros {
            None => fsync_micros,
            Some(baseline) => ((baseline * 7) + fsync_micros) / 8,
        });

        push_bounded(
            &mut self.fsync_samples_micros,
            fsync_micros,
            self.max_samples,
        );
        push_bounded(
            &mut self.validate_samples_micros,
            validate_micros,
            self.max_samples,
        );
    }

    /// Current conformal batch target before availability clamping.
    #[must_use]
    pub fn conformal_batch_size(&self) -> usize {
        let fsync_quantile = upper_quantile(
            &self.fsync_samples_micros,
            self.quantile_numerator,
            self.quantile_denominator,
        );
        let validate_quantile = upper_quantile(
            &self.validate_samples_micros,
            self.quantile_numerator,
            self.quantile_denominator,
        )
        .max(1);

        let raw = rounded_sqrt_ratio(fsync_quantile, validate_quantile);
        raw.clamp(1, self.capacity)
    }

    /// Batch size to drain this cycle based on pending work.
    #[must_use]
    pub fn next_batch_size(&self, available_commits: usize) -> usize {
        if available_commits == 0 {
            return 0;
        }
        self.conformal_batch_size()
            .min(self.capacity)
            .min(available_commits)
            .max(1)
    }
}

/// Group-commit coordinator helper combining observation + batch planning.
#[derive(Debug, Clone)]
pub struct GroupCommitCoordinator {
    controller: ConformalBatchController,
}

impl GroupCommitCoordinator {
    /// Create a coordinator with a bounded in-flight capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            controller: ConformalBatchController::new(capacity),
        }
    }

    /// Observe one cycle and return the next planned batch size.
    #[must_use]
    pub fn observe_and_plan_batch(
        &mut self,
        fsync_micros: u64,
        validate_micros: u64,
        available_commits: usize,
    ) -> usize {
        self.controller
            .observe_samples(fsync_micros, validate_micros);
        self.controller.next_batch_size(available_commits)
    }

    /// Access the underlying controller for diagnostics.
    #[must_use]
    pub const fn controller(&self) -> &ConformalBatchController {
        &self.controller
    }
}

fn push_bounded(samples: &mut VecDeque<u64>, value: u64, max_samples: usize) {
    if samples.len() == max_samples {
        let _ = samples.pop_front();
    }
    samples.push_back(value);
}

fn upper_quantile(samples: &VecDeque<u64>, numerator: usize, denominator: usize) -> u64 {
    if samples.is_empty() {
        return 1;
    }

    let mut ordered: Vec<u64> = samples.iter().copied().collect();
    ordered.sort_unstable();

    let last_index = ordered.len().saturating_sub(1);
    let quantile_index = (last_index
        .saturating_mul(numerator)
        .saturating_add(denominator.saturating_sub(1)))
        / denominator;

    ordered[quantile_index.min(last_index)]
}

fn rounded_sqrt_ratio(numerator: u64, denominator: u64) -> usize {
    if numerator == 0 {
        return 1;
    }
    if denominator == 0 {
        return usize::MAX;
    }

    let mut low = 0_u64;
    let mut high = numerator.max(1);

    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if square_ratio_less_or_equal(mid, numerator, denominator) {
            low = mid;
        } else {
            high = mid.saturating_sub(1);
        }
    }

    let floor_root = low.max(1);
    let ceil_root = floor_root.saturating_add(1);

    let numerator_u128 = u128::from(numerator);
    let denominator_u128 = u128::from(denominator);

    let floor_squared = u128::from(floor_root) * u128::from(floor_root) * denominator_u128;
    let ceil_squared = u128::from(ceil_root) * u128::from(ceil_root) * denominator_u128;

    let floor_error = floor_squared.abs_diff(numerator_u128);
    let ceil_error = ceil_squared.abs_diff(numerator_u128);

    let rounded = if ceil_error < floor_error {
        ceil_root
    } else {
        floor_root
    };

    usize::try_from(rounded.max(1)).unwrap_or(usize::MAX)
}

fn square_ratio_less_or_equal(candidate: u64, numerator: u64, denominator: u64) -> bool {
    let lhs = u128::from(candidate) * u128::from(candidate) * u128::from(denominator);
    lhs <= u128::from(numerator)
}
