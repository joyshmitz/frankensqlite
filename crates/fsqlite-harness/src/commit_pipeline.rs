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
    #[allow(dead_code)]
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

// ── Unit Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const BEAD_ID: &str = "bd-3go.5";

    // ── CommitRequest tests ────────────────────────────────────────────

    #[test]
    fn test_commit_request_fields() {
        let req = CommitRequest::new(42, 7, vec![1, 2, 3]);
        assert_eq!(req.txn_id, 42, "bead_id={BEAD_ID} txn_id");
        assert_eq!(req.reserve_order, 7, "bead_id={BEAD_ID} reserve_order");
        assert_eq!(req.payload, vec![1, 2, 3], "bead_id={BEAD_ID} payload");
    }

    #[test]
    fn test_commit_request_eq() {
        let a = CommitRequest::new(1, 0, vec![10]);
        let b = CommitRequest::new(1, 0, vec![10]);
        let c = CommitRequest::new(2, 0, vec![10]);
        assert_eq!(a, b, "bead_id={BEAD_ID} identical requests");
        assert_ne!(a, c, "bead_id={BEAD_ID} different txn_id");
    }

    // ── Capacity derivation tests ──────────────────────────────────────

    #[test]
    fn test_default_capacity_is_16() {
        assert_eq!(
            DEFAULT_COMMIT_CHANNEL_CAPACITY, 16,
            "bead_id={BEAD_ID} spec §4.5 requires default capacity of 16"
        );
    }

    #[test]
    fn test_resolve_capacity_none() {
        assert_eq!(
            resolve_commit_channel_capacity(None),
            DEFAULT_COMMIT_CHANNEL_CAPACITY,
            "bead_id={BEAD_ID} None→default"
        );
    }

    #[test]
    fn test_resolve_capacity_zero() {
        assert_eq!(
            resolve_commit_channel_capacity(Some(0)),
            DEFAULT_COMMIT_CHANNEL_CAPACITY,
            "bead_id={BEAD_ID} 0→default"
        );
    }

    #[test]
    fn test_resolve_capacity_custom() {
        assert_eq!(
            resolve_commit_channel_capacity(Some(32)),
            32,
            "bead_id={BEAD_ID} custom value"
        );
    }

    // ── Little's Law capacity formula tests ────────────────────────────

    #[test]
    fn test_little_law_spec_derivation() {
        // §4.5: At burst 4× peak (148K/sec), t_commit=40µs, jitter 250%
        // burst_multiplier=1 because lambda already includes the 4× burst.
        let c = little_law_capacity(148_000, 40, 1, 250);
        assert_eq!(c, 15, "bead_id={BEAD_ID} Little's Law §4.5");
        assert!(
            DEFAULT_COMMIT_CHANNEL_CAPACITY >= c,
            "bead_id={BEAD_ID} default covers burst"
        );
    }

    #[test]
    fn test_little_law_zero_lambda() {
        assert_eq!(
            little_law_capacity(0, 40, 1, 250),
            1,
            "bead_id={BEAD_ID} zero lambda"
        );
    }

    #[test]
    fn test_little_law_zero_latency() {
        assert_eq!(
            little_law_capacity(148_000, 0, 1, 250),
            1,
            "bead_id={BEAD_ID} zero latency"
        );
    }

    // ── Pipeline construction tests ────────────────────────────────────

    #[test]
    fn test_pipeline_default_capacity() {
        let (pipeline, _rx) = CommitPipeline::with_default_capacity();
        assert_eq!(
            pipeline.capacity(),
            DEFAULT_COMMIT_CHANNEL_CAPACITY,
            "bead_id={BEAD_ID}"
        );
    }

    #[test]
    fn test_pipeline_custom_capacity() {
        let (pipeline, _rx) = CommitPipeline::new(64);
        assert_eq!(pipeline.capacity(), 64, "bead_id={BEAD_ID}");
    }

    #[test]
    fn test_pipeline_from_pragma_some() {
        let (pipeline, _rx) = CommitPipeline::from_pragma(Some(32));
        assert_eq!(pipeline.capacity(), 32, "bead_id={BEAD_ID}");
    }

    #[test]
    fn test_pipeline_from_pragma_none() {
        let (pipeline, _rx) = CommitPipeline::from_pragma(None);
        assert_eq!(
            pipeline.capacity(),
            DEFAULT_COMMIT_CHANNEL_CAPACITY,
            "bead_id={BEAD_ID}"
        );
    }

    // ── GroupCommitCoordinator tests ───────────────────────────────────

    #[test]
    fn test_group_commit_initial_batch() {
        let mut coordinator = GroupCommitCoordinator::new(16);
        let batch = coordinator.observe_and_plan_batch(2_000, 5, 128);
        assert!(
            (1..=16).contains(&batch),
            "bead_id={BEAD_ID} initial batch={batch} must be in [1,C]"
        );
    }

    #[test]
    fn test_group_commit_n_opt_near_capacity() {
        // N_opt = sqrt(2000/5) = sqrt(400) = 20, clamped to C=16.
        let mut coordinator = GroupCommitCoordinator::new(16);
        for _ in 0..128 {
            let _ = coordinator.observe_and_plan_batch(2_000, 5, 128);
        }
        let batch = coordinator.observe_and_plan_batch(2_000, 5, 128);
        assert!(
            batch >= 14,
            "bead_id={BEAD_ID} batch={batch} should converge near capacity"
        );
    }

    #[test]
    fn test_group_commit_respects_available() {
        let mut coordinator = GroupCommitCoordinator::new(64);
        for _ in 0..128 {
            let _ = coordinator.observe_and_plan_batch(10_000, 5, 3);
        }
        let batch = coordinator.observe_and_plan_batch(10_000, 5, 3);
        assert!(
            batch <= 3,
            "bead_id={BEAD_ID} batch={batch} must not exceed available"
        );
    }

    #[test]
    fn test_group_commit_zero_available() {
        let mut coordinator = GroupCommitCoordinator::new(16);
        coordinator.controller.observe_samples(2_000, 5);
        let batch = coordinator.controller.next_batch_size(0);
        assert_eq!(batch, 0, "bead_id={BEAD_ID} zero available→zero batch");
    }

    #[test]
    fn test_group_commit_batch_always_in_bounds() {
        let mut coordinator = GroupCommitCoordinator::new(16);
        for fsync in [100, 500, 2_000, 10_000, 100_000] {
            for validate in [1, 5, 50, 500] {
                for _ in 0..32 {
                    let batch = coordinator.observe_and_plan_batch(fsync, validate, 128);
                    assert!(
                        (1..=16).contains(&batch),
                        "bead_id={BEAD_ID} batch={batch} out of bounds for fsync={fsync} validate={validate}"
                    );
                }
            }
        }
    }

    // ── ConformalBatchController tests ──────────────────────────────────

    #[test]
    fn test_conformal_regime_shift_detection() {
        let mut ctrl = ConformalBatchController::new(16);
        // Fill with stable 2ms samples.
        for _ in 0..128 {
            ctrl.observe_samples(2_000, 5);
        }
        assert_eq!(
            ctrl.regime_shift_resets(),
            0,
            "bead_id={BEAD_ID} no shift under stable input"
        );
        // Inject 5× regime change.
        for _ in 0..128 {
            ctrl.observe_samples(10_000, 5);
        }
        assert!(
            ctrl.regime_shift_resets() >= 1,
            "bead_id={BEAD_ID} should detect regime shift"
        );
    }

    #[test]
    fn test_conformal_no_false_shift_under_stable_input() {
        let mut ctrl = ConformalBatchController::new(16);
        for _ in 0..512 {
            ctrl.observe_samples(2_000, 5);
        }
        assert_eq!(
            ctrl.regime_shift_resets(),
            0,
            "bead_id={BEAD_ID} stable input must not trigger shift"
        );
    }

    #[test]
    fn test_conformal_batch_size_adapts_upward() {
        let mut ctrl = ConformalBatchController::new(64);
        // Establish baseline with 2ms fsync / 500µs validate.
        for _ in 0..128 {
            ctrl.observe_samples(2_000, 500);
        }
        let before = ctrl.conformal_batch_size();
        // Shift to 10ms fsync (same validate).
        for _ in 0..128 {
            ctrl.observe_samples(10_000, 500);
        }
        let after = ctrl.conformal_batch_size();
        assert!(
            after > before,
            "bead_id={BEAD_ID} batch should increase: before={before} after={after}"
        );
    }

    #[test]
    fn test_conformal_resets_calibration_on_shift() {
        let mut ctrl = ConformalBatchController::new(16);
        for _ in 0..128 {
            ctrl.observe_samples(2_000, 5);
        }
        let resets_before = ctrl.regime_shift_resets();
        // 5× jump should trigger reset.
        for _ in 0..128 {
            ctrl.observe_samples(10_000, 5);
        }
        assert!(
            ctrl.regime_shift_resets() > resets_before,
            "bead_id={BEAD_ID} calibration must reset on shift"
        );
    }

    // ── Internal helper tests ──────────────────────────────────────────

    #[test]
    fn test_upper_quantile_single_element() {
        let mut samples = VecDeque::new();
        samples.push_back(42);
        assert_eq!(upper_quantile(&samples, 9, 10), 42);
    }

    #[test]
    fn test_upper_quantile_empty() {
        let samples = VecDeque::new();
        assert_eq!(upper_quantile(&samples, 9, 10), 1);
    }

    #[test]
    fn test_upper_quantile_ten_elements() {
        let samples: VecDeque<u64> = (1..=10).collect();
        // 90th percentile of [1..=10]: index = (9*9 + 9)/10 = 9 → value 10.
        let q = upper_quantile(&samples, 9, 10);
        assert!(q >= 9, "bead_id={BEAD_ID} q90={q} expected >= 9");
    }

    #[test]
    fn test_rounded_sqrt_ratio_perfect_square() {
        // sqrt(400/1) = 20
        assert_eq!(rounded_sqrt_ratio(400, 1), 20);
    }

    #[test]
    fn test_rounded_sqrt_ratio_non_trivial() {
        // sqrt(2000/5) = sqrt(400) = 20
        assert_eq!(rounded_sqrt_ratio(2_000, 5), 20);
    }

    #[test]
    fn test_rounded_sqrt_ratio_zero_numerator() {
        assert_eq!(rounded_sqrt_ratio(0, 5), 1);
    }

    #[test]
    fn test_rounded_sqrt_ratio_zero_denominator() {
        assert_eq!(rounded_sqrt_ratio(100, 0), usize::MAX);
    }

    // ── E2E-style coordinator test ─────────────────────────────────────

    #[test]
    fn test_e2e_coordinator_sustained_load() {
        // Under sustained load with t_fsync=2ms, t_validate=5µs,
        // coordinator batch size should stay near optimal (16, clamped).
        let mut coordinator = GroupCommitCoordinator::new(DEFAULT_COMMIT_CHANNEL_CAPACITY);
        let mut high_count = 0_usize;
        for _ in 0..256 {
            let batch = coordinator.observe_and_plan_batch(2_000, 5, 128);
            assert!((1..=16).contains(&batch));
            if batch >= 14 {
                high_count += 1;
            }
        }
        assert!(
            high_count >= 200,
            "bead_id={BEAD_ID} batch planner should converge: high_count={high_count}"
        );
    }
}
