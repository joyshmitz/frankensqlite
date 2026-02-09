//! §4.17 Policy Controller: Expected-loss minimization + PRAGMA auto-tune.
//!
//! This controller tunes non-correctness knobs within a bounded safe envelope.
//! It is deterministic for a fixed input stream and keeps an explainability
//! ledger for auditable automatic decisions.

use std::collections::{BTreeMap, VecDeque};

use super::{
    DEFAULT_OVERHEAD_PERCENT, INITIAL_REPAIR_POLICY_EPOCH, MAX_OVERHEAD_PERCENT,
    MIN_OVERHEAD_PERCENT, RepairBudget, RepairObjectClass, compute_repair_budget_for_object,
};

/// Runtime profile exposed through `PRAGMA fsqlite.profile`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AutoTuneProfile {
    /// Balanced default for mixed workloads.
    Balanced,
    /// Favor latency (lower background pressure).
    Latency,
    /// Favor throughput (higher bounded parallelism).
    Throughput,
}

fn clamp_permits(value: usize, min: usize, max: usize) -> usize {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Derive `bg_cpu_max` from profile and available parallelism `P`.
#[must_use]
pub fn derive_bg_cpu_max(profile: AutoTuneProfile, parallelism: usize) -> usize {
    let p = if parallelism == 0 { 1 } else { parallelism };
    match profile {
        AutoTuneProfile::Balanced => clamp_permits(p / 8, 1, 16),
        AutoTuneProfile::Latency => clamp_permits(p / 16, 1, 8),
        AutoTuneProfile::Throughput => clamp_permits(p / 4, 1, 32),
    }
}

/// Derive `remote_max_in_flight` from profile and available parallelism `P`.
#[must_use]
pub fn derive_remote_max_in_flight(profile: AutoTuneProfile, parallelism: usize) -> usize {
    let p = if parallelism == 0 { 1 } else { parallelism };
    match profile {
        AutoTuneProfile::Balanced => clamp_permits(p / 8, 1, 8),
        AutoTuneProfile::Latency => clamp_permits(p / 16, 1, 4),
        AutoTuneProfile::Throughput => clamp_permits(p / 4, 1, 16),
    }
}

/// Derive `commit_encode_max` from profile and available parallelism `P`.
#[must_use]
pub fn derive_commit_encode_max(profile: AutoTuneProfile, parallelism: usize) -> usize {
    let p = if parallelism == 0 { 1 } else { parallelism };
    match profile {
        AutoTuneProfile::Balanced => clamp_permits(p / 4, 1, 16),
        AutoTuneProfile::Latency => clamp_permits(p / 8, 1, 8),
        AutoTuneProfile::Throughput => clamp_permits(p / 2, 1, 32),
    }
}

/// Effective permit caps after profile derivation and hard-cap overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveLimits {
    pub bg_cpu_max: usize,
    pub remote_max_in_flight: usize,
    pub commit_encode_max: usize,
}

/// PRAGMA-backed auto-tuning configuration.
///
/// Integer fields use SQLite-style semantics:
/// - `0` => auto (derived from profile + available parallelism)
/// - `> 0` => hard cap override
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoTunePragmaConfig {
    pub auto_tune: bool,
    pub profile: AutoTuneProfile,
    pub bg_cpu_max: usize,
    pub remote_max_in_flight: usize,
    pub commit_encode_max: usize,
}

impl AutoTunePragmaConfig {
    /// Compute effective limits for a given parallelism.
    #[must_use]
    pub fn effective_limits_with_parallelism(self, parallelism: usize) -> EffectiveLimits {
        let derived = EffectiveLimits {
            bg_cpu_max: derive_bg_cpu_max(self.profile, parallelism),
            remote_max_in_flight: derive_remote_max_in_flight(self.profile, parallelism),
            commit_encode_max: derive_commit_encode_max(self.profile, parallelism),
        };

        EffectiveLimits {
            bg_cpu_max: if self.bg_cpu_max == 0 {
                derived.bg_cpu_max
            } else {
                self.bg_cpu_max
            },
            remote_max_in_flight: if self.remote_max_in_flight == 0 {
                derived.remote_max_in_flight
            } else {
                self.remote_max_in_flight
            },
            commit_encode_max: if self.commit_encode_max == 0 {
                derived.commit_encode_max
            } else {
                self.commit_encode_max
            },
        }
    }

    #[must_use]
    pub fn hard_cap_for_knob(self, knob: PolicyKnob) -> Option<usize> {
        match knob {
            PolicyKnob::BgCpuMax => {
                if self.bg_cpu_max > 0 {
                    Some(self.bg_cpu_max)
                } else {
                    None
                }
            }
            PolicyKnob::RemoteMaxInFlight => {
                if self.remote_max_in_flight > 0 {
                    Some(self.remote_max_in_flight)
                } else {
                    None
                }
            }
            PolicyKnob::CommitEncodeMax => {
                if self.commit_encode_max > 0 {
                    Some(self.commit_encode_max)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl Default for AutoTunePragmaConfig {
    fn default() -> Self {
        Self {
            auto_tune: true,
            profile: AutoTuneProfile::Balanced,
            bg_cpu_max: 0,
            remote_max_in_flight: 0,
            commit_encode_max: 0,
        }
    }
}

/// Version tag for persisted `PRAGMA raptorq_overhead` metadata.
pub const RAPTORQ_OVERHEAD_METADATA_VERSION: u16 = 1;

#[must_use]
fn clamp_overhead_percent(raw_percent: i64) -> u32 {
    let clamped = raw_percent.clamp(
        i64::from(MIN_OVERHEAD_PERCENT),
        i64::from(MAX_OVERHEAD_PERCENT),
    );
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        clamped as u32
    }
}

#[must_use]
fn round_basis_points_to_percent(raw_basis_points: i64) -> i64 {
    // Deterministic rounding to nearest percent, ties away from zero.
    if raw_basis_points >= 0 {
        (raw_basis_points + 50).div_euclid(100)
    } else {
        (raw_basis_points - 50).div_euclid(100)
    }
}

/// Stable key used to persist per-object policy rows in deterministic order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ObjectClassKey {
    CommitMarker,
    CommitProof,
    PageHistory,
    SnapshotBlock,
    WalFecGroup,
    GenericEcs,
}

impl From<RepairObjectClass> for ObjectClassKey {
    fn from(value: RepairObjectClass) -> Self {
        match value {
            RepairObjectClass::CommitMarker => Self::CommitMarker,
            RepairObjectClass::CommitProof => Self::CommitProof,
            RepairObjectClass::PageHistory => Self::PageHistory,
            RepairObjectClass::SnapshotBlock => Self::SnapshotBlock,
            RepairObjectClass::WalFecGroup => Self::WalFecGroup,
            RepairObjectClass::GenericEcs => Self::GenericEcs,
        }
    }
}

/// Source of truth used for effective overhead policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverheadPolicyScope {
    /// SQLite-style connection-local setting (default).
    ConnectionLocal,
    /// Versioned metadata persisted with the database for replay determinism.
    PersistedMetadata,
}

/// Effective overhead policy for one object class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveOverheadPolicy {
    pub object_class: RepairObjectClass,
    pub overhead_percent: u32,
    pub policy_epoch: u64,
    pub metadata_version: u16,
    pub scope: OverheadPolicyScope,
}

/// Persisted metadata representation for `PRAGMA raptorq_overhead`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistedOverheadPolicy {
    pub metadata_version: u16,
    pub policy_epoch: u64,
    pub default_overhead_percent: u32,
    pub object_overrides: BTreeMap<ObjectClassKey, u32>,
}

impl PersistedOverheadPolicy {
    #[must_use]
    pub fn new(policy_epoch: u64, default_overhead_percent: u32) -> Self {
        Self {
            metadata_version: RAPTORQ_OVERHEAD_METADATA_VERSION,
            policy_epoch,
            default_overhead_percent: clamp_overhead_percent(i64::from(default_overhead_percent)),
            object_overrides: BTreeMap::new(),
        }
    }

    pub fn set_override_percent(
        &mut self,
        object_class: RepairObjectClass,
        raw_percent: i64,
    ) -> u32 {
        let clamped = clamp_overhead_percent(raw_percent);
        self.object_overrides
            .insert(ObjectClassKey::from(object_class), clamped);
        clamped
    }

    #[must_use]
    pub fn effective_percent_for_object(&self, object_class: RepairObjectClass) -> u32 {
        self.object_overrides
            .get(&ObjectClassKey::from(object_class))
            .copied()
            .unwrap_or(self.default_overhead_percent)
    }

    #[must_use]
    pub fn override_percent_for_object(&self, object_class: RepairObjectClass) -> Option<u32> {
        self.object_overrides
            .get(&ObjectClassKey::from(object_class))
            .copied()
    }
}

/// Deterministic state for `PRAGMA raptorq_overhead`.
///
/// Behavior:
/// - default scope is SQLite-style connection-local policy,
/// - callers may persist a versioned snapshot into metadata for replay-stable
///   replication/proof behavior,
/// - per-object overrides are supported even when only a global PRAGMA is
///   exposed publicly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaptorQOverheadPragmaState {
    default_overhead_percent: u32,
    object_overrides: BTreeMap<ObjectClassKey, u32>,
    persisted_policy: Option<PersistedOverheadPolicy>,
}

impl Default for RaptorQOverheadPragmaState {
    fn default() -> Self {
        Self {
            default_overhead_percent: DEFAULT_OVERHEAD_PERCENT,
            object_overrides: BTreeMap::new(),
            persisted_policy: None,
        }
    }
}

impl RaptorQOverheadPragmaState {
    #[must_use]
    pub fn default_overhead_percent(&self) -> u32 {
        self.default_overhead_percent
    }

    pub fn set_default_percent_from_pragma(&mut self, raw_percent: i64) -> u32 {
        let clamped = clamp_overhead_percent(raw_percent);
        self.default_overhead_percent = clamped;
        clamped
    }

    /// Parse percent in basis points and store rounded+clamped value.
    ///
    /// Example: `2_549 bps` => `25 %`.
    pub fn set_default_percent_from_basis_points(&mut self, raw_basis_points: i64) -> u32 {
        let rounded = round_basis_points_to_percent(raw_basis_points);
        self.set_default_percent_from_pragma(rounded)
    }

    pub fn set_object_override_from_pragma(
        &mut self,
        object_class: RepairObjectClass,
        raw_percent: i64,
    ) -> u32 {
        let clamped = clamp_overhead_percent(raw_percent);
        self.object_overrides
            .insert(ObjectClassKey::from(object_class), clamped);
        clamped
    }

    pub fn clear_object_override(&mut self, object_class: RepairObjectClass) -> Option<u32> {
        self.object_overrides
            .remove(&ObjectClassKey::from(object_class))
    }

    #[must_use]
    pub fn persisted_policy(&self) -> Option<&PersistedOverheadPolicy> {
        self.persisted_policy.as_ref()
    }

    pub fn apply_persisted_policy(&mut self, policy: PersistedOverheadPolicy) {
        self.persisted_policy = Some(policy);
    }

    pub fn clear_persisted_policy(&mut self) {
        self.persisted_policy = None;
    }

    /// Snapshot connection-local state into versioned persisted metadata.
    pub fn persist_connection_policy(&mut self, policy_epoch: u64) -> PersistedOverheadPolicy {
        let persisted_epoch = policy_epoch.max(INITIAL_REPAIR_POLICY_EPOCH.saturating_add(1));
        let mut persisted =
            PersistedOverheadPolicy::new(persisted_epoch, self.default_overhead_percent);
        persisted.object_overrides = self.object_overrides.clone();
        self.persisted_policy = Some(persisted.clone());
        persisted
    }

    #[must_use]
    pub fn effective_policy_for_object(
        &self,
        object_class: RepairObjectClass,
    ) -> EffectiveOverheadPolicy {
        if let Some(persisted) = &self.persisted_policy {
            return EffectiveOverheadPolicy {
                object_class,
                overhead_percent: persisted.effective_percent_for_object(object_class),
                policy_epoch: persisted.policy_epoch,
                metadata_version: persisted.metadata_version,
                scope: OverheadPolicyScope::PersistedMetadata,
            };
        }

        let overhead_percent = self
            .object_overrides
            .get(&ObjectClassKey::from(object_class))
            .copied()
            .unwrap_or(self.default_overhead_percent);
        EffectiveOverheadPolicy {
            object_class,
            overhead_percent,
            policy_epoch: INITIAL_REPAIR_POLICY_EPOCH,
            metadata_version: RAPTORQ_OVERHEAD_METADATA_VERSION,
            scope: OverheadPolicyScope::ConnectionLocal,
        }
    }

    /// Compute deterministic repair budget and expose effective overhead used.
    #[must_use]
    pub fn compute_budget_for_object(
        &self,
        k_source: u32,
        object_class: RepairObjectClass,
    ) -> RepairBudget {
        let effective = self.effective_policy_for_object(object_class);
        compute_repair_budget_for_object(
            k_source,
            object_class,
            Some(effective.overhead_percent),
            effective.policy_epoch,
        )
    }
}

/// Tunable policy knob.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PolicyKnob {
    RedundancyOverheadPercent,
    GroupCommitBatch,
    RetryBackoffMs,
    TxnMaxDurationMs,
    LeaseDurationMs,
    BgCpuMax,
    RemoteMaxInFlight,
    CommitEncodeMax,
    GcCompactionRate,
}

/// Candidate action evaluated by expected-loss minimization.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateAction {
    pub id: u64,
    pub value: usize,
    pub expected_loss: f64,
    pub description: String,
}

impl CandidateAction {
    #[must_use]
    pub fn new(id: u64, value: usize, expected_loss: f64, description: impl Into<String>) -> Self {
        Self {
            id,
            value,
            expected_loss,
            description: description.into(),
        }
    }
}

/// Signals from monitors/regime detection used by guardrails and annotations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PolicySignals {
    pub symbol_loss_rejects_h0: bool,
    pub bocpd_regime_shift: bool,
    pub regime_id: u64,
}

/// Candidate evaluation details persisted to the evidence ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEvaluation {
    pub id: u64,
    pub value: usize,
    pub expected_loss: f64,
    pub description: String,
    pub blocked: bool,
    pub block_reason: Option<String>,
}

/// Explainability ledger entry for automatic policy evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyEvidenceEntry {
    pub decision_id: u64,
    pub knob: PolicyKnob,
    pub prior_setting: usize,
    pub chosen_candidate_id: Option<u64>,
    pub chosen_setting: usize,
    pub candidates: Vec<CandidateEvaluation>,
    pub expected_losses: BTreeMap<u64, f64>,
    pub top_evidence: Vec<String>,
    pub regime_id: u64,
}

/// Bounded evidence ledger for policy decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyEvidenceLedger {
    capacity: usize,
    entries: VecDeque<PolicyEvidenceEntry>,
}

impl PolicyEvidenceLedger {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::new(),
        }
    }

    pub fn record(&mut self, entry: PolicyEvidenceEntry) {
        if self.entries.len() == self.capacity {
            let _ = self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[must_use]
    pub fn entries(&self) -> &VecDeque<PolicyEvidenceEntry> {
        &self.entries
    }

    #[must_use]
    pub fn latest(&self) -> Option<&PolicyEvidenceEntry> {
        self.entries.back()
    }
}

/// Optional monitor with VOI budgeting metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorSpec {
    pub name: String,
    pub expected_delta_loss: f64,
    pub cost: f64,
    pub correctness_critical: bool,
}

impl MonitorSpec {
    #[must_use]
    pub fn voi(&self) -> f64 {
        if self.correctness_critical {
            f64::INFINITY
        } else {
            self.expected_delta_loss - self.cost
        }
    }
}

/// Result of VOI-aware optional monitor scheduling.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorSchedule {
    pub selected: Vec<String>,
    pub total_optional_cost: f64,
}

/// Select monitors by VOI under a bounded optional-monitor budget.
///
/// Correctness-critical monitors are always selected and excluded from the
/// optional cost budget accounting.
#[must_use]
pub fn schedule_monitors(monitors: &[MonitorSpec], optional_cost_budget: f64) -> MonitorSchedule {
    let mut selected = Vec::new();
    let mut optional_total = 0.0_f64;

    let mut optional = Vec::new();
    for monitor in monitors {
        if monitor.correctness_critical {
            selected.push(monitor.name.clone());
        } else {
            optional.push(monitor.clone());
        }
    }

    optional.sort_by(|left, right| {
        right
            .voi()
            .total_cmp(&left.voi())
            .then_with(|| left.name.cmp(&right.name))
    });

    for monitor in optional {
        let voi = monitor.voi();
        if !(voi > 0.0 && monitor.cost.is_finite()) {
            continue;
        }
        if optional_total + monitor.cost <= optional_cost_budget.max(0.0) {
            optional_total += monitor.cost;
            selected.push(monitor.name);
        }
    }

    selected.sort();
    MonitorSchedule {
        selected,
        total_optional_cost: optional_total,
    }
}

/// Outcome reason for a policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionReason {
    Applied(u64),
    NoAllowedCandidates,
    HysteresisSuppressed,
    FallbackAutoTuneOff,
    FallbackTelemetryUnavailable,
}

/// Result of a policy knob evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyDecisionOutcome {
    pub knob: PolicyKnob,
    pub prior_setting: usize,
    pub applied_setting: usize,
    pub changed: bool,
    pub reason: DecisionReason,
}

/// Policy controller implementing expected-loss minimization with guardrails.
#[derive(Debug, Clone)]
pub struct PolicyController {
    config: AutoTunePragmaConfig,
    effective_limits: EffectiveLimits,
    policy_interval_ticks: u64,
    last_change_tick: BTreeMap<PolicyKnob, u64>,
    ledger: PolicyEvidenceLedger,
    next_decision_id: u64,
}

impl PolicyController {
    #[must_use]
    pub fn new(
        config: AutoTunePragmaConfig,
        available_parallelism: usize,
        policy_interval_ticks: u64,
        ledger_capacity: usize,
    ) -> Self {
        Self {
            effective_limits: config.effective_limits_with_parallelism(available_parallelism),
            config,
            policy_interval_ticks: policy_interval_ticks.max(1),
            last_change_tick: BTreeMap::new(),
            ledger: PolicyEvidenceLedger::new(ledger_capacity),
            next_decision_id: 1,
        }
    }

    #[must_use]
    pub const fn effective_limits(&self) -> EffectiveLimits {
        self.effective_limits
    }

    #[must_use]
    pub fn ledger(&self) -> &PolicyEvidenceLedger {
        &self.ledger
    }

    fn set_knob_value(&mut self, knob: PolicyKnob, value: usize) {
        match knob {
            PolicyKnob::BgCpuMax => {
                self.effective_limits.bg_cpu_max = value;
            }
            PolicyKnob::RemoteMaxInFlight => {
                self.effective_limits.remote_max_in_flight = value;
            }
            PolicyKnob::CommitEncodeMax => {
                self.effective_limits.commit_encode_max = value;
            }
            _ => {}
        }
    }

    fn deterministic_candidate_order(candidates: &[CandidateAction]) -> Vec<CandidateAction> {
        let mut ordered = candidates.to_vec();
        ordered.sort_by(|left, right| {
            left.expected_loss
                .total_cmp(&right.expected_loss)
                .then_with(|| left.id.cmp(&right.id))
        });
        ordered
    }

    fn guardrail_block_reason(
        knob: PolicyKnob,
        prior_setting: usize,
        candidate: &CandidateAction,
        signals: PolicySignals,
        hard_cap: Option<usize>,
    ) -> Option<String> {
        if !candidate.expected_loss.is_finite() {
            return Some("non_finite_expected_loss".to_owned());
        }
        if knob == PolicyKnob::RedundancyOverheadPercent
            && signals.symbol_loss_rejects_h0
            && candidate.value < prior_setting
        {
            return Some("guardrail_symbol_loss_budget".to_owned());
        }
        if let Some(cap) = hard_cap {
            if candidate.value > cap {
                return Some(format!("hard_cap_override({cap})"));
            }
        }
        None
    }

    /// Evaluate a knob update with expected-loss minimization and guardrails.
    pub fn evaluate_knob(
        &mut self,
        knob: PolicyKnob,
        prior_setting: usize,
        candidates: &[CandidateAction],
        signals: PolicySignals,
        telemetry_available: bool,
        tick: u64,
    ) -> PolicyDecisionOutcome {
        if !self.config.auto_tune {
            return PolicyDecisionOutcome {
                knob,
                prior_setting,
                applied_setting: prior_setting,
                changed: false,
                reason: DecisionReason::FallbackAutoTuneOff,
            };
        }
        if !telemetry_available {
            return PolicyDecisionOutcome {
                knob,
                prior_setting,
                applied_setting: prior_setting,
                changed: false,
                reason: DecisionReason::FallbackTelemetryUnavailable,
            };
        }

        let hard_cap = self.config.hard_cap_for_knob(knob);
        let ordered = Self::deterministic_candidate_order(candidates);
        let mut evals = Vec::with_capacity(ordered.len());
        let mut allowed = Vec::new();
        let mut expected_losses = BTreeMap::new();

        for candidate in ordered {
            let block_reason =
                Self::guardrail_block_reason(knob, prior_setting, &candidate, signals, hard_cap);
            let blocked = block_reason.is_some();
            if candidate.expected_loss.is_finite() {
                expected_losses.insert(candidate.id, candidate.expected_loss);
            }
            if !blocked {
                allowed.push(candidate.clone());
            }
            evals.push(CandidateEvaluation {
                id: candidate.id,
                value: candidate.value,
                expected_loss: candidate.expected_loss,
                description: candidate.description,
                blocked,
                block_reason,
            });
        }

        let mut top_evidence = Vec::new();
        if signals.symbol_loss_rejects_h0 {
            top_evidence.push("symbol_loss_eprocess_reject".to_owned());
        }
        if signals.bocpd_regime_shift {
            top_evidence.push("bocpd_regime_shift".to_owned());
        }
        if let Some(cap) = hard_cap {
            top_evidence.push(format!("hard_cap={cap}"));
        }
        top_evidence.sort();

        let mut chosen_candidate_id = None;
        let mut chosen_setting = prior_setting;
        let decision_reason = if let Some(best) = allowed.first() {
            chosen_candidate_id = Some(best.id);
            chosen_setting = best.value;
            DecisionReason::Applied(best.id)
        } else {
            DecisionReason::NoAllowedCandidates
        };

        let change_blocked_by_interval = chosen_setting != prior_setting
            && self.last_change_tick.get(&knob).is_some_and(|previous| {
                tick.saturating_sub(*previous) < self.policy_interval_ticks
            });

        let decision_reason = if change_blocked_by_interval {
            chosen_setting = prior_setting;
            DecisionReason::HysteresisSuppressed
        } else {
            decision_reason
        };

        let changed = chosen_setting != prior_setting;
        if changed {
            self.last_change_tick.insert(knob, tick);
            self.set_knob_value(knob, chosen_setting);
        }

        let entry = PolicyEvidenceEntry {
            decision_id: self.next_decision_id,
            knob,
            prior_setting,
            chosen_candidate_id,
            chosen_setting,
            candidates: evals,
            expected_losses,
            top_evidence,
            regime_id: signals.regime_id,
        };
        self.next_decision_id = self.next_decision_id.saturating_add(1);
        self.ledger.record(entry);

        PolicyDecisionOutcome {
            knob,
            prior_setting,
            applied_setting: chosen_setting,
            changed,
            reason: decision_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_candidates() -> Vec<CandidateAction> {
        vec![
            CandidateAction::new(1, 10, 12.0, "keep"),
            CandidateAction::new(2, 6, 9.0, "reduce"),
            CandidateAction::new(3, 14, 11.0, "increase"),
        ]
    }

    #[test]
    fn test_pragma_auto_tune_on_default() {
        assert!(AutoTunePragmaConfig::default().auto_tune);
    }

    #[test]
    fn test_pragma_profile_balanced_default() {
        assert_eq!(
            AutoTunePragmaConfig::default().profile,
            AutoTuneProfile::Balanced
        );
    }

    #[test]
    fn test_default_derivation_balanced_4_cores() {
        let limits = AutoTunePragmaConfig::default().effective_limits_with_parallelism(4);
        assert_eq!(limits.bg_cpu_max, 1);
        assert_eq!(limits.remote_max_in_flight, 1);
        assert_eq!(limits.commit_encode_max, 1);
    }

    #[test]
    fn test_default_derivation_balanced_64_cores() {
        let limits = AutoTunePragmaConfig::default().effective_limits_with_parallelism(64);
        assert_eq!(limits.bg_cpu_max, 8);
        assert_eq!(limits.remote_max_in_flight, 8);
        assert_eq!(limits.commit_encode_max, 16);
    }

    #[test]
    fn test_default_derivation_throughput_32_cores() {
        let config = AutoTunePragmaConfig {
            profile: AutoTuneProfile::Throughput,
            ..AutoTunePragmaConfig::default()
        };
        let limits = config.effective_limits_with_parallelism(32);
        assert_eq!(limits.bg_cpu_max, 8);
        assert_eq!(limits.remote_max_in_flight, 8);
        assert_eq!(limits.commit_encode_max, 16);
    }

    #[test]
    fn test_default_derivation_latency_128_cores() {
        let config = AutoTunePragmaConfig {
            profile: AutoTuneProfile::Latency,
            ..AutoTunePragmaConfig::default()
        };
        let limits = config.effective_limits_with_parallelism(128);
        assert_eq!(limits.bg_cpu_max, 8);
        assert_eq!(limits.remote_max_in_flight, 4);
        assert_eq!(limits.commit_encode_max, 8);
    }

    #[test]
    fn test_pragma_bg_cpu_max_zero_means_auto() {
        let config = AutoTunePragmaConfig::default();
        let limits = config.effective_limits_with_parallelism(16);
        assert_eq!(limits.bg_cpu_max, 2);
    }

    #[test]
    fn test_pragma_bg_cpu_max_positive_means_hard_cap() {
        let config = AutoTunePragmaConfig {
            bg_cpu_max: 3,
            ..AutoTunePragmaConfig::default()
        };
        let limits = config.effective_limits_with_parallelism(128);
        assert_eq!(limits.bg_cpu_max, 3);
    }

    #[test]
    fn test_policy_argmin_loss() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let out = controller.evaluate_knob(
            PolicyKnob::GroupCommitBatch,
            10,
            &sample_candidates(),
            PolicySignals::default(),
            true,
            10,
        );
        assert_eq!(out.reason, DecisionReason::Applied(2));
        assert_eq!(out.applied_setting, 6);
    }

    #[test]
    fn test_policy_asymmetric_loss() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let candidates = vec![
            CandidateAction::new(1, 20, 5000.0, "lower redundancy"),
            CandidateAction::new(2, 30, 3.0, "higher redundancy"),
        ];
        let out = controller.evaluate_knob(
            PolicyKnob::RedundancyOverheadPercent,
            25,
            &candidates,
            PolicySignals::default(),
            true,
            10,
        );
        assert_eq!(out.reason, DecisionReason::Applied(2));
        assert_eq!(out.applied_setting, 30);
    }

    #[test]
    fn test_policy_candidate_evaluation() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let candidates = sample_candidates();
        let _ = controller.evaluate_knob(
            PolicyKnob::RetryBackoffMs,
            10,
            &candidates,
            PolicySignals::default(),
            true,
            10,
        );
        let entry = controller
            .ledger()
            .latest()
            .expect("ledger entry must exist");
        assert_eq!(entry.candidates.len(), candidates.len());
    }

    #[test]
    fn test_guardrail_blocks_unsafe_action() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let candidates = vec![
            CandidateAction::new(1, 15, 1.0, "unsafe decrease"),
            CandidateAction::new(2, 25, 2.0, "safe increase"),
        ];
        let out = controller.evaluate_knob(
            PolicyKnob::RedundancyOverheadPercent,
            20,
            &candidates,
            PolicySignals {
                symbol_loss_rejects_h0: true,
                bocpd_regime_shift: false,
                regime_id: 9,
            },
            true,
            10,
        );
        assert_eq!(out.applied_setting, 25);
        let entry = controller
            .ledger()
            .latest()
            .expect("ledger entry must exist");
        let blocked = entry
            .candidates
            .iter()
            .find(|candidate| candidate.id == 1)
            .expect("candidate present");
        assert!(blocked.blocked);
    }

    #[test]
    fn test_guardrail_allows_safe_action() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let candidates = vec![CandidateAction::new(2, 24, 2.0, "safe increase")];
        let out = controller.evaluate_knob(
            PolicyKnob::RedundancyOverheadPercent,
            20,
            &candidates,
            PolicySignals {
                symbol_loss_rejects_h0: true,
                bocpd_regime_shift: false,
                regime_id: 9,
            },
            true,
            10,
        );
        assert_eq!(out.reason, DecisionReason::Applied(2));
    }

    #[test]
    fn test_guardrail_bocpd_regime_shift() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let _ = controller.evaluate_knob(
            PolicyKnob::GcCompactionRate,
            2,
            &[CandidateAction::new(1, 3, 1.0, "retune")],
            PolicySignals {
                symbol_loss_rejects_h0: false,
                bocpd_regime_shift: true,
                regime_id: 77,
            },
            true,
            10,
        );
        let entry = controller
            .ledger()
            .latest()
            .expect("ledger entry must exist");
        assert_eq!(entry.regime_id, 77);
        assert!(
            entry
                .top_evidence
                .iter()
                .any(|item| item == "bocpd_regime_shift")
        );
    }

    #[test]
    fn test_policy_change_emits_evidence() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let _ = controller.evaluate_knob(
            PolicyKnob::BgCpuMax,
            2,
            &[CandidateAction::new(11, 3, 0.5, "increase")],
            PolicySignals::default(),
            true,
            10,
        );
        assert_eq!(controller.ledger().len(), 1);
    }

    #[test]
    fn test_evidence_entry_complete() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let _ = controller.evaluate_knob(
            PolicyKnob::BgCpuMax,
            2,
            &[CandidateAction::new(11, 3, 0.5, "increase")],
            PolicySignals::default(),
            true,
            10,
        );
        let entry = controller
            .ledger()
            .latest()
            .expect("ledger entry must exist");
        assert!(entry.decision_id > 0);
        assert_eq!(entry.knob, PolicyKnob::BgCpuMax);
        assert_eq!(entry.prior_setting, 2);
        assert_eq!(entry.chosen_candidate_id, Some(11));
        assert_eq!(entry.chosen_setting, 3);
        assert!(!entry.candidates.is_empty());
        assert!(entry.expected_losses.contains_key(&11));
    }

    #[test]
    fn test_evidence_auditable() {
        let mut ledger = PolicyEvidenceLedger::new(2);
        let first = PolicyEvidenceEntry {
            decision_id: 1,
            knob: PolicyKnob::RetryBackoffMs,
            prior_setting: 5,
            chosen_candidate_id: None,
            chosen_setting: 5,
            candidates: Vec::new(),
            expected_losses: BTreeMap::new(),
            top_evidence: Vec::new(),
            regime_id: 0,
        };
        let second = PolicyEvidenceEntry {
            decision_id: 2,
            knob: PolicyKnob::RetryBackoffMs,
            prior_setting: 5,
            chosen_candidate_id: Some(2),
            chosen_setting: 4,
            candidates: Vec::new(),
            expected_losses: BTreeMap::new(),
            top_evidence: Vec::new(),
            regime_id: 0,
        };
        ledger.record(first);
        ledger.record(second);
        let ids = ledger
            .entries()
            .iter()
            .map(|entry| entry.decision_id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_voi_schedules_high_value_monitors() {
        let schedule = schedule_monitors(
            &[
                MonitorSpec {
                    name: "cheap_high_voi".to_owned(),
                    expected_delta_loss: 10.0,
                    cost: 1.0,
                    correctness_critical: false,
                },
                MonitorSpec {
                    name: "low_voi".to_owned(),
                    expected_delta_loss: 1.5,
                    cost: 2.0,
                    correctness_critical: false,
                },
            ],
            2.0,
        );
        assert!(schedule.selected.contains(&"cheap_high_voi".to_owned()));
        assert!(!schedule.selected.contains(&"low_voi".to_owned()));
    }

    #[test]
    fn test_correctness_monitors_always_on() {
        let schedule = schedule_monitors(
            &[
                MonitorSpec {
                    name: "ssi_invariant".to_owned(),
                    expected_delta_loss: 0.0,
                    cost: 100.0,
                    correctness_critical: true,
                },
                MonitorSpec {
                    name: "optional".to_owned(),
                    expected_delta_loss: 4.0,
                    cost: 3.0,
                    correctness_critical: false,
                },
            ],
            0.0,
        );
        assert!(schedule.selected.contains(&"ssi_invariant".to_owned()));
    }

    #[test]
    fn test_voi_budget_constraint() {
        let schedule = schedule_monitors(
            &[
                MonitorSpec {
                    name: "m1".to_owned(),
                    expected_delta_loss: 8.0,
                    cost: 1.5,
                    correctness_critical: false,
                },
                MonitorSpec {
                    name: "m2".to_owned(),
                    expected_delta_loss: 6.0,
                    cost: 1.0,
                    correctness_critical: false,
                },
            ],
            2.0,
        );
        assert!(schedule.total_optional_cost <= 2.0 + f64::EPSILON);
    }

    #[test]
    fn test_policy_hysteresis_no_thrash() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 5, 32);
        let first = controller.evaluate_knob(
            PolicyKnob::BgCpuMax,
            2,
            &[CandidateAction::new(1, 3, 0.1, "up")],
            PolicySignals::default(),
            true,
            10,
        );
        assert!(first.changed);
        let second = controller.evaluate_knob(
            PolicyKnob::BgCpuMax,
            3,
            &[CandidateAction::new(2, 2, 0.05, "down immediately")],
            PolicySignals::default(),
            true,
            12,
        );
        assert_eq!(second.reason, DecisionReason::HysteresisSuppressed);
        assert!(!second.changed);
    }

    #[test]
    fn test_policy_interval_respected() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 3, 32);
        let _ = controller.evaluate_knob(
            PolicyKnob::RemoteMaxInFlight,
            2,
            &[CandidateAction::new(1, 4, 0.1, "up")],
            PolicySignals::default(),
            true,
            10,
        );
        let out = controller.evaluate_knob(
            PolicyKnob::RemoteMaxInFlight,
            4,
            &[CandidateAction::new(2, 3, 0.05, "down after interval")],
            PolicySignals::default(),
            true,
            13,
        );
        assert!(out.changed);
        assert_eq!(out.applied_setting, 3);
    }

    #[test]
    fn test_permits_not_threads() {
        let mut controller = PolicyController::new(AutoTunePragmaConfig::default(), 16, 2, 32);
        let _ = controller.evaluate_knob(
            PolicyKnob::BgCpuMax,
            2,
            &[CandidateAction::new(1, 4, 0.1, "bump permit cap")],
            PolicySignals::default(),
            true,
            10,
        );
        assert_eq!(controller.effective_limits().bg_cpu_max, 4);
    }

    #[test]
    fn test_lab_mode_deterministic_policy() {
        let config = AutoTunePragmaConfig::default();
        let mut first = PolicyController::new(config, 16, 2, 32);
        let mut second = PolicyController::new(config, 16, 2, 32);
        let candidates = vec![
            CandidateAction::new(3, 7, 3.1, "a"),
            CandidateAction::new(1, 5, 2.0, "b"),
            CandidateAction::new(2, 6, 2.0, "c"),
        ];
        let signals = PolicySignals {
            symbol_loss_rejects_h0: true,
            bocpd_regime_shift: true,
            regime_id: 41,
        };
        let out_a = first.evaluate_knob(
            PolicyKnob::RedundancyOverheadPercent,
            6,
            &candidates,
            signals,
            true,
            10,
        );
        let out_b = second.evaluate_knob(
            PolicyKnob::RedundancyOverheadPercent,
            6,
            &candidates,
            signals,
            true,
            10,
        );
        assert_eq!(out_a, out_b);
        assert_eq!(first.ledger(), second.ledger());
    }

    #[test]
    fn test_lab_mode_no_wall_clock() {
        let config = AutoTunePragmaConfig::default();
        let mut controller = PolicyController::new(config, 16, 2, 32);
        let out1 = controller.evaluate_knob(
            PolicyKnob::RetryBackoffMs,
            10,
            &[CandidateAction::new(1, 9, 0.2, "down")],
            PolicySignals::default(),
            true,
            100,
        );
        let out2 = controller.evaluate_knob(
            PolicyKnob::RetryBackoffMs,
            9,
            &[CandidateAction::new(2, 8, 0.1, "down")],
            PolicySignals::default(),
            true,
            101,
        );
        assert_eq!(out1.reason, DecisionReason::Applied(1));
        assert_eq!(out2.reason, DecisionReason::HysteresisSuppressed);
    }

    #[test]
    fn test_auto_tune_off_uses_defaults() {
        let config = AutoTunePragmaConfig {
            auto_tune: false,
            ..AutoTunePragmaConfig::default()
        };
        let mut controller = PolicyController::new(config, 32, 2, 32);
        let baseline = controller.effective_limits();
        let out = controller.evaluate_knob(
            PolicyKnob::BgCpuMax,
            baseline.bg_cpu_max,
            &[CandidateAction::new(
                1,
                baseline.bg_cpu_max + 1,
                0.0,
                "ignored",
            )],
            PolicySignals::default(),
            true,
            10,
        );
        assert_eq!(out.reason, DecisionReason::FallbackAutoTuneOff);
        assert_eq!(controller.effective_limits(), baseline);
    }

    #[test]
    fn test_missing_telemetry_falls_back() {
        let config = AutoTunePragmaConfig::default();
        let mut controller = PolicyController::new(config, 32, 2, 32);
        let baseline = controller.effective_limits();
        let out = controller.evaluate_knob(
            PolicyKnob::BgCpuMax,
            baseline.bg_cpu_max,
            &[CandidateAction::new(
                1,
                baseline.bg_cpu_max + 1,
                0.0,
                "ignored",
            )],
            PolicySignals::default(),
            false,
            10,
        );
        assert_eq!(out.reason, DecisionReason::FallbackTelemetryUnavailable);
        assert_eq!(controller.effective_limits(), baseline);
    }

    #[test]
    fn test_pragma_raptorq_overhead_bounds_clamped() {
        let mut state = RaptorQOverheadPragmaState::default();
        let low = state.set_default_percent_from_pragma(-100);
        let high = state.set_default_percent_from_pragma(9999);
        assert_eq!(low, MIN_OVERHEAD_PERCENT);
        assert_eq!(high, MAX_OVERHEAD_PERCENT);
    }

    #[test]
    fn test_pragma_raptorq_overhead_rounding_behavior() {
        let mut state = RaptorQOverheadPragmaState::default();
        let rounded = state.set_default_percent_from_basis_points(2_549);
        assert_eq!(rounded, 25);

        let budget = state.compute_budget_for_object(9, RepairObjectClass::PageHistory);
        // ceil(9 * 25 / 100) = ceil(2.25) = 3 (no small-K clamp at K=9).
        assert_eq!(budget.repair_count, 3);
        assert_eq!(budget.overhead_percent_applied, 25);
    }

    #[test]
    fn test_pragma_raptorq_overhead_small_k_no_underprovision() {
        let mut state = RaptorQOverheadPragmaState::default();
        state.set_default_percent_from_pragma(1);

        let budget = state.compute_budget_for_object(1, RepairObjectClass::PageHistory);
        assert!(budget.repair_count >= 3);
        assert!(!budget.underprovisioned);
        assert!(budget.small_k_clamped);
    }

    #[test]
    fn test_pragma_raptorq_overhead_per_object_override_and_exposure() {
        let mut state = RaptorQOverheadPragmaState::default();
        state.set_default_percent_from_pragma(20);
        state.set_object_override_from_pragma(RepairObjectClass::CommitMarker, 60);

        let marker_effective = state.effective_policy_for_object(RepairObjectClass::CommitMarker);
        let history_effective = state.effective_policy_for_object(RepairObjectClass::PageHistory);
        assert_eq!(marker_effective.overhead_percent, 60);
        assert_eq!(history_effective.overhead_percent, 20);

        let marker_budget = state.compute_budget_for_object(10, RepairObjectClass::CommitMarker);
        let history_budget = state.compute_budget_for_object(10, RepairObjectClass::PageHistory);
        assert_eq!(marker_budget.overhead_percent_applied, 60);
        assert_eq!(history_budget.overhead_percent_applied, 20);
        assert!(marker_budget.repair_count > history_budget.repair_count);
    }

    #[test]
    fn test_pragma_raptorq_overhead_persisted_metadata_is_versioned_and_deterministic() {
        let mut state = RaptorQOverheadPragmaState::default();
        state.set_default_percent_from_pragma(33);
        state.set_object_override_from_pragma(RepairObjectClass::SnapshotBlock, 41);
        let persisted = state.persist_connection_policy(9);

        assert_eq!(
            persisted.metadata_version,
            RAPTORQ_OVERHEAD_METADATA_VERSION
        );
        assert_eq!(persisted.policy_epoch, 9);
        assert_eq!(persisted.default_overhead_percent, 33);
        assert_eq!(
            persisted.override_percent_for_object(RepairObjectClass::SnapshotBlock),
            Some(41)
        );

        let effective = state.effective_policy_for_object(RepairObjectClass::SnapshotBlock);
        assert_eq!(effective.scope, OverheadPolicyScope::PersistedMetadata);
        assert_eq!(effective.policy_epoch, 9);
        assert_eq!(
            effective.metadata_version,
            RAPTORQ_OVERHEAD_METADATA_VERSION
        );
    }
}
