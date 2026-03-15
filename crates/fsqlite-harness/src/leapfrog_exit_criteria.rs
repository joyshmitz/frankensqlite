//! Explicit exit criteria for `bd-db300.7.3` leapfrog claims.
//!
//! This module turns the Track G3 decision record into a typed, machine-readable
//! contract that downstream verification and reporting code can consume without
//! reinterpreting prose.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// Bead identifier for the leapfrog exit-criteria gate.
pub const BEAD_ID: &str = "bd-db300.7.3";
/// Stable schema identifier for the exit-criteria contract.
pub const LEAPFROG_EXIT_CRITERIA_SCHEMA_V1: &str = "fsqlite-harness.leapfrog_exit_criteria.v1";
/// Stable schema identifier for the scorecard metric dictionary embedded in the
/// leapfrog contract.
pub const SCORECARD_METRIC_DICTIONARY_SCHEMA_V1: &str =
    "fsqlite-harness.scorecard_metric_dictionary.v1";
/// Stable schema identifier for the transferability rubric embedded in the
/// leapfrog contract.
pub const TRANSFERABILITY_RUBRIC_SCHEMA_V1: &str = "fsqlite-harness.transferability_rubric.v1";
/// Workspace-relative path to the canonical exit-criteria contract.
pub const LEAPFROG_EXIT_CRITERIA_PATH: &str = "leapfrog_exit_criteria.toml";
const CANONICAL_CAMPAIGN_MANIFEST_PATH: &str =
    "sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.json";

const REQUIRED_LOG_FIELDS: [&str; 8] = [
    "throughput_ratio_vs_sqlite",
    "retry_rate",
    "cpu_utilization_pct",
    "p50_latency_ratio_vs_sqlite",
    "p95_latency_ratio_vs_sqlite",
    "p99_latency_ratio_vs_sqlite",
    "responsiveness_regression_ratio_vs_sqlite",
    "topology_reassignments",
];
const REQUIRED_UNIT_TESTS: [&str; 5] = [
    "test_bd_db300_7_3_contract_schema_and_links",
    "test_bd_db300_7_3_required_campaign_surface_exists",
    "test_bd_db300_7_3_cell_targets_are_monotone",
    "test_bd_db300_7_3_verification_plan_is_actionable",
    "test_bd_db300_7_3_transferability_rubric_is_actionable",
];
const REQUIRED_CELL_SUFFIXES: [&str; 3] = ["c1", "c4", "c8"];
const REQUIRED_MODES: [&str; 3] = ["sqlite_reference", "fsqlite_mvcc", "fsqlite_single_writer"];
const REQUIRED_PLACEMENT_PROFILES: [&str; 3] = [
    "baseline_unpinned",
    "recommended_pinned",
    "adversarial_cross_node",
];
const REQUIRED_E2E_SCENARIOS: [&str; 9] = [
    "commutative_inserts_disjoint_keys_c1",
    "commutative_inserts_disjoint_keys_c4",
    "commutative_inserts_disjoint_keys_c8",
    "hot_page_contention_c1",
    "hot_page_contention_c4",
    "hot_page_contention_c8",
    "mixed_read_write_c1",
    "mixed_read_write_c4",
    "mixed_read_write_c8",
];
const REQUIRED_LOGGING_ARTIFACTS: [&str; 7] = [
    "artifacts/{bead_id}/{run_id}/events.jsonl",
    "artifacts/{bead_id}/{run_id}/manifest.json",
    "artifacts/{bead_id}/{run_id}/summary.md",
    "artifacts/{bead_id}/{run_id}/metric_dictionary.json",
    "artifacts/{bead_id}/{run_id}/cell_metrics.jsonl",
    "artifacts/{bead_id}/{run_id}/retry_report.json",
    "artifacts/{bead_id}/{run_id}/topology.json",
];
const REQUIRED_METRIC_FAMILIES: [&str; 12] = [
    "throughput",
    "retry",
    "abort",
    "cpu_efficiency",
    "latency",
    "topology",
    "wait",
    "page_touch",
    "split_path",
    "allocator",
    "cache",
    "copy_allocation",
];
const REQUIRED_TRANSFERABILITY_CLASSES: [&str; 4] = [
    "transferable",
    "profile_specific_but_useful",
    "suspicious",
    "non_claimable",
];
const REQUIRED_HARDWARE_CLASSES: [&str; 3] =
    ["same_host", "same_topology_class", "cross_hardware_class"];
const REQUIRED_RUBRIC_DOWNSTREAM_BEADS: [&str; 2] = ["bd-db300.7.3", "bd-db300.7.4"];

/// Typed decision record for Track G3 leapfrog claims.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeapfrogExitCriteria {
    pub meta: ExitCriteriaMeta,
    pub campaign: CampaignRequirements,
    pub scorecard: ScorecardPolicy,
    pub metric_dictionary: ScorecardMetricDictionary,
    pub transferability_rubric: TransferabilityRubric,
    pub cell_gates: Vec<CellGate>,
    pub verification_plan: VerificationPlan,
    pub references: ExitCriteriaReferences,
}

/// Metadata for the canonical exit-criteria contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitCriteriaMeta {
    pub schema_version: String,
    pub policy_id: String,
    pub bead_id: String,
    pub track_id: String,
    pub generated_at: String,
    pub owner: String,
    pub decision_summary: String,
    pub rationale: String,
}

/// Benchmark campaign surface that the exit criteria bind to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignRequirements {
    pub manifest_path: String,
    pub required_modes: Vec<String>,
    pub required_placement_profiles: Vec<String>,
    pub required_cell_suffixes: Vec<String>,
}

/// Claim-language and placement-profile policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardPolicy {
    pub recommended_profile_id: String,
    pub baseline_profile_id: String,
    pub adversarial_profile_id: String,
    pub claim_language: String,
    pub claim_forbidden_when_any_fail: bool,
}

/// Canonical metric dictionary for db300 scorecard and gate consumers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardMetricDictionary {
    pub schema_version: String,
    pub dictionary_id: String,
    pub notes: String,
    pub metrics: Vec<ScorecardMetricDefinition>,
}

/// One machine-readable metric definition used by scorecards and gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardMetricDefinition {
    pub metric_id: String,
    pub label: String,
    pub family: String,
    pub availability: String,
    pub unit: String,
    pub aggregation: String,
    pub collection_artifact: String,
    pub collection_field: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation: Option<String>,
    pub comparability: String,
    pub required_for_claim: bool,
    pub zero_semantics: String,
    pub missing_semantics: String,
    pub suppressed_semantics: String,
    pub not_applicable_semantics: String,
}

/// Machine-readable rubric for classifying whether a measured win is broadly
/// transferable, narrowly useful, suspicious, or non-claimable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferabilityRubric {
    pub schema_version: String,
    pub rubric_id: String,
    pub required_modes: Vec<String>,
    pub required_hardware_classes: Vec<String>,
    pub classification_order: Vec<String>,
    pub downstream_beads: Vec<String>,
    pub single_writer_role: String,
    pub classes: Vec<TransferabilityClass>,
}

/// One transferability class with the evidence and reporting rules future
/// agents must apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferabilityClass {
    pub classification_id: String,
    pub final_report_label: String,
    pub claimable: bool,
    pub requires_no_catastrophic_regression: bool,
    pub minimum_hardware_evidence: String,
    pub summary: String,
    pub placement_rule: String,
    pub mode_rule: String,
    pub hardware_rule: String,
    pub reporting_requirement: String,
    pub example_profiles: Vec<String>,
    pub example: String,
}

/// Explicit gate for one canonical concurrency cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellGate {
    pub cell: String,
    pub goal: String,
    pub recommended_min_throughput_ratio_vs_sqlite: f64,
    pub baseline_catastrophic_floor_ratio_vs_sqlite: f64,
    pub adversarial_catastrophic_floor_ratio_vs_sqlite: f64,
    pub max_retry_rate: f64,
    pub min_cpu_utilization_pct: f64,
    pub max_p50_latency_ratio_vs_sqlite: f64,
    pub max_p95_latency_ratio_vs_sqlite: f64,
    pub max_p99_latency_ratio_vs_sqlite: f64,
    pub max_responsiveness_regression_ratio_vs_sqlite: f64,
    pub max_topology_reassignments_per_run: u32,
}

/// Follow-on verification obligations that must remain attached to the gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationPlan {
    pub unit_tests: Vec<String>,
    pub e2e_scenarios: Vec<String>,
    pub logging_artifacts: Vec<String>,
    pub required_log_fields: Vec<String>,
}

/// Canonical files that downstream tooling should consume or invoke.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExitCriteriaReferences {
    pub contract_module: String,
    pub verification_test: String,
    pub campaign_matrix_module: String,
    pub score_engine_module: String,
    pub confidence_gates_module: String,
    pub release_certificate_module: String,
    pub verifier_script: String,
}

impl LeapfrogExitCriteria {
    /// Load the canonical contract from the workspace root.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the contract cannot be read or parsed.
    pub fn load_from_workspace_root(workspace_root: &Path) -> Result<Self, String> {
        let path = workspace_root.join(LEAPFROG_EXIT_CRITERIA_PATH);
        Self::load_from_path(&path)
    }

    /// Load the contract from an explicit path.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file cannot be read or parsed.
    pub fn load_from_path(path: &Path) -> Result<Self, String> {
        let raw = fs::read_to_string(path).map_err(|error| {
            format!(
                "leapfrog_exit_criteria_read_failed path={} error={error}",
                path.display()
            )
        })?;
        toml::from_str(&raw).map_err(|error| {
            format!(
                "leapfrog_exit_criteria_parse_failed path={} error={error}",
                path.display()
            )
        })
    }

    /// Return the gate definition for a canonical cell suffix like `c1`.
    #[must_use]
    pub fn cell_gate(&self, cell: &str) -> Option<&CellGate> {
        self.cell_gates.iter().find(|gate| gate.cell == cell)
    }

    /// Validate the contract against internal invariants and the campaign
    /// manifest it references.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the contract is malformed or references missing
    /// campaign/reporting inputs.
    pub fn validate(&self, workspace_root: &Path) -> Result<(), String> {
        require_eq(
            "meta.schema_version",
            &self.meta.schema_version,
            LEAPFROG_EXIT_CRITERIA_SCHEMA_V1,
        )?;
        require_eq("meta.bead_id", &self.meta.bead_id, BEAD_ID)?;
        require_non_empty("meta.policy_id", &self.meta.policy_id)?;
        require_non_empty("meta.track_id", &self.meta.track_id)?;
        require_non_empty("meta.generated_at", &self.meta.generated_at)?;
        require_non_empty("meta.owner", &self.meta.owner)?;
        require_non_empty("meta.decision_summary", &self.meta.decision_summary)?;
        require_non_empty("meta.rationale", &self.meta.rationale)?;

        require_non_empty(
            "scorecard.recommended_profile_id",
            &self.scorecard.recommended_profile_id,
        )?;
        require_non_empty(
            "scorecard.baseline_profile_id",
            &self.scorecard.baseline_profile_id,
        )?;
        require_non_empty(
            "scorecard.adversarial_profile_id",
            &self.scorecard.adversarial_profile_id,
        )?;
        require_non_empty("scorecard.claim_language", &self.scorecard.claim_language)?;
        if !self.scorecard.claim_forbidden_when_any_fail {
            return Err("scorecard.claim_forbidden_when_any_fail must be true".to_owned());
        }
        require_eq(
            "metric_dictionary.schema_version",
            &self.metric_dictionary.schema_version,
            SCORECARD_METRIC_DICTIONARY_SCHEMA_V1,
        )?;
        require_non_empty(
            "metric_dictionary.dictionary_id",
            &self.metric_dictionary.dictionary_id,
        )?;
        require_non_empty("metric_dictionary.notes", &self.metric_dictionary.notes)?;
        require_eq(
            "transferability_rubric.schema_version",
            &self.transferability_rubric.schema_version,
            TRANSFERABILITY_RUBRIC_SCHEMA_V1,
        )?;
        require_non_empty(
            "transferability_rubric.rubric_id",
            &self.transferability_rubric.rubric_id,
        )?;
        require_non_empty(
            "transferability_rubric.single_writer_role",
            &self.transferability_rubric.single_writer_role,
        )?;

        let required_profile_ids = unique_set(
            "campaign.required_placement_profiles",
            &self.campaign.required_placement_profiles,
        )?;
        let required_mode_ids =
            unique_set("campaign.required_modes", &self.campaign.required_modes)?;
        let required_cell_suffixes = unique_set(
            "campaign.required_cell_suffixes",
            &self.campaign.required_cell_suffixes,
        )?;
        require_eq(
            "campaign.manifest_path",
            &self.campaign.manifest_path,
            CANONICAL_CAMPAIGN_MANIFEST_PATH,
        )?;
        if required_mode_ids
            != REQUIRED_MODES
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "campaign.required_modes mismatch actual={required_mode_ids:?} expected={:?}",
                REQUIRED_MODES
            ));
        }
        if required_profile_ids
            != REQUIRED_PLACEMENT_PROFILES
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "campaign.required_placement_profiles mismatch actual={required_profile_ids:?} expected={:?}",
                REQUIRED_PLACEMENT_PROFILES
            ));
        }
        if required_cell_suffixes
            != REQUIRED_CELL_SUFFIXES
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "campaign.required_cell_suffixes mismatch actual={required_cell_suffixes:?} expected={:?}",
                REQUIRED_CELL_SUFFIXES
            ));
        }
        let rubric_mode_ids = unique_set(
            "transferability_rubric.required_modes",
            &self.transferability_rubric.required_modes,
        )?;
        if rubric_mode_ids != required_mode_ids {
            return Err(format!(
                "transferability_rubric.required_modes mismatch actual={rubric_mode_ids:?} expected={required_mode_ids:?}"
            ));
        }
        let rubric_hardware_classes = unique_set(
            "transferability_rubric.required_hardware_classes",
            &self.transferability_rubric.required_hardware_classes,
        )?;
        if rubric_hardware_classes
            != REQUIRED_HARDWARE_CLASSES
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "transferability_rubric.required_hardware_classes mismatch actual={rubric_hardware_classes:?} expected={:?}",
                REQUIRED_HARDWARE_CLASSES
            ));
        }
        let rubric_classification_order = unique_set(
            "transferability_rubric.classification_order",
            &self.transferability_rubric.classification_order,
        )?;
        if rubric_classification_order
            != REQUIRED_TRANSFERABILITY_CLASSES
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "transferability_rubric.classification_order mismatch actual={rubric_classification_order:?} expected={:?}",
                REQUIRED_TRANSFERABILITY_CLASSES
            ));
        }
        if self
            .transferability_rubric
            .classification_order
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != REQUIRED_TRANSFERABILITY_CLASSES
        {
            return Err(format!(
                "transferability_rubric.classification_order order mismatch actual={:?} expected={:?}",
                self.transferability_rubric
                    .classification_order
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
                REQUIRED_TRANSFERABILITY_CLASSES
            ));
        }
        let rubric_downstream_beads = unique_set(
            "transferability_rubric.downstream_beads",
            &self.transferability_rubric.downstream_beads,
        )?;
        if rubric_downstream_beads
            != REQUIRED_RUBRIC_DOWNSTREAM_BEADS
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "transferability_rubric.downstream_beads mismatch actual={rubric_downstream_beads:?} expected={:?}",
                REQUIRED_RUBRIC_DOWNSTREAM_BEADS
            ));
        }
        require_eq(
            "transferability_rubric.single_writer_role",
            &self.transferability_rubric.single_writer_role,
            "comparison_or_fallback_only",
        )?;

        for profile_id in [
            &self.scorecard.recommended_profile_id,
            &self.scorecard.baseline_profile_id,
            &self.scorecard.adversarial_profile_id,
        ] {
            if !required_profile_ids.contains(profile_id.as_str()) {
                return Err(format!(
                    "scorecard profile `{profile_id}` is not listed in campaign.required_placement_profiles"
                ));
            }
        }

        let mut seen_cells = BTreeSet::new();
        for gate in &self.cell_gates {
            if !seen_cells.insert(gate.cell.as_str()) {
                return Err(format!("duplicate cell gate `{}`", gate.cell));
            }
            if !required_cell_suffixes.contains(gate.cell.as_str()) {
                return Err(format!(
                    "cell gate `{}` is not listed in campaign.required_cell_suffixes",
                    gate.cell
                ));
            }
            require_non_empty("cell_gates.goal", &gate.goal)?;
            require_ratio_at_least_one(
                "cell_gates.recommended_min_throughput_ratio_vs_sqlite",
                gate.recommended_min_throughput_ratio_vs_sqlite,
            )?;
            require_ratio_positive(
                "cell_gates.baseline_catastrophic_floor_ratio_vs_sqlite",
                gate.baseline_catastrophic_floor_ratio_vs_sqlite,
            )?;
            require_ratio_positive(
                "cell_gates.adversarial_catastrophic_floor_ratio_vs_sqlite",
                gate.adversarial_catastrophic_floor_ratio_vs_sqlite,
            )?;
            if gate.baseline_catastrophic_floor_ratio_vs_sqlite
                > gate.recommended_min_throughput_ratio_vs_sqlite
            {
                return Err(format!(
                    "cell gate `{}` has baseline catastrophic floor above recommended target",
                    gate.cell
                ));
            }
            if gate.adversarial_catastrophic_floor_ratio_vs_sqlite
                > gate.baseline_catastrophic_floor_ratio_vs_sqlite
            {
                return Err(format!(
                    "cell gate `{}` has adversarial catastrophic floor above baseline catastrophic floor",
                    gate.cell
                ));
            }
            require_pct("cell_gates.max_retry_rate", gate.max_retry_rate, 0.0, 1.0)?;
            require_pct(
                "cell_gates.min_cpu_utilization_pct",
                gate.min_cpu_utilization_pct,
                0.0,
                100.0,
            )?;
            require_ratio_at_least_one(
                "cell_gates.max_p50_latency_ratio_vs_sqlite",
                gate.max_p50_latency_ratio_vs_sqlite,
            )?;
            require_ratio_at_least_one(
                "cell_gates.max_p95_latency_ratio_vs_sqlite",
                gate.max_p95_latency_ratio_vs_sqlite,
            )?;
            require_ratio_at_least_one(
                "cell_gates.max_p99_latency_ratio_vs_sqlite",
                gate.max_p99_latency_ratio_vs_sqlite,
            )?;
            require_ratio_at_least_one(
                "cell_gates.max_responsiveness_regression_ratio_vs_sqlite",
                gate.max_responsiveness_regression_ratio_vs_sqlite,
            )?;
            if gate.max_topology_reassignments_per_run == 0 {
                return Err(format!(
                    "cell gate `{}` must allow at least one topology-reassignment slot",
                    gate.cell
                ));
            }
        }
        if self
            .cell_gates
            .iter()
            .map(|gate| gate.cell.as_str())
            .collect::<Vec<_>>()
            != REQUIRED_CELL_SUFFIXES
        {
            return Err(format!(
                "cell_gates order mismatch actual={:?} expected={:?}",
                self.cell_gates
                    .iter()
                    .map(|gate| gate.cell.as_str())
                    .collect::<Vec<_>>(),
                REQUIRED_CELL_SUFFIXES
            ));
        }

        if seen_cells.len() != required_cell_suffixes.len() {
            return Err(format!(
                "cell gate count mismatch actual={} expected={}",
                seen_cells.len(),
                required_cell_suffixes.len()
            ));
        }

        let unit_tests = unique_set(
            "verification_plan.unit_tests",
            &self.verification_plan.unit_tests,
        )?;
        let e2e_scenarios = unique_set(
            "verification_plan.e2e_scenarios",
            &self.verification_plan.e2e_scenarios,
        )?;
        let logging_artifacts = unique_set(
            "verification_plan.logging_artifacts",
            &self.verification_plan.logging_artifacts,
        )?;
        let log_fields = unique_set(
            "verification_plan.required_log_fields",
            &self.verification_plan.required_log_fields,
        )?;
        let metric_ids = unique_metric_ids(&self.metric_dictionary.metrics)?;
        if unit_tests.is_empty() || e2e_scenarios.is_empty() || logging_artifacts.is_empty() {
            return Err(
                "verification_plan must list unit tests, scenarios, and artifacts".to_owned(),
            );
        }
        if unit_tests
            != REQUIRED_UNIT_TESTS
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "verification_plan.unit_tests mismatch actual={unit_tests:?} expected={:?}",
                REQUIRED_UNIT_TESTS
            ));
        }
        if e2e_scenarios
            != REQUIRED_E2E_SCENARIOS
                .into_iter()
                .map(ToOwned::to_owned)
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "verification_plan.e2e_scenarios mismatch actual={e2e_scenarios:?} expected={:?}",
                REQUIRED_E2E_SCENARIOS
            ));
        }
        for required_field in REQUIRED_LOG_FIELDS {
            if !log_fields.contains(required_field) {
                return Err(format!(
                    "verification_plan.required_log_fields missing `{required_field}`"
                ));
            }
            if !metric_ids.contains(required_field) {
                return Err(format!(
                    "metric_dictionary.metrics missing required log metric `{required_field}`"
                ));
            }
        }
        for log_field in &log_fields {
            if !metric_ids.contains(log_field.as_str()) {
                return Err(format!(
                    "verification_plan.required_log_fields references undefined metric `{log_field}`"
                ));
            }
        }
        for required_artifact in REQUIRED_LOGGING_ARTIFACTS {
            if !logging_artifacts.contains(required_artifact) {
                return Err(format!(
                    "verification_plan.logging_artifacts missing `{required_artifact}`"
                ));
            }
        }

        let mut metric_families = BTreeSet::new();
        for metric in &self.metric_dictionary.metrics {
            require_non_empty("metric_dictionary.metrics.metric_id", &metric.metric_id)?;
            require_non_empty("metric_dictionary.metrics.label", &metric.label)?;
            require_non_empty("metric_dictionary.metrics.family", &metric.family)?;
            require_metric_family(&metric.family)?;
            metric_families.insert(metric.family.as_str());
            require_non_empty(
                "metric_dictionary.metrics.availability",
                &metric.availability,
            )?;
            require_metric_availability(&metric.availability)?;
            require_non_empty("metric_dictionary.metrics.unit", &metric.unit)?;
            require_non_empty("metric_dictionary.metrics.aggregation", &metric.aggregation)?;
            require_non_empty(
                "metric_dictionary.metrics.collection_artifact",
                &metric.collection_artifact,
            )?;
            if !logging_artifacts.contains(metric.collection_artifact.as_str()) {
                return Err(format!(
                    "metric `{}` references unknown collection artifact `{}`",
                    metric.metric_id, metric.collection_artifact
                ));
            }
            require_non_empty(
                "metric_dictionary.metrics.collection_field",
                &metric.collection_field,
            )?;
            if let Some(derivation) = &metric.derivation {
                require_non_empty("metric_dictionary.metrics.derivation", derivation)?;
            }
            require_non_empty(
                "metric_dictionary.metrics.comparability",
                &metric.comparability,
            )?;
            require_metric_comparability(&metric.comparability)?;
            require_non_empty(
                "metric_dictionary.metrics.zero_semantics",
                &metric.zero_semantics,
            )?;
            require_non_empty(
                "metric_dictionary.metrics.missing_semantics",
                &metric.missing_semantics,
            )?;
            require_non_empty(
                "metric_dictionary.metrics.suppressed_semantics",
                &metric.suppressed_semantics,
            )?;
            require_non_empty(
                "metric_dictionary.metrics.not_applicable_semantics",
                &metric.not_applicable_semantics,
            )?;
            if metric.required_for_claim && !log_fields.contains(metric.metric_id.as_str()) {
                return Err(format!(
                    "metric `{}` is marked required_for_claim but is absent from verification_plan.required_log_fields",
                    metric.metric_id
                ));
            }
        }
        for required_family in REQUIRED_METRIC_FAMILIES {
            if !metric_families.contains(required_family) {
                return Err(format!(
                    "metric_dictionary.metrics missing required family `{required_family}`"
                ));
            }
        }
        if self.transferability_rubric.classes.len() != REQUIRED_TRANSFERABILITY_CLASSES.len() {
            return Err(format!(
                "transferability_rubric.classes count mismatch actual={} expected={}",
                self.transferability_rubric.classes.len(),
                REQUIRED_TRANSFERABILITY_CLASSES.len()
            ));
        }
        let mut seen_rubric_classes = BTreeSet::new();
        let mut rubric_example_profiles = BTreeSet::new();
        for class in &self.transferability_rubric.classes {
            require_non_empty(
                "transferability_rubric.classes.classification_id",
                &class.classification_id,
            )?;
            if !seen_rubric_classes.insert(class.classification_id.as_str()) {
                return Err(format!(
                    "transferability_rubric.classes contains duplicate classification_id `{}`",
                    class.classification_id
                ));
            }
            require_non_empty(
                "transferability_rubric.classes.final_report_label",
                &class.final_report_label,
            )?;
            require_non_empty("transferability_rubric.classes.summary", &class.summary)?;
            require_non_empty(
                "transferability_rubric.classes.placement_rule",
                &class.placement_rule,
            )?;
            require_non_empty("transferability_rubric.classes.mode_rule", &class.mode_rule)?;
            require_non_empty(
                "transferability_rubric.classes.hardware_rule",
                &class.hardware_rule,
            )?;
            require_non_empty(
                "transferability_rubric.classes.reporting_requirement",
                &class.reporting_requirement,
            )?;
            require_non_empty("transferability_rubric.classes.example", &class.example)?;
            require_hardware_evidence(&class.minimum_hardware_evidence)?;
            let expected_label = required_transferability_report_label(&class.classification_id)
                .ok_or_else(|| {
                    format!(
                        "unexpected transferability class `{}`",
                        class.classification_id
                    )
                })?;
            require_eq(
                "transferability_rubric.classes.final_report_label",
                &class.final_report_label,
                expected_label,
            )?;
            if class.claimable
                != required_transferability_claimable(&class.classification_id).ok_or_else(
                    || {
                        format!(
                            "unexpected transferability class `{}`",
                            class.classification_id
                        )
                    },
                )?
            {
                return Err(format!(
                    "transferability class `{}` claimable mismatch actual={} expected={}",
                    class.classification_id,
                    class.claimable,
                    required_transferability_claimable(&class.classification_id).unwrap_or(false)
                ));
            }
            if class.requires_no_catastrophic_regression
                != required_transferability_no_catastrophic_regression(&class.classification_id)
                    .ok_or_else(|| {
                        format!(
                            "unexpected transferability class `{}`",
                            class.classification_id
                        )
                    })?
            {
                return Err(format!(
                    "transferability class `{}` requires_no_catastrophic_regression mismatch actual={} expected={}",
                    class.classification_id,
                    class.requires_no_catastrophic_regression,
                    required_transferability_no_catastrophic_regression(&class.classification_id)
                        .unwrap_or(false)
                ));
            }
            let expected_minimum_hardware_evidence =
                required_transferability_minimum_hardware_evidence(&class.classification_id)
                    .ok_or_else(|| {
                        format!(
                            "unexpected transferability class `{}`",
                            class.classification_id
                        )
                    })?;
            require_eq(
                "transferability_rubric.classes.minimum_hardware_evidence",
                &class.minimum_hardware_evidence,
                expected_minimum_hardware_evidence,
            )?;
            let example_profiles = unique_set(
                "transferability_rubric.classes.example_profiles",
                &class.example_profiles,
            )?;
            if example_profiles.is_empty() {
                return Err(format!(
                    "transferability class `{}` must reference at least one example profile",
                    class.classification_id
                ));
            }
            for profile_id in &example_profiles {
                if !required_profile_ids.contains(profile_id.as_str()) {
                    return Err(format!(
                        "transferability class `{}` references unknown example profile `{profile_id}`",
                        class.classification_id
                    ));
                }
                rubric_example_profiles.insert(profile_id.clone());
            }
        }
        if seen_rubric_classes
            != REQUIRED_TRANSFERABILITY_CLASSES
                .into_iter()
                .collect::<BTreeSet<_>>()
        {
            return Err(format!(
                "transferability_rubric.classes mismatch actual={seen_rubric_classes:?} expected={:?}",
                REQUIRED_TRANSFERABILITY_CLASSES
            ));
        }
        if self
            .transferability_rubric
            .classes
            .iter()
            .map(|class| class.classification_id.as_str())
            .collect::<Vec<_>>()
            != REQUIRED_TRANSFERABILITY_CLASSES
        {
            return Err(format!(
                "transferability_rubric.classes order mismatch actual={:?} expected={:?}",
                self.transferability_rubric
                    .classes
                    .iter()
                    .map(|class| class.classification_id.as_str())
                    .collect::<Vec<_>>(),
                REQUIRED_TRANSFERABILITY_CLASSES
            ));
        }
        if rubric_example_profiles != required_profile_ids {
            return Err(format!(
                "transferability_rubric example profile coverage mismatch actual={rubric_example_profiles:?} expected={required_profile_ids:?}"
            ));
        }

        for path in [
            &self.references.contract_module,
            &self.references.verification_test,
            &self.references.campaign_matrix_module,
            &self.references.score_engine_module,
            &self.references.confidence_gates_module,
            &self.references.release_certificate_module,
            &self.references.verifier_script,
        ] {
            require_non_empty("references.path", path)?;
            let resolved = workspace_root.join(path);
            if !resolved.exists() {
                return Err(format!(
                    "referenced path does not exist path={}",
                    resolved.display()
                ));
            }
        }

        for mode_id in &required_mode_ids {
            if !matches!(
                mode_id.as_str(),
                "sqlite_reference" | "fsqlite_mvcc" | "fsqlite_single_writer"
            ) {
                return Err(format!("unexpected required mode `{mode_id}`"));
            }
        }

        Ok(())
    }
}

fn require_non_empty(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    Ok(())
}

fn require_eq(field: &str, value: &str, expected: &str) -> Result<(), String> {
    if value != expected {
        return Err(format!("{field} expected `{expected}` but found `{value}`"));
    }
    Ok(())
}

fn require_ratio_positive(field: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("{field} must be finite and > 0, found {value}"));
    }
    Ok(())
}

fn require_ratio_at_least_one(field: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value < 1.0 {
        return Err(format!("{field} must be finite and >= 1.0, found {value}"));
    }
    Ok(())
}

fn require_pct(field: &str, value: f64, min: f64, max: f64) -> Result<(), String> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(format!(
            "{field} must be finite and in [{min}, {max}], found {value}"
        ));
    }
    Ok(())
}

fn unique_set(field: &str, values: &[String]) -> Result<BTreeSet<String>, String> {
    let mut seen = BTreeSet::new();
    for value in values {
        require_non_empty(field, value)?;
        if !seen.insert(value.clone()) {
            return Err(format!("{field} contains duplicate value `{value}`"));
        }
    }
    Ok(seen)
}

fn unique_metric_ids(metrics: &[ScorecardMetricDefinition]) -> Result<BTreeSet<String>, String> {
    let mut seen = BTreeSet::new();
    if metrics.is_empty() {
        return Err("metric_dictionary.metrics must not be empty".to_owned());
    }
    for metric in metrics {
        require_non_empty("metric_dictionary.metrics.metric_id", &metric.metric_id)?;
        if !seen.insert(metric.metric_id.clone()) {
            return Err(format!(
                "metric_dictionary.metrics contains duplicate metric_id `{}`",
                metric.metric_id
            ));
        }
    }
    Ok(seen)
}

fn require_metric_family(value: &str) -> Result<(), String> {
    if matches!(
        value,
        "throughput"
            | "retry"
            | "abort"
            | "cpu_efficiency"
            | "latency"
            | "topology"
            | "wait"
            | "page_touch"
            | "split_path"
            | "allocator"
            | "cache"
            | "copy_allocation"
    ) {
        Ok(())
    } else {
        Err(format!("unsupported metric family `{value}`"))
    }
}

fn require_metric_availability(value: &str) -> Result<(), String> {
    if matches!(
        value,
        "required_artifact_field" | "supporting_artifact_field" | "planned_follow_on"
    ) {
        Ok(())
    } else {
        Err(format!("unsupported metric availability `{value}`"))
    }
}

fn require_metric_comparability(value: &str) -> Result<(), String> {
    if matches!(
        value,
        "cross_mode_comparable" | "mode_specific" | "topology_sensitive" | "advisory_only"
    ) {
        Ok(())
    } else {
        Err(format!("unsupported metric comparability `{value}`"))
    }
}

fn require_hardware_evidence(value: &str) -> Result<(), String> {
    if matches!(
        value,
        "same_host" | "same_topology_class" | "cross_hardware_class"
    ) {
        Ok(())
    } else {
        Err(format!("unsupported minimum hardware evidence `{value}`"))
    }
}

fn required_transferability_report_label(classification_id: &str) -> Option<&'static str> {
    match classification_id {
        "transferable" => Some("transferable win"),
        "profile_specific_but_useful" => Some("lab-specific win"),
        "suspicious" => Some("topology-sensitive win"),
        "non_claimable" => Some("no-catastrophic-regression failure"),
        _ => None,
    }
}

fn required_transferability_claimable(classification_id: &str) -> Option<bool> {
    match classification_id {
        "transferable" => Some(true),
        "profile_specific_but_useful" | "suspicious" | "non_claimable" => Some(false),
        _ => None,
    }
}

fn required_transferability_no_catastrophic_regression(classification_id: &str) -> Option<bool> {
    match classification_id {
        "transferable" | "profile_specific_but_useful" | "suspicious" => Some(true),
        "non_claimable" => Some(false),
        _ => None,
    }
}

fn required_transferability_minimum_hardware_evidence(
    classification_id: &str,
) -> Option<&'static str> {
    match classification_id {
        "transferable" => Some("same_topology_class"),
        "profile_specific_but_useful" | "suspicious" | "non_claimable" => Some("same_host"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_gate_lookup_returns_expected_gate() {
        let criteria = LeapfrogExitCriteria {
            meta: ExitCriteriaMeta {
                schema_version: LEAPFROG_EXIT_CRITERIA_SCHEMA_V1.to_owned(),
                policy_id: "test".to_owned(),
                bead_id: BEAD_ID.to_owned(),
                track_id: "bd-db300.7".to_owned(),
                generated_at: "2026-03-10".to_owned(),
                owner: "tester".to_owned(),
                decision_summary: "summary".to_owned(),
                rationale: "rationale".to_owned(),
            },
            campaign: CampaignRequirements {
                manifest_path: "manifest.json".to_owned(),
                required_modes: vec!["sqlite_reference".to_owned()],
                required_placement_profiles: vec!["recommended_pinned".to_owned()],
                required_cell_suffixes: vec!["c1".to_owned()],
            },
            scorecard: ScorecardPolicy {
                recommended_profile_id: "recommended_pinned".to_owned(),
                baseline_profile_id: "recommended_pinned".to_owned(),
                adversarial_profile_id: "recommended_pinned".to_owned(),
                claim_language: "claim".to_owned(),
                claim_forbidden_when_any_fail: true,
            },
            metric_dictionary: ScorecardMetricDictionary {
                schema_version: SCORECARD_METRIC_DICTIONARY_SCHEMA_V1.to_owned(),
                dictionary_id: "db300-test-metrics.v1".to_owned(),
                notes: "test".to_owned(),
                metrics: vec![ScorecardMetricDefinition {
                    metric_id: "throughput_ratio_vs_sqlite".to_owned(),
                    label: "Throughput ratio vs SQLite".to_owned(),
                    family: "throughput".to_owned(),
                    availability: "required_artifact_field".to_owned(),
                    unit: "ratio".to_owned(),
                    aggregation: "gate against per-cell median".to_owned(),
                    collection_artifact: "artifacts/{bead_id}/{run_id}/cell_metrics.jsonl"
                        .to_owned(),
                    collection_field: "throughput_ratio_vs_sqlite".to_owned(),
                    derivation: Some("candidate / sqlite_reference".to_owned()),
                    comparability: "cross_mode_comparable".to_owned(),
                    required_for_claim: true,
                    zero_semantics: "collapse".to_owned(),
                    missing_semantics: "fail".to_owned(),
                    suppressed_semantics: "never".to_owned(),
                    not_applicable_semantics: "profiling-only".to_owned(),
                }],
            },
            transferability_rubric: TransferabilityRubric {
                schema_version: TRANSFERABILITY_RUBRIC_SCHEMA_V1.to_owned(),
                rubric_id: "db300-transferability-rubric.v1".to_owned(),
                required_modes: REQUIRED_MODES.iter().map(|id| (*id).to_owned()).collect(),
                required_hardware_classes: REQUIRED_HARDWARE_CLASSES
                    .iter()
                    .map(|id| (*id).to_owned())
                    .collect(),
                classification_order: REQUIRED_TRANSFERABILITY_CLASSES
                    .iter()
                    .map(|id| (*id).to_owned())
                    .collect(),
                downstream_beads: REQUIRED_RUBRIC_DOWNSTREAM_BEADS
                    .iter()
                    .map(|id| (*id).to_owned())
                    .collect(),
                single_writer_role: "comparison_or_fallback_only".to_owned(),
                classes: vec![TransferabilityClass {
                    classification_id: "transferable".to_owned(),
                    final_report_label: "transferable win".to_owned(),
                    claimable: true,
                    requires_no_catastrophic_regression: true,
                    minimum_hardware_evidence: "same_topology_class".to_owned(),
                    summary: "summary".to_owned(),
                    placement_rule: "placement".to_owned(),
                    mode_rule: "mode".to_owned(),
                    hardware_rule: "hardware".to_owned(),
                    reporting_requirement: "report".to_owned(),
                    example_profiles: vec![
                        "recommended_pinned".to_owned(),
                        "baseline_unpinned".to_owned(),
                        "adversarial_cross_node".to_owned(),
                    ],
                    example: "example".to_owned(),
                }],
            },
            cell_gates: vec![CellGate {
                cell: "c1".to_owned(),
                goal: "goal".to_owned(),
                recommended_min_throughput_ratio_vs_sqlite: 1.05,
                baseline_catastrophic_floor_ratio_vs_sqlite: 0.95,
                adversarial_catastrophic_floor_ratio_vs_sqlite: 0.90,
                max_retry_rate: 0.01,
                min_cpu_utilization_pct: 45.0,
                max_p50_latency_ratio_vs_sqlite: 1.10,
                max_p95_latency_ratio_vs_sqlite: 1.20,
                max_p99_latency_ratio_vs_sqlite: 1.30,
                max_responsiveness_regression_ratio_vs_sqlite: 1.15,
                max_topology_reassignments_per_run: 2,
            }],
            verification_plan: VerificationPlan {
                unit_tests: vec!["test".to_owned()],
                e2e_scenarios: vec!["scenario".to_owned()],
                logging_artifacts: vec![
                    "artifact".to_owned(),
                    "artifacts/{bead_id}/{run_id}/cell_metrics.jsonl".to_owned(),
                    "artifacts/{bead_id}/{run_id}/metric_dictionary.json".to_owned(),
                ],
                required_log_fields: REQUIRED_LOG_FIELDS
                    .iter()
                    .map(|field| (*field).to_owned())
                    .collect(),
            },
            references: ExitCriteriaReferences {
                contract_module: "contract".to_owned(),
                verification_test: "test".to_owned(),
                campaign_matrix_module: "fixture_select".to_owned(),
                score_engine_module: "score".to_owned(),
                confidence_gates_module: "gate".to_owned(),
                release_certificate_module: "cert".to_owned(),
                verifier_script: "script".to_owned(),
            },
        };

        let Some(gate) = criteria.cell_gate("c1") else {
            panic!("expected c1 gate");
        };
        assert_eq!(gate.cell, "c1");
        assert!(criteria.cell_gate("c8").is_none());
    }
}
