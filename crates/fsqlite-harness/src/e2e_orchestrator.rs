//! Deterministic full-suite E2E script orchestrator and manifest (bd-mblr.4.5.2).
//!
//! Defines a single orchestrated execution manifest for all required E2E scripts,
//! including ordering, shardability, retries policy, timeout budgets, and required
//! artifacts. Cross-references the traceability matrix from bd-mblr.4.5.1 and
//! the deterministic seed taxonomy from bd-mblr.4.3.1.
//!
//! # Usage
//!
//! ```rust,ignore
//! let manifest = build_execution_manifest(0xDEADBEEF);
//! let errors = manifest.validate();
//! assert!(errors.is_empty());
//! let summary = manifest.dry_run_summary();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::e2e_traceability::{self, ConcurrencyMode, ScriptKind, StorageMode};
use crate::seed_taxonomy::SeedTaxonomy;

#[allow(dead_code)]
const BEAD_ID: &str = "bd-mblr.4.5.2";

/// Root seed for the orchestrator's own seed derivation.
const ORCHESTRATOR_ROOT_SEED: u64 = 0x4F52_4348_4553_5452; // ASCII "ORCHESTR"

/// Default timeout budget per script (seconds).
const DEFAULT_TIMEOUT_SECS: u32 = 300;

/// Maximum retry attempts for flaky scripts.
const MAX_RETRIES: u32 = 2;

// ─── Manifest Types ─────────────────────────────────────────────────────

/// Execution phase for ordering scripts within the manifest.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExecutionPhase {
    /// Infrastructure validation (workspace layering, toolchain checks).
    Preflight,
    /// Core correctness tests (SQL, transactions, storage).
    CoreCorrectness,
    /// Concurrency and MVCC tests.
    Concurrency,
    /// Recovery and WAL tests.
    Recovery,
    /// Compatibility and format tests.
    Compatibility,
    /// Extension tests (FTS, JSON, etc.).
    Extensions,
    /// Performance and stress tests (run last, longest).
    Performance,
    /// Summary and report generation.
    Postflight,
}

/// Retry policy for a script execution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetryPolicy {
    /// No retries; fail immediately.
    NoRetry,
    /// Retry up to N times on failure.
    RetryOnFailure { max_attempts: u32 },
    /// Retry only on specific exit codes (e.g., timeout).
    RetryOnTimeout { max_attempts: u32 },
}

/// Shard assignment for parallel execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShardAssignment {
    /// Shard index (0-based).
    pub shard_id: u32,
    /// Total number of shards.
    pub total_shards: u32,
}

/// A single script entry in the execution manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    /// Workspace-relative path to the script.
    pub path: String,
    /// Execution phase for ordering.
    pub phase: ExecutionPhase,
    /// Execution order within phase (lower = earlier).
    pub order_within_phase: u32,
    /// Deterministic seed for this script's execution.
    pub seed: u64,
    /// Derived seed taxonomy for this script.
    pub seed_taxonomy: SeedFields,
    /// Invocation command.
    pub command: String,
    /// Required environment variables.
    pub env_vars: Vec<EnvVar>,
    /// Timeout budget in seconds.
    pub timeout_secs: u32,
    /// Retry policy.
    pub retry_policy: RetryPolicy,
    /// Whether this script supports JSON structured output.
    pub json_output: bool,
    /// Scenario IDs this script covers.
    pub scenario_ids: Vec<String>,
    /// Expected artifact paths (workspace-relative).
    pub expected_artifacts: Vec<String>,
    /// Whether this script can be sharded across parallel workers.
    pub shardable: bool,
    /// Script kind from traceability matrix.
    pub kind: ScriptKind,
    /// Storage modes tested.
    pub storage_modes: Vec<StorageMode>,
    /// Concurrency modes tested.
    pub concurrency_modes: Vec<ConcurrencyMode>,
}

/// Flattened seed taxonomy fields for serialization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SeedFields {
    pub schedule: u64,
    pub entropy: u64,
    pub fault: u64,
    pub fuzz: u64,
}

/// Environment variable requirement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnvVar {
    pub name: String,
    pub required: bool,
    pub default_value: Option<String>,
}

/// The full execution manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionManifest {
    /// Schema version for the manifest format.
    pub schema_version: String,
    /// Bead ID.
    pub bead_id: String,
    /// Root seed used to derive all per-script seeds.
    pub root_seed: u64,
    /// Total timeout budget across all scripts (seconds).
    pub total_timeout_budget_secs: u64,
    /// All script entries in execution order.
    pub entries: Vec<ManifestEntry>,
    /// Scenario coverage summary.
    pub coverage: ManifestCoverage,
}

/// Coverage summary for the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestCoverage {
    /// Total unique scenario IDs covered.
    pub total_scenario_ids: usize,
    /// Scenario IDs not covered by any script.
    pub uncovered_scenario_ids: Vec<String>,
    /// Total scripts in manifest.
    pub total_scripts: usize,
    /// Scripts per phase.
    pub scripts_per_phase: BTreeMap<String, usize>,
    /// Scripts per kind.
    pub scripts_per_kind: BTreeMap<String, usize>,
}

/// Dry-run summary of what the orchestrator would execute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunSummary {
    pub total_scripts: usize,
    pub total_timeout_budget_secs: u64,
    pub phases: Vec<PhaseSummary>,
    pub total_scenario_ids: usize,
    pub shardable_scripts: usize,
    pub retryable_scripts: usize,
}

/// Summary of a single execution phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseSummary {
    pub phase: String,
    pub script_count: usize,
    pub timeout_budget_secs: u64,
    pub scenario_ids: usize,
}

impl ExecutionManifest {
    /// Validate the manifest for internal consistency.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // No duplicate paths
        let mut seen_paths = BTreeSet::new();
        for entry in &self.entries {
            if !seen_paths.insert(&entry.path) {
                errors.push(format!("Duplicate path in manifest: {}", entry.path));
            }
        }

        // No duplicate seeds (each script gets a unique seed)
        let mut seen_seeds = BTreeSet::new();
        for entry in &self.entries {
            if !seen_seeds.insert(entry.seed) {
                errors.push(format!(
                    "Duplicate seed {} for script {}",
                    entry.seed, entry.path
                ));
            }
        }

        // All entries have at least one scenario ID
        for entry in &self.entries {
            if entry.scenario_ids.is_empty() {
                errors.push(format!(
                    "Script {} has no scenario IDs",
                    entry.path
                ));
            }
        }

        // Timeout budget consistency
        let sum_timeouts: u64 = self.entries.iter().map(|e| u64::from(e.timeout_secs)).sum();
        if sum_timeouts != self.total_timeout_budget_secs {
            errors.push(format!(
                "Total timeout budget mismatch: sum={sum_timeouts} declared={}",
                self.total_timeout_budget_secs
            ));
        }

        // Coverage stats consistency
        let actual_scripts = self.entries.len();
        if actual_scripts != self.coverage.total_scripts {
            errors.push(format!(
                "Coverage script count mismatch: actual={actual_scripts} declared={}",
                self.coverage.total_scripts
            ));
        }

        // Ordering within phases is monotonically increasing
        let mut phase_orders: BTreeMap<ExecutionPhase, Vec<u32>> = BTreeMap::new();
        for entry in &self.entries {
            phase_orders.entry(entry.phase).or_default().push(entry.order_within_phase);
        }
        for (phase, orders) in &phase_orders {
            for window in orders.windows(2) {
                if window[0] > window[1] {
                    errors.push(format!(
                        "Non-monotonic ordering in phase {phase:?}: {} > {}",
                        window[0], window[1]
                    ));
                }
            }
        }

        errors
    }

    /// Generate a dry-run summary without executing anything.
    #[must_use]
    pub fn dry_run_summary(&self) -> DryRunSummary {
        let mut phases: BTreeMap<ExecutionPhase, (usize, u64, BTreeSet<String>)> = BTreeMap::new();

        for entry in &self.entries {
            let phase_data = phases.entry(entry.phase).or_insert_with(|| (0, 0, BTreeSet::new()));
            phase_data.0 += 1;
            phase_data.1 += u64::from(entry.timeout_secs);
            for sid in &entry.scenario_ids {
                phase_data.2.insert(sid.clone());
            }
        }

        let phase_summaries: Vec<PhaseSummary> = phases
            .into_iter()
            .map(|(phase, (count, timeout, sids))| PhaseSummary {
                phase: format!("{phase:?}"),
                script_count: count,
                timeout_budget_secs: timeout,
                scenario_ids: sids.len(),
            })
            .collect();

        let all_sids: BTreeSet<_> = self
            .entries
            .iter()
            .flat_map(|e| e.scenario_ids.iter().cloned())
            .collect();

        let shardable = self.entries.iter().filter(|e| e.shardable).count();
        let retryable = self
            .entries
            .iter()
            .filter(|e| !matches!(e.retry_policy, RetryPolicy::NoRetry))
            .count();

        DryRunSummary {
            total_scripts: self.entries.len(),
            total_timeout_budget_secs: self.total_timeout_budget_secs,
            phases: phase_summaries,
            total_scenario_ids: all_sids.len(),
            shardable_scripts: shardable,
            retryable_scripts: retryable,
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// ─── Phase Classification ───────────────────────────────────────────────

fn classify_phase(path: &str, scenario_ids: &[String], concurrency_modes: &[ConcurrencyMode]) -> ExecutionPhase {
    // Infrastructure and layering checks
    if path.contains("layering") || path.contains("logging_standard") {
        return ExecutionPhase::Preflight;
    }

    // Performance tests
    if path.contains("perf") || path.contains("benchmark") || path.contains("stress")
        || path.contains("hotspot") || path.contains("arc_warmup")
    {
        return ExecutionPhase::Performance;
    }

    // Recovery and WAL tests
    if path.contains("recovery") || path.contains("crash") || path.contains("wal_replay")
        || path.contains("wal_checksum") || path.contains("wal_integrity")
        || scenario_ids.iter().any(|s| s.starts_with("REC-") || s.starts_with("WAL-"))
    {
        return ExecutionPhase::Recovery;
    }

    // Concurrency and MVCC tests
    if path.contains("concurrent") || path.contains("mvcc") || path.contains("ssi")
        || path.contains("write_skew") || path.contains("isolation")
        || concurrency_modes.iter().any(|m| matches!(m, ConcurrencyMode::ConcurrentWriters | ConcurrencyMode::MvccIsolation | ConcurrencyMode::Ssi))
    {
        return ExecutionPhase::Concurrency;
    }

    // Extension tests
    if path.contains("fts") || path.contains("json1") || path.contains("rtree")
        || scenario_ids.iter().any(|s| s.starts_with("EXT-"))
    {
        return ExecutionPhase::Extensions;
    }

    // Compatibility tests
    if path.contains("compat") || path.contains("realdb") || path.contains("format")
        || path.contains("quirks")
        || scenario_ids.iter().any(|s| s.starts_with("COMPAT-"))
    {
        return ExecutionPhase::Compatibility;
    }

    // Default: core correctness
    ExecutionPhase::CoreCorrectness
}

fn classify_retry_policy(kind: ScriptKind, concurrency_modes: &[ConcurrencyMode]) -> RetryPolicy {
    // Concurrency tests may have inherent non-determinism in scheduling
    let has_concurrency = concurrency_modes.iter().any(|m| {
        matches!(
            m,
            ConcurrencyMode::ConcurrentWriters | ConcurrencyMode::MvccIsolation | ConcurrencyMode::Ssi
        )
    });

    if has_concurrency {
        return RetryPolicy::RetryOnFailure {
            max_attempts: MAX_RETRIES,
        };
    }

    // Shell scripts may timeout
    if matches!(kind, ScriptKind::ShellE2e) {
        return RetryPolicy::RetryOnTimeout {
            max_attempts: MAX_RETRIES,
        };
    }

    RetryPolicy::NoRetry
}

fn is_shardable(concurrency_modes: &[ConcurrencyMode]) -> bool {
    // Scripts that don't require concurrent writers can be safely sharded
    // (they don't contend on shared database files)
    !concurrency_modes.iter().any(|m| {
        matches!(
            m,
            ConcurrencyMode::ConcurrentWriters | ConcurrencyMode::MvccIsolation | ConcurrencyMode::Ssi
        )
    })
}

// ─── Build the Manifest ────────────────────────────────────────────────

/// Build the deterministic execution manifest from the traceability matrix.
///
/// The `root_seed` is used to derive unique per-script seeds via the
/// seed taxonomy. Use `ORCHESTRATOR_ROOT_SEED` for the canonical default.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn build_execution_manifest(root_seed: u64) -> ExecutionManifest {
    let matrix = e2e_traceability::build_canonical_inventory();

    // Phase -> entries (for ordering)
    let mut phase_entries: BTreeMap<ExecutionPhase, Vec<ManifestEntry>> = BTreeMap::new();

    for (idx, script) in matrix.scripts.iter().enumerate() {
        // Skip utility scripts that aren't directly executable tests
        if matches!(script.kind, ScriptKind::ShellUtility) {
            continue;
        }

        let phase = classify_phase(
            &script.path,
            &script.scenario_ids,
            &script.concurrency_modes,
        );

        // Derive unique seed for this script using its path as scope
        let taxonomy = SeedTaxonomy::derive(root_seed, &script.path);
        let seed_fields = SeedFields {
            schedule: taxonomy.schedule,
            entropy: taxonomy.entropy,
            fault: taxonomy.fault,
            fuzz: taxonomy.fuzz,
        };

        let timeout = script.invocation.timeout_secs.unwrap_or(DEFAULT_TIMEOUT_SECS);
        let retry_policy = classify_retry_policy(script.kind, &script.concurrency_modes);
        let shardable = is_shardable(&script.concurrency_modes);

        let env_vars: Vec<EnvVar> = script
            .invocation
            .env_vars
            .iter()
            .map(|name| EnvVar {
                name: name.clone(),
                required: true,
                default_value: None,
            })
            .collect();

        // Add FRANKEN_SEED env var for seed propagation
        let mut all_env_vars = vec![EnvVar {
            name: "FRANKEN_SEED".to_owned(),
            required: false,
            default_value: Some(taxonomy.root.to_string()),
        }];
        all_env_vars.extend(env_vars);

        let entry = ManifestEntry {
            path: script.path.clone(),
            phase,
            order_within_phase: u32::try_from(idx).unwrap_or(u32::MAX),
            seed: taxonomy.root,
            seed_taxonomy: seed_fields,
            command: script.invocation.command.clone(),
            env_vars: all_env_vars,
            timeout_secs: timeout,
            retry_policy,
            json_output: script.invocation.json_output,
            scenario_ids: script.scenario_ids.clone(),
            expected_artifacts: script.artifact_paths.clone(),
            shardable,
            kind: script.kind,
            storage_modes: script.storage_modes.clone(),
            concurrency_modes: script.concurrency_modes.clone(),
        };

        phase_entries.entry(phase).or_default().push(entry);
    }

    // Flatten into execution order: phases are ordered by enum variant,
    // within each phase entries keep their original order
    let mut entries = Vec::new();
    for (_phase, mut phase_scripts) in phase_entries {
        // Re-number within phase
        for (i, entry) in phase_scripts.iter_mut().enumerate() {
            entry.order_within_phase = u32::try_from(i).unwrap_or(u32::MAX);
        }
        entries.extend(phase_scripts);
    }

    let total_timeout: u64 = entries.iter().map(|e| u64::from(e.timeout_secs)).sum();
    let coverage = compute_coverage(&entries, &matrix);

    ExecutionManifest {
        schema_version: "1.0.0".to_owned(),
        bead_id: BEAD_ID.to_owned(),
        root_seed,
        total_timeout_budget_secs: total_timeout,
        entries,
        coverage,
    }
}

/// Build the manifest with the default orchestrator root seed.
#[must_use]
pub fn build_default_manifest() -> ExecutionManifest {
    build_execution_manifest(ORCHESTRATOR_ROOT_SEED)
}

fn compute_coverage(entries: &[ManifestEntry], matrix: &e2e_traceability::TraceabilityMatrix) -> ManifestCoverage {
    // Collect all scenario IDs from manifest
    let manifest_sids: BTreeSet<String> = entries
        .iter()
        .flat_map(|e| e.scenario_ids.iter().cloned())
        .collect();

    // Collect all scenario IDs from the traceability matrix
    let matrix_sids: BTreeSet<String> = matrix
        .scripts
        .iter()
        .flat_map(|s| s.scenario_ids.iter().cloned())
        .collect();

    // Uncovered = in matrix but not in manifest (due to filtered-out utility scripts)
    let uncovered: Vec<String> = matrix_sids
        .difference(&manifest_sids)
        .cloned()
        .collect();

    // Scripts per phase
    let mut scripts_per_phase: BTreeMap<String, usize> = BTreeMap::new();
    for entry in entries {
        *scripts_per_phase
            .entry(format!("{:?}", entry.phase))
            .or_insert(0) += 1;
    }

    // Scripts per kind
    let mut scripts_per_kind: BTreeMap<String, usize> = BTreeMap::new();
    for entry in entries {
        *scripts_per_kind
            .entry(format!("{:?}", entry.kind))
            .or_insert(0) += 1;
    }

    ManifestCoverage {
        total_scenario_ids: manifest_sids.len(),
        uncovered_scenario_ids: uncovered,
        total_scripts: entries.len(),
        scripts_per_phase,
        scripts_per_kind,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_builds() {
        let manifest = build_default_manifest();
        assert!(!manifest.entries.is_empty());
        assert_eq!(manifest.schema_version, "1.0.0");
        assert_eq!(manifest.bead_id, BEAD_ID);
    }

    #[test]
    fn manifest_validates() {
        let manifest = build_default_manifest();
        let errors = manifest.validate();
        assert!(
            errors.is_empty(),
            "Validation errors:\n{}",
            errors.join("\n")
        );
    }

    #[test]
    fn manifest_deterministic() {
        let m1 = build_execution_manifest(0x1234);
        let m2 = build_execution_manifest(0x1234);
        assert_eq!(m1.entries.len(), m2.entries.len());
        assert_eq!(m1.total_timeout_budget_secs, m2.total_timeout_budget_secs);
        for (a, b) in m1.entries.iter().zip(m2.entries.iter()) {
            assert_eq!(a.seed, b.seed, "Seeds differ for {}", a.path);
            assert_eq!(a.path, b.path);
        }
    }

    #[test]
    fn different_seeds_produce_different_manifests() {
        let m1 = build_execution_manifest(0x1111);
        let m2 = build_execution_manifest(0x2222);
        // Same structure but different seeds
        assert_eq!(m1.entries.len(), m2.entries.len());
        // At least some seeds should differ
        let differs = m1
            .entries
            .iter()
            .zip(m2.entries.iter())
            .any(|(a, b)| a.seed != b.seed);
        assert!(differs, "Different root seeds should produce different per-script seeds");
    }

    #[test]
    fn unique_seeds_per_script() {
        let manifest = build_default_manifest();
        let mut seeds = BTreeSet::new();
        for entry in &manifest.entries {
            assert!(
                seeds.insert(entry.seed),
                "Duplicate seed {} for script {}",
                entry.seed,
                entry.path
            );
        }
    }

    #[test]
    fn phases_are_ordered() {
        let manifest = build_default_manifest();
        let mut last_phase = None;
        for entry in &manifest.entries {
            if let Some(prev) = last_phase {
                assert!(
                    entry.phase >= prev,
                    "Phase ordering violated: {:?} after {:?} for {}",
                    entry.phase,
                    prev,
                    entry.path
                );
            }
            last_phase = Some(entry.phase);
        }
    }

    #[test]
    fn all_entries_have_scenarios() {
        let manifest = build_default_manifest();
        for entry in &manifest.entries {
            assert!(
                !entry.scenario_ids.is_empty(),
                "Script {} has no scenario IDs",
                entry.path
            );
        }
    }

    #[test]
    fn timeout_budget_consistent() {
        let manifest = build_default_manifest();
        let sum: u64 = manifest.entries.iter().map(|e| u64::from(e.timeout_secs)).sum();
        assert_eq!(sum, manifest.total_timeout_budget_secs);
    }

    #[test]
    fn coverage_stats_consistent() {
        let manifest = build_default_manifest();
        assert_eq!(manifest.coverage.total_scripts, manifest.entries.len());
        let phase_sum: usize = manifest.coverage.scripts_per_phase.values().sum();
        assert_eq!(phase_sum, manifest.entries.len());
        let kind_sum: usize = manifest.coverage.scripts_per_kind.values().sum();
        assert_eq!(kind_sum, manifest.entries.len());
    }

    #[test]
    fn dry_run_summary() {
        let manifest = build_default_manifest();
        let summary = manifest.dry_run_summary();
        assert_eq!(summary.total_scripts, manifest.entries.len());
        assert_eq!(
            summary.total_timeout_budget_secs,
            manifest.total_timeout_budget_secs
        );
        assert!(!summary.phases.is_empty());
    }

    #[test]
    fn json_roundtrip() {
        let manifest = build_default_manifest();
        let json = manifest.to_json().expect("serialize");
        let deserialized: ExecutionManifest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deserialized.entries.len(), manifest.entries.len());
        assert_eq!(deserialized.root_seed, manifest.root_seed);
    }

    #[test]
    fn concurrency_scripts_are_retryable() {
        let manifest = build_default_manifest();
        for entry in &manifest.entries {
            let has_concurrency = entry.concurrency_modes.iter().any(|m| {
                matches!(
                    m,
                    ConcurrencyMode::ConcurrentWriters
                        | ConcurrencyMode::MvccIsolation
                        | ConcurrencyMode::Ssi
                )
            });
            if has_concurrency {
                assert!(
                    !matches!(entry.retry_policy, RetryPolicy::NoRetry),
                    "Concurrent script {} should be retryable",
                    entry.path
                );
            }
        }
    }

    #[test]
    fn seed_taxonomy_fields_populated() {
        let manifest = build_default_manifest();
        for entry in &manifest.entries {
            // All four derived seeds should be non-zero and distinct
            let fields = entry.seed_taxonomy;
            assert_ne!(fields.schedule, 0, "schedule seed is zero for {}", entry.path);
            assert_ne!(fields.entropy, 0, "entropy seed is zero for {}", entry.path);
            assert_ne!(fields.fault, 0, "fault seed is zero for {}", entry.path);
            assert_ne!(fields.fuzz, 0, "fuzz seed is zero for {}", entry.path);
            // All four should differ from each other
            let set: BTreeSet<u64> = [fields.schedule, fields.entropy, fields.fault, fields.fuzz]
                .into_iter()
                .collect();
            assert_eq!(
                set.len(),
                4,
                "Seed taxonomy fields not all distinct for {}",
                entry.path
            );
        }
    }

    #[test]
    fn franken_seed_env_var_present() {
        let manifest = build_default_manifest();
        for entry in &manifest.entries {
            let has_seed_env = entry
                .env_vars
                .iter()
                .any(|v| v.name == "FRANKEN_SEED");
            assert!(
                has_seed_env,
                "Script {} missing FRANKEN_SEED env var",
                entry.path
            );
        }
    }

    #[test]
    fn no_utility_scripts_in_manifest() {
        let manifest = build_default_manifest();
        for entry in &manifest.entries {
            assert_ne!(
                entry.kind,
                ScriptKind::ShellUtility,
                "Utility script {} should not be in execution manifest",
                entry.path
            );
        }
    }
}
