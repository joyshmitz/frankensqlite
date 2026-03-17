//! Performance benchmark matrix runner.
//!
//! Bead: bd-2als.2.2
//!
//! Orchestrates repeated benchmark execution across a matrix of:
//! - **Engine**: sqlite3 (rusqlite) vs fsqlite
//! - **Fixture**: golden database files
//! - **Workload**: OpLog preset names
//! - **Concurrency**: worker counts (1, 2, 4, 8, …)
//!
//! For each matrix cell, the runner uses [`crate::benchmark::run_benchmark`]
//! to perform warmup + measurement iterations, producing a
//! [`crate::benchmark::BenchmarkSummary`] per cell.  Results are collected
//! into a [`PerfResult`] and can be serialized to JSONL for downstream
//! analysis and reporting.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use fsqlite_core::connection::{
    HotPathProfileSnapshot, ParserHotPathProfileSnapshot, hot_path_profile_enabled,
    hot_path_profile_snapshot, reset_hot_path_profile, set_hot_path_profile_enabled,
};

use crate::HarnessSettings;
use crate::benchmark::{BenchmarkConfig, BenchmarkMeta, BenchmarkSummary, run_benchmark};
use crate::fixture_select::{BenchmarkArtifactCommand, BenchmarkArtifactToolVersion};
use crate::fsqlite_executor::run_oplog_fsqlite;
use crate::oplog::{self, OpLog};
use crate::report::EngineRunReport;
use crate::run_workspace::{WorkspaceConfig, create_workspace_with_label};
use crate::sqlite_executor::run_oplog_sqlite;

// ── Configuration ──────────────────────────────────────────────────────

/// Which database engine to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Engine {
    /// C SQLite via rusqlite.
    Sqlite3,
    /// FrankenSQLite.
    Fsqlite,
}

impl Engine {
    /// Returns the canonical string name used in benchmark IDs and reports.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sqlite3 => "sqlite3",
            Self::Fsqlite => "fsqlite",
        }
    }
}

impl std::fmt::Display for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Configuration for the performance benchmark matrix.
#[derive(Debug, Clone)]
pub struct PerfMatrixConfig {
    /// Engines to benchmark.
    pub engines: Vec<Engine>,
    /// Fixture IDs from the golden corpus to test against.
    pub fixture_ids: Vec<String>,
    /// Workload preset names (must match names in `oplog::preset_catalog()`).
    pub workload_presets: Vec<String>,
    /// Concurrency levels to sweep.
    pub concurrency_levels: Vec<u16>,
    /// Seed for deterministic workload generation.
    pub seed: u64,
    /// Scale factor for workload size (row counts, etc.).
    pub scale: u32,
    /// Path to the project root (for locating golden fixtures).
    pub project_root: PathBuf,
    /// Benchmark configuration (warmup, min iterations, time floor).
    pub benchmark_config: BenchmarkConfig,
    /// Harness settings (PRAGMAs, journal mode, etc.).
    pub settings: HarnessSettings,
    /// Cargo profile string for metadata (e.g. `"release"`).
    pub cargo_profile: String,
    /// If true, stop on first error.
    pub fail_fast: bool,
}

/// A single cell in the benchmark matrix.
#[derive(Debug, Clone)]
pub struct MatrixCell {
    pub engine: Engine,
    pub fixture_id: String,
    pub workload: String,
    pub concurrency: u16,
}

impl std::fmt::Display for MatrixCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}:c{}",
            self.engine, self.workload, self.fixture_id, self.concurrency
        )
    }
}

// ── Result types ───────────────────────────────────────────────────────

/// Outcome of a single matrix cell benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellOutcome {
    /// The benchmark summary (present on success).
    pub summary: Option<BenchmarkSummary>,
    /// Error message when the cell is not fully comparable.
    ///
    /// This includes total cell failures and partial measurement runs where at
    /// least one iteration errored.
    pub error: Option<String>,
    /// Engine name.
    pub engine: String,
    /// Fixture ID.
    pub fixture_id: String,
    /// Workload preset name.
    pub workload: String,
    /// Concurrency level.
    pub concurrency: u16,
}

impl CellOutcome {
    /// Whether this cell can participate in engine-to-engine comparisons.
    #[must_use]
    pub fn is_fully_comparable(&self) -> bool {
        self.summary.is_some() && self.error.is_none()
    }

    /// Summary data only when every measurement iteration completed cleanly.
    #[must_use]
    pub fn comparable_summary(&self) -> Option<&BenchmarkSummary> {
        self.summary.as_ref().filter(|_| self.error.is_none())
    }
}

/// Complete result of running the performance matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfResult {
    /// Schema version for forward compatibility.
    pub schema_version: String,
    /// Total cells in the matrix.
    pub total_cells: usize,
    /// Number of cells that completed successfully.
    pub success_count: usize,
    /// Number of cells that errored.
    pub error_count: usize,
    /// Per-cell outcomes.
    pub cells: Vec<CellOutcome>,
}

/// Schema version for the perf result JSONL format.
pub const PERF_RESULT_SCHEMA_V1: &str = "fsqlite-e2e.perf_result.v1";
/// Schema version for hot-path profile reports.
pub const HOT_PATH_PROFILE_SCHEMA_V1: &str = "fsqlite-e2e.hot_path_profile.v1";
/// Schema version for hot-path artifact manifests.
pub const HOT_PATH_PROFILE_MANIFEST_SCHEMA_V1: &str = "fsqlite-e2e.hot_path_profile_manifest.v1";
/// Schema version for raw opcode profile packs.
pub const HOT_PATH_OPCODE_PROFILE_SCHEMA_V1: &str = "fsqlite-e2e.hot_path_opcode_profile.v1";
/// Schema version for raw subsystem profile packs.
pub const HOT_PATH_SUBSYSTEM_PROFILE_SCHEMA_V1: &str = "fsqlite-e2e.hot_path_subsystem_profile.v1";
/// Schema version for structured actionable hotspot and baseline-tax artifacts.
pub const HOT_PATH_PROFILE_ACTIONABLE_RANKING_SCHEMA_V3: &str =
    "fsqlite-e2e.hot_path_actionable_ranking.v3";
/// Bead identifier for the hot-path profiling work.
pub const HOT_PATH_PROFILE_BEAD_ID: &str = "bd-db300.4.1";
/// Canonical scenario identifier prefix for preset-specific hot-path profiles.
pub const HOT_PATH_PROFILE_SCENARIO_PREFIX: &str = HOT_PATH_PROFILE_BEAD_ID;

/// Configuration for a focused FrankenSQLite hot-path profile.
#[derive(Debug, Clone)]
pub struct FsqliteHotPathProfileConfig {
    pub workload: String,
    pub seed: u64,
    pub scale: u32,
    pub concurrency: u16,
    pub exec_config: crate::fsqlite_executor::FsqliteExecConfig,
    pub replay_command: String,
    pub golden_dir: Option<String>,
    pub working_base: Option<String>,
    pub bead_id: Option<String>,
    pub scenario_prefix: Option<String>,
}

impl FsqliteHotPathProfileConfig {
    fn bead_id(&self) -> &str {
        self.bead_id.as_deref().unwrap_or(HOT_PATH_PROFILE_BEAD_ID)
    }

    fn scenario_prefix(&self) -> &str {
        self.scenario_prefix
            .as_deref()
            .unwrap_or_else(|| self.bead_id())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathValueTypeProfile {
    pub total_values: u64,
    pub nulls: u64,
    pub integers: u64,
    pub reals: u64,
    pub texts: u64,
    pub blobs: u64,
    pub text_bytes_total: u64,
    pub blob_bytes_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathParserProfile {
    pub parse_single_calls: u64,
    pub parse_multi_calls: u64,
    pub parse_cache_hits: u64,
    pub parse_cache_misses: u64,
    pub parsed_sql_bytes: u64,
    pub parse_time_ns: u64,
    pub rewrite_calls: u64,
    pub rewrite_time_ns: u64,
    pub compiled_cache_hits: u64,
    pub compiled_cache_misses: u64,
    pub compile_time_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathRecordDecodeProfile {
    pub parse_record_calls: u64,
    pub parse_record_into_calls: u64,
    pub parse_record_column_calls: u64,
    pub record_bytes_scanned: u64,
    pub record_vec_capacity_slots: u64,
    pub decode_time_ns: u64,
    pub decoded_values: HotPathValueTypeProfile,
    pub vdbe_record_decode_calls_total: u64,
    pub vdbe_column_reads_total: u64,
    pub vdbe_decoded_value_heap_bytes_total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathRowMaterializationProfile {
    pub result_rows_total: u64,
    pub result_values_total: u64,
    pub result_value_heap_bytes_total: u64,
    pub result_row_materialization_time_ns_total: u64,
    pub make_record_calls_total: u64,
    pub make_record_blob_bytes_total: u64,
    pub value_types: HotPathValueTypeProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathOpcodeProfileEntry {
    pub opcode: String,
    pub total: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathOpcodeProfilePack {
    pub schema_version: String,
    pub bead_id: String,
    pub run_id: String,
    pub trace_id: String,
    pub scenario_id: String,
    pub fixture_id: String,
    pub workload: String,
    pub seed: u64,
    pub scale: u32,
    pub concurrency: u16,
    pub replay_command: String,
    pub opcodes: Vec<HotPathOpcodeProfileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathTypeProfile {
    pub decoded: HotPathValueTypeProfile,
    pub materialized: HotPathValueTypeProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathSubsystemProfilePack {
    pub schema_version: String,
    pub bead_id: String,
    pub run_id: String,
    pub trace_id: String,
    pub scenario_id: String,
    pub fixture_id: String,
    pub workload: String,
    pub seed: u64,
    pub scale: u32,
    pub concurrency: u16,
    pub replay_command: String,
    pub subsystem_ranking: Vec<HotPathRankingEntry>,
    pub allocator_ranking: Vec<HotPathRankingEntry>,
    pub parser: HotPathParserProfile,
    pub record_decode: HotPathRecordDecodeProfile,
    pub row_materialization: HotPathRowMaterializationProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathRankingEntry {
    pub subsystem: String,
    pub metric_kind: String,
    pub metric_value: u64,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathActionableRankingEntry {
    pub rank: u32,
    pub subsystem: String,
    pub metric_kind: String,
    pub metric_value: u64,
    pub rationale: String,
    pub implication: String,
    pub mapped_beads: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathBaselineReuseLedgerEntry {
    pub rank: u32,
    pub surface: String,
    pub supported: bool,
    pub hits: u64,
    pub misses: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hit_rate_basis_points: Option<u32>,
    pub rationale: String,
    pub implication: String,
    pub mapped_beads: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathBaselineWasteLedgerEntry {
    pub rank: u32,
    pub component: String,
    pub classification: String,
    pub metric_kind: String,
    pub metric_value: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_share_basis_points: Option<u32>,
    pub allocator_pressure_bytes: u64,
    pub activity_count: u64,
    pub rationale: String,
    pub implication: String,
    pub mapped_beads: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathCostComponentEntry {
    pub rank: u32,
    pub component: String,
    pub time_ns: u64,
    pub time_share_basis_points: u32,
    pub allocator_pressure_bytes: u64,
    pub allocator_share_basis_points: u32,
    pub activity_count: u64,
    pub rationale: String,
    pub implication: String,
    pub mapped_beads: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathWallTimeComponentEntry {
    pub rank: u32,
    pub component: String,
    pub time_ns: u64,
    pub wall_share_basis_points: u32,
    pub rationale: String,
    pub implication: String,
    pub mapped_beads: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathMicroarchitecturalSignatureEntry {
    pub rank: u32,
    pub target: String,
    pub primary_signature: String,
    pub secondary_signatures: Vec<String>,
    pub confidence_label: String,
    pub confidence_score_basis_points: u32,
    pub mixed_or_ambiguous: bool,
    pub rationale: String,
    pub evidence_sources: Vec<String>,
    pub fixture_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement_profile_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_class_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_signature: Option<String>,
    pub implication: String,
    pub mapped_beads: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathMicroarchitecturalContext {
    pub fixture_id: String,
    pub row_id: String,
    pub mode_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement_profile_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_class_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathAllocatorPressure {
    pub parser_sql_bytes: u64,
    pub decoded_value_heap_bytes_total: u64,
    pub result_value_heap_bytes_total: u64,
    pub record_vec_capacity_slots: u64,
    pub ranked_sources: Vec<HotPathRankingEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotPathProfileReport {
    pub schema_version: String,
    pub bead_id: String,
    pub scenario_id: String,
    pub run_id: String,
    pub trace_id: String,
    pub fixture_id: String,
    pub workload: String,
    pub seed: u64,
    pub scale: u32,
    pub concurrency: u16,
    pub concurrent_mode: bool,
    pub run_integrity_check: bool,
    pub golden_dir: Option<String>,
    pub working_base: Option<String>,
    pub replay_command: String,
    pub engine_report: EngineRunReport,
    pub parser: HotPathParserProfile,
    pub record_decode: HotPathRecordDecodeProfile,
    pub row_materialization: HotPathRowMaterializationProfile,
    pub opcode_profile: Vec<HotPathOpcodeProfileEntry>,
    pub type_profile: HotPathTypeProfile,
    pub subsystem_ranking: Vec<HotPathRankingEntry>,
    pub allocator_pressure: HotPathAllocatorPressure,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathActionableRanking {
    pub schema_version: String,
    pub bead_id: String,
    pub run_id: String,
    pub trace_id: String,
    pub scenario_id: String,
    pub fixture_id: String,
    pub workload: String,
    pub seed: u64,
    pub scale: u32,
    pub concurrency: u16,
    pub replay_command: String,
    pub baseline_reuse_ledger: Vec<HotPathBaselineReuseLedgerEntry>,
    pub baseline_waste_ledger: Vec<HotPathBaselineWasteLedgerEntry>,
    pub named_hotspots: Vec<HotPathActionableRankingEntry>,
    pub microarchitectural_signatures: Vec<HotPathMicroarchitecturalSignatureEntry>,
    pub wall_time_components: Vec<HotPathWallTimeComponentEntry>,
    pub cost_components: Vec<HotPathCostComponentEntry>,
    pub allocator_pressure: Vec<HotPathActionableRankingEntry>,
    pub top_opcodes: Vec<HotPathOpcodeProfileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathArtifactFile {
    pub path: String,
    pub bytes: u64,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathCounterCaptureManifestSummary {
    pub host_capability_sensitive_captures: Vec<String>,
    pub topology_sensitive_captures: Vec<String>,
    pub fallback_tools: Vec<String>,
    pub fallback_metric_pack: Vec<String>,
    pub fallback_notes: Vec<String>,
    pub raw_output_relpaths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathArtifactProvenance {
    pub row_id: String,
    pub mode_id: String,
    pub artifact_root: String,
    pub command_entrypoint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_root: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub campaign_manifest_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub beads_data_hash: Option<String>,
    pub kernel_release: String,
    pub rustc_version: String,
    pub cargo_profile: String,
    pub commands: Vec<BenchmarkArtifactCommand>,
    pub tool_versions: Vec<BenchmarkArtifactToolVersion>,
    pub fallback_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HotPathArtifactManifest {
    pub schema_version: String,
    pub bead_id: String,
    pub run_id: String,
    pub trace_id: String,
    pub scenario_id: String,
    pub fixture_id: String,
    pub workload: String,
    pub seed: u64,
    pub scale: u32,
    pub concurrency: u16,
    pub concurrent_mode: bool,
    pub run_integrity_check: bool,
    pub golden_dir: Option<String>,
    pub working_base: Option<String>,
    pub replay_command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counter_capture_summary: Option<HotPathCounterCaptureManifestSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<HotPathArtifactProvenance>,
    pub files: Vec<HotPathArtifactFile>,
}

static HOT_PATH_PROFILE_SCOPE_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));
#[cfg(test)]
pub(crate) static HOT_PATH_TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct HotPathProfileScope {
    was_enabled: bool,
    _guard: MutexGuard<'static, ()>,
}

impl HotPathProfileScope {
    fn enable() -> crate::E2eResult<Self> {
        let guard = HOT_PATH_PROFILE_SCOPE_LOCK.try_lock().map_err(|_| {
            crate::E2eError::Io(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "hot-path profiling is already active in this process",
            ))
        })?;
        let was_enabled = hot_path_profile_enabled();
        if was_enabled {
            return Err(crate::E2eError::Io(std::io::Error::other(
                "hot-path profiling was already enabled before scope entry",
            )));
        }
        set_hot_path_profile_enabled(true);
        reset_hot_path_profile();
        Ok(Self {
            was_enabled,
            _guard: guard,
        })
    }
}

impl Drop for HotPathProfileScope {
    fn drop(&mut self) {
        set_hot_path_profile_enabled(self.was_enabled);
    }
}

impl PerfResult {
    /// Returns true if all cells completed successfully.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.error_count == 0
    }

    /// Serialize all cell outcomes to JSONL (one JSON object per line).
    ///
    /// # Errors
    ///
    /// Returns a serialization error if any cell cannot be serialized.
    pub fn to_jsonl(&self) -> Result<String, serde_json::Error> {
        let mut out = String::new();
        for cell in &self.cells {
            out.push_str(&serde_json::to_string(cell)?);
            out.push('\n');
        }
        Ok(out)
    }
}

struct HotPathValueMetrics {
    total_values: u64,
    nulls: u64,
    integers: u64,
    reals: u64,
    texts: u64,
    blobs: u64,
    text_bytes_total: u64,
    blob_bytes_total: u64,
}

#[must_use]
fn hot_path_value_profile(metrics: HotPathValueMetrics) -> HotPathValueTypeProfile {
    HotPathValueTypeProfile {
        total_values: metrics.total_values,
        nulls: metrics.nulls,
        integers: metrics.integers,
        reals: metrics.reals,
        texts: metrics.texts,
        blobs: metrics.blobs,
        text_bytes_total: metrics.text_bytes_total,
        blob_bytes_total: metrics.blob_bytes_total,
    }
}

#[must_use]
fn parser_profile(snapshot: ParserHotPathProfileSnapshot) -> HotPathParserProfile {
    HotPathParserProfile {
        parse_single_calls: snapshot.parse_single_calls,
        parse_multi_calls: snapshot.parse_multi_calls,
        parse_cache_hits: snapshot.parse_cache_hits,
        parse_cache_misses: snapshot.parse_cache_misses,
        parsed_sql_bytes: snapshot.parsed_sql_bytes,
        parse_time_ns: snapshot.parse_time_ns,
        rewrite_calls: snapshot.rewrite_calls,
        rewrite_time_ns: snapshot.rewrite_time_ns,
        compiled_cache_hits: snapshot.compiled_cache_hits,
        compiled_cache_misses: snapshot.compiled_cache_misses,
        compile_time_ns: snapshot.compile_time_ns,
    }
}

#[must_use]
fn build_hot_path_profile_report(
    fixture_id: &str,
    config: &FsqliteHotPathProfileConfig,
    run_id: String,
    trace_id: String,
    engine_report: EngineRunReport,
    snapshot: HotPathProfileSnapshot,
) -> HotPathProfileReport {
    let decoded_types = hot_path_value_profile(HotPathValueMetrics {
        total_values: snapshot.record_decode.decoded_values.total_values(),
        nulls: snapshot.record_decode.decoded_values.null_count,
        integers: snapshot.record_decode.decoded_values.integer_count,
        reals: snapshot.record_decode.decoded_values.float_count,
        texts: snapshot.record_decode.decoded_values.text_count,
        blobs: snapshot.record_decode.decoded_values.blob_count,
        text_bytes_total: snapshot.record_decode.decoded_values.text_bytes,
        blob_bytes_total: snapshot.record_decode.decoded_values.blob_bytes,
    });
    let materialized_types = hot_path_value_profile(HotPathValueMetrics {
        total_values: snapshot.vdbe.result_value_types.total_values,
        nulls: snapshot.vdbe.result_value_types.nulls,
        integers: snapshot.vdbe.result_value_types.integers,
        reals: snapshot.vdbe.result_value_types.reals,
        texts: snapshot.vdbe.result_value_types.texts,
        blobs: snapshot.vdbe.result_value_types.blobs,
        text_bytes_total: snapshot.vdbe.result_value_types.text_bytes_total,
        blob_bytes_total: snapshot.vdbe.result_value_types.blob_bytes_total,
    });

    let opcode_profile = snapshot
        .vdbe
        .opcode_execution_totals
        .iter()
        .map(|entry| HotPathOpcodeProfileEntry {
            opcode: entry.opcode.clone(),
            total: entry.total,
        })
        .collect();

    let parser = parser_profile(snapshot.parser);
    let parser_sql_bytes = parser.parsed_sql_bytes;
    let record_decode = HotPathRecordDecodeProfile {
        parse_record_calls: snapshot.record_decode.parse_record_calls,
        parse_record_into_calls: snapshot.record_decode.parse_record_into_calls,
        parse_record_column_calls: snapshot.record_decode.parse_record_column_calls,
        record_bytes_scanned: snapshot.record_decode.record_bytes_scanned,
        record_vec_capacity_slots: snapshot.record_decode.record_vec_capacity_slots,
        decode_time_ns: snapshot.record_decode.decode_time_ns,
        decoded_values: decoded_types.clone(),
        vdbe_record_decode_calls_total: snapshot.vdbe.record_decode_calls_total,
        vdbe_column_reads_total: snapshot.vdbe.column_reads_total,
        vdbe_decoded_value_heap_bytes_total: snapshot.vdbe.decoded_value_heap_bytes_total,
    };
    let row_materialization = HotPathRowMaterializationProfile {
        result_rows_total: snapshot.vdbe.result_rows_total,
        result_values_total: snapshot.vdbe.result_values_total,
        result_value_heap_bytes_total: snapshot.vdbe.result_value_heap_bytes_total,
        result_row_materialization_time_ns_total: snapshot
            .vdbe
            .result_row_materialization_time_ns_total,
        make_record_calls_total: snapshot.vdbe.make_record_calls_total,
        make_record_blob_bytes_total: snapshot.vdbe.make_record_blob_bytes_total,
        value_types: materialized_types.clone(),
    };

    let parser_time_ns = parser
        .parse_time_ns
        .saturating_add(parser.rewrite_time_ns)
        .saturating_add(parser.compile_time_ns);
    let mut subsystem_ranking = vec![
        HotPathRankingEntry {
            subsystem: "parser_ast_churn".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: parser_time_ns,
            rationale: "parse + rewrite + compile time on the connection path".to_owned(),
        },
        HotPathRankingEntry {
            subsystem: "record_decode".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: record_decode.decode_time_ns,
            rationale: "time spent decoding SQLite record payloads".to_owned(),
        },
        HotPathRankingEntry {
            subsystem: "row_materialization".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: row_materialization.result_row_materialization_time_ns_total,
            rationale: "time spent cloning registers into emitted result rows".to_owned(),
        },
    ];
    subsystem_ranking.sort_by(|lhs, rhs| {
        rhs.metric_value
            .cmp(&lhs.metric_value)
            .then_with(|| lhs.subsystem.cmp(&rhs.subsystem))
    });

    let mut allocator_ranking = vec![
        HotPathRankingEntry {
            subsystem: "result_row_values".to_owned(),
            metric_kind: "heap_bytes".to_owned(),
            metric_value: row_materialization.result_value_heap_bytes_total,
            rationale: "estimated heap carried by emitted result values".to_owned(),
        },
        HotPathRankingEntry {
            subsystem: "record_decode_values".to_owned(),
            metric_kind: "heap_bytes".to_owned(),
            metric_value: record_decode.vdbe_decoded_value_heap_bytes_total,
            rationale: "estimated heap carried by decoded record/column values".to_owned(),
        },
        HotPathRankingEntry {
            subsystem: "parser_sql_bytes".to_owned(),
            metric_kind: "bytes".to_owned(),
            metric_value: parser.parsed_sql_bytes,
            rationale: "SQL text volume parsed on cache-miss paths".to_owned(),
        },
    ];
    allocator_ranking.sort_by(|lhs, rhs| {
        rhs.metric_value
            .cmp(&lhs.metric_value)
            .then_with(|| lhs.subsystem.cmp(&rhs.subsystem))
    });

    HotPathProfileReport {
        schema_version: HOT_PATH_PROFILE_SCHEMA_V1.to_owned(),
        bead_id: config.bead_id().to_owned(),
        scenario_id: hot_path_profile_scenario_id(config.scenario_prefix(), &config.workload),
        run_id,
        trace_id,
        fixture_id: fixture_id.to_owned(),
        workload: config.workload.clone(),
        seed: config.seed,
        scale: config.scale,
        concurrency: config.concurrency,
        concurrent_mode: config.exec_config.concurrent_mode,
        run_integrity_check: config.exec_config.run_integrity_check,
        golden_dir: config.golden_dir.clone(),
        working_base: config.working_base.clone(),
        replay_command: config.replay_command.clone(),
        engine_report,
        parser,
        record_decode,
        row_materialization,
        opcode_profile,
        type_profile: HotPathTypeProfile {
            decoded: decoded_types,
            materialized: materialized_types,
        },
        subsystem_ranking,
        allocator_pressure: HotPathAllocatorPressure {
            parser_sql_bytes,
            decoded_value_heap_bytes_total: snapshot.vdbe.decoded_value_heap_bytes_total,
            result_value_heap_bytes_total: snapshot.vdbe.result_value_heap_bytes_total,
            record_vec_capacity_slots: snapshot.record_decode.record_vec_capacity_slots,
            ranked_sources: allocator_ranking,
        },
    }
}

fn unix_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn hot_path_profile_scenario_id(scenario_prefix: &str, workload: &str) -> String {
    format!("{scenario_prefix}.{workload}")
}

pub fn profile_fsqlite_hot_path(
    db_path: &Path,
    fixture_id: &str,
    config: &FsqliteHotPathProfileConfig,
) -> crate::E2eResult<HotPathProfileReport> {
    if config.concurrency == 0 {
        return Err(crate::E2eError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "hot-path profile concurrency must be >= 1",
        )));
    }

    let now_ms = unix_timestamp_millis();
    let run_id = format!(
        "{}-{}-{fixture_id}-c{}-s{}-{now_ms}",
        config.bead_id(),
        config.workload,
        config.concurrency,
        config.seed
    );
    let scenario_id = hot_path_profile_scenario_id(config.scenario_prefix(), &config.workload);
    let trace_id = format!("{scenario_id}:{fixture_id}:c{}", config.concurrency);
    let oplog = generate_oplog(
        &config.workload,
        fixture_id,
        config.seed,
        config.concurrency,
        config.scale,
    )
    .ok_or_else(|| {
        crate::E2eError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("hot-path profile preset unavailable: {}", config.workload),
        ))
    })?;

    let _scope = HotPathProfileScope::enable()?;
    let engine_report = run_oplog_fsqlite(db_path, &oplog, &config.exec_config)?;
    let snapshot = hot_path_profile_snapshot();
    Ok(build_hot_path_profile_report(
        fixture_id,
        config,
        run_id,
        trace_id,
        engine_report,
        snapshot,
    ))
}

#[must_use]
pub fn render_hot_path_profile_markdown(report: &HotPathProfileReport) -> String {
    let actionable_ranking = build_hot_path_actionable_ranking(report, None, None);
    let mut out = String::with_capacity(4096);
    let _ = writeln!(out, "# Hot-Path Profile\n");
    let _ = writeln!(out, "- Bead: `{}`", report.bead_id);
    let _ = writeln!(out, "- Run ID: `{}`", report.run_id);
    let _ = writeln!(out, "- Trace ID: `{}`", report.trace_id);
    let _ = writeln!(out, "- Scenario: `{}`", report.scenario_id);
    let _ = writeln!(out, "- Fixture: `{}`", report.fixture_id);
    let _ = writeln!(out, "- Workload: `{}`", report.workload);
    let _ = writeln!(out, "- Seed: `{}`", report.seed);
    let _ = writeln!(out, "- Concurrency: `{}`", report.concurrency);
    let _ = writeln!(out, "- Scale: `{}`", report.scale);
    let _ = writeln!(
        out,
        "- Concurrent mode: `{}`",
        if report.concurrent_mode { "ON" } else { "OFF" }
    );
    let _ = writeln!(
        out,
        "- Integrity check: `{}`",
        if report.run_integrity_check {
            "enabled"
        } else {
            "disabled"
        }
    );
    if let Some(golden_dir) = &report.golden_dir {
        let _ = writeln!(out, "- Golden dir: `{golden_dir}`");
    }
    if let Some(working_base) = &report.working_base {
        let _ = writeln!(out, "- Working base: `{working_base}`");
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Engine Summary\n");
    let _ = writeln!(
        out,
        "- Wall time (ms): {}",
        report.engine_report.wall_time_ms
    );
    let _ = writeln!(out, "- Ops total: {}", report.engine_report.ops_total);
    let _ = writeln!(out, "- Ops/sec: {:.2}", report.engine_report.ops_per_sec);
    let _ = writeln!(out, "- Retries: {}", report.engine_report.retries);
    let _ = writeln!(out, "- Aborts: {}", report.engine_report.aborts);
    let _ = writeln!(
        out,
        "- Integrity check: {}",
        report
            .engine_report
            .correctness
            .integrity_check_ok
            .map_or_else(|| "skipped".to_owned(), |ok| ok.to_string())
    );
    if let Some(error) = &report.engine_report.error {
        let _ = writeln!(out, "- Error: `{error}`");
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Ranked Hotspots\n");
    for entry in &actionable_ranking.named_hotspots {
        let _ = writeln!(
            out,
            "- rank {} {}: {}={} -> {} [{}]",
            entry.rank,
            entry.subsystem,
            entry.metric_kind,
            entry.metric_value,
            entry.implication,
            entry.mapped_beads.join(", ")
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Baseline Reuse Ledger\n");
    for entry in &actionable_ranking.baseline_reuse_ledger {
        let hit_rate = entry
            .hit_rate_basis_points
            .map_or_else(|| "n/a".to_owned(), |value| value.to_string());
        let _ = writeln!(
            out,
            "- rank {} {}: supported={}, hits={}, misses={}, hit_rate_bps={} -> {} [{}]",
            entry.rank,
            entry.surface,
            entry.supported,
            entry.hits,
            entry.misses,
            hit_rate,
            entry.implication,
            entry.mapped_beads.join(", ")
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Baseline Waste Ledger\n");
    for entry in &actionable_ranking.baseline_waste_ledger {
        let wall_share = entry
            .wall_share_basis_points
            .map_or_else(|| "n/a".to_owned(), |value| value.to_string());
        let _ = writeln!(
            out,
            "- rank {} {}: class={}, {}={}, wall_share_bps={}, allocator_pressure_bytes={}, activity_count={} -> {} [{}]",
            entry.rank,
            entry.component,
            entry.classification,
            entry.metric_kind,
            entry.metric_value,
            wall_share,
            entry.allocator_pressure_bytes,
            entry.activity_count,
            entry.implication,
            entry.mapped_beads.join(", ")
        );
    }
    let _ = writeln!(
        out,
        "- note: baseline and structural spillover entries are intentionally listed together here so low-retry rows can be separated from contention-driven wall time without hiding either class of cost."
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Quantified Cost Components\n");
    for entry in &actionable_ranking.cost_components {
        let _ = writeln!(
            out,
            "- rank {} {}: time_ns={}, time_share_bps={}, allocator_pressure_bytes={}, allocator_share_bps={}, activity_count={} -> {} [{}]",
            entry.rank,
            entry.component,
            entry.time_ns,
            entry.time_share_basis_points,
            entry.allocator_pressure_bytes,
            entry.allocator_share_basis_points,
            entry.activity_count,
            entry.implication,
            entry.mapped_beads.join(", ")
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Wall-Time Decomposition\n");
    for entry in &actionable_ranking.wall_time_components {
        let _ = writeln!(
            out,
            "- rank {} {}: time_ns={}, wall_share_bps={} -> {} [{}]",
            entry.rank,
            entry.component,
            entry.time_ns,
            entry.wall_share_basis_points,
            entry.implication,
            entry.mapped_beads.join(", ")
        );
    }
    let _ = writeln!(
        out,
        "- note: component shares are evidence-backed but may overlap on multi-worker runs, so they should steer classification rather than be treated as an exclusive partition."
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "## Microarchitectural Signatures\n");
    for entry in &actionable_ranking.microarchitectural_signatures {
        let secondary = if entry.secondary_signatures.is_empty() {
            String::from("none")
        } else {
            entry.secondary_signatures.join(", ")
        };
        let evidence = if entry.evidence_sources.is_empty() {
            String::from("none")
        } else {
            entry.evidence_sources.join(", ")
        };
        let _ = writeln!(
            out,
            "- rank {} {}: primary={}, secondary={}, confidence={} ({}bp), mixed={} -> {} [{}] evidence={}",
            entry.rank,
            entry.target,
            entry.primary_signature,
            secondary,
            entry.confidence_label,
            entry.confidence_score_basis_points,
            entry.mixed_or_ambiguous,
            entry.implication,
            entry.mapped_beads.join(", "),
            evidence
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Allocator Pressure\n");
    for entry in &actionable_ranking.allocator_pressure {
        let _ = writeln!(
            out,
            "- rank {} {}: {}={} -> {} [{}]",
            entry.rank,
            entry.subsystem,
            entry.metric_kind,
            entry.metric_value,
            entry.implication,
            entry.mapped_beads.join(", ")
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Top Opcodes\n");
    for entry in report.opcode_profile.iter().take(12) {
        let _ = writeln!(out, "- {}: {}", entry.opcode, entry.total);
    }
    let _ = writeln!(out);

    let _ = writeln!(out, "## Replay\n");
    let _ = writeln!(out, "```sh\n{}\n```", report.replay_command);
    let _ = writeln!(out);
    let _ = writeln!(out, "## Structured Artifacts\n");
    let _ = writeln!(out, "- `profile.json` — raw scenario profile");
    let _ = writeln!(
        out,
        "- `opcode_profile.json` — raw opcode totals for this profiled run"
    );
    let _ = writeln!(
        out,
        "- `subsystem_profile.json` — raw execution-subsystem timing and heap profile for this run"
    );
    let _ = writeln!(
        out,
        "- `actionable_ranking.json` — hotspot, reuse, and baseline-waste ledger for follow-on Track I/Track J work"
    );
    let _ = writeln!(
        out,
        "- `manifest.json` — replay metadata + artifact inventory"
    );
    out
}

fn hotspot_implication(subsystem: &str) -> (&'static str, &'static [&'static str]) {
    match subsystem {
        "parser_ast_churn" => (
            "J2/J4 target: parser, AST, and compile churn should be reduced through prepared-artifact reuse and arena-backed scratch.",
            &["bd-db300.10.2", "bd-db300.10.4"],
        ),
        "record_decode" => (
            "J2/J5 target: row decode work is expensive enough to justify scratch-space reuse, decode caching, and copy reduction.",
            &["bd-db300.10.2", "bd-db300.10.5"],
        ),
        "row_materialization" => (
            "J2/J6/J7 target: result-row materialization is still paying avoidable clone/allocation and reusable-frame cost in the mixed hot path.",
            &["bd-db300.10.2", "bd-db300.10.6", "bd-db300.10.7"],
        ),
        _ => (
            "Secondary follow-up bucket after the named Track J hotspots.",
            &[],
        ),
    }
}

fn allocator_implication(subsystem: &str) -> (&'static str, &'static [&'static str]) {
    match subsystem {
        "result_row_values" => (
            "J2/J6/J7 target: emitted result rows are carrying most of the transient heap pressure and should benefit from ownership and frame reuse.",
            &["bd-db300.10.2", "bd-db300.10.6", "bd-db300.10.7"],
        ),
        "record_decode_values" => (
            "J2/J5 target: decoded record values create enough heap churn to justify scratch buffers and decode caching.",
            &["bd-db300.10.2", "bd-db300.10.5"],
        ),
        "parser_sql_bytes" => (
            "J2/J4 target: parse-volume churn is visible and should be reduced with reuse rather than repeated prepare work.",
            &["bd-db300.10.2", "bd-db300.10.4"],
        ),
        _ => (
            "Secondary allocator-pressure source after the named Track J hotspots.",
            &[],
        ),
    }
}

fn cost_component_implication(component: &str) -> (&'static str, &'static [&'static str]) {
    match component {
        "parser_ast_churn" => (
            "J2/J4 target: parser and compile reuse still dominate this component enough to justify prepared-artifact work next.",
            &["bd-db300.10.2", "bd-db300.10.4"],
        ),
        "record_decode" => (
            "J2/J5 target: decode cost is large enough to justify scratch buffers and decode-cache work.",
            &["bd-db300.10.2", "bd-db300.10.5"],
        ),
        "row_materialization" => (
            "J2/J6/J7 target: emitted-row cloning and transient value ownership remain a first-class hot-path cost.",
            &["bd-db300.10.2", "bd-db300.10.6", "bd-db300.10.7"],
        ),
        _ => (
            "Secondary cost component after the named Track J follow-on buckets.",
            &[],
        ),
    }
}

fn wall_time_component_implication(component: &str) -> (&'static str, &'static [&'static str]) {
    match component {
        "queueing" => (
            "A/B target: retried BUSY attempts are consuming visible wall time before useful work resumes, so the handoff policy should be tightened before scaling further.",
            &["bd-db300.2.4"],
        ),
        "synchronization" => (
            "A/E target: transaction-boundary coordination is still a first-class wall-time tax and should be pushed toward narrower residual serialized regions.",
            &["bd-db300.1.5", "bd-db300.5.1"],
        ),
        "retry" => (
            "B target: sleep-based retry backoff is still showing up in the hot cell and should be replaced with a bounded handoff strategy.",
            &["bd-db300.2.4"],
        ),
        "service" => (
            "J target: useful body execution still dominates enough wall time that parser, decode, row-path, and residual VDBE/page-motion optimizations remain the main throughput lever once contention is under control.",
            &[
                "bd-db300.10.2",
                "bd-db300.10.4",
                "bd-db300.10.5",
                "bd-db300.10.6",
                "bd-db300.10.7",
                "bd-db300.10.8",
            ],
        ),
        "allocator_copy" => (
            "J target: allocator and copy work is large enough to justify scratch-space reuse, row-value ownership reduction, and reusable buffers.",
            &[
                "bd-db300.10.2",
                "bd-db300.10.3",
                "bd-db300.10.6",
                "bd-db300.10.7",
            ],
        ),
        "durability" => (
            "E target: commit durability is now an explicit measured lane, so future architecture work can separate durable ordering from general executor service cost.",
            &["bd-db300.5.1"],
        ),
        _ => (
            "Secondary wall-time bucket after the named Track A steering categories.",
            &[],
        ),
    }
}

fn hot_path_signature_confidence_label(confidence_score_basis_points: u32) -> &'static str {
    match confidence_score_basis_points {
        8_000..=u32::MAX => "high",
        5_500..=7_999 => "medium",
        _ => "low",
    }
}

fn push_unique_signature_evidence(values: &mut Vec<String>, value: impl Into<String>) {
    let value = value.into();
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn top_subsystem_for_service(report: &HotPathProfileReport) -> &'static str {
    report
        .subsystem_ranking
        .first()
        .map_or("service_mixed", |entry| match entry.subsystem.as_str() {
            "parser_ast_churn" => "parser_ast_churn",
            "record_decode" => "record_decode",
            "row_materialization" => "row_materialization",
            _ => "service_mixed",
        })
}

fn component_signature_shape(
    report: &HotPathProfileReport,
    component: &str,
    counter_capture_summary: Option<&HotPathCounterCaptureManifestSummary>,
) -> (
    &'static str,
    Vec<&'static str>,
    u32,
    bool,
    Vec<String>,
    String,
) {
    let mut evidence_sources = Vec::new();
    push_unique_signature_evidence(
        &mut evidence_sources,
        format!("wall_time_component:{component}"),
    );
    let has_capture = |capture: &str| {
        counter_capture_summary.is_some_and(|summary| {
            summary
                .host_capability_sensitive_captures
                .iter()
                .chain(summary.topology_sensitive_captures.iter())
                .any(|entry| entry == capture)
        })
    };
    let has_fallback_metric = |metric: &str| {
        counter_capture_summary.is_some_and(|summary| {
            summary
                .fallback_metric_pack
                .iter()
                .any(|entry| entry == metric)
        })
    };
    if has_capture("cache_to_cache") {
        push_unique_signature_evidence(
            &mut evidence_sources,
            "counter_capture:cache_to_cache(perf-c2c/perf-mem fallback)",
        );
    }
    if has_capture("remote_access") {
        push_unique_signature_evidence(
            &mut evidence_sources,
            "counter_capture:remote_access(perf-mem/perf-stat fallback)",
        );
    }
    if has_capture("migration") {
        push_unique_signature_evidence(
            &mut evidence_sources,
            "counter_capture:migration(cpu-migrations/context-switch fallback)",
        );
    }
    if has_capture("topdown") {
        push_unique_signature_evidence(
            &mut evidence_sources,
            "counter_capture:topdown(TopdownL1/core-event fallback)",
        );
    }
    if has_fallback_metric("cache-misses") {
        push_unique_signature_evidence(&mut evidence_sources, "fallback_metric:cache-misses");
    }

    match component {
        "durability" => {
            push_unique_signature_evidence(
                &mut evidence_sources,
                "wal_runtime:group_commit_latency_us_total",
            );
            push_unique_signature_evidence(
                &mut evidence_sources,
                "wal_runtime:checkpoint_duration_us_total",
            );
            (
                "durability_pressure",
                Vec::new(),
                9_000,
                false,
                evidence_sources,
                String::from(
                    "WAL group-commit/checkpoint latency is directly measured on the commit-finalize path, so this cell is a genuine durability-pressure lane rather than a generic executor slowdown.",
                ),
            )
        }
        "synchronization" => {
            let mut secondary = Vec::new();
            if has_capture("remote_access") || has_capture("migration") {
                secondary.push("remote_numa_traffic");
            }
            (
                if has_capture("cache_to_cache") {
                    "ownership_ping_pong_hitm"
                } else {
                    "mixed_or_ambiguous"
                },
                secondary,
                if has_capture("cache_to_cache") {
                    7_000
                } else {
                    4_000
                },
                true,
                evidence_sources,
                String::from(
                    "BEGIN/COMMIT/ROLLBACK boundary time is visible, and the topology-aware capture set points at coordination traffic, but without raw counter values this remains an honest mixed synchronization story rather than a fabricated single cause.",
                ),
            )
        }
        "queueing" | "retry" => {
            let mut secondary = Vec::new();
            if has_capture("cache_to_cache") {
                secondary.push("ownership_ping_pong_hitm");
            }
            if has_capture("remote_access") || has_capture("migration") {
                secondary.push("remote_numa_traffic");
            }
            (
                "mixed_or_ambiguous",
                secondary,
                if has_capture("cache_to_cache") || has_capture("remote_access") {
                    5_500
                } else {
                    3_500
                },
                true,
                evidence_sources,
                String::from(
                    "BUSY/retry wall time proves contention is present, but the available evidence only narrows it to topology-sensitive coordination candidates rather than a single definitive microarchitectural culprit.",
                ),
            )
        }
        "allocator_copy" => (
            "llc_pressure",
            vec!["tlb_pressure"],
            if has_fallback_metric("cache-misses") {
                6_200
            } else {
                5_400
            },
            true,
            evidence_sources,
            String::from(
                "Copy-heavy row materialization is the clearest cache/memory lane in the hot path, and the capture pack includes cache-oriented evidence, so LLC/TLB pressure is the most defensible signature pair here.",
            ),
        ),
        "service" => match top_subsystem_for_service(report) {
            "parser_ast_churn" => (
                "front_end_starvation",
                vec!["branch_waste"],
                if has_capture("topdown") { 6_300 } else { 4_800 },
                true,
                {
                    push_unique_signature_evidence(
                        &mut evidence_sources,
                        "subsystem_hotspot:parser_ast_churn",
                    );
                    evidence_sources
                },
                String::from(
                    "Service time is dominated by parser/compile work, so front-end starvation with branch waste as a secondary signature is the most honest interpretation of the current evidence pack.",
                ),
            ),
            "record_decode" => (
                "llc_pressure",
                vec!["tlb_pressure"],
                if has_fallback_metric("cache-misses") || has_capture("remote_access") {
                    6_400
                } else {
                    5_000
                },
                true,
                {
                    push_unique_signature_evidence(
                        &mut evidence_sources,
                        "subsystem_hotspot:record_decode",
                    );
                    evidence_sources
                },
                String::from(
                    "Useful service work is concentrated in record decode, which is the strongest cache/memory consumer in the measured subsystem mix, so LLC/TLB pressure is the least speculative signature pair.",
                ),
            ),
            "row_materialization" => (
                "llc_pressure",
                vec!["remote_numa_traffic"],
                if has_capture("remote_access") || has_capture("cache_to_cache") {
                    5_900
                } else {
                    4_900
                },
                true,
                {
                    push_unique_signature_evidence(
                        &mut evidence_sources,
                        "subsystem_hotspot:row_materialization",
                    );
                    evidence_sources
                },
                String::from(
                    "Service time is being spent cloning row values, which is predominantly a cache/copy path; remote-access evidence keeps the cross-node possibility explicit instead of hiding it.",
                ),
            ),
            _ => (
                "mixed_or_ambiguous",
                Vec::new(),
                3_500,
                true,
                evidence_sources,
                String::from(
                    "Service time is present but the current profile does not justify a stronger microarchitectural story than an explicit mixed/ambiguous tag.",
                ),
            ),
        },
        _ => (
            "mixed_or_ambiguous",
            Vec::new(),
            3_000,
            true,
            evidence_sources,
            String::from(
                "This wall-time bucket is not yet strong enough to support a more specific microarchitectural classification.",
            ),
        ),
    }
}

fn build_hot_path_microarchitectural_signatures(
    report: &HotPathProfileReport,
    counter_capture_summary: Option<&HotPathCounterCaptureManifestSummary>,
    microarchitectural_context: Option<&HotPathMicroarchitecturalContext>,
) -> Vec<HotPathMicroarchitecturalSignatureEntry> {
    let fixture_id = microarchitectural_context.map_or_else(
        || report.fixture_id.clone(),
        |value| value.fixture_id.clone(),
    );
    let row_id = microarchitectural_context.map(|value| value.row_id.clone());
    let mode_id = microarchitectural_context.map(|value| value.mode_id.clone());
    let placement_profile_id =
        microarchitectural_context.and_then(|value| value.placement_profile_id.clone());
    let hardware_class_id =
        microarchitectural_context.and_then(|value| value.hardware_class_id.clone());
    let hardware_signature =
        microarchitectural_context.and_then(|value| value.hardware_signature.clone());
    let wall_time_components = build_hot_path_wall_time_components(report);
    let mut entries = Vec::with_capacity(wall_time_components.len());

    for component in wall_time_components {
        let (
            primary_signature,
            secondary_signatures,
            confidence_score_basis_points,
            mixed,
            evidence_sources,
            rationale,
        ) = component_signature_shape(report, &component.component, counter_capture_summary);
        let (implication, mapped_beads) = wall_time_component_implication(&component.component);
        entries.push(HotPathMicroarchitecturalSignatureEntry {
            rank: 0,
            target: component.component,
            primary_signature: primary_signature.to_owned(),
            secondary_signatures: secondary_signatures
                .into_iter()
                .map(str::to_owned)
                .collect(),
            confidence_label: hot_path_signature_confidence_label(confidence_score_basis_points)
                .to_owned(),
            confidence_score_basis_points,
            mixed_or_ambiguous: mixed,
            rationale,
            evidence_sources,
            fixture_id: fixture_id.clone(),
            row_id: row_id.clone(),
            mode_id: mode_id.clone(),
            placement_profile_id: placement_profile_id.clone(),
            hardware_class_id: hardware_class_id.clone(),
            hardware_signature: hardware_signature.clone(),
            implication: implication.to_owned(),
            mapped_beads: mapped_beads.iter().map(|bead| (*bead).to_owned()).collect(),
        });
    }

    entries.sort_by(|lhs, rhs| {
        rhs.confidence_score_basis_points
            .cmp(&lhs.confidence_score_basis_points)
            .then_with(|| lhs.target.cmp(&rhs.target))
    });
    for (rank, entry) in entries.iter_mut().enumerate() {
        entry.rank = u32::try_from(rank + 1).unwrap_or(u32::MAX);
    }
    entries
}

fn ratio_basis_points(value: u64, total: u64) -> u32 {
    if total == 0 {
        return 0;
    }
    let numerator = u128::from(value).saturating_mul(10_000);
    let denominator = u128::from(total);
    let rounded = numerator.saturating_add(denominator / 2) / denominator;
    u32::try_from(rounded).unwrap_or(u32::MAX)
}

fn actionable_entry(
    rank: usize,
    entry: &HotPathRankingEntry,
    implication: &'static str,
    mapped_beads: &'static [&'static str],
) -> HotPathActionableRankingEntry {
    HotPathActionableRankingEntry {
        rank: u32::try_from(rank + 1).unwrap_or(u32::MAX),
        subsystem: entry.subsystem.clone(),
        metric_kind: entry.metric_kind.clone(),
        metric_value: entry.metric_value,
        rationale: entry.rationale.clone(),
        implication: implication.to_owned(),
        mapped_beads: mapped_beads.iter().map(|bead| (*bead).to_owned()).collect(),
    }
}

fn baseline_reuse_implication(surface: &str) -> (&'static str, &'static [&'static str]) {
    match surface {
        "statement_parse_cache" => (
            "J4 target: repeated parse misses still show avoidable prepare churn on the low-contention path.",
            &["bd-db300.10.4"],
        ),
        "compiled_plan_cache" => (
            "J4 target: compiled-plan misses are direct evidence for statement/plan caching work.",
            &["bd-db300.10.4"],
        ),
        "record_decode_cache" => (
            "J5 target: decode reuse is still invisible, so the report should assume repeated record parsing until explicit cache hits exist.",
            &["bd-db300.10.5"],
        ),
        "cursor_frame_reuse" => (
            "J7 target: missing cursor/frame reuse evidence means VDBE setup churn is still treated as open baseline tax.",
            &["bd-db300.10.7"],
        ),
        "page_buffer_pool_reuse" => (
            "J3/J8 target: page-buffer reuse remains opaque in the hot-path report and needs explicit pool-hit evidence.",
            &["bd-db300.10.3", "bd-db300.10.8"],
        ),
        "page_data_ownership_reuse" => (
            "J6 target: PageData ownership reuse is still not measured directly, so clone/vec/arc churn remains a first-class suspect.",
            &["bd-db300.10.6"],
        ),
        _ => (
            "Secondary baseline reuse surface after the named Track J cache/reuse buckets.",
            &[],
        ),
    }
}

fn build_hot_path_baseline_reuse_ledger(
    report: &HotPathProfileReport,
) -> Vec<HotPathBaselineReuseLedgerEntry> {
    let parser_calls = report
        .parser
        .parse_single_calls
        .saturating_add(report.parser.parse_multi_calls);
    let decode_calls = report
        .record_decode
        .parse_record_into_calls
        .saturating_add(report.record_decode.parse_record_column_calls);
    let result_rows = report.row_materialization.result_rows_total;
    let mut entries = vec![
        HotPathBaselineReuseLedgerEntry {
            rank: 0,
            surface: "statement_parse_cache".to_owned(),
            supported: true,
            hits: report.parser.parse_cache_hits,
            misses: report.parser.parse_cache_misses,
            hit_rate_basis_points: Some(ratio_basis_points(
                report.parser.parse_cache_hits,
                report
                    .parser
                    .parse_cache_hits
                    .saturating_add(report.parser.parse_cache_misses),
            )),
            rationale: format!(
                "parse cache hits/misses are measured directly across {} parse calls",
                parser_calls
            ),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineReuseLedgerEntry {
            rank: 0,
            surface: "compiled_plan_cache".to_owned(),
            supported: true,
            hits: report.parser.compiled_cache_hits,
            misses: report.parser.compiled_cache_misses,
            hit_rate_basis_points: Some(ratio_basis_points(
                report.parser.compiled_cache_hits,
                report
                    .parser
                    .compiled_cache_hits
                    .saturating_add(report.parser.compiled_cache_misses),
            )),
            rationale:
                "compiled-plan cache hits/misses are captured directly at the prepare/compile boundary"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineReuseLedgerEntry {
            rank: 0,
            surface: "record_decode_cache".to_owned(),
            supported: false,
            hits: 0,
            misses: 0,
            hit_rate_basis_points: None,
            rationale: format!(
                "record decode activity is measured ({} decode calls), but decode-cache hits/misses are not yet surfaced in the report layer",
                decode_calls
            ),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineReuseLedgerEntry {
            rank: 0,
            surface: "cursor_frame_reuse".to_owned(),
            supported: false,
            hits: 0,
            misses: 0,
            hit_rate_basis_points: None,
            rationale:
                "runtime phase timing exists, but cursor/frame reuse versus fresh setup is not yet emitted as a machine-readable counter"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineReuseLedgerEntry {
            rank: 0,
            surface: "page_buffer_pool_reuse".to_owned(),
            supported: false,
            hits: 0,
            misses: 0,
            hit_rate_basis_points: None,
            rationale:
                "pager-side hit counters exist elsewhere, but this hot-path report does not yet expose page-buffer pool reuse directly"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineReuseLedgerEntry {
            rank: 0,
            surface: "page_data_ownership_reuse".to_owned(),
            supported: false,
            hits: 0,
            misses: 0,
            hit_rate_basis_points: None,
            rationale: format!(
                "row materialization emitted {} rows, but PageData ownership reuse versus clone/vec/arc churn is still uninstrumented",
                result_rows
            ),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
    ];
    entries.sort_by(|lhs, rhs| {
        let lhs_supported_rank = if lhs.supported { 1u8 } else { 0u8 };
        let rhs_supported_rank = if rhs.supported { 1u8 } else { 0u8 };
        lhs_supported_rank
            .cmp(&rhs_supported_rank)
            .then_with(|| {
                lhs.hit_rate_basis_points
                    .unwrap_or(0)
                    .cmp(&rhs.hit_rate_basis_points.unwrap_or(0))
            })
            .then_with(|| {
                rhs.hits
                    .saturating_add(rhs.misses)
                    .cmp(&lhs.hits.saturating_add(lhs.misses))
            })
            .then_with(|| lhs.surface.cmp(&rhs.surface))
    });
    for (rank, entry) in entries.iter_mut().enumerate() {
        entry.rank = u32::try_from(rank + 1).unwrap_or(u32::MAX);
        let (implication, mapped_beads) = baseline_reuse_implication(&entry.surface);
        entry.implication = implication.to_owned();
        entry.mapped_beads = mapped_beads.iter().map(|bead| (*bead).to_owned()).collect();
    }
    entries
}

fn baseline_waste_implication(component: &str) -> (&'static str, &'static [&'static str]) {
    match component {
        "parser_prepare_churn" => (
            "J2/J4 target: parse/rewrite/compile work is still visible enough that caching and arena-backed scratch should move first.",
            &["bd-db300.10.2", "bd-db300.10.4"],
        ),
        "record_decode" => (
            "J2/J5 target: decode time and decoded-value heap churn remain direct baseline-tax candidates.",
            &["bd-db300.10.2", "bd-db300.10.5"],
        ),
        "row_materialization" => (
            "J2/J6/J7 target: emitted-row cloning is still paying avoidable heap and ownership cost on the common path.",
            &["bd-db300.10.2", "bd-db300.10.6", "bd-db300.10.7"],
        ),
        "executor_body_residual" => (
            "J6/J7/J8 target: residual service time beyond decode/materialization is where VDBE setup, cursor motion, page fetch, and ownership churn likely still hide.",
            &["bd-db300.10.6", "bd-db300.10.7", "bd-db300.10.8"],
        ),
        "boundary_coordination" => (
            "Mixed lane: boundary coordination should stay visible so baseline fixes do not accidentally absorb residual commit/path coordination into the wrong bucket.",
            &["bd-db300.1.5", "bd-db300.5.1"],
        ),
        "busy_retry_queueing" => (
            "Structural spillover: retry and BUSY queueing are not baseline tax and should steer Track A work instead of J-lane fixes.",
            &["bd-db300.2.4"],
        ),
        "durability" => (
            "Mixed lane: durability must stay explicit so later baseline work does not overclaim gains that really belong to WAL/commit-path changes.",
            &["bd-db300.5.1"],
        ),
        _ => (
            "Secondary baseline or spillover bucket after the named Track J categories.",
            &[],
        ),
    }
}

fn build_hot_path_baseline_waste_ledger(
    report: &HotPathProfileReport,
) -> Vec<HotPathBaselineWasteLedgerEntry> {
    let runtime_phase_timing = report
        .engine_report
        .runtime_phase_timing
        .unwrap_or_default();
    let wall_time_ns = report.engine_report.wall_time_ms.saturating_mul(1_000_000);
    let parser_time_ns = report
        .parser
        .parse_time_ns
        .saturating_add(report.parser.rewrite_time_ns)
        .saturating_add(report.parser.compile_time_ns);
    let decode_calls = report
        .record_decode
        .parse_record_into_calls
        .saturating_add(report.record_decode.parse_record_column_calls);
    let parser_calls = report
        .parser
        .parse_single_calls
        .saturating_add(report.parser.parse_multi_calls);
    let wal_durability_time_ns =
        report
            .engine_report
            .hot_path_profile
            .as_ref()
            .map_or(0, |hot_path_profile| {
                hot_path_profile
                    .wal
                    .group_commit_latency_us_total
                    .saturating_add(hot_path_profile.wal.checkpoint_duration_us_total)
                    .saturating_mul(1_000)
            });
    let durability_time_ns =
        wal_durability_time_ns.min(runtime_phase_timing.commit_finalize_time_ns);
    let boundary_coordination_time_ns = runtime_phase_timing
        .begin_boundary_time_ns
        .saturating_add(runtime_phase_timing.rollback_time_ns)
        .saturating_add(
            runtime_phase_timing
                .commit_finalize_time_ns
                .saturating_sub(durability_time_ns),
        );
    let executor_body_residual = runtime_phase_timing
        .body_execution_time_ns
        .saturating_sub(report.record_decode.decode_time_ns)
        .saturating_sub(
            report
                .row_materialization
                .result_row_materialization_time_ns_total,
        );
    let busy_retry_queueing_time_ns = runtime_phase_timing
        .busy_attempt_time_ns
        .saturating_add(runtime_phase_timing.retry_backoff_time_ns);
    let mut entries = vec![
        HotPathBaselineWasteLedgerEntry {
            rank: 0,
            component: "parser_prepare_churn".to_owned(),
            classification: "baseline_tax".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: parser_time_ns,
            wall_share_basis_points: Some(ratio_basis_points(parser_time_ns, wall_time_ns)),
            allocator_pressure_bytes: report.allocator_pressure.parser_sql_bytes,
            activity_count: parser_calls,
            rationale:
                "parse, rewrite, and compile time represent repeated prepare-path work on the common path"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineWasteLedgerEntry {
            rank: 0,
            component: "record_decode".to_owned(),
            classification: "baseline_tax".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: report.record_decode.decode_time_ns,
            wall_share_basis_points: Some(ratio_basis_points(
                report.record_decode.decode_time_ns,
                wall_time_ns,
            )),
            allocator_pressure_bytes: report.record_decode.vdbe_decoded_value_heap_bytes_total,
            activity_count: decode_calls,
            rationale:
                "record decode time and decoded-value heap bytes are measured directly in the VDBE hot path"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineWasteLedgerEntry {
            rank: 0,
            component: "row_materialization".to_owned(),
            classification: "baseline_tax".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: report
                .row_materialization
                .result_row_materialization_time_ns_total,
            wall_share_basis_points: Some(ratio_basis_points(
                report
                    .row_materialization
                    .result_row_materialization_time_ns_total,
                wall_time_ns,
            )),
            allocator_pressure_bytes: report.row_materialization.result_value_heap_bytes_total,
            activity_count: report.row_materialization.result_rows_total,
            rationale:
                "result-row materialization isolates emitted-value cloning and transient ownership work"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineWasteLedgerEntry {
            rank: 0,
            component: "executor_body_residual".to_owned(),
            classification: "mixed_or_residual".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: executor_body_residual,
            wall_share_basis_points: Some(ratio_basis_points(
                executor_body_residual,
                wall_time_ns,
            )),
            allocator_pressure_bytes: 0,
            activity_count: report.engine_report.ops_total,
            rationale:
                "body execution time that remains after explicit decode and row-materialization accounting; likely mixes VDBE dispatch, cursor motion, page fetch, and PageData ownership churn"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineWasteLedgerEntry {
            rank: 0,
            component: "boundary_coordination".to_owned(),
            classification: "mixed_or_residual".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: boundary_coordination_time_ns,
            wall_share_basis_points: Some(ratio_basis_points(
                boundary_coordination_time_ns,
                wall_time_ns,
            )),
            allocator_pressure_bytes: 0,
            activity_count: report.engine_report.ops_total,
            rationale:
                "BEGIN/ROLLBACK boundaries plus non-durable COMMIT coordination are real wall time, but they are not pure baseline tax and should stay separated"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineWasteLedgerEntry {
            rank: 0,
            component: "busy_retry_queueing".to_owned(),
            classification: "structural_side_effect".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: busy_retry_queueing_time_ns,
            wall_share_basis_points: Some(ratio_basis_points(
                busy_retry_queueing_time_ns,
                wall_time_ns,
            )),
            allocator_pressure_bytes: 0,
            activity_count: report.engine_report.retries,
            rationale:
                "BUSY-attempt time and retry-backoff sleep are contention spillover, not evidence of intrinsic single-writer baseline tax"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathBaselineWasteLedgerEntry {
            rank: 0,
            component: "durability".to_owned(),
            classification: "mixed_or_residual".to_owned(),
            metric_kind: "time_ns".to_owned(),
            metric_value: durability_time_ns,
            wall_share_basis_points: Some(ratio_basis_points(durability_time_ns, wall_time_ns)),
            allocator_pressure_bytes: 0,
            activity_count: report.engine_report.ops_total,
            rationale:
                "WAL durability is an explicit measured lane and must remain visible so baseline optimization claims stay honest"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
    ];
    entries.sort_by(|lhs, rhs| {
        rhs.metric_value
            .cmp(&lhs.metric_value)
            .then_with(|| lhs.component.cmp(&rhs.component))
    });
    for (rank, entry) in entries.iter_mut().enumerate() {
        entry.rank = u32::try_from(rank + 1).unwrap_or(u32::MAX);
        let (implication, mapped_beads) = baseline_waste_implication(&entry.component);
        entry.implication = implication.to_owned();
        entry.mapped_beads = mapped_beads.iter().map(|bead| (*bead).to_owned()).collect();
    }
    entries
}

fn build_hot_path_cost_components(report: &HotPathProfileReport) -> Vec<HotPathCostComponentEntry> {
    let parser_time_ns = report
        .parser
        .parse_time_ns
        .saturating_add(report.parser.rewrite_time_ns)
        .saturating_add(report.parser.compile_time_ns);
    let parser_activity_count = report
        .parser
        .parse_single_calls
        .saturating_add(report.parser.parse_multi_calls);
    let record_decode_activity_count = report
        .record_decode
        .parse_record_into_calls
        .saturating_add(report.record_decode.parse_record_column_calls);
    let total_time_ns = parser_time_ns
        .saturating_add(report.record_decode.decode_time_ns)
        .saturating_add(
            report
                .row_materialization
                .result_row_materialization_time_ns_total,
        );
    let total_allocator_pressure_bytes = report
        .allocator_pressure
        .decoded_value_heap_bytes_total
        .saturating_add(report.allocator_pressure.result_value_heap_bytes_total);

    let mut entries = vec![
        HotPathCostComponentEntry {
            rank: 0,
            component: "parser_ast_churn".to_owned(),
            time_ns: parser_time_ns,
            time_share_basis_points: ratio_basis_points(parser_time_ns, total_time_ns),
            allocator_pressure_bytes: 0,
            allocator_share_basis_points: 0,
            activity_count: parser_activity_count,
            rationale:
                "parse, rewrite, and compile time on the connection path; parser heap pressure is not directly instrumented"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathCostComponentEntry {
            rank: 0,
            component: "record_decode".to_owned(),
            time_ns: report.record_decode.decode_time_ns,
            time_share_basis_points: ratio_basis_points(
                report.record_decode.decode_time_ns,
                total_time_ns,
            ),
            allocator_pressure_bytes: report.record_decode.vdbe_decoded_value_heap_bytes_total,
            allocator_share_basis_points: ratio_basis_points(
                report.record_decode.vdbe_decoded_value_heap_bytes_total,
                total_allocator_pressure_bytes,
            ),
            activity_count: record_decode_activity_count,
            rationale: "record/column decode time paired with decoded-value heap pressure"
                .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathCostComponentEntry {
            rank: 0,
            component: "row_materialization".to_owned(),
            time_ns: report
                .row_materialization
                .result_row_materialization_time_ns_total,
            time_share_basis_points: ratio_basis_points(
                report
                    .row_materialization
                    .result_row_materialization_time_ns_total,
                total_time_ns,
            ),
            allocator_pressure_bytes: report.row_materialization.result_value_heap_bytes_total,
            allocator_share_basis_points: ratio_basis_points(
                report.row_materialization.result_value_heap_bytes_total,
                total_allocator_pressure_bytes,
            ),
            activity_count: report.row_materialization.result_rows_total,
            rationale: "result-row cloning time paired with emitted-value allocator pressure"
                .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
    ];
    entries.sort_by(|lhs, rhs| {
        rhs.time_ns
            .cmp(&lhs.time_ns)
            .then_with(|| {
                rhs.allocator_pressure_bytes
                    .cmp(&lhs.allocator_pressure_bytes)
            })
            .then_with(|| lhs.component.cmp(&rhs.component))
    });
    for (rank, entry) in entries.iter_mut().enumerate() {
        entry.rank = u32::try_from(rank + 1).unwrap_or(u32::MAX);
        let (implication, mapped_beads) = cost_component_implication(&entry.component);
        entry.implication = implication.to_owned();
        entry.mapped_beads = mapped_beads.iter().map(|bead| (*bead).to_owned()).collect();
    }
    entries
}

fn build_hot_path_wall_time_components(
    report: &HotPathProfileReport,
) -> Vec<HotPathWallTimeComponentEntry> {
    let wall_time_ns = report.engine_report.wall_time_ms.saturating_mul(1_000_000);
    let allocator_copy_time_ns = report
        .row_materialization
        .result_row_materialization_time_ns_total;
    let runtime_phase_timing = report
        .engine_report
        .runtime_phase_timing
        .unwrap_or_default();
    let wal_durability_time_ns =
        report
            .engine_report
            .hot_path_profile
            .as_ref()
            .map_or(0, |hot_path_profile| {
                hot_path_profile
                    .wal
                    .group_commit_latency_us_total
                    .saturating_add(hot_path_profile.wal.checkpoint_duration_us_total)
                    .saturating_mul(1_000)
            });
    let durability_time_ns =
        wal_durability_time_ns.min(runtime_phase_timing.commit_finalize_time_ns);
    let synchronization_time_ns = runtime_phase_timing
        .begin_boundary_time_ns
        .saturating_add(runtime_phase_timing.rollback_time_ns)
        .saturating_add(
            runtime_phase_timing
                .commit_finalize_time_ns
                .saturating_sub(durability_time_ns),
        );
    let service_time_ns = runtime_phase_timing
        .body_execution_time_ns
        .saturating_sub(allocator_copy_time_ns);

    let mut entries = vec![
        HotPathWallTimeComponentEntry {
            rank: 0,
            component: "queueing".to_owned(),
            time_ns: runtime_phase_timing.busy_attempt_time_ns,
            wall_share_basis_points: ratio_basis_points(
                runtime_phase_timing.busy_attempt_time_ns,
                wall_time_ns,
            ),
            rationale:
                "elapsed wall time spent inside batch attempts that ended in BUSY and had to queue for a later retry"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathWallTimeComponentEntry {
            rank: 0,
            component: "synchronization".to_owned(),
            time_ns: synchronization_time_ns,
            wall_share_basis_points: ratio_basis_points(synchronization_time_ns, wall_time_ns),
            rationale:
                "BEGIN/ROLLBACK boundary time plus non-durable COMMIT coordination observed directly by the executor"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathWallTimeComponentEntry {
            rank: 0,
            component: "retry".to_owned(),
            time_ns: runtime_phase_timing.retry_backoff_time_ns,
            wall_share_basis_points: ratio_basis_points(
                runtime_phase_timing.retry_backoff_time_ns,
                wall_time_ns,
            ),
            rationale:
                "configured backoff sleep requested after BUSY-family retries in the hot cell"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathWallTimeComponentEntry {
            rank: 0,
            component: "service".to_owned(),
            time_ns: service_time_ns,
            wall_share_basis_points: ratio_basis_points(service_time_ns, wall_time_ns),
            rationale:
                "batch body execution time outside the explicit allocator/copy bucket, capturing useful in-engine service work"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathWallTimeComponentEntry {
            rank: 0,
            component: "allocator_copy".to_owned(),
            time_ns: allocator_copy_time_ns,
            wall_share_basis_points: ratio_basis_points(allocator_copy_time_ns, wall_time_ns),
            rationale:
                "row materialization time measured inline at the VDBE boundary, isolating value cloning and copy-heavy result emission"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
        HotPathWallTimeComponentEntry {
            rank: 0,
            component: "durability".to_owned(),
            time_ns: durability_time_ns,
            wall_share_basis_points: ratio_basis_points(durability_time_ns, wall_time_ns),
            rationale:
                "WAL group-commit and checkpoint latency captured by runtime telemetry, clipped to observed COMMIT finalize time"
                    .to_owned(),
            implication: String::new(),
            mapped_beads: Vec::new(),
        },
    ];
    entries.sort_by(|lhs, rhs| {
        rhs.time_ns
            .cmp(&lhs.time_ns)
            .then_with(|| lhs.component.cmp(&rhs.component))
    });
    for (rank, entry) in entries.iter_mut().enumerate() {
        entry.rank = u32::try_from(rank + 1).unwrap_or(u32::MAX);
        let (implication, mapped_beads) = wall_time_component_implication(&entry.component);
        entry.implication = implication.to_owned();
        entry.mapped_beads = mapped_beads.iter().map(|bead| (*bead).to_owned()).collect();
    }
    entries
}

#[must_use]
pub fn build_hot_path_opcode_profile(report: &HotPathProfileReport) -> HotPathOpcodeProfilePack {
    let mut opcodes = report.opcode_profile.clone();
    opcodes.sort_by(|lhs, rhs| {
        rhs.total
            .cmp(&lhs.total)
            .then_with(|| lhs.opcode.cmp(&rhs.opcode))
    });

    HotPathOpcodeProfilePack {
        schema_version: HOT_PATH_OPCODE_PROFILE_SCHEMA_V1.to_owned(),
        bead_id: report.bead_id.clone(),
        run_id: report.run_id.clone(),
        trace_id: report.trace_id.clone(),
        scenario_id: report.scenario_id.clone(),
        fixture_id: report.fixture_id.clone(),
        workload: report.workload.clone(),
        seed: report.seed,
        scale: report.scale,
        concurrency: report.concurrency,
        replay_command: report.replay_command.clone(),
        opcodes,
    }
}

#[must_use]
pub fn build_hot_path_subsystem_profile(
    report: &HotPathProfileReport,
) -> HotPathSubsystemProfilePack {
    HotPathSubsystemProfilePack {
        schema_version: HOT_PATH_SUBSYSTEM_PROFILE_SCHEMA_V1.to_owned(),
        bead_id: report.bead_id.clone(),
        run_id: report.run_id.clone(),
        trace_id: report.trace_id.clone(),
        scenario_id: report.scenario_id.clone(),
        fixture_id: report.fixture_id.clone(),
        workload: report.workload.clone(),
        seed: report.seed,
        scale: report.scale,
        concurrency: report.concurrency,
        replay_command: report.replay_command.clone(),
        subsystem_ranking: report.subsystem_ranking.clone(),
        allocator_ranking: report.allocator_pressure.ranked_sources.clone(),
        parser: report.parser.clone(),
        record_decode: report.record_decode.clone(),
        row_materialization: report.row_materialization.clone(),
    }
}

#[must_use]
pub fn build_hot_path_actionable_ranking(
    report: &HotPathProfileReport,
    counter_capture_summary: Option<&HotPathCounterCaptureManifestSummary>,
    microarchitectural_context: Option<&HotPathMicroarchitecturalContext>,
) -> HotPathActionableRanking {
    let named_hotspots = report
        .subsystem_ranking
        .iter()
        .enumerate()
        .map(|(rank, entry)| {
            let (implication, mapped_beads) = hotspot_implication(&entry.subsystem);
            actionable_entry(rank, entry, implication, mapped_beads)
        })
        .collect();
    let baseline_reuse_ledger = build_hot_path_baseline_reuse_ledger(report);
    let baseline_waste_ledger = build_hot_path_baseline_waste_ledger(report);
    let allocator_pressure = report
        .allocator_pressure
        .ranked_sources
        .iter()
        .enumerate()
        .map(|(rank, entry)| {
            let (implication, mapped_beads) = allocator_implication(&entry.subsystem);
            actionable_entry(rank, entry, implication, mapped_beads)
        })
        .collect();
    let microarchitectural_signatures = build_hot_path_microarchitectural_signatures(
        report,
        counter_capture_summary,
        microarchitectural_context,
    );
    let wall_time_components = build_hot_path_wall_time_components(report);
    let cost_components = build_hot_path_cost_components(report);
    let top_opcodes = build_hot_path_opcode_profile(report)
        .opcodes
        .into_iter()
        .take(12)
        .collect();

    HotPathActionableRanking {
        schema_version: HOT_PATH_PROFILE_ACTIONABLE_RANKING_SCHEMA_V3.to_owned(),
        bead_id: report.bead_id.clone(),
        run_id: report.run_id.clone(),
        trace_id: report.trace_id.clone(),
        scenario_id: report.scenario_id.clone(),
        fixture_id: report.fixture_id.clone(),
        workload: report.workload.clone(),
        seed: report.seed,
        scale: report.scale,
        concurrency: report.concurrency,
        replay_command: report.replay_command.clone(),
        baseline_reuse_ledger,
        baseline_waste_ledger,
        named_hotspots,
        microarchitectural_signatures,
        wall_time_components,
        cost_components,
        allocator_pressure,
        top_opcodes,
    }
}

pub fn write_hot_path_profile_artifacts(
    report: &HotPathProfileReport,
    output_dir: &Path,
    counter_capture_summary: Option<HotPathCounterCaptureManifestSummary>,
    provenance: Option<HotPathArtifactProvenance>,
    microarchitectural_context: Option<HotPathMicroarchitecturalContext>,
) -> std::io::Result<HotPathArtifactManifest> {
    std::fs::create_dir_all(output_dir)?;

    let report_json = serde_json::to_string_pretty(report)
        .map_err(|error| std::io::Error::other(format!("profile JSON: {error}")))?;
    let opcode_profile = build_hot_path_opcode_profile(report);
    let opcode_profile_json = serde_json::to_string_pretty(&opcode_profile)
        .map_err(|error| std::io::Error::other(format!("opcode profile JSON: {error}")))?;
    let subsystem_profile = build_hot_path_subsystem_profile(report);
    let subsystem_profile_json = serde_json::to_string_pretty(&subsystem_profile)
        .map_err(|error| std::io::Error::other(format!("subsystem profile JSON: {error}")))?;
    let actionable_ranking = build_hot_path_actionable_ranking(
        report,
        counter_capture_summary.as_ref(),
        microarchitectural_context.as_ref(),
    );
    let actionable_ranking_json = serde_json::to_string_pretty(&actionable_ranking)
        .map_err(|error| std::io::Error::other(format!("actionable ranking JSON: {error}")))?;
    let summary_md = render_hot_path_profile_markdown(report);

    let report_path = output_dir.join("profile.json");
    let opcode_profile_path = output_dir.join("opcode_profile.json");
    let subsystem_profile_path = output_dir.join("subsystem_profile.json");
    let actionable_ranking_path = output_dir.join("actionable_ranking.json");
    let summary_path = output_dir.join("summary.md");
    std::fs::write(&report_path, report_json.as_bytes())?;
    std::fs::write(&opcode_profile_path, opcode_profile_json.as_bytes())?;
    std::fs::write(&subsystem_profile_path, subsystem_profile_json.as_bytes())?;
    std::fs::write(&actionable_ranking_path, actionable_ranking_json.as_bytes())?;
    std::fs::write(&summary_path, summary_md.as_bytes())?;

    let manifest = HotPathArtifactManifest {
        schema_version: HOT_PATH_PROFILE_MANIFEST_SCHEMA_V1.to_owned(),
        bead_id: report.bead_id.clone(),
        run_id: report.run_id.clone(),
        trace_id: report.trace_id.clone(),
        scenario_id: report.scenario_id.clone(),
        fixture_id: report.fixture_id.clone(),
        workload: report.workload.clone(),
        seed: report.seed,
        scale: report.scale,
        concurrency: report.concurrency,
        concurrent_mode: report.concurrent_mode,
        run_integrity_check: report.run_integrity_check,
        golden_dir: report.golden_dir.clone(),
        working_base: report.working_base.clone(),
        replay_command: report.replay_command.clone(),
        counter_capture_summary,
        provenance,
        files: vec![
            HotPathArtifactFile {
                path: "profile.json".to_owned(),
                bytes: u64::try_from(report_json.len()).unwrap_or(u64::MAX),
                description: "structured hot-path profile report".to_owned(),
            },
            HotPathArtifactFile {
                path: "opcode_profile.json".to_owned(),
                bytes: u64::try_from(opcode_profile_json.len()).unwrap_or(u64::MAX),
                description: "raw opcode totals for the profiled run".to_owned(),
            },
            HotPathArtifactFile {
                path: "subsystem_profile.json".to_owned(),
                bytes: u64::try_from(subsystem_profile_json.len()).unwrap_or(u64::MAX),
                description: "raw execution-subsystem timing and heap profile for the run"
                    .to_owned(),
            },
            HotPathArtifactFile {
                path: "summary.md".to_owned(),
                bytes: u64::try_from(summary_md.len()).unwrap_or(u64::MAX),
                description: "human-readable hotspot ranking summary".to_owned(),
            },
            HotPathArtifactFile {
                path: "actionable_ranking.json".to_owned(),
                bytes: u64::try_from(actionable_ranking_json.len()).unwrap_or(u64::MAX),
                description:
                    "structured hotspot, reuse, and baseline-waste ledger for follow-on implementation work"
                        .to_owned(),
            },
        ],
    };
    let manifest_json = serde_json::to_string_pretty(&manifest)
        .map_err(|error| std::io::Error::other(format!("artifact manifest: {error}")))?;
    std::fs::write(output_dir.join("manifest.json"), manifest_json.as_bytes())?;

    let mut files = manifest.files.clone();
    files.push(HotPathArtifactFile {
        path: "manifest.json".to_owned(),
        bytes: u64::try_from(manifest_json.len()).unwrap_or(u64::MAX),
        description: "artifact manifest with replay metadata".to_owned(),
    });
    Ok(HotPathArtifactManifest { files, ..manifest })
}

// ── Matrix expansion ───────────────────────────────────────────────────

/// Expand the configuration into a flat list of matrix cells.
#[must_use]
pub fn expand_matrix(config: &PerfMatrixConfig) -> Vec<MatrixCell> {
    let mut cells = Vec::new();
    for engine in &config.engines {
        for fixture_id in &config.fixture_ids {
            for workload in &config.workload_presets {
                for &concurrency in &config.concurrency_levels {
                    cells.push(MatrixCell {
                        engine: *engine,
                        fixture_id: fixture_id.clone(),
                        workload: workload.clone(),
                        concurrency,
                    });
                }
            }
        }
    }
    cells
}

// ── OpLog generation ───────────────────────────────────────────────────

/// Generate an `OpLog` for the given preset name and parameters.
///
/// Returns `None` if the preset name is unrecognized.
#[must_use]
pub fn generate_oplog(
    preset_name: &str,
    fixture_id: &str,
    seed: u64,
    concurrency: u16,
    scale: u32,
) -> Option<OpLog> {
    match preset_name {
        "commutative_inserts_disjoint_keys" => Some(
            oplog::preset_commutative_inserts_disjoint_keys(fixture_id, seed, concurrency, scale),
        ),
        "hot_page_contention" => Some(oplog::preset_hot_page_contention(
            fixture_id,
            seed,
            concurrency,
            scale,
        )),
        "mixed_read_write" => Some(oplog::preset_mixed_read_write(
            fixture_id,
            seed,
            concurrency,
            scale,
        )),
        "deterministic_transform" => Some(oplog::preset_deterministic_transform(
            fixture_id, seed, scale,
        )),
        "large_txn" => Some(oplog::preset_large_txn(
            fixture_id,
            seed,
            concurrency,
            scale,
        )),
        "schema_migration" => Some(oplog::preset_schema_migration(fixture_id, seed, scale)),
        "btree_stress_sequential" => Some(oplog::preset_btree_stress_sequential(
            fixture_id, seed, scale,
        )),
        "wide_row_overflow" => Some(oplog::preset_wide_row_overflow(
            fixture_id, seed, scale, 2000,
        )),
        "bulk_delete_reinsert" => Some(oplog::preset_bulk_delete_reinsert(fixture_id, seed, scale)),
        "scatter_write" => Some(oplog::preset_scatter_write(
            fixture_id,
            seed,
            concurrency,
            scale,
        )),
        "multi_table_foreign_keys" => Some(oplog::preset_multi_table_foreign_keys(
            fixture_id, seed, scale,
        )),
        _ => None,
    }
}

// ── Cell execution ─────────────────────────────────────────────────────

/// Run a single matrix cell through the benchmark runner.
fn run_cell(cell: &MatrixCell, config: &PerfMatrixConfig) -> CellOutcome {
    let Some(oplog) = generate_oplog(
        &cell.workload,
        &cell.fixture_id,
        config.seed,
        cell.concurrency,
        config.scale,
    ) else {
        return CellOutcome {
            summary: None,
            error: Some(format!("unknown workload preset: {}", cell.workload)),
            engine: cell.engine.as_str().to_owned(),
            fixture_id: cell.fixture_id.clone(),
            workload: cell.workload.clone(),
            concurrency: cell.concurrency,
        };
    };

    let meta = BenchmarkMeta {
        engine: cell.engine.as_str().to_owned(),
        workload: cell.workload.clone(),
        fixture_id: cell.fixture_id.clone(),
        concurrency: cell.concurrency,
        cargo_profile: config.cargo_profile.clone(),
    };

    let workspace_config = WorkspaceConfig {
        golden_dir: config.project_root.join("sample_sqlite_db_files/golden"),
        working_base: config.project_root.join("sample_sqlite_db_files/working"),
    };

    let settings = config.settings.clone();
    let engine = cell.engine;
    let fixture_id = cell.fixture_id.clone();

    let summary = run_benchmark(&config.benchmark_config, &meta, |iteration_idx| {
        run_single_iteration(
            engine,
            &fixture_id,
            &oplog,
            &workspace_config,
            &settings,
            iteration_idx,
        )
    });
    let error = benchmark_summary_error(&summary);

    CellOutcome {
        summary: Some(summary),
        error,
        engine: cell.engine.as_str().to_owned(),
        fixture_id: cell.fixture_id.clone(),
        workload: cell.workload.clone(),
        concurrency: cell.concurrency,
    }
}

fn benchmark_summary_error(summary: &BenchmarkSummary) -> Option<String> {
    let failed_iterations = summary
        .iterations
        .iter()
        .filter_map(|iteration| iteration.error.as_deref())
        .collect::<Vec<_>>();
    if failed_iterations.is_empty() {
        return None;
    }

    let failure_count = failed_iterations.len();
    let measurement_count = summary.iterations.len();
    let first_error = failed_iterations[0];
    if failure_count == measurement_count {
        Some(format!(
            "all {measurement_count} measurement iterations failed; first={first_error}"
        ))
    } else {
        Some(format!(
            "{failure_count}/{measurement_count} measurement iterations failed; first={first_error}"
        ))
    }
}

/// Execute a single benchmark iteration for one engine.
///
/// Creates a fresh working copy, applies fairness PRAGMAs, runs the OpLog,
/// and returns the engine report.
fn run_single_iteration(
    engine: Engine,
    fixture_id: &str,
    oplog: &OpLog,
    workspace_config: &WorkspaceConfig,
    settings: &HarnessSettings,
    iteration_idx: u32,
) -> Result<EngineRunReport, crate::E2eError> {
    let label = format!("perf_{engine}_{fixture_id}_iter{iteration_idx}");
    let workspace = create_workspace_with_label(workspace_config, &[fixture_id], &label)?;

    let db = workspace.databases.first().ok_or_else(|| {
        crate::E2eError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("fixture not found in workspace: {fixture_id}"),
        ))
    })?;

    match engine {
        Engine::Sqlite3 => {
            let exec_config = settings.to_sqlite_exec_config();
            run_oplog_sqlite(&db.db_path, oplog, &exec_config)
        }
        Engine::Fsqlite => {
            let exec_config = settings.to_fsqlite_exec_config();
            run_oplog_fsqlite(&db.db_path, oplog, &exec_config)
        }
    }
}

// ── Matrix runner ──────────────────────────────────────────────────────

/// Run the full performance benchmark matrix.
///
/// Iterates over every (engine × fixture × workload × concurrency) cell,
/// runs the benchmark for each, and collects results.
#[must_use]
pub fn run_perf_matrix(config: &PerfMatrixConfig) -> PerfResult {
    let cells = expand_matrix(config);
    let total = cells.len();
    let mut outcomes = Vec::with_capacity(total);
    let mut success_count = 0usize;
    let mut error_count = 0usize;

    for (i, cell) in cells.iter().enumerate() {
        eprintln!("[perf] ({}/{}) running {}", i + 1, total, cell,);

        let outcome = run_cell(cell, config);

        if outcome.error.is_some() {
            error_count += 1;
            if config.fail_fast {
                outcomes.push(outcome);
                break;
            }
        } else {
            success_count += 1;
        }

        outcomes.push(outcome);
    }

    PerfResult {
        schema_version: PERF_RESULT_SCHEMA_V1.to_owned(),
        total_cells: total,
        success_count,
        error_count,
        cells: outcomes,
    }
}

/// Write all per-cell benchmark outcomes to a JSONL file.
///
/// Each line is a serialized [`CellOutcome`], including failed cells.
///
/// # Errors
///
/// Returns an I/O error if the file cannot be written.
pub fn write_results_jsonl(result: &PerfResult, path: &Path) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    for cell in &result.cells {
        let line = serde_json::to_string(cell).map_err(std::io::Error::other)?;
        writeln!(file, "{line}")?;
    }
    Ok(())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::StorageWiringReport;
    use fsqlite_types::record::{RecordHotPathProfileSnapshot, ValueTypeProfileSnapshot};
    use fsqlite_vdbe::engine::{
        OpcodeExecutionCount, ValueTypeMetricsSnapshot, VdbeMetricsSnapshot,
    };

    fn hot_path_test_guard() -> std::sync::MutexGuard<'static, ()> {
        super::HOT_PATH_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn sample_engine_report() -> EngineRunReport {
        EngineRunReport {
            wall_time_ms: 42,
            ops_total: 17,
            ops_per_sec: 404.0,
            retries: 0,
            aborts: 0,
            correctness: crate::report::CorrectnessReport {
                raw_sha256_match: None,
                dump_match: None,
                canonical_sha256_match: None,
                integrity_check_ok: None,
                raw_sha256: None,
                canonical_sha256: None,
                logical_sha256: None,
                notes: None,
            },
            latency_ms: None,
            error: None,
            first_failure_diagnostic: None,
            storage_wiring: None,
            runtime_phase_timing: None,
            hot_path_profile: None,
        }
    }

    fn sample_hot_path_profile_report(
        config: &FsqliteHotPathProfileConfig,
    ) -> HotPathProfileReport {
        let storage_wiring = StorageWiringReport {
            backend_kind: "unix".to_owned(),
            backend_mode: "parity_cert_strict".to_owned(),
            backend_identity: "unix:parity_cert_strict".to_owned(),
        };
        let engine_report = EngineRunReport {
            wall_time_ms: 42,
            ops_total: 17,
            ops_per_sec: 404.0,
            retries: 0,
            aborts: 0,
            correctness: crate::report::CorrectnessReport {
                raw_sha256_match: None,
                dump_match: None,
                canonical_sha256_match: None,
                integrity_check_ok: None,
                raw_sha256: None,
                canonical_sha256: None,
                logical_sha256: None,
                notes: Some(
                    "mode=concurrent (MVCC); single-threaded sequential execution; backend_identity=unix:parity_cert_strict"
                        .to_owned(),
                ),
            },
            latency_ms: None,
            error: None,
            first_failure_diagnostic: None,
            storage_wiring: Some(storage_wiring),
            runtime_phase_timing: Some(crate::report::RuntimePhaseTimingEvidence {
                retry_backoff_time_ns: 800_000,
                busy_attempt_time_ns: 1_200_000,
                begin_boundary_time_ns: 500_000,
                body_execution_time_ns: 7_500,
                commit_finalize_time_ns: 700_000,
                rollback_time_ns: 300_000,
            }),
            hot_path_profile: None,
        };
        let snapshot = HotPathProfileSnapshot {
            parser: ParserHotPathProfileSnapshot {
                parse_single_calls: 3,
                parse_multi_calls: 1,
                parse_cache_hits: 2,
                parse_cache_misses: 2,
                parsed_sql_bytes: 256,
                parse_time_ns: 1_500,
                rewrite_calls: 2,
                rewrite_time_ns: 600,
                compiled_cache_hits: 1,
                compiled_cache_misses: 2,
                compile_time_ns: 900,
            },
            background_status_checks: 2,
            op_cx_background_gates: 1,
            statement_dispatch_background_gates: 1,
            prepared_schema_refreshes: 1,
            pager_publication_refreshes: 1,
            memory_autocommit_fast_path_begins: 1,
            column_default_evaluation_passes: 2,
            prepared_table_engine_fresh_allocs: 1,
            prepared_table_engine_reuses: 1,
            record_decode: RecordHotPathProfileSnapshot {
                parse_record_calls: 4,
                parse_record_into_calls: 2,
                parse_record_column_calls: 6,
                record_bytes_scanned: 512,
                record_vec_capacity_slots: 12,
                decode_time_ns: 3_000,
                decoded_values: ValueTypeProfileSnapshot {
                    null_count: 1,
                    integer_count: 5,
                    float_count: 1,
                    text_count: 2,
                    blob_count: 1,
                    text_bytes: 64,
                    blob_bytes: 32,
                },
            },
            vdbe: VdbeMetricsSnapshot {
                opcodes_executed_total: 120,
                statements_total: 8,
                statement_duration_us_total: 3_400,
                sort_rows_total: 0,
                sort_spill_pages_total: 0,
                opcode_execution_totals: vec![
                    OpcodeExecutionCount {
                        opcode: "Column".to_owned(),
                        total: 10,
                    },
                    OpcodeExecutionCount {
                        opcode: "ResultRow".to_owned(),
                        total: 8,
                    },
                    OpcodeExecutionCount {
                        opcode: "MakeRecord".to_owned(),
                        total: 4,
                    },
                ],
                type_coercions_total: 0,
                type_coercion_changes_total: 0,
                column_reads_total: 6,
                record_decode_calls_total: 4,
                decoded_values_total: 10,
                decoded_value_heap_bytes_total: 96,
                result_rows_total: 4,
                result_values_total: 8,
                result_value_heap_bytes_total: 120,
                result_row_materialization_time_ns_total: 2_500,
                make_record_calls_total: 3,
                make_record_blob_bytes_total: 48,
                decoded_value_types: ValueTypeMetricsSnapshot {
                    total_values: 10,
                    nulls: 1,
                    integers: 5,
                    reals: 1,
                    texts: 2,
                    blobs: 1,
                    text_bytes_total: 64,
                    blob_bytes_total: 32,
                },
                result_value_types: ValueTypeMetricsSnapshot {
                    total_values: 8,
                    nulls: 0,
                    integers: 4,
                    reals: 1,
                    texts: 2,
                    blobs: 1,
                    text_bytes_total: 48,
                    blob_bytes_total: 24,
                },
            },
        };

        build_hot_path_profile_report(
            "smoke",
            config,
            format!("{}-smoke", config.bead_id()),
            format!(
                "{}:smoke:c1",
                hot_path_profile_scenario_id(config.scenario_prefix(), &config.workload)
            ),
            engine_report,
            snapshot,
        )
    }

    #[test]
    fn benchmark_summary_error_reports_partial_measurement_failure() {
        let config = BenchmarkConfig {
            warmup_iterations: 0,
            min_iterations: 3,
            measurement_time_secs: 0,
        };
        let meta = BenchmarkMeta {
            engine: "sqlite3".to_owned(),
            workload: "inserts".to_owned(),
            fixture_id: "db-a".to_owned(),
            concurrency: 1,
            cargo_profile: "test".to_owned(),
        };
        let mut call_count = 0u32;
        let summary = run_benchmark(&config, &meta, |_| {
            call_count = call_count.saturating_add(1);
            if call_count == 2 {
                Err("setup failed")
            } else {
                Ok::<_, &str>(sample_engine_report())
            }
        });

        assert_eq!(
            benchmark_summary_error(&summary).as_deref(),
            Some("1/3 measurement iterations failed; first=setup failed")
        );
    }

    #[test]
    fn benchmark_summary_error_reports_total_measurement_failure() {
        let config = BenchmarkConfig {
            warmup_iterations: 0,
            min_iterations: 2,
            measurement_time_secs: 0,
        };
        let meta = BenchmarkMeta {
            engine: "sqlite3".to_owned(),
            workload: "inserts".to_owned(),
            fixture_id: "db-a".to_owned(),
            concurrency: 1,
            cargo_profile: "test".to_owned(),
        };
        let summary = run_benchmark(&config, &meta, |_| Err::<EngineRunReport, _>("open failed"));

        assert_eq!(
            benchmark_summary_error(&summary).as_deref(),
            Some("all 2 measurement iterations failed; first=open failed")
        );
    }

    #[test]
    fn test_engine_display() {
        assert_eq!(Engine::Sqlite3.as_str(), "sqlite3");
        assert_eq!(Engine::Fsqlite.as_str(), "fsqlite");
        assert_eq!(format!("{}", Engine::Sqlite3), "sqlite3");
    }

    #[test]
    fn test_expand_matrix() {
        let config = PerfMatrixConfig {
            engines: vec![Engine::Sqlite3, Engine::Fsqlite],
            fixture_ids: vec!["fix1".to_owned(), "fix2".to_owned()],
            workload_presets: vec!["deterministic_transform".to_owned()],
            concurrency_levels: vec![1, 4],
            seed: 42,
            scale: 10,
            project_root: PathBuf::from("/tmp"),
            benchmark_config: BenchmarkConfig::default(),
            settings: HarnessSettings::default(),
            cargo_profile: "release".to_owned(),
            fail_fast: false,
        };

        let cells = expand_matrix(&config);
        // 2 engines × 2 fixtures × 1 workload × 2 concurrencies = 8
        assert_eq!(cells.len(), 8);

        // Verify all combinations are present.
        assert!(
            cells.iter().any(|c| c.engine == Engine::Sqlite3
                && c.fixture_id == "fix1"
                && c.concurrency == 1)
        );
        assert!(
            cells.iter().any(|c| c.engine == Engine::Fsqlite
                && c.fixture_id == "fix2"
                && c.concurrency == 4)
        );
    }

    fn focused_perf_workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(std::path::Path::parent)
            .expect("workspace root")
            .to_path_buf()
    }

    fn run_focused_disjoint_c8_case(
        mode_id: &str,
        engine: Engine,
        concurrent_mode: bool,
    ) -> BenchmarkSummary {
        let oplog = generate_oplog(
            "commutative_inserts_disjoint_keys",
            "frankensqlite",
            42,
            8,
            50,
        )
        .expect("known preset");
        let workspace_root = focused_perf_workspace_root();
        let workspace_config = WorkspaceConfig {
            golden_dir: workspace_root.join("sample_sqlite_db_files/golden"),
            working_base: workspace_root.join("sample_sqlite_db_files/working"),
        };
        let settings = HarnessSettings {
            concurrent_mode,
            ..HarnessSettings::default()
        };
        let benchmark_config = BenchmarkConfig {
            warmup_iterations: 0,
            min_iterations: 1,
            measurement_time_secs: 0,
        };
        let meta = BenchmarkMeta {
            engine: mode_id.to_owned(),
            workload: "commutative_inserts_disjoint_keys".to_owned(),
            fixture_id: "frankensqlite".to_owned(),
            concurrency: 8,
            cargo_profile: "release-perf".to_owned(),
        };
        let summary = run_benchmark(&benchmark_config, &meta, |iteration_idx| {
            run_single_iteration(
                engine,
                "frankensqlite",
                &oplog,
                &workspace_config,
                &settings,
                iteration_idx,
            )
        });
        let error = benchmark_summary_error(&summary);
        assert!(
            error.is_none(),
            "focused disjoint c8 run failed for {mode_id}: {error:?}"
        );
        summary
    }

    #[test]
    #[ignore = "manual remote perf evidence for the flagship c8 row"]
    fn focused_disjoint_c8_release_perf_triplet_emits_results() {
        let cases = [
            ("sqlite3", Engine::Sqlite3, true),
            ("fsqlite_mvcc", Engine::Fsqlite, true),
            ("fsqlite_single_writer", Engine::Fsqlite, false),
        ];

        let summaries = cases
            .into_iter()
            .map(|(mode_id, engine, concurrent_mode)| {
                eprintln!("[focused_disjoint_c8] starting {mode_id}");
                let summary = run_focused_disjoint_c8_case(mode_id, engine, concurrent_mode);
                eprintln!(
                    "[focused_disjoint_c8] finished {mode_id} median_ops_per_sec={:.2} mean_ops_per_sec={:.2}",
                    summary.throughput.median_ops_per_sec,
                    summary.throughput.mean_ops_per_sec
                );
                (
                    mode_id,
                    summary,
                )
            })
            .collect::<Vec<_>>();

        let artifact_dir =
            focused_perf_workspace_root().join("artifacts/perf/20260317_targeted_disjoint_c8");
        std::fs::create_dir_all(&artifact_dir).expect("create artifact dir");

        let payload = serde_json::json!({
            "schema_version": "focused_disjoint_c8_release_perf_test.v1",
            "fixture_id": "frankensqlite",
            "workload": "commutative_inserts_disjoint_keys",
            "concurrency": 8,
            "seed": 42,
            "scale": 50,
            "cases": summaries.iter().map(|(mode_id, summary)| {
                serde_json::json!({
                    "mode_id": mode_id,
                    "benchmark_id": summary.benchmark_id,
                    "mean_ops_per_sec": summary.throughput.mean_ops_per_sec,
                    "median_ops_per_sec": summary.throughput.median_ops_per_sec,
                    "peak_ops_per_sec": summary.throughput.peak_ops_per_sec,
                    "iterations": summary.iterations,
                })
            }).collect::<Vec<_>>(),
        });
        std::fs::write(
            artifact_dir.join("focused_disjoint_c8_release_perf_test.json"),
            serde_json::to_string_pretty(&payload).expect("serialize focused result"),
        )
        .expect("write focused artifact");
        eprintln!(
            "FOCUSED_DISJOINT_C8_RELEASE_PERF_TEST_JSON:{}",
            serde_json::to_string(&payload).expect("serialize focused stdout payload")
        );
    }

    #[test]
    fn test_generate_oplog_known_presets() {
        let known = [
            "commutative_inserts_disjoint_keys",
            "hot_page_contention",
            "mixed_read_write",
            "deterministic_transform",
            "large_txn",
            "schema_migration",
            "btree_stress_sequential",
            "wide_row_overflow",
            "bulk_delete_reinsert",
            "scatter_write",
            "multi_table_foreign_keys",
        ];

        for name in &known {
            let log = generate_oplog(name, "test-fix", 42, 2, 20);
            assert!(
                log.is_some(),
                "generate_oplog should recognize preset '{name}'"
            );
            let log = log.unwrap();
            assert_eq!(log.header.preset.as_deref(), Some(*name));
        }
    }

    #[test]
    fn test_generate_oplog_unknown_preset() {
        assert!(generate_oplog("nonexistent_preset", "fix", 42, 1, 10).is_none());
    }

    #[test]
    fn test_matrix_cell_display() {
        let cell = MatrixCell {
            engine: Engine::Sqlite3,
            fixture_id: "mydb".to_owned(),
            workload: "hot_page_contention".to_owned(),
            concurrency: 8,
        };
        assert_eq!(format!("{cell}"), "sqlite3:hot_page_contention:mydb:c8");
    }

    #[test]
    fn test_perf_result_schema() {
        let result = PerfResult {
            schema_version: PERF_RESULT_SCHEMA_V1.to_owned(),
            total_cells: 0,
            success_count: 0,
            error_count: 0,
            cells: Vec::new(),
        };
        assert!(result.all_succeeded());
        assert_eq!(result.to_jsonl().unwrap(), "");
    }

    #[test]
    fn test_cell_outcome_with_error() {
        let outcome = CellOutcome {
            summary: None,
            error: Some("test error".to_owned()),
            engine: "sqlite3".to_owned(),
            fixture_id: "fix1".to_owned(),
            workload: "test".to_owned(),
            concurrency: 1,
        };
        let json = serde_json::to_string(&outcome).unwrap();
        let parsed: CellOutcome = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.error.as_deref(), Some("test error"));
        assert!(parsed.summary.is_none());
    }

    #[test]
    fn test_perf_jsonl_includes_error_cells() {
        let result = PerfResult {
            schema_version: PERF_RESULT_SCHEMA_V1.to_owned(),
            total_cells: 1,
            success_count: 0,
            error_count: 1,
            cells: vec![CellOutcome {
                summary: None,
                error: Some("boom".to_owned()),
                engine: "fsqlite".to_owned(),
                fixture_id: "fix1".to_owned(),
                workload: "hot_page_contention".to_owned(),
                concurrency: 4,
            }],
        };

        let jsonl = result.to_jsonl().unwrap();
        assert!(jsonl.contains("\"error\":\"boom\""));
        assert!(jsonl.contains("\"engine\":\"fsqlite\""));
    }

    #[test]
    fn hot_path_profile_smoke_writes_artifacts() {
        let tempdir = tempfile::tempdir().unwrap();
        let config = FsqliteHotPathProfileConfig {
            workload: "hot_page_contention".to_owned(),
            seed: 7,
            scale: 6,
            concurrency: 1,
            exec_config: crate::fsqlite_executor::FsqliteExecConfig {
                concurrent_mode: true,
                run_integrity_check: false,
                ..crate::fsqlite_executor::FsqliteExecConfig::default()
            },
            replay_command:
                "cargo run -p fsqlite-e2e --bin realdb-e2e -- hot-profile --db smoke --workload hot_page_contention"
                    .to_owned(),
            golden_dir: Some("sample_sqlite_db_files/golden".to_owned()),
            working_base: Some("sample_sqlite_db_files/working".to_owned()),
            bead_id: None,
            scenario_prefix: None,
        };

        let report = sample_hot_path_profile_report(&config);

        assert_eq!(report.schema_version, HOT_PATH_PROFILE_SCHEMA_V1);
        assert_eq!(report.fixture_id, "smoke");
        assert_eq!(report.workload, "hot_page_contention");
        assert_eq!(report.scenario_id, "bd-db300.4.1.hot_page_contention");
        assert!(report.concurrent_mode);
        assert!(!report.run_integrity_check);
        assert_eq!(
            report.golden_dir.as_deref(),
            Some("sample_sqlite_db_files/golden")
        );
        assert_eq!(
            report.working_base.as_deref(),
            Some("sample_sqlite_db_files/working")
        );
        assert!(!report.opcode_profile.is_empty());
        assert!(
            report.record_decode.parse_record_column_calls > 0
                || report.record_decode.parse_record_into_calls > 0
        );

        let artifact_dir = tempdir.path().join("artifacts");
        let counter_capture_summary = HotPathCounterCaptureManifestSummary {
            host_capability_sensitive_captures: vec!["topdown".to_owned()],
            topology_sensitive_captures: vec!["cache_to_cache".to_owned()],
            fallback_tools: vec!["perf-stat".to_owned()],
            fallback_metric_pack: vec!["cache-misses".to_owned()],
            fallback_notes: vec!["cache_to_cache: perf c2c unavailable".to_owned()],
            raw_output_relpaths: vec!["profiles/perf-c2c.profiler_safe.data".to_owned()],
        };
        let provenance = HotPathArtifactProvenance {
            row_id: "hot_page_contention_c4".to_owned(),
            mode_id: "fsqlite_mvcc".to_owned(),
            artifact_root: artifact_dir.display().to_string(),
            command_entrypoint: report.replay_command.clone(),
            workspace_root: Some(tempdir.path().display().to_string()),
            campaign_manifest_path: Some(
                "sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.json".to_owned(),
            ),
            source_revision: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            beads_data_hash: Some("a".repeat(64)),
            kernel_release: "Linux 6.13.5-test".to_owned(),
            rustc_version: "rustc 1.91.0-nightly".to_owned(),
            cargo_profile: "release-perf".to_owned(),
            commands: vec![crate::fixture_select::BenchmarkArtifactCommand {
                tool: "realdb-e2e".to_owned(),
                command_line: report.replay_command.clone(),
            }],
            tool_versions: vec![crate::fixture_select::BenchmarkArtifactToolVersion {
                tool: "cargo".to_owned(),
                version: "cargo 1.91.0-nightly".to_owned(),
            }],
            fallback_notes: counter_capture_summary.fallback_notes.clone(),
        };
        let microarchitectural_context = HotPathMicroarchitecturalContext {
            fixture_id: report.fixture_id.clone(),
            row_id: provenance.row_id.clone(),
            mode_id: provenance.mode_id.clone(),
            placement_profile_id: Some("baseline_unpinned".to_owned()),
            hardware_class_id: Some("linux_x86_64_any".to_owned()),
            hardware_signature: Some("linux:x86_64:any".to_owned()),
        };
        let manifest = write_hot_path_profile_artifacts(
            &report,
            &artifact_dir,
            Some(counter_capture_summary.clone()),
            Some(provenance.clone()),
            Some(microarchitectural_context.clone()),
        )
        .unwrap();
        let opcode_profile: HotPathOpcodeProfilePack = serde_json::from_slice(
            &std::fs::read(artifact_dir.join("opcode_profile.json")).unwrap(),
        )
        .unwrap();
        let subsystem_profile: HotPathSubsystemProfilePack = serde_json::from_slice(
            &std::fs::read(artifact_dir.join("subsystem_profile.json")).unwrap(),
        )
        .unwrap();
        let actionable_ranking: HotPathActionableRanking = serde_json::from_slice(
            &std::fs::read(artifact_dir.join("actionable_ranking.json")).unwrap(),
        )
        .unwrap();

        assert_eq!(manifest.schema_version, HOT_PATH_PROFILE_MANIFEST_SCHEMA_V1);
        assert!(manifest.concurrent_mode);
        assert!(!manifest.run_integrity_check);
        assert_eq!(
            manifest.golden_dir.as_deref(),
            Some("sample_sqlite_db_files/golden")
        );
        assert_eq!(
            manifest.working_base.as_deref(),
            Some("sample_sqlite_db_files/working")
        );
        assert!(artifact_dir.join("profile.json").exists());
        assert!(artifact_dir.join("opcode_profile.json").exists());
        assert!(artifact_dir.join("subsystem_profile.json").exists());
        assert!(artifact_dir.join("summary.md").exists());
        assert!(artifact_dir.join("actionable_ranking.json").exists());
        assert!(artifact_dir.join("manifest.json").exists());
        assert_eq!(manifest.files.len(), 6);
        assert_eq!(
            manifest.counter_capture_summary,
            Some(counter_capture_summary.clone())
        );
        assert_eq!(manifest.provenance, Some(provenance.clone()));
        let disk_manifest: HotPathArtifactManifest =
            serde_json::from_slice(&std::fs::read(artifact_dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            disk_manifest.counter_capture_summary,
            Some(counter_capture_summary)
        );
        assert_eq!(disk_manifest.provenance, Some(provenance));
        assert_eq!(
            opcode_profile.schema_version,
            HOT_PATH_OPCODE_PROFILE_SCHEMA_V1
        );
        assert_eq!(
            subsystem_profile.schema_version,
            HOT_PATH_SUBSYSTEM_PROFILE_SCHEMA_V1
        );
        assert_eq!(
            actionable_ranking.schema_version,
            HOT_PATH_PROFILE_ACTIONABLE_RANKING_SCHEMA_V3
        );
        assert!(!opcode_profile.opcodes.is_empty());
        assert!(!subsystem_profile.subsystem_ranking.is_empty());
        assert!(!actionable_ranking.baseline_reuse_ledger.is_empty());
        assert!(!actionable_ranking.baseline_waste_ledger.is_empty());
        assert!(!actionable_ranking.named_hotspots.is_empty());
        assert_eq!(actionable_ranking.microarchitectural_signatures.len(), 6);
        assert_eq!(actionable_ranking.wall_time_components.len(), 6);
        assert!(!actionable_ranking.cost_components.is_empty());
        assert_eq!(actionable_ranking.allocator_pressure.len(), 3);
        assert_eq!(
            actionable_ranking.top_opcodes,
            opcode_profile
                .opcodes
                .iter()
                .take(actionable_ranking.top_opcodes.len())
                .cloned()
                .collect::<Vec<_>>()
        );
        assert!(
            std::fs::read_to_string(artifact_dir.join("summary.md"))
                .unwrap()
                .contains("## Baseline Reuse Ledger")
        );
        assert!(
            std::fs::read_to_string(artifact_dir.join("summary.md"))
                .unwrap()
                .contains("## Baseline Waste Ledger")
        );
        assert!(
            std::fs::read_to_string(artifact_dir.join("summary.md"))
                .unwrap()
                .contains("## Wall-Time Decomposition")
        );
        assert!(
            actionable_ranking
                .named_hotspots
                .iter()
                .flat_map(|entry| entry.mapped_beads.iter())
                .any(|bead| bead == "bd-db300.10.2"
                    || bead == "bd-db300.10.4"
                    || bead == "bd-db300.10.5"
                    || bead == "bd-db300.10.6"
                    || bead == "bd-db300.10.7")
        );
        assert_eq!(actionable_ranking.cost_components.len(), 3);
        let wall_component_names = actionable_ranking
            .wall_time_components
            .iter()
            .map(|entry| entry.component.as_str())
            .collect::<Vec<_>>();
        assert!(wall_component_names.contains(&"queueing"));
        assert!(wall_component_names.contains(&"synchronization"));
        assert!(wall_component_names.contains(&"retry"));
        assert!(wall_component_names.contains(&"service"));
        assert!(wall_component_names.contains(&"allocator_copy"));
        assert!(wall_component_names.contains(&"durability"));
        assert!(
            actionable_ranking
                .microarchitectural_signatures
                .iter()
                .all(|entry| entry.fixture_id == report.fixture_id)
        );
        assert!(
            actionable_ranking
                .microarchitectural_signatures
                .iter()
                .all(|entry| entry.row_id.as_deref()
                    == Some(microarchitectural_context.row_id.as_str()))
        );
        assert!(
            actionable_ranking
                .microarchitectural_signatures
                .iter()
                .all(|entry| entry.mode_id.as_deref()
                    == Some(microarchitectural_context.mode_id.as_str()))
        );
        assert!(
            actionable_ranking
                .microarchitectural_signatures
                .iter()
                .all(
                    |entry| entry.placement_profile_id.as_deref() == Some("baseline_unpinned")
                        && entry.hardware_class_id.as_deref() == Some("linux_x86_64_any")
                        && entry.hardware_signature.as_deref() == Some("linux:x86_64:any")
                        && !entry.evidence_sources.is_empty()
                )
        );
        let component_names = actionable_ranking
            .cost_components
            .iter()
            .map(|entry| entry.component.as_str())
            .collect::<Vec<_>>();
        assert!(component_names.contains(&"parser_ast_churn"));
        assert!(component_names.contains(&"record_decode"));
        assert!(component_names.contains(&"row_materialization"));
        assert!(
            actionable_ranking
                .allocator_pressure
                .iter()
                .all(|entry| !entry.implication.is_empty() && !entry.mapped_beads.is_empty())
        );

        let parser_component = actionable_ranking
            .cost_components
            .iter()
            .find(|entry| entry.component == "parser_ast_churn")
            .unwrap();
        assert_eq!(
            parser_component.mapped_beads,
            vec!["bd-db300.10.2".to_owned(), "bd-db300.10.4".to_owned()]
        );
        assert!(!parser_component.implication.is_empty());

        let record_decode_component = actionable_ranking
            .cost_components
            .iter()
            .find(|entry| entry.component == "record_decode")
            .unwrap();
        assert_eq!(
            record_decode_component.mapped_beads,
            vec!["bd-db300.10.2".to_owned(), "bd-db300.10.5".to_owned()]
        );
        assert!(!record_decode_component.implication.is_empty());

        let row_materialization_component = actionable_ranking
            .cost_components
            .iter()
            .find(|entry| entry.component == "row_materialization")
            .unwrap();
        assert_eq!(
            row_materialization_component.mapped_beads,
            vec![
                "bd-db300.10.2".to_owned(),
                "bd-db300.10.6".to_owned(),
                "bd-db300.10.7".to_owned(),
            ]
        );
        assert!(!row_materialization_component.implication.is_empty());

        let parse_cache_reuse = actionable_ranking
            .baseline_reuse_ledger
            .iter()
            .find(|entry| entry.surface == "statement_parse_cache")
            .unwrap();
        assert!(parse_cache_reuse.supported);
        assert_eq!(parse_cache_reuse.hit_rate_basis_points, Some(5_000));
        assert_eq!(
            parse_cache_reuse.mapped_beads,
            vec!["bd-db300.10.4".to_owned()]
        );

        let page_data_reuse_gap = actionable_ranking
            .baseline_reuse_ledger
            .iter()
            .find(|entry| entry.surface == "page_data_ownership_reuse")
            .unwrap();
        assert!(!page_data_reuse_gap.supported);
        assert_eq!(page_data_reuse_gap.hit_rate_basis_points, None);
        assert_eq!(
            page_data_reuse_gap.mapped_beads,
            vec!["bd-db300.10.6".to_owned()]
        );

        let parser_waste = actionable_ranking
            .baseline_waste_ledger
            .iter()
            .find(|entry| entry.component == "parser_prepare_churn")
            .unwrap();
        assert_eq!(parser_waste.classification, "baseline_tax");
        assert_eq!(parser_waste.metric_kind, "time_ns");
        assert_eq!(
            parser_waste.mapped_beads,
            vec!["bd-db300.10.2".to_owned(), "bd-db300.10.4".to_owned()]
        );

        let busy_retry_spillover = actionable_ranking
            .baseline_waste_ledger
            .iter()
            .find(|entry| entry.component == "busy_retry_queueing")
            .unwrap();
        assert_eq!(
            busy_retry_spillover.classification,
            "structural_side_effect"
        );
        assert_eq!(busy_retry_spillover.metric_value, 2_000_000);
        assert_eq!(
            busy_retry_spillover.mapped_beads,
            vec!["bd-db300.2.4".to_owned()]
        );
    }

    #[test]
    fn hot_path_profile_smoke_respects_bead_override() {
        let config = FsqliteHotPathProfileConfig {
            workload: "hot_page_contention".to_owned(),
            seed: 7,
            scale: 6,
            concurrency: 1,
            exec_config: crate::fsqlite_executor::FsqliteExecConfig {
                concurrent_mode: true,
                run_integrity_check: false,
                ..crate::fsqlite_executor::FsqliteExecConfig::default()
            },
            replay_command:
                "cargo run -p fsqlite-e2e --bin realdb-e2e -- hot-profile --db smoke --workload hot_page_contention"
                    .to_owned(),
            golden_dir: Some("sample_sqlite_db_files/golden".to_owned()),
            working_base: Some("sample_sqlite_db_files/working".to_owned()),
            bead_id: Some("bd-db300.1.3".to_owned()),
            scenario_prefix: None,
        };

        let report = sample_hot_path_profile_report(&config);

        assert_eq!(report.bead_id, "bd-db300.1.3");
        assert_eq!(report.scenario_id, "bd-db300.1.3.hot_page_contention");
    }

    #[test]
    fn hot_path_profile_scope_rejects_reentrant_entry() {
        let _guard = hot_path_test_guard();
        reset_hot_path_profile();
        set_hot_path_profile_enabled(false);

        let first_scope = HotPathProfileScope::enable().unwrap();

        let error = match HotPathProfileScope::enable() {
            Ok(_) => panic!("reentrant profiling scope should fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("already active in this process"));

        drop(first_scope);
        set_hot_path_profile_enabled(false);
        reset_hot_path_profile();
    }
}
