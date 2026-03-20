//! RealDB E2E runner — differential testing of FrankenSQLite vs C SQLite
//! using real-world database fixtures discovered from `/dp`.
//!
//! # Subcommands
//!
//! - `corpus scan` — Discover SQLite databases under `/dp` and list candidates.
//! - `corpus import` — Copy selected databases into `sample_sqlite_db_files/golden/`.
//! - `corpus verify` — Verify golden copies against `sample_sqlite_db_files/checksums.sha256`.
//! - `run` — Execute an OpLog workload against a chosen engine.
//! - `bench` — Run a Criterion-style benchmark matrix.
//! - `hot-profile` — Capture a structured hot-path profile for a benchmark preset.
//! - `corrupt` — Inject corruption into a working copy for recovery testing.
//! - `compare` — Tiered comparison of two database files (bd-2als.3.2).

use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read as _, Write as _};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use rusqlite::{Connection, DatabaseName, OpenFlags};
use serde::{Deserialize, Serialize};

use fsqlite_types::{DATABASE_HEADER_SIZE, DatabaseHeader};

use fsqlite_e2e::benchmark::{BenchmarkConfig, BenchmarkMeta, BenchmarkSummary, run_benchmark};
use fsqlite_e2e::corruption::{CorruptionStrategy, inject_corruption};
use fsqlite_e2e::fixture_metadata::{
    ColumnProfileV1, FIXTURE_METADATA_SCHEMA_VERSION_V1, FixtureFeaturesV1, FixtureMetadataV1,
    FixtureSafetyV1, RiskLevel, SqliteMetaV1, TableProfileV1, normalize_tags, size_bucket_tag,
};
use fsqlite_e2e::fixture_select::{
    BEADS_BENCHMARK_CAMPAIGN_PATH_RELATIVE, BeadsBenchmarkCampaign, BenchmarkArtifactCommand,
    BenchmarkArtifactToolVersion, BenchmarkMode, PLACEMENT_PROFILE_BASELINE_UNPINNED,
    load_beads_benchmark_campaign,
};
use fsqlite_e2e::fsqlite_executor::{FsqliteExecConfig, run_oplog_fsqlite};
use fsqlite_e2e::golden::{format_mismatch_diagnostic, verify_databases};
use fsqlite_e2e::methodology::EnvironmentMeta;
use fsqlite_e2e::oplog::{self, OpLog};
#[cfg(test)]
use fsqlite_e2e::perf_runner::HotPathConnectionCeremonyProfile;
use fsqlite_e2e::perf_runner::{
    FsqliteHotPathProfileConfig, HotPathArtifactFile, HotPathArtifactManifest,
    HotPathArtifactProvenance, HotPathCounterCaptureManifestSummary,
    HotPathMicroarchitecturalContext, HotPathProfileReport, build_hot_path_actionable_ranking,
    build_hot_path_opcode_profile, build_hot_path_subsystem_profile, profile_fsqlite_hot_path,
    render_hot_path_profile_markdown, write_hot_path_profile_artifacts,
};
use fsqlite_e2e::report::{EngineInfo, RunRecordV1, RunRecordV1Args};
use fsqlite_e2e::report_render::render_benchmark_summaries_markdown;
use fsqlite_e2e::run_workspace::{WorkspaceConfig, create_workspace_with_label};
use fsqlite_e2e::sqlite_executor::{SqliteExecConfig, run_oplog_sqlite};

const HOT_PATH_INLINE_BUNDLE_SCHEMA_V1: &str = "fsqlite-e2e.hot_path_inline_bundle.v1";
const HOT_PATH_INLINE_BUNDLE_PREFIX: &str = "HOT_PATH_INLINE_BUNDLE_JSON=";
const HOT_PATH_COMMAND_PACK_SCHEMA_V2: &str = "fsqlite-e2e.hot_path_command_pack.v2";
const HOT_PATH_COMMAND_PACK_NAME: &str = "command_pack.json";

#[derive(Debug, Clone, PartialEq, Eq)]
struct BenchCampaignDefaults {
    golden_dir: PathBuf,
    fixture_ids: Vec<String>,
    fixture_paths: HashMap<String, PathBuf>,
    presets: Vec<String>,
    concurrency: Vec<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RunModeOptions {
    run_integrity_check: bool,
    capture_environment_metadata: bool,
    hot_path_profile: bool,
}

impl Default for RunModeOptions {
    fn default() -> Self {
        Self {
            run_integrity_check: true,
            capture_environment_metadata: true,
            hot_path_profile: false,
        }
    }
}

impl RunModeOptions {
    #[must_use]
    const fn from_flags(
        profile_only: bool,
        skip_integrity_check: bool,
        skip_environment_metadata: bool,
    ) -> Self {
        Self {
            run_integrity_check: !(profile_only || skip_integrity_check),
            capture_environment_metadata: !(profile_only || skip_environment_metadata),
            hot_path_profile: profile_only,
        }
    }

    fn environment(self, cargo_profile: &str) -> EnvironmentMeta {
        if self.capture_environment_metadata {
            EnvironmentMeta::capture(cargo_profile)
        } else {
            EnvironmentMeta::suppressed(cargo_profile)
        }
    }
}

fn default_bench_presets() -> Vec<String> {
    vec![
        "commutative_inserts_disjoint_keys".to_owned(),
        "hot_page_contention".to_owned(),
        "mixed_read_write".to_owned(),
    ]
}

fn find_bench_workspace_root(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join(BEADS_BENCHMARK_CAMPAIGN_PATH_RELATIVE).is_file())
        .map(Path::to_path_buf)
}

fn resolve_hot_path_workspace_root_candidate(candidate: &Path) -> Option<PathBuf> {
    let normalized = candidate
        .canonicalize()
        .unwrap_or_else(|_| candidate.to_path_buf());
    if normalized
        .join(BEADS_BENCHMARK_CAMPAIGN_PATH_RELATIVE)
        .is_file()
    {
        return Some(normalized);
    }
    find_bench_workspace_root(&normalized)
}

fn canonical_bench_defaults(workspace_root: &Path) -> Result<BenchCampaignDefaults, String> {
    let campaign = load_beads_benchmark_campaign(workspace_root)?;
    canonical_bench_defaults_from_campaign(workspace_root, &campaign)
}

fn canonical_bench_defaults_from_campaign(
    workspace_root: &Path,
    campaign: &BeadsBenchmarkCampaign,
) -> Result<BenchCampaignDefaults, String> {
    let Some(first_fixture) = campaign.fixtures.first() else {
        return Err("canonical Beads benchmark campaign has no fixtures".to_owned());
    };
    let first_path = workspace_root.join(&first_fixture.working_copy_relpath);
    let Some(golden_dir) = first_path.parent().map(Path::to_path_buf) else {
        return Err(format!(
            "canonical fixture `{}` has no parent directory: {}",
            first_fixture.fixture_id,
            first_path.display()
        ));
    };

    for fixture in &campaign.fixtures[1..] {
        let fixture_path = workspace_root.join(&fixture.working_copy_relpath);
        let Some(parent) = fixture_path.parent() else {
            return Err(format!(
                "canonical fixture `{}` has no parent directory: {}",
                fixture.fixture_id,
                fixture_path.display()
            ));
        };
        if parent != golden_dir {
            return Err(format!(
                "canonical Beads benchmark fixtures must share one golden directory: {} vs {}",
                golden_dir.display(),
                parent.display()
            ));
        }
    }

    let fixture_ids = campaign
        .fixtures
        .iter()
        .map(|fixture| fixture.fixture_id.clone())
        .collect::<Vec<_>>();
    let fixture_paths = campaign
        .fixtures
        .iter()
        .map(|fixture| {
            (
                fixture.fixture_id.clone(),
                workspace_root.join(&fixture.working_copy_relpath),
            )
        })
        .collect::<HashMap<_, _>>();

    let mut presets = Vec::new();
    let mut concurrency = Vec::new();
    for row in &campaign.matrix_rows {
        if !presets.iter().any(|preset| preset == &row.workload) {
            presets.push(row.workload.clone());
        }
        if !concurrency.contains(&row.concurrency) {
            concurrency.push(row.concurrency);
        }
    }

    if presets.is_empty() {
        return Err("canonical Beads benchmark campaign has no workloads".to_owned());
    }
    if concurrency.is_empty() {
        return Err("canonical Beads benchmark campaign has no concurrency rows".to_owned());
    }

    Ok(BenchCampaignDefaults {
        golden_dir,
        fixture_ids,
        fixture_paths,
        presets,
        concurrency,
    })
}

fn resolve_path_from_base(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn resolve_path_from_current_dir(path: PathBuf) -> PathBuf {
    match std::env::current_dir() {
        Ok(current_dir) => resolve_path_from_base(&current_dir, &path),
        Err(_) => path,
    }
}

fn shell_escape(raw: &str) -> String {
    if raw
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || b"/._-*=+".contains(&byte))
    {
        return raw.to_owned();
    }
    let escaped = raw.replace('\'', "'\"'\"'");
    format!("'{escaped}'")
}

#[derive(Debug, Clone, Copy)]
struct HotProfileReplayCommand<'a> {
    db: &'a str,
    workload: &'a str,
    golden_dir: &'a Path,
    working_base: &'a Path,
    concurrency: u16,
    seed: u64,
    scale: u32,
    output_dir: &'a Path,
    mvcc: bool,
    run_integrity_check: bool,
}

const HOT_PATH_BEAD_ID_ENV: &str = "FSQLITE_HOT_PATH_BEAD_ID";
const HOT_PATH_CAMPAIGN_MANIFEST_PATH_ENV: &str = "FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH";
const HOT_PATH_CARGO_PROFILE_ENV: &str = "FSQLITE_HOT_PATH_CARGO_PROFILE";
const HOT_PATH_WORKSPACE_ROOT_ENV: &str = "FSQLITE_HOT_PATH_WORKSPACE_ROOT";
const HOT_PATH_SOURCE_REVISION_ENV: &str = "FSQLITE_HOT_PATH_SOURCE_REVISION";
const HOT_PATH_BEADS_DATA_HASH_ENV: &str = "FSQLITE_HOT_PATH_BEADS_DATA_HASH";
const HOT_PATH_PLACEMENT_PROFILE_ID_ENV: &str = "FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID";
const HOT_PATH_HARDWARE_CLASS_ID_ENV: &str = "FSQLITE_HOT_PATH_HARDWARE_CLASS_ID";
const HOT_PATH_HARDWARE_SIGNATURE_ENV: &str = "FSQLITE_HOT_PATH_HARDWARE_SIGNATURE";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct HotPathEvidenceCommandPack {
    schema_version: String,
    bead_id: String,
    run_id: String,
    trace_id: String,
    scenario_id: String,
    fixture_id: String,
    workload: String,
    seed: u64,
    scale: u32,
    concurrency: u16,
    concurrent_mode: bool,
    artifact_root: String,
    profiler_safe_replay_command: String,
    full_validation_replay_command: String,
    commands: Vec<HotPathEvidenceCommand>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct HotPathCounterPackMetadata {
    host_capability_sensitive: bool,
    topology_sensitive: bool,
    primary_tool: String,
    fallback_tools: Vec<String>,
    primary_selection: String,
    fallback_selections: Vec<String>,
    capability_probe: String,
    fallback_event_pack: Vec<String>,
    fallback_reason_hints: Vec<String>,
    raw_output_relpaths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct HotPathEvidenceCommand {
    capture: String,
    mode: String,
    tool: String,
    output_relpath: String,
    command_line: String,
    description: String,
    counter_pack: Option<HotPathCounterPackMetadata>,
}

fn hot_path_override_env(name: &'static str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn current_hot_path_replay_env_overrides() -> Vec<(&'static str, String)> {
    [
        HOT_PATH_BEAD_ID_ENV,
        HOT_PATH_CAMPAIGN_MANIFEST_PATH_ENV,
        HOT_PATH_CARGO_PROFILE_ENV,
        HOT_PATH_WORKSPACE_ROOT_ENV,
        HOT_PATH_SOURCE_REVISION_ENV,
        HOT_PATH_BEADS_DATA_HASH_ENV,
        HOT_PATH_PLACEMENT_PROFILE_ID_ENV,
        HOT_PATH_HARDWARE_CLASS_ID_ENV,
        HOT_PATH_HARDWARE_SIGNATURE_ENV,
    ]
    .into_iter()
    .filter_map(|name| hot_path_override_env(name).map(|value| (name, value)))
    .collect()
}

fn format_hot_profile_replay_command_with_env_overrides(
    command: &HotProfileReplayCommand<'_>,
    env_overrides: &[(&'static str, String)],
) -> String {
    let mut rendered = String::from("rch exec --");
    if !env_overrides.is_empty() {
        rendered.push_str(" env");
        for (name, value) in env_overrides {
            rendered.push(' ');
            rendered.push_str(name);
            rendered.push('=');
            rendered.push_str(&shell_escape(value));
        }
    }
    rendered.push_str(" cargo run -p fsqlite-e2e --bin realdb-e2e -- hot-profile");
    for (flag, value) in [
        ("--db", command.db.to_owned()),
        ("--workload", command.workload.to_owned()),
        (
            "--golden-dir",
            command
                .golden_dir
                .as_os_str()
                .to_string_lossy()
                .into_owned(),
        ),
        (
            "--working-base",
            command
                .working_base
                .as_os_str()
                .to_string_lossy()
                .into_owned(),
        ),
        ("--concurrency", command.concurrency.to_string()),
        ("--seed", command.seed.to_string()),
        ("--scale", command.scale.to_string()),
        (
            "--output-dir",
            command
                .output_dir
                .as_os_str()
                .to_string_lossy()
                .into_owned(),
        ),
    ] {
        rendered.push(' ');
        rendered.push_str(flag);
        rendered.push(' ');
        rendered.push_str(&shell_escape(&value));
    }
    rendered.push(' ');
    rendered.push_str(if command.mvcc { "--mvcc" } else { "--no-mvcc" });
    if command.run_integrity_check {
        rendered.push_str(" --integrity-check");
    }
    rendered
}

fn format_hot_profile_replay_command(command: &HotProfileReplayCommand<'_>) -> String {
    format_hot_profile_replay_command_with_env_overrides(
        command,
        &current_hot_path_replay_env_overrides(),
    )
}

fn hot_path_output_path(output_dir: &Path, relpath: &str) -> String {
    output_dir.join(relpath).display().to_string()
}

fn mkdir_prefixed_capture_command(output_dir: &Path, relpath: &str, body: String) -> String {
    let parent = output_dir
        .join(relpath)
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| output_dir.to_path_buf());
    format!(
        "mkdir -p {} && {body}",
        shell_escape(&parent.display().to_string())
    )
}

fn format_hot_path_hyperfine_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    mkdir_prefixed_capture_command(
        output_dir,
        relpath,
        format!(
            "hyperfine --warmup 1 --runs 5 --export-json {} {}",
            shell_escape(&output_path),
            shell_escape(replay_command),
        ),
    )
}

fn format_hot_path_perf_record_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    mkdir_prefixed_capture_command(
        output_dir,
        relpath,
        format!(
            "perf record -g -o {} -- {replay_command}",
            shell_escape(&output_path),
        ),
    )
}

fn format_hot_path_perf_sched_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    mkdir_prefixed_capture_command(
        output_dir,
        relpath,
        format!(
            "perf sched record -o {} -- {replay_command}",
            shell_escape(&output_path),
        ),
    )
}

fn format_hot_path_strace_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    mkdir_prefixed_capture_command(
        output_dir,
        relpath,
        format!(
            "strace -f -o {} {replay_command}",
            shell_escape(&output_path)
        ),
    )
}

fn format_hot_path_heaptrack_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    mkdir_prefixed_capture_command(
        output_dir,
        relpath,
        format!(
            "heaptrack -o {} {replay_command}",
            shell_escape(&output_path)
        ),
    )
}

fn format_hot_path_bash_command(output_dir: &Path, relpath: &str, script: String) -> String {
    mkdir_prefixed_capture_command(
        output_dir,
        relpath,
        format!(
            "bash -lc {}",
            shell_escape(&format!("set -euo pipefail; {script}"))
        ),
    )
}

fn hot_path_sidecar_relpath(relpath: &str, suffix: &str) -> String {
    relpath.strip_suffix(".summary.txt").map_or_else(
        || format!("{relpath}{suffix}"),
        |prefix| format!("{prefix}{suffix}"),
    )
}

fn format_hot_path_perf_topdown_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    format_hot_path_bash_command(
        output_dir,
        relpath,
        format!(
            "out={}; tmp=$(mktemp); trap 'rm -f \"$tmp\"' EXIT; \
             if perf stat -M TopdownL1 -o \"$tmp\" -- true >/dev/null 2>&1; then \
                 printf '%s\\n' 'capture=topdown' 'selected=TopdownL1' 'fallback=false' > \"$out\"; \
                 perf stat -x, -M TopdownL1 -o \"$tmp\" -- {replay_command}; \
             else \
                 printf '%s\\n' 'capture=topdown' 'selected=core_event_fallback' 'fallback=true' 'reason=TopdownL1 unsupported on this host' > \"$out\"; \
                 perf stat -x, -e cycles,instructions,branches,branch-misses,cache-references,cache-misses -o \"$tmp\" -- {replay_command}; \
             fi; \
             cat \"$tmp\" >> \"$out\"",
            shell_escape(&output_path),
        ),
    )
}

fn format_hot_path_perf_c2c_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    let raw_path = hot_path_output_path(output_dir, &hot_path_sidecar_relpath(relpath, ".data"));
    format_hot_path_bash_command(
        output_dir,
        relpath,
        format!(
            "out={}; raw={}; tmp=$(mktemp); trap 'rm -f \"$tmp\"' EXIT; \
             rm -f \"$raw\"; \
             if perf c2c record -o \"$raw\" -- {replay_command} >/dev/null 2>&1 && \
                 perf c2c report --stats -i \"$raw\" > \"$tmp\" 2>&1; then \
                 printf '%s\\n' 'capture=cache_to_cache' 'selected=perf_c2c' 'fallback=false' > \"$out\"; \
                 cat \"$tmp\" >> \"$out\"; \
             elif perf mem record -o \"$raw\" -- {replay_command} >/dev/null 2>&1 && \
                 perf mem report -i \"$raw\" --stdio > \"$tmp\" 2>&1; then \
                 printf '%s\\n' 'capture=cache_to_cache' 'selected=perf_mem_fallback' 'fallback=true' 'reason=perf c2c unavailable or failed on this host' > \"$out\"; \
                 cat \"$tmp\" >> \"$out\"; \
             else \
                 rm -f \"$raw\"; \
                 printf '%s\\n' 'capture=cache_to_cache' 'selected=cache_event_fallback' 'fallback=true' 'reason=perf c2c and perf mem unavailable or failed on this host' > \"$out\"; \
                 perf stat -x, -e cache-references,cache-misses -o \"$tmp\" -- {replay_command}; \
                 cat \"$tmp\" >> \"$out\"; \
             fi",
            shell_escape(&output_path),
            shell_escape(&raw_path),
        ),
    )
}

fn format_hot_path_perf_migration_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    format_hot_path_bash_command(
        output_dir,
        relpath,
        format!(
            "out={}; tmp=$(mktemp); trap 'rm -f \"$tmp\"' EXIT; \
             if perf stat -e cpu-migrations,context-switches -o \"$tmp\" -- true >/dev/null 2>&1; then \
                 printf '%s\\n' 'capture=migration' 'selected=cpu_migrations' 'fallback=false' > \"$out\"; \
                 perf stat -x, -e cpu-migrations,context-switches -o \"$tmp\" -- {replay_command}; \
             else \
                 printf '%s\\n' 'capture=migration' 'selected=context_switch_fallback' 'fallback=true' 'reason=cpu-migrations unsupported on this host' > \"$out\"; \
                 perf stat -x, -e context-switches,task-clock -o \"$tmp\" -- {replay_command}; \
             fi; \
             cat \"$tmp\" >> \"$out\"",
            shell_escape(&output_path),
        ),
    )
}

fn format_hot_path_perf_mem_remote_access_command(
    output_dir: &Path,
    relpath: &str,
    replay_command: &str,
) -> String {
    let output_path = hot_path_output_path(output_dir, relpath);
    let raw_path = hot_path_output_path(output_dir, &hot_path_sidecar_relpath(relpath, ".data"));
    format_hot_path_bash_command(
        output_dir,
        relpath,
        format!(
            "out={}; raw={}; tmp=$(mktemp); trap 'rm -f \"$tmp\"' EXIT; \
             rm -f \"$raw\"; \
             if perf mem record -o \"$raw\" -- {replay_command} >/dev/null 2>&1 && \
                 perf mem report -i \"$raw\" --stdio > \"$tmp\" 2>&1; then \
                 printf '%s\\n' 'capture=remote_access' 'selected=perf_mem' 'fallback=false' > \"$out\"; \
                 cat \"$tmp\" >> \"$out\"; \
             else \
                 rm -f \"$raw\"; \
                 printf '%s\\n' 'capture=remote_access' 'selected=cache_event_fallback' 'fallback=true' 'reason=perf mem unavailable or failed on this host' > \"$out\"; \
                 perf stat -x, -e cache-references,cache-misses,page-faults -o \"$tmp\" -- {replay_command}; \
                 cat \"$tmp\" >> \"$out\"; \
             fi",
            shell_escape(&output_path),
            shell_escape(&raw_path),
        ),
    )
}

fn build_hot_path_command_pack(
    report: &HotPathProfileReport,
    replay_command: &HotProfileReplayCommand<'_>,
) -> HotPathEvidenceCommandPack {
    let profiler_safe_replay_command =
        format_hot_profile_replay_command(&HotProfileReplayCommand {
            run_integrity_check: false,
            ..*replay_command
        });
    let full_validation_replay_command =
        format_hot_profile_replay_command(&HotProfileReplayCommand {
            run_integrity_check: true,
            ..*replay_command
        });
    let mut commands = Vec::with_capacity(18);
    for (mode, replay) in [
        ("profiler_safe", profiler_safe_replay_command.as_str()),
        ("full_validation", full_validation_replay_command.as_str()),
    ] {
        let c2c_output_relpath = format!("profiles/perf-c2c.{mode}.summary.txt");
        let c2c_raw_relpath = hot_path_sidecar_relpath(&c2c_output_relpath, ".data");
        let remote_output_relpath = format!("profiles/perf-mem-remote-access.{mode}.summary.txt");
        let remote_raw_relpath = hot_path_sidecar_relpath(&remote_output_relpath, ".data");
        for (capture, tool, output_relpath, description, command_line, counter_pack) in [
            (
                "wall_clock",
                "hyperfine",
                format!("profiles/hyperfine.{mode}.json"),
                format!("wall-clock benchmark replay for {mode} capture"),
                format_hot_path_hyperfine_command(
                    replay_command.output_dir,
                    &format!("profiles/hyperfine.{mode}.json"),
                    replay,
                ),
                None,
            ),
            (
                "on_cpu",
                "perf-record",
                format!("profiles/perf-record.{mode}.data"),
                format!("sampled on-CPU profile for {mode} capture"),
                format_hot_path_perf_record_command(
                    replay_command.output_dir,
                    &format!("profiles/perf-record.{mode}.data"),
                    replay,
                ),
                None,
            ),
            (
                "scheduler",
                "perf-sched-record",
                format!("profiles/perf-sched.{mode}.data"),
                format!("scheduler and off-CPU trace for {mode} capture"),
                format_hot_path_perf_sched_command(
                    replay_command.output_dir,
                    &format!("profiles/perf-sched.{mode}.data"),
                    replay,
                ),
                None,
            ),
            (
                "syscall",
                "strace",
                format!("logs/strace.{mode}.log"),
                format!("syscall trace for {mode} capture"),
                format_hot_path_strace_command(
                    replay_command.output_dir,
                    &format!("logs/strace.{mode}.log"),
                    replay,
                ),
                None,
            ),
            (
                "allocation",
                "heaptrack",
                format!("profiles/heaptrack.{mode}.gz"),
                format!("allocation profile for {mode} capture"),
                format_hot_path_heaptrack_command(
                    replay_command.output_dir,
                    &format!("profiles/heaptrack.{mode}.gz"),
                    replay,
                ),
                None,
            ),
            (
                "topdown",
                "perf-stat",
                format!("profiles/perf-stat-topdown.{mode}.summary.txt"),
                format!("topdown counter pack with comparable fallback metrics for {mode} capture"),
                format_hot_path_perf_topdown_command(
                    replay_command.output_dir,
                    &format!("profiles/perf-stat-topdown.{mode}.summary.txt"),
                    replay,
                ),
                Some(HotPathCounterPackMetadata {
                    host_capability_sensitive: true,
                    topology_sensitive: false,
                    primary_tool: "perf-stat".to_owned(),
                    fallback_tools: vec!["perf-stat".to_owned()],
                    primary_selection: "TopdownL1".to_owned(),
                    fallback_selections: vec!["core_event_fallback".to_owned()],
                    capability_probe: "perf stat -M TopdownL1 -o \"$tmp\" -- true".to_owned(),
                    fallback_event_pack: vec![
                        "cycles".to_owned(),
                        "instructions".to_owned(),
                        "branches".to_owned(),
                        "branch-misses".to_owned(),
                        "cache-references".to_owned(),
                        "cache-misses".to_owned(),
                    ],
                    fallback_reason_hints: vec!["TopdownL1 unsupported on this host".to_owned()],
                    raw_output_relpaths: Vec::new(),
                }),
            ),
            (
                "cache_to_cache",
                "perf-c2c",
                c2c_output_relpath.clone(),
                format!(
                    "cache-to-cache or HITM evidence with perf mem/cache-stat fallbacks for {mode} capture"
                ),
                format_hot_path_perf_c2c_command(
                    replay_command.output_dir,
                    &c2c_output_relpath,
                    replay,
                ),
                Some(HotPathCounterPackMetadata {
                    host_capability_sensitive: true,
                    topology_sensitive: true,
                    primary_tool: "perf-c2c".to_owned(),
                    fallback_tools: vec!["perf-mem".to_owned(), "perf-stat".to_owned()],
                    primary_selection: "perf_c2c".to_owned(),
                    fallback_selections: vec![
                        "perf_mem_fallback".to_owned(),
                        "cache_event_fallback".to_owned(),
                    ],
                    capability_probe: "perf c2c record/report, else perf mem record/report"
                        .to_owned(),
                    fallback_event_pack: vec![
                        "cache-references".to_owned(),
                        "cache-misses".to_owned(),
                    ],
                    fallback_reason_hints: vec![
                        "perf c2c unavailable or failed on this host".to_owned(),
                        "perf c2c and perf mem unavailable or failed on this host".to_owned(),
                    ],
                    raw_output_relpaths: vec![c2c_raw_relpath],
                }),
            ),
            (
                "migration",
                "perf-stat",
                format!("profiles/perf-stat-migration.{mode}.summary.txt"),
                format!(
                    "migration and scheduling counter pack with fallback metrics for {mode} capture"
                ),
                format_hot_path_perf_migration_command(
                    replay_command.output_dir,
                    &format!("profiles/perf-stat-migration.{mode}.summary.txt"),
                    replay,
                ),
                Some(HotPathCounterPackMetadata {
                    host_capability_sensitive: true,
                    topology_sensitive: true,
                    primary_tool: "perf-stat".to_owned(),
                    fallback_tools: vec!["perf-stat".to_owned()],
                    primary_selection: "cpu_migrations".to_owned(),
                    fallback_selections: vec!["context_switch_fallback".to_owned()],
                    capability_probe:
                        "perf stat -e cpu-migrations,context-switches -o \"$tmp\" -- true"
                            .to_owned(),
                    fallback_event_pack: vec![
                        "context-switches".to_owned(),
                        "task-clock".to_owned(),
                    ],
                    fallback_reason_hints: vec![
                        "cpu-migrations unsupported on this host".to_owned(),
                    ],
                    raw_output_relpaths: Vec::new(),
                }),
            ),
            (
                "remote_access",
                "perf-mem",
                remote_output_relpath.clone(),
                format!(
                    "remote-access memory evidence with cache-stat fallback metrics for {mode} capture"
                ),
                format_hot_path_perf_mem_remote_access_command(
                    replay_command.output_dir,
                    &remote_output_relpath,
                    replay,
                ),
                Some(HotPathCounterPackMetadata {
                    host_capability_sensitive: true,
                    topology_sensitive: true,
                    primary_tool: "perf-mem".to_owned(),
                    fallback_tools: vec!["perf-stat".to_owned()],
                    primary_selection: "perf_mem".to_owned(),
                    fallback_selections: vec!["cache_event_fallback".to_owned()],
                    capability_probe: "perf mem record/report".to_owned(),
                    fallback_event_pack: vec![
                        "cache-references".to_owned(),
                        "cache-misses".to_owned(),
                        "page-faults".to_owned(),
                    ],
                    fallback_reason_hints: vec![
                        "perf mem unavailable or failed on this host".to_owned(),
                    ],
                    raw_output_relpaths: vec![remote_raw_relpath],
                }),
            ),
        ] {
            commands.push(HotPathEvidenceCommand {
                capture: capture.to_owned(),
                mode: mode.to_owned(),
                tool: tool.to_owned(),
                output_relpath,
                command_line,
                description,
                counter_pack,
            });
        }
    }

    HotPathEvidenceCommandPack {
        schema_version: HOT_PATH_COMMAND_PACK_SCHEMA_V2.to_owned(),
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
        artifact_root: replay_command.output_dir.display().to_string(),
        profiler_safe_replay_command,
        full_validation_replay_command,
        commands,
    }
}

fn write_hot_path_command_pack(
    output_dir: &Path,
    command_pack: &HotPathEvidenceCommandPack,
) -> io::Result<HotPathArtifactFile> {
    let command_pack_json = serde_json::to_string_pretty(command_pack)
        .map_err(|error| io::Error::other(format!("command pack JSON: {error}")))?;
    fs::write(
        output_dir.join(HOT_PATH_COMMAND_PACK_NAME),
        command_pack_json.as_bytes(),
    )?;
    Ok(HotPathArtifactFile {
        path: HOT_PATH_COMMAND_PACK_NAME.to_owned(),
        bytes: u64::try_from(command_pack_json.len()).unwrap_or(u64::MAX),
        sha256: hot_path_artifact_sha256(command_pack_json.as_bytes()),
        description:
            "structured replay/evidence capture commands for profiler-safe and full-validation runs"
                .to_owned(),
    })
}

fn push_unique_string(values: &mut Vec<String>, value: impl Into<String>) {
    let value = value.into();
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn build_hot_path_counter_capture_summary(
    command_pack: &HotPathEvidenceCommandPack,
) -> Option<HotPathCounterCaptureManifestSummary> {
    let mut host_capability_sensitive_captures = Vec::new();
    let mut topology_sensitive_captures = Vec::new();
    let mut fallback_tools = Vec::new();
    let mut fallback_metric_pack = Vec::new();
    let mut fallback_notes = Vec::new();
    let mut raw_output_relpaths = Vec::new();

    for command in &command_pack.commands {
        let Some(counter_pack) = &command.counter_pack else {
            continue;
        };
        if counter_pack.host_capability_sensitive {
            push_unique_string(
                &mut host_capability_sensitive_captures,
                command.capture.clone(),
            );
        }
        if counter_pack.topology_sensitive {
            push_unique_string(&mut topology_sensitive_captures, command.capture.clone());
        }
        for tool in &counter_pack.fallback_tools {
            push_unique_string(&mut fallback_tools, tool.clone());
        }
        for event in &counter_pack.fallback_event_pack {
            push_unique_string(&mut fallback_metric_pack, event.clone());
        }
        for hint in &counter_pack.fallback_reason_hints {
            push_unique_string(
                &mut fallback_notes,
                format!("{}:{}: {hint}", command.capture, command.mode),
            );
        }
        for relpath in &counter_pack.raw_output_relpaths {
            push_unique_string(&mut raw_output_relpaths, relpath.clone());
        }
    }

    (!host_capability_sensitive_captures.is_empty()
        || !topology_sensitive_captures.is_empty()
        || !fallback_tools.is_empty()
        || !fallback_metric_pack.is_empty()
        || !fallback_notes.is_empty()
        || !raw_output_relpaths.is_empty())
    .then_some(HotPathCounterCaptureManifestSummary {
        host_capability_sensitive_captures,
        topology_sensitive_captures,
        fallback_tools,
        fallback_metric_pack,
        fallback_notes,
        raw_output_relpaths,
    })
}

#[derive(Debug, Clone)]
struct HotPathArtifactProvenanceInputs {
    artifact_root: String,
    workspace_root: Option<String>,
    campaign_manifest_path: Option<String>,
    source_revision: Option<String>,
    beads_data_hash: Option<String>,
    kernel_release: String,
    rustc_version: String,
    cargo_profile: String,
    tool_versions: Vec<BenchmarkArtifactToolVersion>,
}

fn hot_path_row_id(workload: &str, concurrency: u16) -> String {
    format!("{workload}_c{concurrency}")
}

fn hot_path_mode_id(concurrent_mode: bool) -> &'static str {
    if concurrent_mode {
        "fsqlite_mvcc"
    } else {
        "fsqlite_single_writer"
    }
}

fn hot_path_benchmark_mode(concurrent_mode: bool) -> BenchmarkMode {
    if concurrent_mode {
        BenchmarkMode::FsqliteMvcc
    } else {
        BenchmarkMode::FsqliteSingleWriter
    }
}

fn hot_path_hardware_signature(
    campaign: &BeadsBenchmarkCampaign,
    hardware_class_id: &str,
) -> Option<String> {
    let hardware_class = campaign
        .hardware_classes
        .iter()
        .find(|hardware| hardware.id == hardware_class_id)?;
    Some(format!(
        "{}:{}:{}",
        hardware_class.id_fields.os_family.as_str(),
        hardware_class.id_fields.cpu_arch.as_str(),
        hardware_class.id_fields.topology_class.as_str()
    ))
}

fn hot_path_fixture_matches_report_fixture(
    campaign: &BeadsBenchmarkCampaign,
    campaign_fixture_id: &str,
    report_fixture_id: &str,
) -> bool {
    if campaign_fixture_id == report_fixture_id {
        return true;
    }
    campaign
        .fixtures
        .iter()
        .find(|fixture| fixture.fixture_id == campaign_fixture_id)
        .and_then(|fixture| {
            Path::new(&fixture.working_copy_relpath)
                .file_stem()
                .and_then(|stem| stem.to_str())
        })
        .is_some_and(|stem| stem == report_fixture_id)
}

fn resolve_hot_path_microarchitectural_context(
    workspace_root: Option<&Path>,
    report: &HotPathProfileReport,
) -> HotPathMicroarchitecturalContext {
    let row_id = hot_path_row_id(&report.workload, report.concurrency);
    let mode_id = hot_path_mode_id(report.concurrent_mode).to_owned();
    let mut context = HotPathMicroarchitecturalContext {
        fixture_id: report.fixture_id.clone(),
        row_id: row_id.clone(),
        mode_id,
        placement_profile_id: None,
        hardware_class_id: None,
        hardware_signature: None,
    };
    let placement_profile_id_override = hot_path_override_env(HOT_PATH_PLACEMENT_PROFILE_ID_ENV);
    let hardware_class_id_override = hot_path_override_env(HOT_PATH_HARDWARE_CLASS_ID_ENV);
    let hardware_signature_override = hot_path_override_env(HOT_PATH_HARDWARE_SIGNATURE_ENV);
    if placement_profile_id_override.is_some()
        || hardware_class_id_override.is_some()
        || hardware_signature_override.is_some()
    {
        context.placement_profile_id = placement_profile_id_override;
        context.hardware_class_id = hardware_class_id_override;
        context.hardware_signature = hardware_signature_override;
        return context;
    }
    let Some(workspace_root) = workspace_root else {
        return context;
    };
    let Ok(campaign) = load_beads_benchmark_campaign(workspace_root) else {
        return context;
    };
    let Some(row) = campaign.matrix_rows.iter().find(|candidate| {
        candidate.row_id == row_id
            && candidate.workload == report.workload
            && candidate.concurrency == report.concurrency
            && candidate.fixtures.iter().any(|fixture| {
                hot_path_fixture_matches_report_fixture(&campaign, fixture, &report.fixture_id)
            })
            && candidate
                .modes
                .contains(&hot_path_benchmark_mode(report.concurrent_mode))
    }) else {
        return context;
    };
    let Some(placement_variant) = row
        .placement_variants
        .iter()
        .find(|variant| variant.placement_profile_id == PLACEMENT_PROFILE_BASELINE_UNPINNED)
        .or_else(|| {
            row.placement_variants
                .iter()
                .find(|variant| variant.required)
        })
        .or_else(|| row.placement_variants.first())
    else {
        return context;
    };
    context.placement_profile_id = Some(placement_variant.placement_profile_id.clone());
    context.hardware_class_id = Some(placement_variant.hardware_class_id.clone());
    context.hardware_signature =
        hot_path_hardware_signature(&campaign, &placement_variant.hardware_class_id);
    context
}

fn current_hot_path_cargo_profile() -> String {
    hot_path_override_env(HOT_PATH_CARGO_PROFILE_ENV)
        .or_else(|| std::env::var("PROFILE").ok())
        .unwrap_or_else(|| "unknown".to_owned())
}

fn resolve_hot_path_workspace_root(
    output_dir: &Path,
    golden_dir: &Path,
    working_base: &Path,
) -> Option<PathBuf> {
    if let Some(workspace_root) = hot_path_override_env(HOT_PATH_WORKSPACE_ROOT_ENV) {
        if let Some(workspace_root) =
            resolve_hot_path_workspace_root_candidate(Path::new(&workspace_root))
        {
            return Some(workspace_root);
        }
    }
    let mut candidates = Vec::new();
    if let Ok(current_dir) = std::env::current_dir() {
        candidates.push(current_dir);
    }
    candidates.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    for path in [output_dir, golden_dir, working_base] {
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else if let Ok(current_dir) = std::env::current_dir() {
            current_dir.join(path)
        } else {
            path.to_path_buf()
        };
        candidates.push(candidate);
    }

    for candidate in candidates {
        if let Some(workspace_root) = resolve_hot_path_workspace_root_candidate(&candidate) {
            return Some(workspace_root);
        }
    }
    None
}

fn tool_version_command(tool: &str) -> Option<(&'static str, &'static [&'static str])> {
    match tool {
        "cargo" => Some(("cargo", &["--version"])),
        "git" => Some(("git", &["--version"])),
        "heaptrack" => Some(("heaptrack", &["--version"])),
        "hyperfine" => Some(("hyperfine", &["--version"])),
        "rch" => Some(("rch", &["--version"])),
        "rustc" => Some(("rustc", &["--version"])),
        "strace" => Some(("strace", &["-V"])),
        "perf-record" | "perf-stat" | "perf-c2c" | "perf-mem" | "perf-sched-record" => {
            Some(("perf", &["--version"]))
        }
        _ => None,
    }
}

fn resolve_hot_path_tool_version(tool: &str) -> Option<String> {
    let (program, args) = tool_version_command(tool)?;
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !stdout.is_empty() {
        return Some(stdout);
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !stderr.is_empty() {
        return Some(stderr);
    }
    Some("available".to_owned())
}

fn push_hot_path_tool_version(tool_versions: &mut Vec<BenchmarkArtifactToolVersion>, tool: &str) {
    if tool_versions.iter().any(|entry| entry.tool == tool) {
        return;
    }
    let version = resolve_hot_path_tool_version(tool).unwrap_or_else(|| "unavailable".to_owned());
    tool_versions.push(BenchmarkArtifactToolVersion {
        tool: tool.to_owned(),
        version,
    });
}

fn resolve_hot_path_source_revision(workspace_root: &Path) -> Option<String> {
    if let Some(source_revision) = hot_path_override_env(HOT_PATH_SOURCE_REVISION_ENV) {
        return Some(source_revision);
    }
    let output = Command::new("git")
        .args([
            "-C",
            &workspace_root.display().to_string(),
            "rev-parse",
            "HEAD",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!value.is_empty()).then_some(value)
}

fn resolve_hot_path_beads_data_hash(workspace_root: &Path) -> Option<String> {
    if let Some(beads_data_hash) = hot_path_override_env(HOT_PATH_BEADS_DATA_HASH_ENV) {
        return Some(beads_data_hash);
    }
    let campaign = load_beads_benchmark_campaign(workspace_root).ok()?;
    let beads_path = workspace_root.join(campaign.beads_data_relpath);
    let bytes = fs::read(beads_path).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

fn resolve_hot_path_artifact_provenance_inputs(
    command_pack: &HotPathEvidenceCommandPack,
    output_dir: &Path,
    golden_dir: &Path,
    working_base: &Path,
) -> HotPathArtifactProvenanceInputs {
    let cargo_profile = current_hot_path_cargo_profile();
    let environment = EnvironmentMeta::capture(&cargo_profile);
    let workspace_root = resolve_hot_path_workspace_root(output_dir, golden_dir, working_base);
    let workspace_root_string = workspace_root
        .as_ref()
        .map(|root| root.display().to_string())
        .or_else(|| hot_path_override_env(HOT_PATH_WORKSPACE_ROOT_ENV));
    let campaign_manifest_path = hot_path_override_env(HOT_PATH_CAMPAIGN_MANIFEST_PATH_ENV)
        .or_else(|| {
            workspace_root.as_ref().and_then(|root| {
                root.join(BEADS_BENCHMARK_CAMPAIGN_PATH_RELATIVE)
                    .is_file()
                    .then(|| BEADS_BENCHMARK_CAMPAIGN_PATH_RELATIVE.to_owned())
            })
        });
    let source_revision = hot_path_override_env(HOT_PATH_SOURCE_REVISION_ENV).or_else(|| {
        workspace_root
            .as_ref()
            .and_then(|root| resolve_hot_path_source_revision(root))
    });
    let beads_data_hash = hot_path_override_env(HOT_PATH_BEADS_DATA_HASH_ENV).or_else(|| {
        workspace_root
            .as_ref()
            .and_then(|root| resolve_hot_path_beads_data_hash(root))
    });
    let mut tool_versions = Vec::new();
    for tool in ["cargo", "git", "rch", "rustc"] {
        push_hot_path_tool_version(&mut tool_versions, tool);
    }
    for command in &command_pack.commands {
        push_hot_path_tool_version(&mut tool_versions, &command.tool);
    }
    tool_versions.sort_by(|left, right| left.tool.cmp(&right.tool));
    HotPathArtifactProvenanceInputs {
        artifact_root: output_dir.display().to_string(),
        workspace_root: workspace_root_string,
        campaign_manifest_path,
        source_revision,
        beads_data_hash,
        kernel_release: environment.os,
        rustc_version: environment.rustc_version,
        cargo_profile,
        tool_versions,
    }
}

fn build_hot_path_artifact_provenance(
    report: &HotPathProfileReport,
    command_pack: &HotPathEvidenceCommandPack,
    counter_capture_summary: Option<&HotPathCounterCaptureManifestSummary>,
    inputs: HotPathArtifactProvenanceInputs,
) -> HotPathArtifactProvenance {
    let mut commands = Vec::with_capacity(command_pack.commands.len() + 1);
    commands.push(BenchmarkArtifactCommand {
        tool: "realdb-e2e".to_owned(),
        command_line: report.replay_command.clone(),
    });
    commands.extend(
        command_pack
            .commands
            .iter()
            .map(|command| BenchmarkArtifactCommand {
                tool: command.tool.clone(),
                command_line: command.command_line.clone(),
            }),
    );
    HotPathArtifactProvenance {
        row_id: hot_path_row_id(&report.workload, report.concurrency),
        mode_id: hot_path_mode_id(report.concurrent_mode).to_owned(),
        artifact_root: inputs.artifact_root,
        command_entrypoint: report.replay_command.clone(),
        workspace_root: inputs.workspace_root,
        campaign_manifest_path: inputs.campaign_manifest_path,
        source_revision: inputs.source_revision,
        beads_data_hash: inputs.beads_data_hash,
        kernel_release: inputs.kernel_release,
        rustc_version: inputs.rustc_version,
        cargo_profile: inputs.cargo_profile,
        commands,
        tool_versions: inputs.tool_versions,
        fallback_notes: counter_capture_summary
            .map_or_else(Vec::new, |summary| summary.fallback_notes.clone()),
    }
}

fn hot_path_artifact_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn refresh_hot_path_artifact_metadata(
    output_dir: &Path,
    file: &mut HotPathArtifactFile,
) -> io::Result<()> {
    let bytes = fs::read(output_dir.join(&file.path))?;
    file.bytes = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    file.sha256 = hot_path_artifact_sha256(&bytes);
    Ok(())
}

fn finalize_hot_path_manifest(
    output_dir: &Path,
    manifest: HotPathArtifactManifest,
    counter_capture_summary: Option<HotPathCounterCaptureManifestSummary>,
    extra_files: Vec<HotPathArtifactFile>,
) -> io::Result<HotPathArtifactManifest> {
    let mut files: Vec<HotPathArtifactFile> = manifest
        .files
        .into_iter()
        .filter(|file| file.path != "manifest.json")
        .collect();
    for extra in extra_files {
        if let Some(existing) = files.iter_mut().find(|file| file.path == extra.path) {
            *existing = extra;
        } else {
            files.push(extra);
        }
    }
    for file in &mut files {
        refresh_hot_path_artifact_metadata(output_dir, file)?;
    }
    let mut disk_manifest = HotPathArtifactManifest {
        files,
        counter_capture_summary,
        ..manifest
    };
    let manifest_json = serde_json::to_string_pretty(&disk_manifest)
        .map_err(|error| io::Error::other(format!("artifact manifest: {error}")))?;
    fs::write(output_dir.join("manifest.json"), manifest_json.as_bytes())?;
    disk_manifest.files.push(HotPathArtifactFile {
        path: "manifest.json".to_owned(),
        bytes: u64::try_from(manifest_json.len()).unwrap_or(u64::MAX),
        sha256: hot_path_artifact_sha256(manifest_json.as_bytes()),
        description: "artifact manifest with replay metadata".to_owned(),
    });
    Ok(disk_manifest)
}

fn read_hot_path_disk_manifest(output_dir: &Path) -> io::Result<HotPathArtifactManifest> {
    let manifest_json = fs::read_to_string(output_dir.join("manifest.json"))?;
    serde_json::from_str(&manifest_json)
        .map_err(|error| io::Error::other(format!("artifact manifest JSON: {error}")))
}

#[allow(clippy::too_many_arguments)]
fn serialize_run_output(
    recorded_unix_ms: u64,
    environment: &EnvironmentMeta,
    engine: EngineInfo,
    fixture_id: &str,
    golden_path: &Path,
    golden_sha256: Option<String>,
    workload: &str,
    concurrency: u16,
    ops_count: usize,
    report: fsqlite_e2e::report::EngineRunReport,
    pretty: bool,
) -> Result<String, serde_json::Error> {
    let ops_count = u64::try_from(ops_count).unwrap_or(u64::MAX);
    let record = RunRecordV1::new(RunRecordV1Args {
        recorded_unix_ms,
        environment: environment.clone(),
        engine,
        fixture_id: fixture_id.to_owned(),
        golden_path: Some(golden_path.display().to_string()),
        golden_sha256,
        workload: workload.to_owned(),
        concurrency,
        ops_count,
        report,
    });

    if pretty {
        record.to_pretty_json()
    } else {
        record.to_jsonl_line()
    }
}

fn main() {
    let exit_code = run_cli(std::env::args_os());
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn run_cli<I>(os_args: I) -> i32
where
    I: IntoIterator<Item = OsString>,
{
    let raw: Vec<String> = os_args
        .into_iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();

    // Skip program name (raw[0]).
    let tail = if raw.len() > 1 { &raw[1..] } else { &[] };

    if tail.is_empty() || tail.iter().any(|a| a == "-h" || a == "--help") {
        print_top_level_help();
        return 0;
    }

    match tail[0].as_str() {
        "corpus" => cmd_corpus(&tail[1..]),
        "run" => cmd_run(&tail[1..]),
        "bench" => cmd_bench(&tail[1..]),
        "hot-profile" => cmd_hot_profile(&tail[1..]),
        "corrupt" => cmd_corrupt(&tail[1..]),
        "compare" => cmd_compare(&tail[1..]),
        other => {
            eprintln!("error: unknown subcommand `{other}`");
            eprintln!();
            print_top_level_help();
            2
        }
    }
}

// ── Top-level help ──────────────────────────────────────────────────────

fn print_top_level_help() {
    let text = "\
realdb-e2e — Differential testing of FrankenSQLite vs C SQLite

USAGE:
    realdb-e2e <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    corpus scan             Discover SQLite databases under /dp
    corpus import           Copy selected DBs into golden/ with checksums
    corpus verify           Verify golden copies against checksums.sha256
    run                     Execute an OpLog workload against an engine
    bench                   Run the benchmark matrix (Criterion)
    hot-profile             Capture hot-path evidence for a benchmark preset
    corrupt                 Inject corruption into a working copy
    compare                 Tiered comparison of two database files

OPTIONS:
    -h, --help              Show this help message

EXAMPLES:
    realdb-e2e corpus scan
    realdb-e2e corpus scan --root /dp --max-depth 4
    realdb-e2e corpus import --db beads.db --tag beads
    realdb-e2e corpus verify
    realdb-e2e run --engine sqlite3 --db beads-proj-a --workload commutative_inserts --concurrency 4
    realdb-e2e run --engine fsqlite --db beads-proj-a --workload hot_page_contention --concurrency 8
    realdb-e2e bench --db beads-proj-a --preset all
    realdb-e2e hot-profile --db beads-proj-a --workload mixed_read_write --concurrency 4
    realdb-e2e corrupt --db beads-proj-a --strategy page --page 1 --seed 42
";
    let _ = io::stdout().write_all(text.as_bytes());
}

// ── corpus ──────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn cmd_corpus(argv: &[String]) -> i32 {
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print_corpus_help();
        return if argv.is_empty() { 2 } else { 0 };
    }

    match argv[0].as_str() {
        "scan" => cmd_corpus_scan(&argv[1..]),
        "import" => cmd_corpus_import(&argv[1..]),
        "verify" => cmd_corpus_verify(&argv[1..]),
        other => {
            eprintln!("error: unknown corpus subcommand `{other}`");
            eprintln!();
            print_corpus_help();
            2
        }
    }
}

fn print_corpus_help() {
    let text = "\
realdb-e2e corpus — Manage the SQLite database fixture corpus

USAGE:
    realdb-e2e corpus <ACTION> [OPTIONS]

ACTIONS:
    scan        Discover SQLite databases under configured roots
    import      Copy a discovered database into golden/ with checksums
    verify      Verify all golden copies match their checksums entries

SCAN OPTIONS:
    --root <DIR>            Root directory to scan (default: /dp)
    --max-depth <N>         Maximum traversal depth (default: 6)
    --min-bytes <N>         Skip files smaller than N bytes (default: 0)
    --max-bytes <N>         Skip files larger than N bytes (default: 536870912).
                            Use 0 to disable the size cap (not recommended).
    --max-file-size-mib <N> Alias for --max-bytes, expressed in MiB (default: 512).
                            Use 0 to disable the size cap (not recommended).
    --header-only           Only include files with valid SQLite magic header
                            (alias: --require-header-ok)
    --require-header-ok     Alias for --header-only
    --json                  Emit machine-readable JSON describing candidates

IMPORT OPTIONS:
    --db <PATH|NAME>        Source database path (preferred) or discovery filename/stem
    --id <DB_ID>            Override destination fixture id (default: sanitized stem)
    --tag <LABEL>           Classification tag (stored in metadata).
                            Stable tags: asupersync, frankentui, flywheel, frankensqlite,
                            agent-mail, beads, misc
    --pii-risk <LEVEL>      PII risk classification for metadata
                            (unknown|unlikely|possible|likely; default: unknown)
    --secrets-risk <LEVEL>  Secrets risk classification for metadata
                            (unknown|unlikely|possible|likely; default: unknown)
    --allow-for-ci          Mark fixture as allowed_for_ci=true in metadata
                            (default: false; implicitly true when both risks are unlikely)
    --golden-dir <DIR>      Destination golden directory
                            (default: sample_sqlite_db_files/golden)
    --metadata-dir <DIR>    Destination metadata directory
                            (default: sample_sqlite_db_files/metadata)
    --checksums <PATH>      Checksums file to update
                            (default: sample_sqlite_db_files/checksums.sha256)
    --root <DIR>            Discovery root (only used when resolving NAME)
                            (default: /dp)
    --max-depth <N>         Discovery max-depth (only used when resolving NAME)
                            (default: 6)
    --max-file-size-mib <N>
                            Refuse to import files larger than N MiB unless overridden
                            (default: 512). Use 0 to disable the size cap (not recommended).
    --allow-bad-header      Allow importing files failing SQLite magic header check
    --no-metadata           Skip metadata generation

VERIFY OPTIONS:
    --checksums <PATH>    Path to checksums file (default: sample_sqlite_db_files/checksums.sha256)
    --golden-dir <DIR>    Directory containing golden DB copies
                          (default: sample_sqlite_db_files/golden)
    --json                Emit machine-readable JSON instead of human text
";
    let _ = io::stdout().write_all(text.as_bytes());
}

#[derive(Debug, Serialize)]
struct CorpusScanReportV1 {
    /// Stable contract identifier for scan output.
    schema_version: String,
    candidates: Vec<CorpusScanCandidateV1>,
}

#[derive(Debug, Serialize)]
struct CorpusScanCandidateV1 {
    /// Absolute path to the discovered file.
    path: String,
    /// Discovered filename (basename).
    file_name: String,
    /// Inferred id candidate (sanitized stem).
    db_id: String,
    /// File size in bytes.
    size_bytes: u64,
    /// Whether the file begins with the SQLite header magic bytes.
    header_ok: bool,
    /// Sidecar suffixes present (`-wal`, `-shm`, `-journal`).
    sidecars_present: Vec<String>,
    /// Tags inferred from path heuristics (sorted, deduped).
    tags: Vec<String>,
}

#[allow(clippy::too_many_lines)]
fn cmd_corpus_scan(argv: &[String]) -> i32 {
    let mut root = PathBuf::from("/dp");
    let mut max_depth: usize = 6;
    let mut min_bytes: u64 = 0;
    let mut max_bytes: u64 = 512 * 1024 * 1024;
    let mut header_only = false;
    let mut json = false;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--root" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --root requires a directory argument");
                    return 2;
                }
                root = PathBuf::from(&argv[i]);
            }
            "--max-depth" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --max-depth requires an integer argument");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<usize>() else {
                    eprintln!("error: invalid integer for --max-depth: `{}`", argv[i]);
                    return 2;
                };
                max_depth = n;
            }
            "--min-bytes" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --min-bytes requires an integer argument");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u64>() else {
                    eprintln!("error: invalid integer for --min-bytes: `{}`", argv[i]);
                    return 2;
                };
                min_bytes = n;
            }
            "--max-bytes" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --max-bytes requires an integer argument");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u64>() else {
                    eprintln!("error: invalid integer for --max-bytes: `{}`", argv[i]);
                    return 2;
                };
                max_bytes = if n == 0 { u64::MAX } else { n };
            }
            "--max-file-size-mib" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --max-file-size-mib requires an integer argument");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u64>() else {
                    eprintln!(
                        "error: invalid integer for --max-file-size-mib: `{}`",
                        argv[i]
                    );
                    return 2;
                };
                max_bytes = match mib_to_bytes(n) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return 2;
                    }
                };
            }
            "--header-only" | "--require-header-ok" => header_only = true,
            "--json" => json = true,
            "-h" | "--help" => {
                print_corpus_help();
                return 0;
            }
            other => {
                eprintln!("error: unknown option `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    let mut config = fsqlite_harness::fixture_discovery::DiscoveryConfig {
        roots: vec![root],
        max_depth,
        min_file_size: min_bytes,
        header_only,
        ..fsqlite_harness::fixture_discovery::DiscoveryConfig::default()
    };
    config.max_file_size = max_bytes;

    match fsqlite_harness::fixture_discovery::discover_sqlite_files(&config) {
        Ok(candidates) => {
            if json {
                let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                let report = CorpusScanReportV1 {
                    schema_version: "corpus_scan_v1".to_owned(),
                    candidates: candidates
                        .iter()
                        .map(|c| {
                            let abs = if c.path.is_absolute() {
                                c.path.clone()
                            } else {
                                cwd.join(&c.path)
                            };
                            CorpusScanCandidateV1 {
                                path: abs.to_string_lossy().into_owned(),
                                file_name: c
                                    .path
                                    .file_name()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or_default()
                                    .to_owned(),
                                db_id: c.db_id.clone(),
                                size_bytes: c.size_bytes,
                                header_ok: c.header_ok,
                                sidecars_present: c.sidecars_present.clone(),
                                tags: c.tags.clone(),
                            }
                        })
                        .collect(),
                };

                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{text}"),
                    Err(e) => {
                        eprintln!("error: failed to serialize scan report as JSON: {e}");
                        return 2;
                    }
                }
            } else {
                println!("Found {} candidate(s):", candidates.len());
                for c in &candidates {
                    let mut line = String::new();
                    let _ = write!(&mut line, "  {c}");
                    if !c.sidecars_present.is_empty() {
                        let _ = write!(&mut line, "\tsidecars={}", c.sidecars_present.join(","));
                    }
                    let _ = write!(&mut line, "\tdb_id={}", c.db_id);
                    println!("{line}");
                }
            }
            0
        }
        Err(e) => {
            eprintln!("error: corpus scan failed: {e}");
            1
        }
    }
}

#[allow(clippy::too_many_lines)]
fn cmd_corpus_import(argv: &[String]) -> i32 {
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print_corpus_help();
        return if argv.is_empty() { 2 } else { 0 };
    }

    let mut db_arg: Option<String> = None;
    let mut id_override: Option<String> = None;
    let mut tag: Option<String> = None;
    let mut pii_risk = RiskLevel::Unknown;
    let mut secrets_risk = RiskLevel::Unknown;
    let mut allow_for_ci = false;
    let mut golden_dir = PathBuf::from(DEFAULT_GOLDEN_DIR);
    let mut metadata_dir = PathBuf::from(DEFAULT_METADATA_DIR);
    let mut checksums_path = PathBuf::from(DEFAULT_CHECKSUMS_PATH);
    let mut root = PathBuf::from("/dp");
    let mut max_depth: usize = 6;
    let mut max_file_size_mib: u64 = 512;
    let mut allow_bad_header = false;
    let mut write_metadata = true;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--db" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --db requires a path or discovery name");
                    return 2;
                }
                db_arg = Some(argv[i].clone());
            }
            "--id" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --id requires a fixture identifier");
                    return 2;
                }
                id_override = Some(argv[i].clone());
            }
            "--tag" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --tag requires a label");
                    return 2;
                }
                tag = Some(argv[i].clone());
            }
            "--pii-risk" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --pii-risk requires a level");
                    return 2;
                }
                match RiskLevel::parse(&argv[i]) {
                    Ok(v) => pii_risk = v,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return 2;
                    }
                }
            }
            "--secrets-risk" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --secrets-risk requires a level");
                    return 2;
                }
                match RiskLevel::parse(&argv[i]) {
                    Ok(v) => secrets_risk = v,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return 2;
                    }
                }
            }
            "--allow-for-ci" => allow_for_ci = true,
            "--golden-dir" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --golden-dir requires a directory path");
                    return 2;
                }
                golden_dir = PathBuf::from(&argv[i]);
            }
            "--metadata-dir" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --metadata-dir requires a directory path");
                    return 2;
                }
                metadata_dir = PathBuf::from(&argv[i]);
            }
            "--checksums" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --checksums requires a file path");
                    return 2;
                }
                checksums_path = PathBuf::from(&argv[i]);
            }
            "--root" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --root requires a directory path");
                    return 2;
                }
                root = PathBuf::from(&argv[i]);
            }
            "--max-depth" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --max-depth requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<usize>() else {
                    eprintln!("error: invalid integer for --max-depth: `{}`", argv[i]);
                    return 2;
                };
                max_depth = n;
            }
            "--max-file-size-mib" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --max-file-size-mib requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u64>() else {
                    eprintln!(
                        "error: invalid integer for --max-file-size-mib: `{}`",
                        argv[i]
                    );
                    return 2;
                };
                max_file_size_mib = n;
            }
            "--allow-bad-header" => allow_bad_header = true,
            "--no-metadata" => write_metadata = false,
            other => {
                eprintln!("error: unknown option `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    let Some(db_arg) = db_arg.as_deref() else {
        eprintln!("error: --db is required");
        return 2;
    };

    let max_file_size = match mib_to_bytes(max_file_size_mib) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };

    if let Some(tag) = tag.as_deref() {
        if !fsqlite_harness::fixture_discovery::is_stable_corpus_tag(tag) {
            eprintln!("error: unknown --tag `{tag}`");
            eprintln!(
                "help: allowed tags: {}",
                fsqlite_harness::fixture_discovery::STABLE_CORPUS_TAGS.join(", ")
            );
            return 2;
        }
    }

    // Resolve source DB path. Prefer literal paths; otherwise do a bounded discovery scan.
    let (source_path, source_tags, header_ok) =
        match resolve_source_db(db_arg, &root, max_depth, max_file_size) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                return 1;
            }
        };

    if !allow_bad_header && !header_ok {
        eprintln!(
            "error: source does not look like a SQLite database (bad magic header): {}",
            source_path.display()
        );
        return 1;
    }

    // Enforce size cap for literal paths too (discovery scan already does this).
    let source_meta = match fs::metadata(&source_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: cannot stat {}: {e}", source_path.display());
            return 1;
        }
    };
    if source_meta.len() > max_file_size {
        eprintln!(
            "error: refusing to import {} ({} bytes) because it exceeds max size cap ({} MiB).",
            source_path.display(),
            source_meta.len(),
            max_file_size_mib
        );
        eprintln!("help: pass --max-file-size-mib to override (0 disables the cap)");
        return 2;
    }
    if source_meta.len() > 64 * 1024 * 1024 {
        eprintln!(
            "warning: importing a relatively large DB ({} bytes). \
CI and local runs may be slow; prefer smaller fixtures when possible.",
            source_meta.len()
        );
    }

    let source_sidecars_present = detect_sidecars(&source_path);

    // Determine destination fixture id.
    let raw_id = id_override.as_deref().unwrap_or_else(|| {
        source_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("fixture")
    });
    let fixture_id = match sanitize_db_id(raw_id) {
        Ok(id) => id,
        Err(e) => {
            eprintln!("error: invalid fixture id `{raw_id}`: {e}");
            return 2;
        }
    };

    if let Err(e) = fs::create_dir_all(&golden_dir) {
        eprintln!(
            "error: failed to create golden dir {}: {e}",
            golden_dir.display()
        );
        return 1;
    }

    let dest_db = golden_dir.join(format!("{fixture_id}.db"));

    if dest_db.exists() {
        // Golden copies are immutable. Never overwrite in-place; use --id for a new fixture.
        println!("Golden already exists: {}", dest_db.display());
    } else {
        // Safety policy (sample_sqlite_db_files/FIXTURES.md): never raw-copy /dp inputs.
        // Use SQLite's backup API to capture a consistent snapshot.
        if let Err(e) = backup_sqlite_file(&source_path, &dest_db) {
            eprintln!(
                "error: failed to back up {} to {}: {e}",
                source_path.display(),
                dest_db.display()
            );
            return 1;
        }
    }

    // Verify integrity immediately after capture (or for existing golden).
    if let Err(e) = sqlite_integrity_check(&dest_db) {
        eprintln!(
            "error: golden DB failed PRAGMA integrity_check: {}: {e}",
            dest_db.display()
        );
        return 1;
    }

    // Best-effort: mark golden copies read-only.
    if let Err(e) = set_read_only(&dest_db) {
        eprintln!(
            "warning: failed to mark read-only {}: {e}",
            dest_db.display()
        );
    }

    // Update checksums file (DB only, not sidecars).
    let dest_sha = match sha256_file(&dest_db) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: cannot hash golden db {}: {e}", dest_db.display());
            return 1;
        }
    };
    if let Err(e) = upsert_checksum(&checksums_path, &dest_db, &dest_sha) {
        eprintln!("error: failed to update checksums: {e}");
        return 1;
    }

    // Generate/update metadata JSON unless disabled.
    if write_metadata {
        if let Err(e) = fs::create_dir_all(&metadata_dir) {
            eprintln!(
                "error: failed to create metadata dir {}: {e}",
                metadata_dir.display()
            );
            return 1;
        }

        let Some(golden_filename) = dest_db.file_name().and_then(|s| s.to_str()) else {
            eprintln!("error: invalid golden filename");
            return 1;
        };

        let allowed_for_ci = allow_for_ci
            || (pii_risk == RiskLevel::Unlikely && secrets_risk == RiskLevel::Unlikely);

        match profile_database_for_metadata(
            &dest_db,
            &fixture_id,
            Some(&source_path),
            golden_filename,
            &dest_sha,
            tag.as_deref(),
            &source_tags,
            &source_sidecars_present,
            FixtureSafetyV1 {
                pii_risk,
                secrets_risk,
                allowed_for_ci,
            },
        ) {
            Ok(profile) => {
                let out_path = metadata_dir.join(format!("{fixture_id}.json"));
                match serde_json::to_string_pretty(&profile) {
                    Ok(json) => {
                        if let Err(e) = fs::write(&out_path, json.as_bytes()) {
                            eprintln!(
                                "error: failed to write metadata {}: {e}",
                                out_path.display()
                            );
                            return 1;
                        }
                        println!("Wrote metadata: {}", out_path.display());
                    }
                    Err(e) => {
                        eprintln!("error: failed to serialize metadata: {e}");
                        return 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("error: failed to profile imported DB: {e}");
                return 1;
            }
        }
    }

    // Final summary.
    println!("Imported fixture:");
    println!("  id: {fixture_id}");
    println!("  source: {}", source_path.display());
    println!("  golden: {}", dest_db.display());
    println!("  sha256: {dest_sha}");
    if let Some(tag) = tag.as_deref() {
        println!("  tag: {tag}");
    }
    if !source_tags.is_empty() {
        println!("  tags: {}", source_tags.join(", "));
    }
    if !source_sidecars_present.is_empty() {
        println!("  sidecars: {}", source_sidecars_present.join(", "));
    }

    0
}

/// Default path for the checksums file (relative to workspace root).
const DEFAULT_CHECKSUMS_PATH: &str = "sample_sqlite_db_files/checksums.sha256";

/// Default directory containing golden database copies.
const DEFAULT_GOLDEN_DIR: &str = "sample_sqlite_db_files/golden";

/// Default directory containing per-fixture metadata JSON.
const DEFAULT_METADATA_DIR: &str = "sample_sqlite_db_files/metadata";

/// Default base directory for per-run working copies.
const DEFAULT_WORKING_DIR: &str = "sample_sqlite_db_files/working";

fn cmd_corpus_verify(argv: &[String]) -> i32 {
    let mut checksums_path = PathBuf::from(DEFAULT_CHECKSUMS_PATH);
    let mut golden_dir = PathBuf::from(DEFAULT_GOLDEN_DIR);
    let mut json = false;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--checksums" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --checksums requires a path argument");
                    return 2;
                }
                checksums_path = PathBuf::from(&argv[i]);
            }
            "--golden-dir" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --golden-dir requires a path argument");
                    return 2;
                }
                golden_dir = PathBuf::from(&argv[i]);
            }
            "--json" => {
                json = true;
            }
            "-h" | "--help" => {
                print_corpus_help();
                return 0;
            }
            other => {
                eprintln!("error: unknown option `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    let report = match verify_golden_checksums(&checksums_path, &golden_dir) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };

    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{text}"),
            Err(e) => {
                eprintln!("error: failed to serialize verify report as JSON: {e}");
                return 2;
            }
        }
    } else {
        print_verify_report_human(&report);
        println!(
            "\n{} ok, {} mismatch, {} missing, {} error, {} extra",
            report.summary.ok,
            report.summary.mismatch,
            report.summary.missing,
            report.summary.error,
            report.summary.extra,
        );
    }

    i32::from(report.summary.has_failures())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum VerifyStatus {
    Ok,
    Missing,
    Mismatch,
    Error,
    Extra,
}

#[derive(Debug, Serialize)]
struct VerifyFileResult {
    filename: String,
    status: VerifyStatus,
    expected_sha256: Option<String>,
    actual_sha256: Option<String>,
    file_size_bytes: Option<u64>,
    modified_unix_ms: Option<u64>,
    error: Option<String>,
    hint: Option<String>,
}

#[derive(Debug, Default, Serialize)]
struct VerifySummary {
    ok: usize,
    mismatch: usize,
    missing: usize,
    error: usize,
    extra: usize,
}

impl VerifySummary {
    fn has_failures(&self) -> bool {
        self.mismatch > 0 || self.missing > 0 || self.error > 0 || self.extra > 0
    }
}

#[derive(Debug, Serialize)]
struct VerifyReport {
    checksums_path: String,
    golden_dir: String,
    summary: VerifySummary,
    files: Vec<VerifyFileResult>,
}

fn print_verify_report_human(report: &VerifyReport) {
    for file in &report.files {
        match file.status {
            VerifyStatus::Ok => {
                println!(
                    "OK       {}{}",
                    file.filename,
                    fmt_size_mtime(file.file_size_bytes, file.modified_unix_ms),
                );
            }
            VerifyStatus::Missing => {
                eprintln!(
                    "MISSING  {}{}",
                    file.filename,
                    fmt_size_mtime(file.file_size_bytes, file.modified_unix_ms),
                );
                if let Some(hint) = &file.hint {
                    eprintln!("  hint: {hint}");
                }
            }
            VerifyStatus::Mismatch => {
                eprintln!(
                    "MISMATCH {}{}",
                    file.filename,
                    fmt_size_mtime(file.file_size_bytes, file.modified_unix_ms),
                );
                if let Some(expected) = &file.expected_sha256 {
                    eprintln!("  expected: {expected}");
                }
                if let Some(actual) = &file.actual_sha256 {
                    eprintln!("  actual:   {actual}");
                }
                if let Some(hint) = &file.hint {
                    eprintln!("  hint: {hint}");
                }
            }
            VerifyStatus::Error => {
                eprintln!(
                    "ERROR    {}{}",
                    file.filename,
                    fmt_size_mtime(file.file_size_bytes, file.modified_unix_ms),
                );
                if let Some(err) = &file.error {
                    eprintln!("  error: {err}");
                }
                if let Some(hint) = &file.hint {
                    eprintln!("  hint: {hint}");
                }
            }
            VerifyStatus::Extra => {
                eprintln!(
                    "EXTRA    {}{}",
                    file.filename,
                    fmt_size_mtime(file.file_size_bytes, file.modified_unix_ms),
                );
                if let Some(hint) = &file.hint {
                    eprintln!("  hint: {hint}");
                }
            }
        }
    }
}

fn fmt_size_mtime(file_size_bytes: Option<u64>, modified_unix_ms: Option<u64>) -> String {
    let (Some(size), Some(mtime)) = (file_size_bytes, modified_unix_ms) else {
        return String::new();
    };
    // Keep this compact; the human path is mainly for quick scanning.
    format!("  ({} B, mtime_ms={mtime})", size)
}

#[derive(Debug)]
struct ChecksumEntry {
    expected_sha256: String,
    filename: String,
}

/// Read `checksums.sha256`, recompute each hash, and compare.
#[allow(clippy::too_many_lines)]
fn verify_golden_checksums(
    checksums_path: &Path,
    golden_dir: &Path,
) -> Result<VerifyReport, String> {
    let contents = fs::read_to_string(checksums_path)
        .map_err(|e| format!("cannot read {}: {e}", checksums_path.display()))?;

    let mut expected_entries: Vec<ChecksumEntry> = Vec::new();
    let mut expected_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    for (line_no, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let display_line_no = line_no + 1;
        let (expected_hex, filename) = parse_checksum_line(line, display_line_no)?;
        if !expected_names.insert(filename.to_owned()) {
            return Err(format!(
                "duplicate filename in checksums file on line {display_line_no}: {filename}"
            ));
        }

        expected_entries.push(ChecksumEntry {
            expected_sha256: expected_hex.to_owned(),
            filename: filename.to_owned(),
        });
    }

    // We intentionally avoid following symlinks or attempting to open DBs here.
    let mut files: Vec<VerifyFileResult> = Vec::with_capacity(expected_entries.len());
    let mut summary = VerifySummary::default();

    for entry in expected_entries {
        if !is_safe_golden_filename(&entry.filename) {
            return Err(format!(
                "invalid fixture filename in checksums file: `{}` (must be a simple filename)",
                entry.filename
            ));
        }

        let file_path = golden_dir.join(&entry.filename);
        if !file_path.exists() {
            summary.missing += 1;
            files.push(VerifyFileResult {
                filename: entry.filename,
                status: VerifyStatus::Missing,
                expected_sha256: Some(entry.expected_sha256),
                actual_sha256: None,
                file_size_bytes: None,
                modified_unix_ms: None,
                error: None,
                hint: Some(
                    "Re-import the fixture or remove the stale entry from checksums.sha256."
                        .to_owned(),
                ),
            });
            continue;
        }

        let (size_bytes, modified_unix_ms) = file_size_and_mtime(&file_path);

        let actual_hex = match sha256_file(&file_path) {
            Ok(h) => h,
            Err(e) => {
                summary.error += 1;
                files.push(VerifyFileResult {
                    filename: entry.filename,
                    status: VerifyStatus::Error,
                    expected_sha256: Some(entry.expected_sha256),
                    actual_sha256: None,
                    file_size_bytes: size_bytes,
                    modified_unix_ms,
                    error: Some(e),
                    hint: Some(
                        "Fix filesystem permissions/IO errors, then re-run corpus verify."
                            .to_owned(),
                    ),
                });
                continue;
            }
        };

        if actual_hex == entry.expected_sha256 {
            summary.ok += 1;
            files.push(VerifyFileResult {
                filename: entry.filename,
                status: VerifyStatus::Ok,
                expected_sha256: Some(entry.expected_sha256),
                actual_sha256: Some(actual_hex),
                file_size_bytes: size_bytes,
                modified_unix_ms,
                error: None,
                hint: None,
            });
        } else {
            summary.mismatch += 1;
            files.push(VerifyFileResult {
                filename: entry.filename,
                status: VerifyStatus::Mismatch,
                expected_sha256: Some(entry.expected_sha256),
                actual_sha256: Some(actual_hex),
                file_size_bytes: size_bytes,
                modified_unix_ms,
                error: None,
                hint: Some(
                    "Golden bytes drifted. Investigate accidental writes to golden/, or recapture under a new fixture id and update checksums."
                        .to_owned(),
                ),
            });
        }
    }

    // EXTRA: any on-disk golden files not referenced by checksums.sha256.
    let dir = fs::read_dir(golden_dir)
        .map_err(|e| format!("cannot read golden dir {}: {e}", golden_dir.display()))?;
    let mut extra: Vec<VerifyFileResult> = Vec::new();
    for entry in dir {
        let entry = entry
            .map_err(|e| format!("cannot read golden dir entry {}: {e}", golden_dir.display()))?;
        let meta = entry
            .metadata()
            .map_err(|e| format!("cannot stat golden file {}: {e}", entry.path().display()))?;
        if !meta.is_file() {
            continue;
        }

        let filename = entry.file_name().to_string_lossy().into_owned();
        // Ignore local dotfiles and SQLite sidecars; checksums cover only the golden DB bytes.
        if filename.starts_with('.') || is_sqlite_sidecar_filename(&filename) {
            continue;
        }
        if expected_names.contains(&filename) {
            continue;
        }

        let modified_unix_ms = system_time_to_unix_ms(meta.modified().ok());
        extra.push(VerifyFileResult {
            filename,
            status: VerifyStatus::Extra,
            expected_sha256: None,
            actual_sha256: None,
            file_size_bytes: Some(meta.len()),
            modified_unix_ms,
            error: None,
            hint: Some(
                "Add this file to checksums.sha256 (if it is intended to be golden), or remove it from golden/ (if it is stray)."
                    .to_owned(),
            ),
        });
    }
    extra.sort_by(|a, b| a.filename.cmp(&b.filename));
    summary.extra += extra.len();
    files.extend(extra);

    Ok(VerifyReport {
        checksums_path: checksums_path.display().to_string(),
        golden_dir: golden_dir.display().to_string(),
        summary,
        files,
    })
}

/// Compute the SHA-256 hex digest of a file.
fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;

    let mut hasher = Sha256::new();
    let mut buf = vec![0_u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sqlite_page_size_or_default(db_bytes: &[u8]) -> u32 {
    if db_bytes.len() < DATABASE_HEADER_SIZE {
        return 4096;
    }
    let Ok(header_bytes) =
        <[u8; DATABASE_HEADER_SIZE]>::try_from(&db_bytes[..DATABASE_HEADER_SIZE])
    else {
        return 4096;
    };
    let Ok(header) = DatabaseHeader::from_bytes(&header_bytes) else {
        return 4096;
    };
    header.page_size.get()
}

fn diff_modified_ranges(before: &[u8], after: &[u8], page_size: u32) -> Vec<CorruptModification> {
    let ps = u64::from(page_size.max(1));
    let common_len = before.len().min(after.len());

    let mut mods = Vec::new();

    let mut i = 0usize;
    while i < common_len {
        if before[i] == after[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < common_len && before[i] != after[i] {
            i += 1;
        }
        let end = i;

        let offset = u64::try_from(start).unwrap_or(u64::MAX);
        let length = u64::try_from(end - start).unwrap_or(u64::MAX);
        let page_first = u32::try_from(offset / ps + 1).unwrap_or(u32::MAX);
        let page_last = u32::try_from((offset + length - 1) / ps + 1).unwrap_or(u32::MAX);
        mods.push(CorruptModification {
            offset,
            length,
            page_first,
            page_last,
            sha256_before: sha256_bytes(&before[start..end]),
            sha256_after: Some(sha256_bytes(&after[start..end])),
        });
    }

    // Handle truncation (tail removed) or append (tail added).
    if after.len() < before.len() {
        let start = after.len();
        let end = before.len();
        let offset = u64::try_from(start).unwrap_or(u64::MAX);
        let length = u64::try_from(end - start).unwrap_or(u64::MAX);
        let page_first = u32::try_from(offset / ps + 1).unwrap_or(u32::MAX);
        let page_last = u32::try_from((offset + length - 1) / ps + 1).unwrap_or(u32::MAX);
        mods.push(CorruptModification {
            offset,
            length,
            page_first,
            page_last,
            sha256_before: sha256_bytes(&before[start..end]),
            sha256_after: None,
        });
    } else if after.len() > before.len() {
        let start = before.len();
        let end = after.len();
        let offset = u64::try_from(start).unwrap_or(u64::MAX);
        let length = u64::try_from(end - start).unwrap_or(u64::MAX);
        let page_first = u32::try_from(offset / ps + 1).unwrap_or(u32::MAX);
        let page_last = u32::try_from((offset + length - 1) / ps + 1).unwrap_or(u32::MAX);
        mods.push(CorruptModification {
            offset,
            length,
            page_first,
            page_last,
            sha256_before: sha256_bytes(&[]),
            sha256_after: Some(sha256_bytes(&after[start..end])),
        });
    }

    mods
}

fn parse_checksum_line(line: &str, line_no: usize) -> Result<(&str, &str), String> {
    // Format: "<hex>  <filename>" (two-space separator, sha256sum convention).
    let Some((expected_hex, filename)) = line.split_once("  ") else {
        return Err(format!(
            "malformed checksums line {line_no}: expected `<sha256>  <filename>`"
        ));
    };

    let expected_hex = expected_hex.trim();
    let filename = filename.trim();

    if expected_hex.len() != 64 || !expected_hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "malformed checksums line {line_no}: invalid sha256 hex: `{expected_hex}`"
        ));
    }
    if filename.is_empty() {
        return Err(format!(
            "malformed checksums line {line_no}: empty filename after sha256"
        ));
    }

    Ok((expected_hex, filename))
}

fn is_safe_golden_filename(filename: &str) -> bool {
    let path = Path::new(filename);
    let mut components = path.components();
    matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
}

fn file_size_and_mtime(path: &Path) -> (Option<u64>, Option<u64>) {
    let Ok(meta) = fs::metadata(path) else {
        return (None, None);
    };
    (
        Some(meta.len()),
        system_time_to_unix_ms(meta.modified().ok()),
    )
}

fn system_time_to_unix_ms(st: Option<SystemTime>) -> Option<u64> {
    let st = st?;
    let dur = st.duration_since(UNIX_EPOCH).ok()?;
    u64::try_from(dur.as_millis()).ok()
}

// ── run ─────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn cmd_run(argv: &[String]) -> i32 {
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        print_run_help();
        return 0;
    }

    let mut engine: Option<String> = None;
    let mut db: Option<String> = None;
    let mut workload: Option<String> = None;
    let mut concurrency: Vec<u16> = vec![1];
    let mut repeat: usize = 1;
    let mut fsqlite_mvcc: bool = true;
    let mut profile_only = false;
    let mut skip_integrity_check = false;
    let mut skip_environment_metadata = false;
    let mut pretty: bool = false;
    let mut output_jsonl: Option<PathBuf> = None;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--engine" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --engine requires an argument (sqlite3|fsqlite)");
                    return 2;
                }
                engine = Some(argv[i].clone());
            }
            "--db" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --db requires a database identifier");
                    return 2;
                }
                db = Some(argv[i].clone());
            }
            "--workload" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --workload requires a preset name");
                    return 2;
                }
                workload = Some(argv[i].clone());
            }
            "--concurrency" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --concurrency requires an integer or comma-separated list");
                    return 2;
                }
                match parse_u16_list(&argv[i]) {
                    Ok(v) => concurrency = v,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return 2;
                    }
                }
            }
            "--repeat" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --repeat requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<usize>() else {
                    eprintln!("error: invalid integer for --repeat: `{}`", argv[i]);
                    return 2;
                };
                if n == 0 {
                    eprintln!("error: --repeat must be >= 1");
                    return 2;
                }
                repeat = n;
            }
            "--mvcc" => {
                fsqlite_mvcc = true;
            }
            "--no-mvcc" => {
                fsqlite_mvcc = false;
            }
            "--profile-only" => {
                profile_only = true;
            }
            "--skip-integrity-check" => {
                skip_integrity_check = true;
            }
            "--skip-environment-metadata" => {
                skip_environment_metadata = true;
            }
            "--pretty" => {
                pretty = true;
            }
            "--output-jsonl" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --output-jsonl requires a path");
                    return 2;
                }
                output_jsonl = Some(PathBuf::from(argv[i].clone()));
            }
            other => {
                eprintln!("error: unknown option `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    let Some(engine_str) = engine.as_deref() else {
        eprintln!("error: --engine is required (sqlite3|fsqlite)");
        return 2;
    };
    let Some(db_name) = db.as_deref() else {
        eprintln!("error: --db is required (golden database identifier)");
        return 2;
    };
    let Some(workload_name) = workload.as_deref() else {
        eprintln!("error: --workload is required (preset name)");
        return 2;
    };
    let run_mode = RunModeOptions::from_flags(
        profile_only,
        skip_integrity_check,
        skip_environment_metadata,
    );

    match engine_str {
        "sqlite3" => run_sqlite3_engine(
            db_name,
            workload_name,
            &concurrency,
            repeat,
            run_mode,
            pretty,
            output_jsonl.as_deref(),
        ),
        "fsqlite" => run_fsqlite_engine(FsqliteRunArgs {
            db_name,
            workload_name,
            concurrency: &concurrency,
            repeat,
            mvcc: fsqlite_mvcc,
            run_mode,
            pretty,
            output_jsonl: output_jsonl.as_deref(),
        }),
        other => {
            eprintln!("error: unknown engine `{other}` (expected sqlite3 or fsqlite)");
            2
        }
    }
}

/// Resolve a database identifier to its golden copy path.
///
/// Accepts either a bare name (e.g. `"frankensqlite"`) which maps to
/// `sample_sqlite_db_files/golden/frankensqlite.db`, or an absolute/relative
/// path to an existing `.db` file.
fn resolve_golden_db(db_name: &str) -> Result<PathBuf, String> {
    // If it looks like a path and exists, use it directly.
    let as_path = PathBuf::from(db_name);
    if as_path.exists() {
        return Ok(as_path);
    }

    // Try golden directory with .db extension.
    let golden = PathBuf::from(DEFAULT_GOLDEN_DIR).join(format!("{db_name}.db"));
    if golden.exists() {
        return Ok(golden);
    }

    // Try golden directory without adding .db (user may have included it).
    let golden_bare = PathBuf::from(DEFAULT_GOLDEN_DIR).join(db_name);
    if golden_bare.exists() {
        return Ok(golden_bare);
    }

    Err(format!(
        "cannot find database `{db_name}` (tried {}, {}, and literal path)",
        golden.display(),
        golden_bare.display(),
    ))
}

/// Generate an OpLog from a preset name and concurrency setting.
fn resolve_workload(preset: &str, fixture_id: &str, concurrency: u16) -> Result<OpLog, String> {
    match preset {
        "commutative_inserts_disjoint_keys" | "commutative_inserts" => Ok(
            oplog::preset_commutative_inserts_disjoint_keys(fixture_id, 42, concurrency, 100),
        ),
        "hot_page_contention" | "hot_page" => Ok(oplog::preset_hot_page_contention(
            fixture_id,
            42,
            concurrency,
            10,
        )),
        "mixed_read_write" | "mixed" => Ok(oplog::preset_mixed_read_write(
            fixture_id,
            42,
            concurrency,
            50,
        )),
        other => Err(format!(
            "unknown workload preset `{other}`. Available: \
             commutative_inserts_disjoint_keys, hot_page_contention, mixed_read_write"
        )),
    }
}

/// Execute a workload against C SQLite via rusqlite and print JSON results.
fn report_has_failure(report: &fsqlite_e2e::report::EngineRunReport) -> bool {
    report.error.is_some() || report.correctness.integrity_check_ok == Some(false)
}

/// Execute a workload against C SQLite via rusqlite and print JSON results.
#[allow(clippy::too_many_lines)]
fn run_sqlite3_engine(
    db_name: &str,
    workload_name: &str,
    concurrency: &[u16],
    repeat: usize,
    run_mode: RunModeOptions,
    pretty: bool,
    output_jsonl: Option<&Path>,
) -> i32 {
    // Resolve golden DB path.
    let golden_path = match resolve_golden_db(db_name) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    // Copy golden to a working directory so we don't modify the original.
    let work_dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: failed to create temp dir: {e}");
            return 1;
        }
    };
    let work_db = work_dir.path().join("work.db");
    if let Err(e) = fs::copy(&golden_path, &work_db) {
        eprintln!(
            "error: failed to copy {} to {}: {e}",
            golden_path.display(),
            work_db.display()
        );
        return 1;
    }

    let config = SqliteExecConfig {
        run_integrity_check: run_mode.run_integrity_check,
        ..SqliteExecConfig::default()
    };
    let sqlite_version = rusqlite::version().to_owned();
    let environment = run_mode.environment(cargo_profile_name());

    let golden_sha256 = match sha256_file(&golden_path) {
        Ok(h) => Some(h),
        Err(e) => {
            eprintln!("warning: failed to compute golden sha256: {e}");
            None
        }
    };

    let mut results: Vec<RunAgg> = Vec::new();
    let mut any_error = false;

    for &c in concurrency {
        let mut agg = RunAgg::new(c);
        for rep in 0..repeat {
            // Copy golden to a fresh working directory so we don't modify the original.
            let work_dir = match tempfile::tempdir() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: failed to create temp dir: {e}");
                    return 1;
                }
            };
            let work_db = work_dir.path().join("work.db");
            if let Err(e) = fs::copy(&golden_path, &work_db) {
                eprintln!(
                    "error: failed to copy {} to {}: {e}",
                    golden_path.display(),
                    work_db.display()
                );
                return 1;
            }

            let oplog = match resolve_workload(workload_name, db_name, c) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("error: {e}");
                    return 1;
                }
            };

            eprintln!(
                "Running: engine=sqlite3 (v{sqlite_version}) db={db_name} workload={workload_name} \
                 concurrency={c} rep={rep}/{repeat}"
            );
            eprintln!("  golden: {}", golden_path.display());
            eprintln!("  working: {}", work_db.display());
            eprintln!("  ops: {}", oplog.records.len());

            let report = match run_oplog_sqlite(&work_db, &oplog, &config) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: execution failed: {e}");
                    return 1;
                }
            };
            agg.record(&report);
            any_error |= report_has_failure(&report);

            let recorded_unix_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));

            let text = match serialize_run_output(
                recorded_unix_ms,
                &environment,
                EngineInfo {
                    name: "sqlite3".to_owned(),
                    sqlite_version: Some(sqlite_version.clone()),
                    fsqlite_git: None,
                },
                db_name,
                &golden_path,
                golden_sha256.clone(),
                workload_name,
                c,
                oplog.records.len(),
                report,
                pretty,
            ) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error: failed to serialize report: {e}");
                    return 1;
                }
            };

            if let Some(path) = output_jsonl {
                if let Err(e) = append_jsonl_line(path, &text) {
                    eprintln!("error: failed to append JSONL output: {e}");
                    return 1;
                }
            }
            println!("{text}");
        }
        results.push(agg);
    }

    if results.len() > 1 || repeat > 1 {
        eprintln!("{}", format_scaling_summary("sqlite3", repeat, &results));
    }

    i32::from(any_error)
}

/// Execute a workload against FrankenSQLite and print JSON results.
#[allow(clippy::too_many_lines)]
struct FsqliteRunArgs<'a> {
    db_name: &'a str,
    workload_name: &'a str,
    concurrency: &'a [u16],
    repeat: usize,
    mvcc: bool,
    run_mode: RunModeOptions,
    pretty: bool,
    output_jsonl: Option<&'a Path>,
}

#[allow(clippy::too_many_lines)]
fn run_fsqlite_engine(args: FsqliteRunArgs<'_>) -> i32 {
    let FsqliteRunArgs {
        db_name,
        workload_name,
        concurrency,
        repeat,
        mvcc,
        run_mode,
        pretty,
        output_jsonl,
    } = args;
    let golden_path = match resolve_golden_db(db_name) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    let golden_sha256 = match sha256_file(&golden_path) {
        Ok(h) => Some(h),
        Err(e) => {
            eprintln!("warning: failed to compute golden sha256: {e}");
            None
        }
    };

    let config = FsqliteExecConfig {
        concurrent_mode: mvcc,
        run_integrity_check: run_mode.run_integrity_check,
        collect_hot_path_profile: run_mode.hot_path_profile,
        ..FsqliteExecConfig::default()
    };
    let environment = run_mode.environment(cargo_profile_name());

    let mut results: Vec<RunAgg> = Vec::new();
    let mut any_error = false;

    for &c in concurrency {
        let mut agg = RunAgg::new(c);
        for rep in 0..repeat {
            let work_dir = match tempfile::tempdir() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("error: failed to create temp dir: {e}");
                    return 1;
                }
            };
            let work_db = work_dir.path().join("work.db");
            if let Err(e) = fs::copy(&golden_path, &work_db) {
                eprintln!(
                    "error: failed to copy {} to {}: {e}",
                    golden_path.display(),
                    work_db.display()
                );
                return 1;
            }

            let oplog = match resolve_workload(workload_name, db_name, c) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("error: {e}");
                    return 1;
                }
            };

            let mode = if mvcc { "mvcc" } else { "single-writer" };
            eprintln!(
                "Running: engine=fsqlite mode={mode} db={db_name} workload={workload_name} \
                 concurrency={c} rep={rep}/{repeat}"
            );
            eprintln!("  golden: {}", golden_path.display());
            eprintln!("  working: {}", work_db.display());
            eprintln!("  ops: {}", oplog.records.len());

            let report = match run_oplog_fsqlite(&work_db, &oplog, &config) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("error: execution failed: {e}");
                    return 1;
                }
            };
            agg.record(&report);
            any_error |= report_has_failure(&report);

            let recorded_unix_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));

            let text = match serialize_run_output(
                recorded_unix_ms,
                &environment,
                EngineInfo {
                    name: "fsqlite".to_owned(),
                    sqlite_version: None,
                    fsqlite_git: None,
                },
                db_name,
                &golden_path,
                golden_sha256.clone(),
                workload_name,
                c,
                oplog.records.len(),
                report,
                pretty,
            ) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("error: failed to serialize report: {e}");
                    return 1;
                }
            };

            if let Some(path) = output_jsonl {
                if let Err(e) = append_jsonl_line(path, &text) {
                    eprintln!("error: failed to append JSONL output: {e}");
                    return 1;
                }
            }
            println!("{text}");
        }
        results.push(agg);
    }

    if results.len() > 1 || repeat > 1 {
        eprintln!("{}", format_scaling_summary("fsqlite", repeat, &results));
    }

    i32::from(any_error)
}

fn append_jsonl_line(path: &Path, line: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")?;
    Ok(())
}

#[derive(Debug, Clone)]
struct RunAgg {
    concurrency: u16,
    wall_time_ms: Vec<u64>,
    ops_per_sec: Vec<f64>,
    retries: Vec<u64>,
    aborts: Vec<u64>,
}

impl RunAgg {
    fn new(concurrency: u16) -> Self {
        Self {
            concurrency,
            wall_time_ms: Vec::new(),
            ops_per_sec: Vec::new(),
            retries: Vec::new(),
            aborts: Vec::new(),
        }
    }

    fn record(&mut self, report: &fsqlite_e2e::report::EngineRunReport) {
        self.wall_time_ms.push(report.wall_time_ms);
        self.ops_per_sec.push(report.ops_per_sec);
        self.retries.push(report.retries);
        self.aborts.push(report.aborts);
    }
}

fn format_scaling_summary(engine: &str, repeat: usize, results: &[RunAgg]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "\n{}", "-".repeat(72));
    let _ = writeln!(out, "  Scaling summary: engine={engine} repeat={repeat}");
    let _ = writeln!(out, "{}", "-".repeat(72));
    let _ = writeln!(
        out,
        "  {:>10} {:>12} {:>12} {:>10} {:>10}",
        "Conc", "p50 ops/s", "p95 ops/s", "p50 ms", "p50 retries"
    );
    let _ = writeln!(out, "  {:-<72}", "");

    for r in results {
        let p50_ops = percentile_f64(&r.ops_per_sec, 50);
        let p95_ops = percentile_f64(&r.ops_per_sec, 95);
        let p50_ms = percentile_u64(&r.wall_time_ms, 50);
        let p50_retries = percentile_u64(&r.retries, 50);
        let _ = writeln!(
            out,
            "  {:>10} {:>12.1} {:>12.1} {:>10} {:>10}",
            r.concurrency, p50_ops, p95_ops, p50_ms, p50_retries
        );
    }

    let _ = writeln!(out, "{}", "-".repeat(72));
    out
}

fn percentile_u64(data: &[u64], pct: u32) -> u64 {
    if data.is_empty() {
        return 0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = ((f64::from(pct) / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn percentile_f64(data: &[f64], pct: u32) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(f64::total_cmp);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let idx = ((f64::from(pct) / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn parse_u16_list(raw: &str) -> Result<Vec<u16>, String> {
    let mut out: Vec<u16> = Vec::new();
    for part in raw.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return Err(format!("invalid --concurrency list: `{raw}`"));
        }
        let Ok(n) = part.parse::<u16>() else {
            return Err(format!("invalid integer in --concurrency list: `{part}`"));
        };
        if n == 0 {
            return Err("concurrency values must be >= 1".to_owned());
        }
        out.push(n);
    }
    if out.is_empty() {
        Err(format!("invalid --concurrency list: `{raw}`"))
    } else {
        Ok(out)
    }
}

fn print_run_help() {
    let text = "\
realdb-e2e run — Execute an OpLog workload against an engine

USAGE:
    realdb-e2e run --engine <ENGINE> --db <DB_ID> --workload <NAME> [OPTIONS]

OPTIONS:
    --engine <ENGINE>       Engine to use: sqlite3 | fsqlite
    --db <DB_ID>            Database fixture identifier
    --workload <NAME>       OpLog preset name (e.g. commutative_inserts_disjoint_keys)
    --concurrency <N|LIST>  Number of workers, or comma-separated list (default: 1)
    --repeat <N>            Repetitions per concurrency (default: 1)
    --mvcc                  For fsqlite: force MVCC concurrent_mode on (default)
    --no-mvcc               For fsqlite: disable MVCC concurrent_mode
    --profile-only          Skip integrity_check, suppress environment probing,
                            and capture hot-path metrics for fsqlite runs
    --skip-integrity-check  Disable post-run PRAGMA integrity_check explicitly
    --skip-environment-metadata
                            Keep RunRecordV1 output but mark environment as suppressed
    --output-jsonl <PATH>   Append a single JSONL record to PATH
    --pretty                Pretty-print JSON to stdout (default: JSONL)
    -h, --help              Show this help message
";
    let _ = io::stdout().write_all(text.as_bytes());
}

// ── bench ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn cmd_bench(argv: &[String]) -> i32 {
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        print_bench_help();
        return 0;
    }

    let mut golden_dir = PathBuf::from(DEFAULT_GOLDEN_DIR);
    let mut fixture_ids: Vec<String> = Vec::new();
    let mut presets: Vec<String> = Vec::new();
    let mut concurrency: Vec<u16> = vec![1, 2, 4, 8];
    let mut engine = "both".to_owned(); // sqlite3|fsqlite|both
    let mut mvcc = true;
    let defaults = BenchmarkConfig::default();
    let mut warmup_iterations = defaults.warmup_iterations;
    let mut min_iterations = defaults.min_iterations;
    let mut measurement_time_secs = defaults.measurement_time_secs;
    let mut output_jsonl: Option<PathBuf> = None;
    let mut output_md: Option<PathBuf> = None;
    let mut pretty = false;
    let mut golden_dir_overridden = false;
    let mut db_overridden = false;
    let mut concurrency_overridden = false;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--golden-dir" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --golden-dir requires a directory path");
                    return 2;
                }
                golden_dir = PathBuf::from(&argv[i]);
                golden_dir_overridden = true;
            }
            "--db" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --db requires a fixture id or comma-separated list");
                    return 2;
                }
                db_overridden = true;
                for part in argv[i].split(',') {
                    let part = part.trim();
                    if !part.is_empty() {
                        fixture_ids.push(part.to_owned());
                    }
                }
            }
            "--preset" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --preset requires a preset name or comma-separated list");
                    return 2;
                }
                for part in argv[i].split(',') {
                    let part = part.trim();
                    if !part.is_empty() {
                        presets.push(part.to_owned());
                    }
                }
            }
            "--concurrency" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --concurrency requires an integer or comma-separated list");
                    return 2;
                }
                match parse_u16_list(&argv[i]) {
                    Ok(v) => {
                        concurrency = v;
                        concurrency_overridden = true;
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        return 2;
                    }
                }
            }
            "--engine" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --engine requires sqlite3|fsqlite|both");
                    return 2;
                }
                engine.clone_from(&argv[i]);
            }
            "--mvcc" => mvcc = true,
            "--no-mvcc" => mvcc = false,
            "--warmup" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --warmup requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u32>() else {
                    eprintln!("error: invalid integer for --warmup: `{}`", argv[i]);
                    return 2;
                };
                warmup_iterations = n;
            }
            "--repeat" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --repeat requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u32>() else {
                    eprintln!("error: invalid integer for --repeat: `{}`", argv[i]);
                    return 2;
                };
                if n == 0 {
                    eprintln!("error: --repeat must be >= 1");
                    return 2;
                }
                min_iterations = n;
                measurement_time_secs = 0;
            }
            "--min-iters" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --min-iters requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u32>() else {
                    eprintln!("error: invalid integer for --min-iters: `{}`", argv[i]);
                    return 2;
                };
                min_iterations = n;
            }
            "--time-secs" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --time-secs requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u64>() else {
                    eprintln!("error: invalid integer for --time-secs: `{}`", argv[i]);
                    return 2;
                };
                measurement_time_secs = n;
            }
            "--output-jsonl" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --output-jsonl requires a path");
                    return 2;
                }
                output_jsonl = Some(PathBuf::from(&argv[i]));
            }
            "--output" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --output requires a path");
                    return 2;
                }
                output_jsonl = Some(PathBuf::from(&argv[i]));
            }
            "--output-md" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --output-md requires a path");
                    return 2;
                }
                output_md = Some(PathBuf::from(&argv[i]));
            }
            "--pretty" => pretty = true,
            other => {
                eprintln!("error: unknown option `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    let needs_canonical_defaults = !golden_dir_overridden
        || !db_overridden
        || !concurrency_overridden
        || presets.is_empty()
        || presets.iter().any(|preset| preset == "all");

    let canonical_defaults = if needs_canonical_defaults {
        match std::env::current_dir() {
            Ok(cwd) => match find_bench_workspace_root(&cwd) {
                Some(workspace_root) => match canonical_bench_defaults(&workspace_root) {
                    Ok(defaults) => Some(defaults),
                    Err(e) => {
                        eprintln!("error: failed to load canonical Beads benchmark defaults: {e}");
                        return 1;
                    }
                },
                None => None,
            },
            Err(_) => None,
        }
    } else {
        None
    };

    if !golden_dir_overridden {
        if let Some(defaults) = &canonical_defaults {
            golden_dir.clone_from(&defaults.golden_dir);
        }
    }

    if !concurrency_overridden {
        if let Some(defaults) = &canonical_defaults {
            concurrency.clone_from(&defaults.concurrency);
        }
    }

    if presets.is_empty() || presets.iter().any(|p| p == "all") {
        presets = canonical_defaults
            .as_ref()
            .map_or_else(default_bench_presets, |defaults| defaults.presets.clone());
    }

    if fixture_ids.is_empty() {
        if !golden_dir_overridden {
            if let Some(defaults) = &canonical_defaults {
                fixture_ids.clone_from(&defaults.fixture_ids);
            }
        }
        if fixture_ids.is_empty() {
            match discover_golden_fixture_ids(&golden_dir) {
                Ok(ids) => fixture_ids = ids,
                Err(e) => {
                    eprintln!("error: {e}");
                    return 1;
                }
            }
        }
    }

    let bench_cfg = BenchmarkConfig {
        warmup_iterations,
        min_iterations,
        measurement_time_secs,
    };

    let cargo_profile = cargo_profile_name();
    let mut summaries: Vec<BenchmarkSummary> = Vec::new();
    let mut any_iteration_error = false;

    // If an output file is specified, truncate it up front so this run produces a
    // clean report artifact (rather than appending to an existing file).
    if let Some(ref path) = output_jsonl {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = fs::create_dir_all(parent) {
                    eprintln!(
                        "error: failed to create output directory {}: {e}",
                        parent.display()
                    );
                    return 1;
                }
            }
        }
        if let Err(e) = fs::File::create(path) {
            eprintln!(
                "error: failed to create output file {}: {e}",
                path.display()
            );
            return 1;
        }
    }

    let canonical_fixture_paths = canonical_defaults
        .as_ref()
        .filter(|defaults| golden_dir == defaults.golden_dir)
        .map_or_else(HashMap::new, |defaults| defaults.fixture_paths.clone());

    for fixture_id in &fixture_ids {
        let golden_path = if let Some(path) = canonical_fixture_paths.get(fixture_id) {
            path.clone()
        } else {
            match resolve_golden_db_in(&golden_dir, fixture_id) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error: {e}");
                    return 1;
                }
            }
        };

        for preset in &presets {
            for &c in &concurrency {
                let engines: Vec<(&str, bool)> = match engine.as_str() {
                    "sqlite3" => vec![("sqlite3", false)],
                    "fsqlite" => vec![("fsqlite", mvcc)],
                    "both" => vec![("sqlite3", false), ("fsqlite", mvcc)],
                    other => {
                        eprintln!(
                            "error: unknown --engine `{other}` (expected sqlite3|fsqlite|both)"
                        );
                        return 2;
                    }
                };

                for (engine_name, fsqlite_mvcc) in engines {
                    let engine_label = match (engine_name, fsqlite_mvcc) {
                        ("fsqlite", true) => "fsqlite_mvcc",
                        ("fsqlite", false) => "fsqlite_single_writer",
                        _ => engine_name,
                    };

                    let meta = BenchmarkMeta {
                        engine: engine_label.to_owned(),
                        workload: preset.to_owned(),
                        fixture_id: fixture_id.to_owned(),
                        concurrency: c,
                        cargo_profile: cargo_profile.to_owned(),
                    };

                    let sqlite_cfg = SqliteExecConfig {
                        run_integrity_check: false,
                        ..SqliteExecConfig::default()
                    };
                    let fsqlite_cfg = FsqliteExecConfig {
                        concurrent_mode: fsqlite_mvcc,
                        run_integrity_check: false,
                        ..FsqliteExecConfig::default()
                    };

                    let summary = run_benchmark(&bench_cfg, &meta, |global_idx| {
                        let _ = global_idx; // currently unused, but kept for future run-id tagging.
                        let td = tempfile::tempdir()
                            .map_err(|e| format!("failed to create temp dir: {e}"))?;
                        let work_db = td.path().join("work.db");
                        copy_db_with_sidecars(&golden_path, &work_db)?;

                        let oplog = resolve_workload(preset, fixture_id, c)?;

                        if engine_name == "sqlite3" {
                            run_oplog_sqlite(&work_db, &oplog, &sqlite_cfg)
                                .map_err(|e| format!("{e}"))
                        } else {
                            run_oplog_fsqlite(&work_db, &oplog, &fsqlite_cfg)
                                .map_err(|e| format!("{e}"))
                        }
                    });

                    any_iteration_error |= summary.iterations.iter().any(|it| it.error.is_some());

                    let line = if pretty {
                        summary
                            .to_pretty_json()
                            .map_err(|e| format!("serialize benchmark: {e}"))
                    } else {
                        summary
                            .to_jsonl()
                            .map_err(|e| format!("serialize benchmark: {e}"))
                    };

                    let text = match line {
                        Ok(t) => t,
                        Err(e) => {
                            eprintln!("error: {e}");
                            return 1;
                        }
                    };

                    if let Some(ref path) = output_jsonl {
                        let compact = match summary.to_jsonl() {
                            Ok(t) => t,
                            Err(e) => {
                                eprintln!(
                                    "error: failed to serialize benchmark for JSONL output: {e}"
                                );
                                return 1;
                            }
                        };
                        if let Err(e) = append_jsonl_line(path, &compact) {
                            eprintln!("error: failed to append JSONL output: {e}");
                            return 1;
                        }
                    }

                    println!("{text}");
                    summaries.push(summary);
                }
            }
        }
    }

    if let Some(path) = output_md.as_deref() {
        let md = render_benchmark_summaries_markdown(&summaries);
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = fs::create_dir_all(parent) {
                    eprintln!(
                        "error: failed to create output directory {}: {e}",
                        parent.display()
                    );
                    return 1;
                }
            }
        }
        if let Err(e) = fs::write(path, md.as_bytes()) {
            eprintln!(
                "error: failed to write markdown report {}: {e}",
                path.display()
            );
            return 1;
        }
        eprintln!("Wrote markdown report: {}", path.display());
    }

    i32::from(any_iteration_error)
}

fn print_bench_help() {
    let text = "\
realdb-e2e bench — Run the comparative benchmark matrix

USAGE:
    realdb-e2e bench [OPTIONS]

OPTIONS:
    --golden-dir <DIR>      Golden directory (default: canonical Beads campaign working golden/,
                            else sample_sqlite_db_files/golden)
    --db <DB_ID>            Database fixture id, or comma-separated list
                            (default: canonical campaign fixtures, else all DBs in golden dir)
    --preset <NAME>         Workload preset, or comma-separated list
                            (default: canonical campaign workloads, else all wired presets)
    --concurrency <N|LIST>  Concurrency levels (default: canonical campaign matrix 1,4,8,
                            else 1,2,4,8)
    --engine <NAME>         sqlite3 | fsqlite | both
                            (default: both = sqlite3 + current fsqlite mode)
    --mvcc                  For fsqlite: force MVCC concurrent_mode on (default)
    --no-mvcc               For fsqlite: disable MVCC concurrent_mode (reports fsqlite_single_writer)
    --warmup <N>            Warmup iterations discarded (default: methodology default)
    --repeat <N>            Exact measurement iterations (sets --min-iters=N and --time-secs=0)
    --min-iters <N>         Minimum measurement iterations (default: methodology default)
    --time-secs <N>         Measurement time floor in seconds (default: methodology default)
    --output <PATH>         Alias for --output-jsonl
    --output-jsonl <PATH>   Append compact JSONL BenchmarkSummary records to PATH
    --output-md <PATH>      Write a Markdown report to PATH (rendered from summaries)
    --pretty                Pretty-print JSON to stdout (default: JSONL)
    -h, --help              Show this help message
";
    let _ = io::stdout().write_all(text.as_bytes());
}

fn cmd_hot_profile(argv: &[String]) -> i32 {
    if argv.iter().any(|arg| arg == "-h" || arg == "--help") {
        print_hot_profile_help();
        return 0;
    }

    let mut golden_dir = PathBuf::from(DEFAULT_GOLDEN_DIR);
    let mut working_base = PathBuf::from(DEFAULT_WORKING_DIR);
    let mut db: Option<String> = None;
    let mut workload = "mixed_read_write".to_owned();
    let mut concurrency: u16 = 4;
    let mut seed: u64 = 42;
    let mut scale: u32 = 50;
    let mut output_dir: Option<PathBuf> = None;
    let mut mvcc = true;
    let mut run_integrity_check = false;
    let mut pretty = false;
    let mut emit_inline_bundle = false;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--golden-dir" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --golden-dir requires a directory path");
                    return 2;
                }
                golden_dir = PathBuf::from(&argv[i]);
            }
            "--working-base" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --working-base requires a directory path");
                    return 2;
                }
                working_base = PathBuf::from(&argv[i]);
            }
            "--db" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --db requires a fixture id");
                    return 2;
                }
                db = Some(argv[i].clone());
            }
            "--workload" | "--preset" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --workload requires a preset name");
                    return 2;
                }
                workload.clone_from(&argv[i]);
            }
            "--concurrency" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --concurrency requires an integer");
                    return 2;
                }
                let Ok(value) = argv[i].parse::<u16>() else {
                    eprintln!("error: invalid integer for --concurrency: `{}`", argv[i]);
                    return 2;
                };
                concurrency = value;
            }
            "--seed" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --seed requires an integer");
                    return 2;
                }
                let Ok(value) = argv[i].parse::<u64>() else {
                    eprintln!("error: invalid integer for --seed: `{}`", argv[i]);
                    return 2;
                };
                seed = value;
            }
            "--scale" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --scale requires an integer");
                    return 2;
                }
                let Ok(value) = argv[i].parse::<u32>() else {
                    eprintln!("error: invalid integer for --scale: `{}`", argv[i]);
                    return 2;
                };
                scale = value;
            }
            "--output-dir" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --output-dir requires a directory path");
                    return 2;
                }
                output_dir = Some(PathBuf::from(&argv[i]));
            }
            "--mvcc" => mvcc = true,
            "--no-mvcc" => mvcc = false,
            "--integrity-check" => run_integrity_check = true,
            "--pretty" => pretty = true,
            "--emit-inline-bundle" => emit_inline_bundle = true,
            other => {
                eprintln!("error: unknown option `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    let Some(db) = db else {
        eprintln!("error: --db is required");
        return 2;
    };
    if concurrency == 0 {
        eprintln!("error: --concurrency must be >= 1");
        return 2;
    }

    golden_dir = resolve_path_from_current_dir(golden_dir);
    working_base = resolve_path_from_current_dir(working_base);

    let workspace_config = WorkspaceConfig {
        golden_dir: golden_dir.clone(),
        working_base: working_base.clone(),
    };
    let label = format!("hot_profile_{workload}_{db}_c{concurrency}_s{seed}");
    let workspace = match create_workspace_with_label(&workspace_config, &[db.as_str()], &label) {
        Ok(workspace) => workspace,
        Err(error) => {
            eprintln!("error: {error}");
            return 1;
        }
    };
    let Some(profile_db) = workspace.databases.first() else {
        eprintln!("error: fixture workspace did not contain `{db}`");
        return 1;
    };

    let output_dir = resolve_path_from_current_dir(output_dir.unwrap_or_else(|| {
        PathBuf::from("artifacts")
            .join("bd-db300.4.1")
            .join(&workload)
            .join(format!("{db}_c{concurrency}_s{seed}"))
    }));
    let replay_command = HotProfileReplayCommand {
        db: &db,
        workload: &workload,
        golden_dir: &golden_dir,
        working_base: &working_base,
        concurrency,
        seed,
        scale,
        output_dir: &output_dir,
        mvcc,
        run_integrity_check,
    };
    let config = FsqliteHotPathProfileConfig {
        workload: workload.clone(),
        seed,
        scale,
        concurrency,
        exec_config: FsqliteExecConfig {
            concurrent_mode: mvcc,
            run_integrity_check,
            ..FsqliteExecConfig::default()
        },
        replay_command: format_hot_profile_replay_command(&replay_command),
        golden_dir: Some(golden_dir.display().to_string()),
        working_base: Some(working_base.display().to_string()),
        bead_id: hot_path_override_env(HOT_PATH_BEAD_ID_ENV),
        scenario_prefix: None,
    };

    let report = match profile_fsqlite_hot_path(&profile_db.db_path, &db, &config) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("error: {error}");
            return 1;
        }
    };
    let workspace_root = resolve_hot_path_workspace_root(&output_dir, &golden_dir, &working_base);
    let microarchitectural_context =
        resolve_hot_path_microarchitectural_context(workspace_root.as_deref(), &report);
    let command_pack = build_hot_path_command_pack(&report, &replay_command);
    let counter_capture_summary = build_hot_path_counter_capture_summary(&command_pack);
    let provenance_inputs = resolve_hot_path_artifact_provenance_inputs(
        &command_pack,
        replay_command.output_dir,
        replay_command.golden_dir,
        replay_command.working_base,
    );
    let provenance = build_hot_path_artifact_provenance(
        &report,
        &command_pack,
        counter_capture_summary.as_ref(),
        provenance_inputs,
    );
    let manifest = match write_hot_path_profile_artifacts(
        &report,
        &output_dir,
        counter_capture_summary.clone(),
        Some(provenance),
        Some(microarchitectural_context.clone()),
    ) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!("error: failed to write hot-path artifacts: {error}");
            return 1;
        }
    };
    let command_pack_file = match write_hot_path_command_pack(&output_dir, &command_pack) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("error: failed to write hot-path command pack: {error}");
            return 1;
        }
    };
    if let Err(error) = finalize_hot_path_manifest(
        &output_dir,
        manifest,
        counter_capture_summary,
        vec![command_pack_file],
    ) {
        eprintln!("error: failed to finalize hot-path manifest: {error}");
        return 1;
    }
    let disk_manifest = match read_hot_path_disk_manifest(&output_dir) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!("error: failed to read hot-path manifest: {error}");
            return 1;
        }
    };

    if emit_inline_bundle {
        match serialize_hot_path_inline_bundle(
            &report,
            &disk_manifest,
            &command_pack,
            Some(&microarchitectural_context),
        ) {
            Ok(json) => eprintln!("{HOT_PATH_INLINE_BUNDLE_PREFIX}{json}"),
            Err(error) => {
                eprintln!("error: failed to serialize hot-path inline bundle: {error}");
                return 1;
            }
        }
    }

    if pretty {
        match serde_json::to_string_pretty(&report) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("error: failed to serialize hot-path report: {error}");
                return 1;
            }
        }
    } else {
        match serde_json::to_string(&disk_manifest) {
            Ok(json) => println!("{json}"),
            Err(error) => {
                eprintln!("error: failed to serialize hot-path manifest: {error}");
                return 1;
            }
        }
    }

    eprintln!("Wrote hot-path artifacts: {}", output_dir.display());
    0
}

fn print_hot_profile_help() {
    let text = "\
realdb-e2e hot-profile — Capture hot-path evidence for a benchmark preset

USAGE:
    realdb-e2e hot-profile [OPTIONS]

OPTIONS:
    --golden-dir <DIR>      Golden directory (default: sample_sqlite_db_files/golden)
    --working-base <DIR>    Working-copy directory (default: sample_sqlite_db_files/working)
    --db <DB_ID>            Database fixture id (required)
    --workload <NAME>       Workload preset to profile (default: mixed_read_write)
    --preset <NAME>         Alias for --workload
    --concurrency <N>       Worker count for the selected preset (default: 4)
    --seed <N>              Deterministic workload seed (default: 42)
    --scale <N>             Workload scale passed to the preset generator (default: 50)
    --output-dir <DIR>      Artifact output directory
    --mvcc                  Force concurrent mode on (default)
    --no-mvcc               Disable concurrent mode for forced serialized comparison
    --integrity-check       Run post-run integrity check (default: off for lower profiling noise)
    --emit-inline-bundle    Print a tagged single-line artifact bundle for remote wrappers
    --pretty                Print the full report JSON instead of the manifest JSON
    -h, --help              Show this help message
";
    let _ = io::stdout().write_all(text.as_bytes());
}

fn serialize_hot_path_inline_bundle(
    report: &HotPathProfileReport,
    manifest: &HotPathArtifactManifest,
    command_pack: &HotPathEvidenceCommandPack,
    microarchitectural_context: Option<&HotPathMicroarchitecturalContext>,
) -> Result<String, serde_json::Error> {
    let opcode_profile = build_hot_path_opcode_profile(report);
    let subsystem_profile = build_hot_path_subsystem_profile(report);
    let actionable_ranking = build_hot_path_actionable_ranking(
        report,
        manifest.counter_capture_summary.as_ref(),
        microarchitectural_context,
    );
    let summary_markdown = render_hot_path_profile_markdown(report);
    serde_json::to_string(&serde_json::json!({
        "schema_version": HOT_PATH_INLINE_BUNDLE_SCHEMA_V1,
        "profile": report,
        "opcode_profile": opcode_profile,
        "subsystem_profile": subsystem_profile,
        "actionable_ranking": actionable_ranking,
        "summary_markdown": summary_markdown,
        "manifest": manifest,
        "command_pack": command_pack,
    }))
}

// ── corrupt ─────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
fn cmd_corrupt(argv: &[String]) -> i32 {
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        print_corrupt_help();
        return 0;
    }

    let mut golden_dir = PathBuf::from(DEFAULT_GOLDEN_DIR);
    let mut working_base = PathBuf::from(DEFAULT_WORKING_DIR);

    let mut db: Option<String> = None;
    let mut strategy: Option<String> = None;
    let mut seed: u64 = 0;
    let mut count: usize = 1;
    let mut offset: Option<usize> = None;
    let mut length: Option<usize> = None;
    let mut page: Option<u32> = None;
    let mut json = false;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--golden-dir" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --golden-dir requires a directory path");
                    return 2;
                }
                golden_dir = PathBuf::from(&argv[i]);
            }
            "--working-base" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --working-base requires a directory path");
                    return 2;
                }
                working_base = PathBuf::from(&argv[i]);
            }
            "--db" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --db requires a fixture id");
                    return 2;
                }
                db = Some(argv[i].clone());
            }
            "--strategy" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --strategy requires bitflip|zero|page");
                    return 2;
                }
                strategy = Some(argv[i].clone());
            }
            "--seed" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --seed requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u64>() else {
                    eprintln!("error: invalid integer for --seed: `{}`", argv[i]);
                    return 2;
                };
                seed = n;
            }
            "--count" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --count requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<usize>() else {
                    eprintln!("error: invalid integer for --count: `{}`", argv[i]);
                    return 2;
                };
                count = n;
            }
            "--offset" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --offset requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<usize>() else {
                    eprintln!("error: invalid integer for --offset: `{}`", argv[i]);
                    return 2;
                };
                offset = Some(n);
            }
            "--length" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --length requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<usize>() else {
                    eprintln!("error: invalid integer for --length: `{}`", argv[i]);
                    return 2;
                };
                length = Some(n);
            }
            "--page" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --page requires an integer");
                    return 2;
                }
                let Ok(n) = argv[i].parse::<u32>() else {
                    eprintln!("error: invalid integer for --page: `{}`", argv[i]);
                    return 2;
                };
                page = Some(n);
            }
            "--json" => json = true,
            other => {
                eprintln!("error: unknown option `{other}`");
                return 2;
            }
        }
        i += 1;
    }

    let Some(db_id) = db.as_deref() else {
        eprintln!("error: --db is required");
        return 2;
    };
    let Some(strategy) = strategy.as_deref() else {
        eprintln!("error: --strategy is required");
        return 2;
    };

    let (scenario_id, strategy_desc, strat) = match strategy {
        "bitflip" => (
            format!("bitflip_count_{count}_seed_{seed}"),
            format!("bitflip(count={count}, seed={seed})"),
            CorruptionStrategy::RandomBitFlip { count },
        ),
        "zero" => {
            let Some(off) = offset else {
                eprintln!("error: zero strategy requires --offset");
                return 2;
            };
            let Some(len) = length else {
                eprintln!("error: zero strategy requires --length");
                return 2;
            };
            (
                format!("zero_off_{off}_len_{len}"),
                format!("zero(offset={off}, length={len})"),
                CorruptionStrategy::ZeroRange {
                    offset: off,
                    length: len,
                },
            )
        }
        "page" => {
            let Some(pg) = page else {
                eprintln!("error: page strategy requires --page");
                return 2;
            };
            (
                format!("page_pg_{pg}_seed_{seed}"),
                format!("page(page_number={pg}, seed={seed})"),
                CorruptionStrategy::PageCorrupt { page_number: pg },
            )
        }
        other => {
            eprintln!("error: unknown strategy `{other}` (expected bitflip|zero|page)");
            return 2;
        }
    };

    // Create a working workspace containing the selected golden DB.
    let ws_cfg = WorkspaceConfig {
        golden_dir,
        working_base,
    };

    let ws = match create_workspace_with_label(&ws_cfg, &[db_id], &scenario_id) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: failed to create workspace: {e}");
            return 1;
        }
    };
    let Some(db) = ws.databases.first() else {
        eprintln!("error: workspace contains no databases");
        return 1;
    };

    let work_db = db.db_path.clone();
    let before_bytes = match fs::read(&work_db) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read working db {}: {e}", work_db.display());
            return 1;
        }
    };
    let before = sha256_bytes(&before_bytes);
    let page_size = sqlite_page_size_or_default(&before_bytes);

    if let Err(e) = inject_corruption(&work_db, strat, seed) {
        eprintln!("error: corruption injection failed: {e}");
        return 1;
    }

    let after_bytes = match fs::read(&work_db) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read corrupted db {}: {e}", work_db.display());
            return 1;
        }
    };
    let after = sha256_bytes(&after_bytes);

    let modifications = diff_modified_ranges(&before_bytes, &after_bytes, page_size);
    let modified_bytes: u64 = modifications.iter().map(|m| m.length).sum();

    let report = CorruptReport {
        fixture_id: db_id.to_owned(),
        scenario_id,
        strategy: strategy_desc,
        workspace_dir: ws.run_dir.display().to_string(),
        db_path: work_db.display().to_string(),
        page_size,
        modified_bytes,
        modifications,
        sha256_before: before,
        sha256_after: after,
    };

    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{text}"),
            Err(e) => {
                eprintln!("error: failed to serialize report: {e}");
                return 1;
            }
        }
    } else {
        println!("Corruption injected:");
        println!("  fixture: {}", report.fixture_id);
        println!("  scenario_id: {}", report.scenario_id);
        println!("  strategy: {}", report.strategy);
        println!("  workspace: {}", report.workspace_dir);
        println!("  db: {}", report.db_path);
        println!("  page_size: {}", report.page_size);
        println!("  modified_bytes: {}", report.modified_bytes);
        println!("  modifications: {}", report.modifications.len());
        println!("  sha256(before): {}", report.sha256_before);
        println!("  sha256(after):  {}", report.sha256_after);
    }

    // Ensure the corruption actually changed bytes (sanity).
    i32::from(report.sha256_before == report.sha256_after)
}

fn print_corrupt_help() {
    let text = "\
realdb-e2e corrupt — Inject corruption into a working copy

USAGE:
    realdb-e2e corrupt --db <DB_ID> --strategy <STRATEGY> [OPTIONS]

STRATEGIES:
    bitflip             Flip random bits (--count N)
    zero                Zero out a byte range (--offset N --length N)
    page                Corrupt an entire page (--page N)

OPTIONS:
    --golden-dir <DIR>      Golden directory (default: sample_sqlite_db_files/golden)
    --working-base <DIR>    Base directory for working copies
                            (default: sample_sqlite_db_files/working)
    --db <DB_ID>            Database fixture to corrupt (copied from golden/)
    --strategy <STRATEGY>   Corruption strategy (bitflip|zero|page)
    --seed <N>              RNG seed for deterministic corruption (default: 0)
    --count <N>             Number of bits to flip (bitflip strategy)
    --offset <N>            Byte offset (zero strategy)
    --length <N>            Byte count (zero strategy)
    --page <N>              Page number to corrupt (page strategy)
    --json                  Output a structured JSON report
    -h, --help              Show this help message
";
    let _ = io::stdout().write_all(text.as_bytes());
}

// ── Types: corpus import metadata + corrupt report ─────────────────────

#[derive(Debug, Serialize)]
struct CorruptReport {
    fixture_id: String,
    scenario_id: String,
    strategy: String,
    workspace_dir: String,
    db_path: String,
    page_size: u32,
    modified_bytes: u64,
    modifications: Vec<CorruptModification>,
    sha256_before: String,
    sha256_after: String,
}

#[derive(Debug, Serialize)]
struct CorruptModification {
    offset: u64,
    length: u64,
    page_first: u32,
    page_last: u32,
    sha256_before: String,
    sha256_after: Option<String>,
}

// Fixture metadata is emitted using `fsqlite_e2e::fixture_metadata::FixtureMetadataV1`.

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn profile_database_for_metadata(
    db_path: &Path,
    fixture_id: &str,
    source_path: Option<&Path>,
    golden_filename: &str,
    sha256_golden: &str,
    tag: Option<&str>,
    discovery_tags: &[String],
    sidecars_present: &[String],
    safety: FixtureSafetyV1,
) -> Result<FixtureMetadataV1, String> {
    let meta =
        fs::metadata(db_path).map_err(|e| format!("cannot stat {}: {e}", db_path.display()))?;

    let conn = Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| format!("cannot open {}: {e}", db_path.display()))?;

    let encoding: String = conn
        .query_row("PRAGMA encoding", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA encoding: {e}"))?;
    let page_size: u32 = conn
        .query_row("PRAGMA page_size", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA page_size: {e}"))?;
    let page_count: u32 = conn
        .query_row("PRAGMA page_count", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA page_count: {e}"))?;
    let freelist_count: u32 = conn
        .query_row("PRAGMA freelist_count", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA freelist_count: {e}"))?;
    let schema_version: u32 = conn
        .query_row("PRAGMA schema_version", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA schema_version: {e}"))?;
    let journal_mode: String = conn
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA journal_mode: {e}"))?;
    let user_version: u32 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA user_version: {e}"))?;
    let application_id: u32 = conn
        .query_row("PRAGMA application_id", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA application_id: {e}"))?;
    let auto_vacuum: u32 = conn
        .query_row("PRAGMA auto_vacuum", [], |r| r.get(0))
        .map_err(|e| format!("PRAGMA auto_vacuum: {e}"))?;

    let tables = collect_tables(&conn)?;
    let indices = collect_names(&conn, "index")?;
    let triggers = collect_names(&conn, "trigger")?;
    let views = collect_names(&conn, "view")?;

    let has_fts = sqlite_master_sql_contains(&conn, "using fts")?;
    let has_rtree = sqlite_master_sql_contains(&conn, "using rtree")?;
    let has_foreign_keys = has_foreign_keys(&conn, &tables)?;

    let has_wal_sidecars_observed = sidecars_present.iter().any(|s| s == "-wal" || s == "-shm");

    let features = FixtureFeaturesV1 {
        has_wal_sidecars_observed,
        has_fts,
        has_rtree,
        has_triggers: !triggers.is_empty(),
        has_views: !views.is_empty(),
        has_foreign_keys,
    };

    let mut tags: Vec<String> = Vec::new();
    if let Some(t) = tag {
        tags.push(t.to_owned());
    }
    tags.extend(discovery_tags.iter().cloned());
    tags.push(size_bucket_tag(meta.len()).to_owned());
    tags.push(format!("page-size-{page_size}"));
    if journal_mode.eq_ignore_ascii_case("wal") {
        tags.push("wal".to_owned());
    }
    if features.has_fts {
        tags.push("fts".to_owned());
    }
    if features.has_rtree {
        tags.push("rtree".to_owned());
    }
    if indices.len() > 20 {
        tags.push("many-indexes".to_owned());
    }
    if tables.len() > 20 {
        tags.push("many-tables".to_owned());
    }
    if tags.is_empty() {
        tags.push("misc".to_owned());
    }

    Ok(FixtureMetadataV1 {
        schema_version: FIXTURE_METADATA_SCHEMA_VERSION_V1,
        db_id: fixture_id.to_owned(),
        source_path: source_path.map(|p| p.to_string_lossy().into_owned()),
        golden_filename: golden_filename.to_owned(),
        sha256_golden: sha256_golden.to_owned(),
        size_bytes: meta.len(),
        sidecars_present: sidecars_present.to_vec(),
        sqlite_meta: SqliteMetaV1 {
            page_size,
            page_count,
            freelist_count,
            schema_version,
            encoding,
            user_version,
            application_id,
            journal_mode,
            auto_vacuum,
        },
        features,
        tags: normalize_tags(tags),
        safety,
        tables,
        indices,
        triggers,
        views,
    })
}

fn sqlite_master_sql_contains(conn: &Connection, needle_lower: &str) -> Result<bool, String> {
    let pattern = format!("%{needle_lower}%");
    let mut stmt = conn
        .prepare(
            "SELECT 1 FROM sqlite_master \
             WHERE sql IS NOT NULL AND lower(sql) LIKE ?1 \
             LIMIT 1",
        )
        .map_err(|e| format!("sqlite_master sql prepare: {e}"))?;
    let mut rows = stmt
        .query([pattern])
        .map_err(|e| format!("sqlite_master sql query: {e}"))?;
    Ok(rows
        .next()
        .map_err(|e| format!("sqlite_master sql next: {e}"))?
        .is_some())
}

fn has_foreign_keys(conn: &Connection, tables: &[TableProfileV1]) -> Result<bool, String> {
    for t in tables {
        let sql = format!("PRAGMA foreign_key_list({})", quote_ident(&t.name));
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| format!("PRAGMA foreign_key_list({}) prepare: {e}", t.name))?;
        let mut rows = stmt
            .query([])
            .map_err(|e| format!("PRAGMA foreign_key_list({}) query: {e}", t.name))?;
        if rows
            .next()
            .map_err(|e| format!("PRAGMA foreign_key_list({}) next: {e}", t.name))?
            .is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn collect_names(conn: &Connection, ty: &str) -> Result<Vec<String>, String> {
    let sql = format!(
        "SELECT name FROM sqlite_master \
         WHERE type='{ty}' AND name NOT LIKE 'sqlite_%' \
         ORDER BY name"
    );
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("sqlite_master({ty}) prepare: {e}"))?;
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(|e| format!("sqlite_master({ty}) query: {e}"))?;
    Ok(rows.flatten().collect())
}

fn collect_tables(conn: &Connection) -> Result<Vec<TableProfileV1>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT name FROM sqlite_master \
             WHERE type='table' AND name NOT LIKE 'sqlite_%' \
             ORDER BY name",
        )
        .map_err(|e| format!("sqlite_master(table) prepare: {e}"))?;
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(|e| format!("sqlite_master(table) query: {e}"))?;

    let mut out: Vec<TableProfileV1> = Vec::new();
    for row in rows {
        let Ok(table) = row else { continue };
        let cols = collect_table_columns(conn, &table)?;
        let row_count = count_rows(conn, &table)?;
        out.push(TableProfileV1 {
            name: table,
            row_count,
            columns: cols,
        });
    }
    Ok(out)
}

fn collect_table_columns(conn: &Connection, table: &str) -> Result<Vec<ColumnProfileV1>, String> {
    let sql = format!("PRAGMA table_info({})", quote_ident(table));
    let mut stmt = conn
        .prepare(&sql)
        .map_err(|e| format!("PRAGMA table_info({table}) prepare: {e}"))?;

    let mut cols = Vec::new();
    let mut rows = stmt
        .query([])
        .map_err(|e| format!("PRAGMA table_info({table}) query: {e}"))?;

    while let Some(r) = rows
        .next()
        .map_err(|e| format!("PRAGMA table_info({table}) next: {e}"))?
    {
        let name: String = r.get(1).map_err(|e| format!("col.name: {e}"))?;
        let col_type: String = r.get(2).map_err(|e| format!("col.type: {e}"))?;
        let not_null_raw: i32 = r.get(3).map_err(|e| format!("col.not_null flag: {e}"))?;
        let not_null: bool = not_null_raw != 0;
        let default_value: Option<String> =
            r.get(4).map_err(|e| format!("col.default_value: {e}"))?;
        let primary_key_raw: i32 = r.get(5).map_err(|e| format!("col.pk flag: {e}"))?;
        let primary_key: bool = primary_key_raw != 0;
        cols.push(ColumnProfileV1 {
            name,
            col_type,
            primary_key,
            not_null,
            default_value,
        });
    }

    Ok(cols)
}

fn count_rows(conn: &Connection, table: &str) -> Result<u64, String> {
    let sql = format!("SELECT count(*) FROM {}", quote_ident(table));
    conn.query_row(&sql, [], |r| r.get::<_, u64>(0))
        .map_err(|e| format!("count_rows({table}): {e}"))
}

fn quote_ident(name: &str) -> String {
    let escaped = name.replace('"', "\"\"");
    format!("\"{escaped}\"")
}

fn cargo_profile_name() -> &'static str {
    if cfg!(debug_assertions) {
        "dev"
    } else {
        "release"
    }
}

fn sanitize_db_id(raw: &str) -> Result<String, &'static str> {
    let s = raw.trim();
    if s.is_empty() {
        return Err("empty");
    }
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        let c = ch.to_ascii_lowercase();
        if c.is_ascii_alphanumeric() {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    // Trim underscores.
    let trimmed = out.trim_matches('_').to_owned();
    if trimmed.is_empty() {
        Err("no usable characters after sanitization")
    } else {
        Ok(trimmed)
    }
}

fn mib_to_bytes(mib: u64) -> Result<u64, String> {
    if mib == 0 {
        return Ok(u64::MAX);
    }
    mib.checked_mul(1024 * 1024)
        .ok_or_else(|| format!("--max-file-size-mib value {mib} is too large"))
}

fn detect_sidecars(db_path: &Path) -> Vec<String> {
    const SIDECARS: [&str; 3] = ["-wal", "-shm", "-journal"];
    let mut present = Vec::new();

    for suffix in SIDECARS {
        let mut os = db_path.as_os_str().to_os_string();
        os.push(suffix);
        let path = PathBuf::from(os);
        if path.exists() {
            present.push(suffix.to_owned());
        }
    }

    present
}

fn is_sqlite_sidecar_filename(filename: &str) -> bool {
    filename.ends_with("-wal") || filename.ends_with("-shm") || filename.ends_with("-journal")
}

fn resolve_source_db(
    db_arg: &str,
    root: &Path,
    max_depth: usize,
    max_file_size: u64,
) -> Result<(PathBuf, Vec<String>, bool), String> {
    let as_path = PathBuf::from(db_arg);
    if as_path.exists() {
        let header_ok =
            sqlite_magic_header_ok(&as_path).map_err(|e| format!("header check failed: {e}"))?;
        return Ok((as_path, Vec::new(), header_ok));
    }

    let config = fsqlite_harness::fixture_discovery::DiscoveryConfig {
        roots: vec![root.to_path_buf()],
        max_depth,
        max_file_size,
        ..fsqlite_harness::fixture_discovery::DiscoveryConfig::default()
    };

    let candidates = fsqlite_harness::fixture_discovery::discover_sqlite_files(&config)
        .map_err(|e| format!("discovery scan failed: {e}"))?;

    let mut matches = Vec::new();
    for c in candidates {
        let filename = c.path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let stem = c.path.file_stem().and_then(|n| n.to_str()).unwrap_or("");

        if filename == db_arg || stem == db_arg {
            matches.push(c);
        }
    }

    if matches.is_empty() {
        return Err(format!(
            "cannot resolve `{db_arg}`. Provide a literal path, or run `realdb-e2e corpus scan` and pass an exact filename/stem."
        ));
    }
    if matches.len() > 1 {
        eprintln!("error: `{db_arg}` is ambiguous; matches:");
        for m in &matches {
            eprintln!("  {m}");
        }
        return Err("ambiguous discovery name".to_owned());
    }

    let chosen = matches.remove(0);
    Ok((chosen.path, chosen.tags, chosen.header_ok))
}

fn sqlite_magic_header_ok(path: &Path) -> io::Result<bool> {
    use std::io::Read as _;
    const MAGIC: &[u8; 16] = b"SQLite format 3\0";
    let mut f = std::fs::File::open(path)?;
    let mut buf = [0u8; 16];
    if f.read_exact(&mut buf).is_err() {
        return Ok(false);
    }
    Ok(&buf == MAGIC)
}

fn backup_sqlite_file(src: &Path, dst: &Path) -> Result<(), String> {
    let src_conn = Connection::open_with_flags(src, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("cannot open source DB {} (read-only): {e}", src.display()))?;

    // Uses SQLite backup API (same semantics as `sqlite3 "$SRC" ".backup '$DST'"`).
    src_conn
        .backup(DatabaseName::Main, dst, None)
        .map_err(|e| format!("sqlite backup API failed: {e}"))
}

fn sqlite_integrity_check(db: &Path) -> Result<(), String> {
    let conn = Connection::open_with_flags(db, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(|e| format!("cannot open {} for integrity_check: {e}", db.display()))?;

    let mut stmt = conn
        .prepare("PRAGMA integrity_check;")
        .map_err(|e| format!("prepare integrity_check: {e}"))?;

    let mut rows = stmt
        .query([])
        .map_err(|e| format!("query integrity_check: {e}"))?;

    let mut lines: Vec<String> = Vec::new();
    while let Some(row) = rows
        .next()
        .map_err(|e| format!("read integrity_check row: {e}"))?
    {
        let msg: String = row.get(0).map_err(|e| format!("read row text: {e}"))?;
        lines.push(msg);
    }

    if lines.len() == 1 && lines[0].trim() == "ok" {
        return Ok(());
    }

    let mut out = String::new();
    for l in &lines {
        let _ = writeln!(out, "{l}");
    }
    Err(format!(
        "integrity_check reported {} line(s):\n{out}",
        lines.len()
    ))
}

fn copy_sidecars(src_db: &Path, dest_db: &Path) -> Result<Vec<PathBuf>, String> {
    const SIDECARS: [&str; 3] = ["-wal", "-shm", "-journal"];
    let mut copied = Vec::new();

    for suffix in SIDECARS {
        let mut src_os = src_db.as_os_str().to_os_string();
        src_os.push(suffix);
        let src = PathBuf::from(src_os);
        if !src.exists() {
            continue;
        }

        let mut dest_os = dest_db.as_os_str().to_os_string();
        dest_os.push(suffix);
        let dest = PathBuf::from(dest_os);

        if dest.exists() {
            // Idempotent: skip if already present.
            copied.push(dest);
            continue;
        }

        fs::copy(&src, &dest).map_err(|e| {
            format!(
                "failed to copy sidecar {} -> {}: {e}",
                src.display(),
                dest.display()
            )
        })?;
        make_path_writable(&dest)?;
        copied.push(dest);
    }

    Ok(copied)
}

fn copy_db_with_sidecars(src_db: &Path, dest_db: &Path) -> Result<(), String> {
    fs::copy(src_db, dest_db).map_err(|e| {
        format!(
            "failed to copy {} -> {}: {e}",
            src_db.display(),
            dest_db.display()
        )
    })?;
    make_path_writable(dest_db)?;
    let _ = copy_sidecars(src_db, dest_db)?;
    Ok(())
}

fn make_path_writable(path: &Path) -> Result<(), String> {
    let mut permissions = fs::metadata(path)
        .map_err(|e| format!("failed to read permissions for {}: {e}", path.display()))?
        .permissions();
    #[cfg(unix)]
    permissions.set_mode(permissions.mode() | 0o200);
    #[cfg(not(unix))]
    permissions.set_readonly(false);
    fs::set_permissions(path, permissions)
        .map_err(|e| format!("failed to mark {} writable: {e}", path.display()))
}

fn upsert_checksum(
    checksums_path: &Path,
    golden_db: &Path,
    sha256_hex: &str,
) -> Result<(), String> {
    let filename = golden_db
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or("golden db has no filename")?
        .to_owned();

    let mut lines: Vec<(String, String)> = Vec::new();
    if checksums_path.exists() {
        let contents = fs::read_to_string(checksums_path)
            .map_err(|e| format!("cannot read {}: {e}", checksums_path.display()))?;
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((hex, name)) = line.split_once("  ") else {
                continue;
            };
            lines.push((name.trim().to_owned(), hex.trim().to_owned()));
        }
    }

    for (name, hex) in &lines {
        if name == &filename {
            if hex == sha256_hex {
                // Idempotent: already recorded.
                return Ok(());
            }
            return Err(format!(
                "{} already contains an entry for {filename} with a different sha256.\n\
Refusing to overwrite provenance. Golden files are immutable; ingest under a new --id instead.\n\
existing: {hex}\n\
current:  {sha256_hex}",
                checksums_path.display()
            ));
        }
    }

    lines.push((filename, sha256_hex.to_owned()));
    lines.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out = String::new();
    for (name, hex) in &lines {
        let _ = writeln!(out, "{hex}  {name}");
    }
    fs::write(checksums_path, out.as_bytes())
        .map_err(|e| format!("cannot write {}: {e}", checksums_path.display()))?;

    Ok(())
}

fn discover_golden_fixture_ids(golden_dir: &Path) -> Result<Vec<String>, String> {
    let mut ids = Vec::new();
    let entries = fs::read_dir(golden_dir)
        .map_err(|e| format!("cannot read golden dir {}: {e}", golden_dir.display()))?;
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("db") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if !stem.is_empty() {
                    ids.push(stem.to_owned());
                }
            }
        }
    }
    ids.sort();
    Ok(ids)
}

fn resolve_golden_db_in(golden_dir: &Path, db_name: &str) -> Result<PathBuf, String> {
    // If it looks like a path and exists, use it directly.
    let as_path = PathBuf::from(db_name);
    if as_path.exists() {
        return Ok(as_path);
    }

    // Try golden directory with .db extension.
    let golden = golden_dir.join(format!("{db_name}.db"));
    if golden.exists() {
        return Ok(golden);
    }

    // Try golden directory without adding .db (user may have included it).
    let golden_bare = golden_dir.join(db_name);
    if golden_bare.exists() {
        return Ok(golden_bare);
    }

    Err(format!(
        "cannot find database `{db_name}` (tried {}, {}, and literal path)",
        golden.display(),
        golden_bare.display(),
    ))
}

#[cfg(unix)]
fn set_read_only(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .permissions();
    perms.set_mode(0o444);
    fs::set_permissions(path, perms)
        .map_err(|e| format!("cannot chmod {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_read_only(_path: &Path) -> Result<(), String> {
    Ok(())
}

// ── compare ─────────────────────────────────────────────────────────────

fn print_compare_help() {
    let text = "\
realdb-e2e compare — Tiered comparison of two database files (bd-2als.3.2)

Compares two SQLite database files using a three-tier equivalence oracle:

  Tier 1 (canonical_sha256): VACUUM INTO + SHA-256 byte-for-byte identity.
  Tier 2 (logical):          Schema + row-level comparison with stable ordering.
  Tier 3 (data_complete):    Row counts + spot checks + integrity_check.

When a mismatch is detected, emits diagnostics: which tier failed, SHA-256
values, key PRAGMAs, schema diffs, and logical dump diffs.

USAGE:
    realdb-e2e compare --db-a <PATH> --db-b <PATH> [OPTIONS]

OPTIONS:
    --db-a <PATH>      Path to the first database file
    --db-b <PATH>      Path to the second database file
    --json             Output comparison report as JSON
    -h, --help         Show this help message

EXIT CODES:
    0   Match (databases are equivalent at canonical or logical tier)
    1   Mismatch (databases differ)
    2   Error (insufficient data or I/O failure)
";
    let _ = io::stdout().write_all(text.as_bytes());
}

#[allow(clippy::too_many_lines)]
fn cmd_compare(argv: &[String]) -> i32 {
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        print_compare_help();
        return 0;
    }

    let mut db_a: Option<PathBuf> = None;
    let mut db_b: Option<PathBuf> = None;
    let mut json_output = false;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--db-a" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --db-a requires a path argument");
                    return 2;
                }
                db_a = Some(PathBuf::from(&argv[i]));
            }
            "--db-b" => {
                i += 1;
                if i >= argv.len() {
                    eprintln!("error: --db-b requires a path argument");
                    return 2;
                }
                db_b = Some(PathBuf::from(&argv[i]));
            }
            "--json" => {
                json_output = true;
            }
            other => {
                eprintln!("error: unknown option `{other}`");
                print_compare_help();
                return 2;
            }
        }
        i += 1;
    }

    let Some(path_a) = db_a else {
        eprintln!("error: --db-a is required");
        return 2;
    };
    let Some(path_b) = db_b else {
        eprintln!("error: --db-b is required");
        return 2;
    };

    if !path_a.exists() {
        eprintln!("error: database A not found: {}", path_a.display());
        return 2;
    }
    if !path_b.exists() {
        eprintln!("error: database B not found: {}", path_b.display());
        return 2;
    }

    let (report, diagnostic) = match verify_databases(&path_a, &path_b) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: comparison failed: {e}");
            return 2;
        }
    };

    if json_output {
        #[derive(Serialize)]
        struct CompareOutput<'a> {
            verdict: String,
            explanation: String,
            tiers: &'a fsqlite_e2e::report::EqualityTiersReport,
            diagnostic: Option<&'a fsqlite_e2e::golden::MismatchDiagnostic>,
        }

        let out = CompareOutput {
            verdict: format!("{:?}", report.verdict),
            explanation: report.explanation.clone(),
            tiers: &report.tiers,
            diagnostic: diagnostic.as_ref(),
        };
        match serde_json::to_string_pretty(&out) {
            Ok(json) => println!("{json}"),
            Err(e) => {
                eprintln!("error: failed to serialize report: {e}");
                return 2;
            }
        }
    } else {
        println!("Verdict: {:?}", report.verdict);
        println!("Explanation: {}", report.explanation);
        println!();
        println!("Tiers:");
        println!(
            "  raw_sha256_match:       {:?}",
            report.tiers.raw_sha256_match
        );
        println!(
            "  canonical_sha256_match: {:?}",
            report.tiers.canonical_sha256_match
        );
        println!("  logical_match:          {:?}", report.tiers.logical_match);

        if let Some(ref diag) = diagnostic {
            println!();
            print!("{}", format_mismatch_diagnostic(diag));
        }
    }

    match report.verdict {
        fsqlite_e2e::report::ComparisonVerdict::Match => 0,
        fsqlite_e2e::report::ComparisonVerdict::Mismatch => 1,
        fsqlite_e2e::report::ComparisonVerdict::Error => 2,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fsqlite_e2e::fixture_select::{
        BEADS_BENCHMARK_ARTIFACT_MANIFEST_SCHEMA_V1, BEADS_BENCHMARK_CAMPAIGN_SCHEMA_V1,
        BeadsBenchmarkCampaign, BeadsBenchmarkFixture, BeadsBenchmarkMatrixRow,
        BenchmarkArtifactContract, BenchmarkArtifactProvenanceCapture,
        BenchmarkArtifactRetentionClass, BenchmarkArtifactRetentionPolicy, BenchmarkMode,
        BuildProfile, ExpandedBenchmarkCell, HARDWARE_CLASS_LINUX_X86_64_ANY, HardwareClass,
        HardwareClassIdFields, HardwareCpuArchitecture, HardwareOsFamily, HardwareTopologyClass,
        PlacementAvailability, PlacementClaimContract, PlacementCpuAffinityPolicy,
        PlacementExecutionContract, PlacementFocusedRerunContract,
        PlacementFocusedRerunSelectorKind, PlacementHelperLanePolicy, PlacementMemoryPolicy,
        PlacementProfile, PlacementProfileKind, PlacementSmtPolicy,
        PlacementSuiteSelectionContract, PlacementSuiteSelectorKind, PlacementVariant,
        PlacementViolationDisposition, RetryPolicy, SeedPolicy, build_benchmark_artifact_manifest,
    };
    use fsqlite_e2e::perf_runner::{
        HOT_PATH_OPCODE_PROFILE_SCHEMA_V1, HOT_PATH_PROFILE_ACTIONABLE_RANKING_SCHEMA_V3,
        HOT_PATH_PROFILE_MANIFEST_SCHEMA_V1, HOT_PATH_PROFILE_SCHEMA_V1,
        HOT_PATH_SUBSYSTEM_PROFILE_SCHEMA_V1, HotPathAllocatorPressure, HotPathArtifactFile,
        HotPathArtifactManifest, HotPathMvccWriteProfile, HotPathOpcodeProfileEntry,
        HotPathPageDataMotionProfile, HotPathParserProfile, HotPathProfileReport,
        HotPathRankingEntry, HotPathRecordDecodeCallsiteBreakdown,
        HotPathRecordDecodeCallsiteCounters, HotPathRecordDecodeProfile,
        HotPathRowMaterializationProfile, HotPathTypeProfile, HotPathValueTypeProfile,
    };
    use fsqlite_e2e::report::{CorrectnessReport, EngineRunReport};
    use jsonschema::{Draft, options};
    use serde_json::Value;

    const HOT_PATH_MANIFEST_SCHEMA_PATH: &str =
        "sample_sqlite_db_files/manifests/hot_path_profile_manifest.v1.schema.json";
    const BENCHMARK_ARTIFACT_MANIFEST_SCHEMA_PATH: &str =
        "sample_sqlite_db_files/manifests/beads_benchmark_artifact_manifest.v1.schema.json";
    #[allow(clippy::needless_raw_string_hashes)]
    const HOT_PATH_MANIFEST_SCHEMA_RAW: &str = r###"{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://frankensqlite.dev/schemas/hot_path_profile_manifest.v1.schema.json","title":"FrankenSQLite Hot Path Profile Manifest v1","type":"object","additionalProperties":false,"required":["schema_version","bead_id","run_id","trace_id","scenario_id","fixture_id","workload","seed","scale","concurrency","concurrent_mode","run_integrity_check","replay_command","files"],"properties":{"schema_version":{"type":"string","const":"fsqlite-e2e.hot_path_profile_manifest.v1"},"bead_id":{"$ref":"#/$defs/id"},"run_id":{"$ref":"#/$defs/id"},"trace_id":{"$ref":"#/$defs/id"},"scenario_id":{"$ref":"#/$defs/id"},"fixture_id":{"$ref":"#/$defs/id"},"workload":{"$ref":"#/$defs/non_empty_string"},"seed":{"type":"integer","minimum":0},"scale":{"type":"integer","minimum":1},"concurrency":{"type":"integer","minimum":1},"concurrent_mode":{"type":"boolean"},"run_integrity_check":{"type":"boolean"},"golden_dir":{"$ref":"#/$defs/nullable_non_empty_string"},"working_base":{"$ref":"#/$defs/nullable_non_empty_string"},"replay_command":{"$ref":"#/$defs/non_empty_string"},"counter_capture_summary":{"$ref":"#/$defs/counter_capture_summary"},"provenance":{"$ref":"#/$defs/provenance"},"files":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/artifact_file"}}},"$defs":{"id":{"type":"string","pattern":"^[A-Za-z0-9][A-Za-z0-9._:-]*$"},"non_empty_string":{"type":"string","minLength":1},"nullable_non_empty_string":{"type":["string","null"],"minLength":1},"string_list":{"type":"array","items":{"$ref":"#/$defs/non_empty_string"}},"counter_capture_summary":{"type":"object","additionalProperties":false,"required":["host_capability_sensitive_captures","topology_sensitive_captures","fallback_tools","fallback_metric_pack","fallback_notes","raw_output_relpaths"],"properties":{"host_capability_sensitive_captures":{"$ref":"#/$defs/string_list"},"topology_sensitive_captures":{"$ref":"#/$defs/string_list"},"fallback_tools":{"$ref":"#/$defs/string_list"},"fallback_metric_pack":{"$ref":"#/$defs/string_list"},"fallback_notes":{"$ref":"#/$defs/string_list"},"raw_output_relpaths":{"$ref":"#/$defs/string_list"}}},"command":{"type":"object","additionalProperties":false,"required":["tool","command_line"],"properties":{"tool":{"$ref":"#/$defs/non_empty_string"},"command_line":{"$ref":"#/$defs/non_empty_string"}}},"tool_version":{"type":"object","additionalProperties":false,"required":["tool","version"],"properties":{"tool":{"$ref":"#/$defs/non_empty_string"},"version":{"$ref":"#/$defs/non_empty_string"}}},"sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},"provenance":{"type":"object","additionalProperties":false,"required":["row_id","mode_id","artifact_root","command_entrypoint","kernel_release","rustc_version","cargo_profile","commands","tool_versions","fallback_notes"],"properties":{"row_id":{"$ref":"#/$defs/id"},"mode_id":{"$ref":"#/$defs/id"},"artifact_root":{"$ref":"#/$defs/non_empty_string"},"command_entrypoint":{"$ref":"#/$defs/non_empty_string"},"workspace_root":{"$ref":"#/$defs/nullable_non_empty_string"},"campaign_manifest_path":{"$ref":"#/$defs/nullable_non_empty_string"},"source_revision":{"$ref":"#/$defs/nullable_non_empty_string"},"beads_data_hash":{"oneOf":[{"$ref":"#/$defs/sha256"},{"type":"null"}]},"kernel_release":{"$ref":"#/$defs/non_empty_string"},"rustc_version":{"$ref":"#/$defs/non_empty_string"},"cargo_profile":{"$ref":"#/$defs/non_empty_string"},"commands":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/command"}},"tool_versions":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/tool_version"}},"fallback_notes":{"$ref":"#/$defs/string_list"}}},"artifact_file":{"type":"object","additionalProperties":false,"required":["path","bytes","sha256","description"],"properties":{"path":{"$ref":"#/$defs/non_empty_string"},"bytes":{"type":"integer","minimum":0},"sha256":{"$ref":"#/$defs/sha256"},"description":{"$ref":"#/$defs/non_empty_string"}}}}}"###;
    #[allow(clippy::needless_raw_string_hashes)]
    const BENCHMARK_ARTIFACT_MANIFEST_SCHEMA_RAW: &str = r###"{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"https://frankensqlite.local/schemas/sample_sqlite_db_files/beads_benchmark_artifact_manifest.v1.schema.json","title":"FrankenSQLite Beads Benchmark Artifact Manifest (v1)","type":"object","additionalProperties":false,"required":["schema_version","campaign_id","campaign_manifest_path","row_id","fixture_id","workload","concurrency","mode","placement_profile_id","hardware_class_id","retry_policy_id","build_profile_id","seed_policy_id","run_id","artifact_bundle_key","artifact_bundle_name","artifact_bundle_dir","artifact_bundle_relpath","artifact_names","retention_policy","provenance"],"properties":{"schema_version":{"type":"string","const":"fsqlite-e2e.beads_benchmark_artifact_manifest.v1"},"campaign_id":{"$ref":"#/$defs/id"},"campaign_manifest_path":{"$ref":"#/$defs/non_empty_string"},"row_id":{"$ref":"#/$defs/id"},"fixture_id":{"$ref":"#/$defs/id"},"workload":{"$ref":"#/$defs/id"},"concurrency":{"type":"integer","minimum":1},"mode":{"type":"string","enum":["sqlite_reference","fsqlite_mvcc","fsqlite_single_writer"]},"placement_profile_id":{"$ref":"#/$defs/id"},"hardware_class_id":{"$ref":"#/$defs/id"},"retry_policy_id":{"$ref":"#/$defs/id"},"build_profile_id":{"$ref":"#/$defs/id"},"seed_policy_id":{"$ref":"#/$defs/id"},"run_id":{"$ref":"#/$defs/id"},"artifact_bundle_key":{"$ref":"#/$defs/non_empty_string"},"artifact_bundle_name":{"$ref":"#/$defs/non_empty_string"},"artifact_bundle_dir":{"$ref":"#/$defs/non_empty_string"},"artifact_bundle_relpath":{"$ref":"#/$defs/non_empty_string"},"artifact_names":{"$ref":"#/$defs/artifact_names"},"retention_policy":{"$ref":"#/$defs/retention_policy"},"provenance":{"$ref":"#/$defs/provenance"}},"$defs":{"id":{"type":"string","pattern":"^[A-Za-z0-9][A-Za-z0-9._:-]*$"},"non_empty_string":{"type":"string","minLength":1},"sha256":{"type":"string","pattern":"^[0-9a-f]{64}$"},"retention_class":{"type":"string","enum":["quick_run","full_proof","failure_bundle","final_scorecard"]},"artifact_names":{"type":"object","additionalProperties":false,"required":["result_jsonl","summary_md","manifest_json","hardware_discovery_bundle_json","hardware_discovery_summary_md","logs_dir","profiles_dir"],"properties":{"result_jsonl":{"$ref":"#/$defs/non_empty_string"},"summary_md":{"$ref":"#/$defs/non_empty_string"},"manifest_json":{"$ref":"#/$defs/non_empty_string"},"hardware_discovery_bundle_json":{"$ref":"#/$defs/non_empty_string"},"hardware_discovery_summary_md":{"$ref":"#/$defs/non_empty_string"},"logs_dir":{"$ref":"#/$defs/non_empty_string"},"profiles_dir":{"$ref":"#/$defs/non_empty_string"}}},"retention_policy":{"type":"object","additionalProperties":false,"required":["class","description","superseded_by_newer","immutable","authoritative"],"properties":{"class":{"$ref":"#/$defs/retention_class"},"description":{"$ref":"#/$defs/non_empty_string"},"superseded_by_newer":{"type":"boolean"},"immutable":{"type":"boolean"},"authoritative":{"type":"boolean"}}},"command":{"type":"object","additionalProperties":false,"required":["tool","command_line"],"properties":{"tool":{"$ref":"#/$defs/non_empty_string"},"command_line":{"$ref":"#/$defs/non_empty_string"}}},"tool_version":{"type":"object","additionalProperties":false,"required":["tool","version"],"properties":{"tool":{"$ref":"#/$defs/non_empty_string"},"version":{"$ref":"#/$defs/non_empty_string"}}},"placement_cpu_affinity_policy":{"type":"string","enum":["scheduler_default","dedicated_local_one_thread_per_core","split_across_locality_domains"]},"placement_smt_policy":{"type":"string","enum":["host_default","one_thread_per_core","avoid_primary_sibling_reuse"]},"placement_memory_policy":{"type":"string","enum":["host_default","bind_local","match_cross_domain_placement"]},"placement_helper_lane_policy":{"type":"string","enum":["disclose_host_default","same_locality_housekeeping_core","outside_primary_worker_domains"]},"placement_suite_selector_kind":{"type":"string","enum":["matrix_placement_variant"]},"placement_focused_rerun_selector_kind":{"type":"string","enum":["explicit_bindings"]},"placement_violation_disposition":{"type":"string","enum":["not_comparable"]},"placement_suite_selection_contract":{"type":"object","additionalProperties":false,"required":["selector_kind","selector_field"],"properties":{"selector_kind":{"$ref":"#/$defs/placement_suite_selector_kind"},"selector_field":{"$ref":"#/$defs/non_empty_string"}}},"placement_focused_rerun_contract":{"type":"object","additionalProperties":false,"required":["selector_kind","required_bindings"],"properties":{"selector_kind":{"$ref":"#/$defs/placement_focused_rerun_selector_kind"},"required_bindings":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/non_empty_string"}}}},"placement_claim_contract":{"type":"object","additionalProperties":false,"required":["mandatory_for","optional_for","avoid_for"],"properties":{"mandatory_for":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/non_empty_string"}},"optional_for":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/non_empty_string"}},"avoid_for":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/non_empty_string"}}}},"placement_execution_contract":{"type":"object","additionalProperties":false,"required":["cpu_affinity_policy","smt_policy","memory_policy","helper_lane_policy","required_environment_disclosures","suite_selection","focused_rerun","fixed_knobs","optional_knobs","claim_contract","violation_disposition"],"properties":{"cpu_affinity_policy":{"$ref":"#/$defs/placement_cpu_affinity_policy"},"smt_policy":{"$ref":"#/$defs/placement_smt_policy"},"memory_policy":{"$ref":"#/$defs/placement_memory_policy"},"helper_lane_policy":{"$ref":"#/$defs/placement_helper_lane_policy"},"required_environment_disclosures":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/non_empty_string"}},"suite_selection":{"$ref":"#/$defs/placement_suite_selection_contract"},"focused_rerun":{"$ref":"#/$defs/placement_focused_rerun_contract"},"fixed_knobs":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/non_empty_string"}},"optional_knobs":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/non_empty_string"}},"claim_contract":{"$ref":"#/$defs/placement_claim_contract"},"violation_disposition":{"$ref":"#/$defs/placement_violation_disposition"}}},"hardware_os_family":{"type":"string","enum":["linux"]},"hardware_cpu_architecture":{"type":"string","enum":["x86_64"]},"hardware_topology_class":{"type":"string","enum":["any","many_core_numa"]},"hardware_class_id_fields":{"type":"object","additionalProperties":false,"required":["os_family","cpu_arch","topology_class"],"properties":{"os_family":{"$ref":"#/$defs/hardware_os_family"},"cpu_arch":{"$ref":"#/$defs/hardware_cpu_architecture"},"topology_class":{"$ref":"#/$defs/hardware_topology_class"}}},"hardware_class":{"type":"object","additionalProperties":false,"required":["id","id_fields","min_logical_cores","min_numa_nodes","description"],"properties":{"id":{"$ref":"#/$defs/id"},"id_fields":{"$ref":"#/$defs/hardware_class_id_fields"},"min_logical_cores":{"type":"integer","minimum":1},"min_numa_nodes":{"type":["integer","null"],"minimum":1},"description":{"$ref":"#/$defs/non_empty_string"}}},"build_profile":{"type":"object","additionalProperties":false,"required":["id","cargo_profile","cargo_args","notes"],"properties":{"id":{"$ref":"#/$defs/id"},"cargo_profile":{"$ref":"#/$defs/non_empty_string"},"cargo_args":{"type":"array","items":{"$ref":"#/$defs/non_empty_string"}},"notes":{"$ref":"#/$defs/non_empty_string"}}},"fixture":{"type":"object","additionalProperties":false,"required":["fixture_id","source_path","source_sha256","source_size_bytes","working_copy_relpath","working_copy_sha256","working_copy_size_bytes","page_size","journal_mode","capture_rule"],"properties":{"fixture_id":{"$ref":"#/$defs/id"},"source_path":{"$ref":"#/$defs/non_empty_string"},"source_sha256":{"$ref":"#/$defs/sha256"},"source_size_bytes":{"type":"integer","minimum":1},"working_copy_relpath":{"$ref":"#/$defs/non_empty_string"},"working_copy_sha256":{"$ref":"#/$defs/sha256"},"working_copy_size_bytes":{"type":"integer","minimum":1},"page_size":{"type":"integer","minimum":1},"journal_mode":{"$ref":"#/$defs/non_empty_string"},"capture_rule":{"$ref":"#/$defs/non_empty_string"}}},"placement_policy":{"type":"object","additionalProperties":false,"required":["placement_profile_id","hardware_class_id","availability","command_hint","required","execution_contract"],"properties":{"placement_profile_id":{"$ref":"#/$defs/id"},"hardware_class_id":{"$ref":"#/$defs/id"},"availability":{"type":"string","enum":["universal","topology_aware"]},"command_hint":{"$ref":"#/$defs/non_empty_string"},"required":{"type":"boolean"},"execution_contract":{"$ref":"#/$defs/placement_execution_contract"}}},"provenance":{"type":"object","additionalProperties":false,"required":["command_entrypoint","source_revision","beads_data_hash","kernel_release","fixture","build_profile","hardware_class","placement_policy","commands","tool_versions","fallback_notes"],"properties":{"command_entrypoint":{"$ref":"#/$defs/non_empty_string"},"source_revision":{"$ref":"#/$defs/non_empty_string"},"beads_data_hash":{"$ref":"#/$defs/sha256"},"kernel_release":{"$ref":"#/$defs/non_empty_string"},"fixture":{"$ref":"#/$defs/fixture"},"build_profile":{"$ref":"#/$defs/build_profile"},"hardware_class":{"$ref":"#/$defs/hardware_class"},"placement_policy":{"$ref":"#/$defs/placement_policy"},"commands":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/command"}},"tool_versions":{"type":"array","minItems":1,"items":{"$ref":"#/$defs/tool_version"}},"fallback_notes":{"type":"array","items":{"$ref":"#/$defs/non_empty_string"}}}}}}"###;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(Path::parent)
            .expect("workspace root")
            .to_path_buf()
    }

    fn assert_embedded_schema_matches_tracked_file(relative_path: &str, embedded_schema_raw: &str) {
        let schema_path = workspace_root().join(relative_path);
        if !schema_path.is_file() {
            return;
        }
        let tracked_schema_raw = fs::read_to_string(&schema_path).expect("read tracked schema");
        let embedded_schema_json: serde_json::Value =
            serde_json::from_str(embedded_schema_raw).expect("parse embedded schema fallback");
        let tracked_schema_json: serde_json::Value =
            serde_json::from_str(&tracked_schema_raw).expect("parse tracked schema");
        assert_eq!(
            tracked_schema_json,
            embedded_schema_json,
            "embedded schema fallback drifted from tracked file {}",
            schema_path.display()
        );
    }

    fn assert_json_schema_valid(schema_raw: &str, doc_raw: &str) {
        let schema_json: serde_json::Value =
            serde_json::from_str(schema_raw).expect("parse hot_path_profile_manifest schema");
        let doc_json: serde_json::Value =
            serde_json::from_str(doc_raw).expect("parse hot_path_profile manifest");
        let validator = options()
            .with_draft(Draft::Draft202012)
            .build(&schema_json)
            .expect("build hot_path_profile_manifest validator");
        let errors: Vec<String> = validator
            .iter_errors(&doc_json)
            .map(|error| error.to_string())
            .collect();
        assert!(
            errors.is_empty(),
            "hot_path_profile_manifest schema validation failed:\n- {}",
            errors.join("\n- ")
        );
    }

    fn run_with(args: &[&str]) -> i32 {
        let os_args: Vec<OsString> = args.iter().map(OsString::from).collect();
        run_cli(os_args)
    }

    fn sample_engine_report() -> EngineRunReport {
        EngineRunReport {
            wall_time_ms: 0,
            ops_total: 0,
            ops_per_sec: 0.0,
            retries: 0,
            aborts: 0,
            correctness: CorrectnessReport {
                raw_sha256_match: None,
                dump_match: None,
                canonical_sha256_match: None,
                integrity_check_ok: Some(true),
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

    fn sample_value_type_profile() -> HotPathValueTypeProfile {
        HotPathValueTypeProfile {
            total_values: 1,
            nulls: 0,
            integers: 1,
            reals: 0,
            texts: 0,
            blobs: 0,
            text_bytes_total: 0,
            blob_bytes_total: 0,
        }
    }

    fn sample_hot_path_report() -> HotPathProfileReport {
        let decoded_values = sample_value_type_profile();
        HotPathProfileReport {
            schema_version: HOT_PATH_PROFILE_SCHEMA_V1.to_owned(),
            bead_id: "bd-db300.4.1".to_owned(),
            scenario_id: "bd-db300.4.1.mixed_read_write".to_owned(),
            run_id: "run-1".to_owned(),
            trace_id: "trace-1".to_owned(),
            fixture_id: "fixture-a".to_owned(),
            workload: "mixed_read_write".to_owned(),
            seed: 42,
            scale: 50,
            concurrency: 4,
            concurrent_mode: true,
            run_integrity_check: false,
            golden_dir: Some("/tmp/golden".to_owned()),
            working_base: Some("/tmp/working".to_owned()),
            replay_command:
                "rch exec -- cargo run -p fsqlite-e2e --bin realdb-e2e -- hot-profile --db fixture-a --workload mixed_read_write --golden-dir /tmp/golden --working-base /tmp/working --concurrency 4 --seed 42 --scale 50 --output-dir /tmp/out --mvcc"
                    .to_owned(),
            engine_report: sample_engine_report(),
            parser: HotPathParserProfile {
                parse_single_calls: 1,
                parse_multi_calls: 0,
                parse_cache_hits: 0,
                parse_cache_misses: 1,
                parsed_sql_bytes: 32,
                parse_time_ns: 10,
                rewrite_calls: 0,
                rewrite_time_ns: 0,
                compiled_cache_hits: 0,
                compiled_cache_misses: 1,
                prepared_cache_hits: 0,
                prepared_cache_misses: 1,
                compile_time_ns: 10,
            },
            record_decode: HotPathRecordDecodeProfile {
                parse_record_calls: 1,
                parse_record_into_calls: 0,
                parse_record_column_calls: 1,
                record_bytes_scanned: 64,
                record_vec_capacity_slots: 4,
                decode_time_ns: 12,
                decoded_values: decoded_values.clone(),
                vdbe_record_decode_calls_total: 1,
                vdbe_column_reads_total: 1,
                vdbe_decoded_value_heap_bytes_total: 16,
                decode_cache_hits_total: 0,
                decode_cache_misses_total: 1,
                decode_cache_invalidations_position_total: 0,
                decode_cache_invalidations_write_total: 0,
                decode_cache_invalidations_pseudo_total: 0,
                callsite_breakdown: HotPathRecordDecodeCallsiteBreakdown {
                    unattributed: HotPathRecordDecodeCallsiteCounters::default(),
                    core_connection: HotPathRecordDecodeCallsiteCounters::default(),
                    core_compat_persist: HotPathRecordDecodeCallsiteCounters::default(),
                    vdbe_engine: HotPathRecordDecodeCallsiteCounters {
                        parse_record_calls: 1,
                        parse_record_into_calls: 0,
                        parse_record_column_calls: 1,
                    },
                    vdbe_vectorized_scan: HotPathRecordDecodeCallsiteCounters::default(),
                    btree_cursor: HotPathRecordDecodeCallsiteCounters::default(),
                },
            },
            row_materialization: HotPathRowMaterializationProfile {
                result_rows_total: 1,
                result_values_total: 1,
                result_value_heap_bytes_total: 8,
                result_row_materialization_time_ns_total: 14,
                make_record_calls_total: 0,
                make_record_blob_bytes_total: 0,
                value_types: decoded_values.clone(),
            },
            mvcc_write: HotPathMvccWriteProfile {
                total_write_attempts: 4,
                tier0_already_owned_writes_total: 1,
                tier1_first_touch_writes_total: 2,
                tier2_commit_surface_writes_total: 1,
                page_lock_waits_total: 2,
                page_lock_wait_time_ns_total: 48,
                write_busy_retries_total: 1,
                write_busy_timeouts_total: 0,
                stale_snapshot_rejects_total: 0,
                page_one_conflict_tracks_total: 1,
                page_one_conflict_track_time_ns_total: 12,
                pending_commit_surface_clears_total: 1,
                pending_commit_surface_clear_time_ns_total: 8,
                runtime_retry: fsqlite_e2e::report::HotPathRetryBreakdown {
                    total_retries: 3,
                    total_aborts: 3,
                    kind: fsqlite_e2e::report::HotPathRetryKindBreakdown {
                        busy: 1,
                        busy_snapshot: 2,
                        busy_recovery: 0,
                        busy_other: 0,
                    },
                    phase: fsqlite_e2e::report::HotPathRetryPhaseBreakdown {
                        begin: 0,
                        body: 2,
                        commit: 1,
                        rollback: 0,
                    },
                    max_batch_attempts: 4,
                    top_snapshot_conflict_pages: vec![fsqlite_e2e::report::HotPathConflictPageCount {
                        page_no: 9,
                        retries: 2,
                    }],
                    last_busy_message: Some(
                        "SQLITE_BUSY_SNAPSHOT on page 9".to_owned(),
                    ),
                },
            },
            page_data_motion: HotPathPageDataMotionProfile {
                borrowed_write_normalization_calls_total: 1,
                borrowed_exact_size_copies_total: 1,
                owned_write_normalization_calls_total: 2,
                owned_passthrough_total: 1,
                owned_resized_copies_total: 1,
                normalized_payload_bytes_total: 96,
                normalized_zero_fill_bytes_total: 32,
                normalized_bytes_total: 128,
            },
            connection_ceremony: HotPathConnectionCeremonyProfile {
                background_status_checks: 1,
                op_cx_background_gates: 1,
                statement_dispatch_background_gates: 0,
                prepared_schema_refreshes: 0,
                prepared_schema_lightweight_refreshes: 0,
                prepared_schema_full_reloads: 0,
                pager_publication_refreshes: 1,
                memory_autocommit_fast_path_begins: 1,
                cached_read_snapshot_reuses: 1,
                cached_read_snapshot_parks: 0,
                column_default_evaluation_passes: 0,
                prepared_table_engine_fresh_allocs: 1,
                prepared_table_engine_reuses: 2,
                prepared_insert_fast_lane_hits: 0,
                prepared_insert_instrumented_lane_hits: 0,
                prepared_update_delete_fast_lane_hits: 0,
                prepared_update_delete_instrumented_lane_hits: 0,
                prepared_update_delete_fallback_returning: 0,
                prepared_update_delete_fallback_sqlite_sequence: 0,
                prepared_update_delete_fallback_without_rowid: 0,
                prepared_update_delete_fallback_live_vtab: 0,
                prepared_update_delete_fallback_trigger: 0,
                prepared_update_delete_fallback_foreign_key: 0,
                prepared_table_dml_affected_only_runs: 0,
                autoincrement_sequence_fast_path_updates: 1,
                autoincrement_sequence_scan_refreshes: 0,
            },
            opcode_profile: vec![HotPathOpcodeProfileEntry {
                opcode: "Column".to_owned(),
                total: 3,
            }],
            type_profile: HotPathTypeProfile {
                decoded: decoded_values.clone(),
                materialized: decoded_values,
            },
            subsystem_ranking: vec![HotPathRankingEntry {
                subsystem: "record_decode".to_owned(),
                metric_kind: "time_ns".to_owned(),
                metric_value: 12,
                rationale: "test hotspot".to_owned(),
            }],
            allocator_pressure: HotPathAllocatorPressure {
                parser_sql_bytes: 32,
                decoded_value_heap_bytes_total: 16,
                result_value_heap_bytes_total: 8,
                record_vec_capacity_slots: 4,
                page_data_normalization_bytes_total: 128,
                ranked_sources: vec![HotPathRankingEntry {
                    subsystem: "record_decode_values".to_owned(),
                    metric_kind: "bytes".to_owned(),
                    metric_value: 16,
                    rationale: "test allocator pressure".to_owned(),
                }],
            },
        }
    }

    fn sample_benchmark_campaign() -> BeadsBenchmarkCampaign {
        BeadsBenchmarkCampaign {
            schema_version: BEADS_BENCHMARK_CAMPAIGN_SCHEMA_V1.to_owned(),
            campaign_id: "bd-db300.1.2".to_owned(),
            title: "sample".to_owned(),
            working_benchmark_root_relpath: "sample_sqlite_db_files/working/beads_bench_20260310"
                .to_owned(),
            beads_data_relpath: ".beads/issues.jsonl".to_owned(),
            fixtures: vec![BeadsBenchmarkFixture {
                fixture_id: "frankensqlite".to_owned(),
                source_path: "/data/projects/frankensqlite/.beads/beads.db".to_owned(),
                source_sha256: "a".repeat(64),
                source_size_bytes: 13,
                working_copy_relpath:
                    "sample_sqlite_db_files/working/beads_bench_20260310/golden/frankensqlite_beads.db"
                        .to_owned(),
                working_copy_sha256: "b".repeat(64),
                working_copy_size_bytes: 13,
                page_size: 4096,
                journal_mode: "wal".to_owned(),
                capture_rule: "copy pinned working copy".to_owned(),
            }],
            placement_profiles: vec![PlacementProfile {
                id: "baseline_unpinned".to_owned(),
                kind: PlacementProfileKind::Baseline,
                description: "scheduler default".to_owned(),
                command_hint: "run directly".to_owned(),
                availability: PlacementAvailability::Universal,
                execution_contract: PlacementExecutionContract {
                    cpu_affinity_policy: PlacementCpuAffinityPolicy::SchedulerDefault,
                    smt_policy: PlacementSmtPolicy::HostDefault,
                    memory_policy: PlacementMemoryPolicy::HostDefault,
                    helper_lane_policy: PlacementHelperLanePolicy::DiscloseHostDefault,
                    required_environment_disclosures: vec![
                        "placement_profile_id".to_owned(),
                        "hardware_class_id".to_owned(),
                        "hardware_signature".to_owned(),
                        "cpu_affinity_mask".to_owned(),
                        "smt_policy_state".to_owned(),
                        "memory_policy".to_owned(),
                        "helper_lane_cpu_set".to_owned(),
                        "numa_balancing_state".to_owned(),
                    ],
                    suite_selection: PlacementSuiteSelectionContract {
                        selector_kind: PlacementSuiteSelectorKind::MatrixPlacementVariant,
                        selector_field:
                            "matrix_rows[].placement_variants[].placement_profile_id".to_owned(),
                    },
                    focused_rerun: PlacementFocusedRerunContract {
                        selector_kind: PlacementFocusedRerunSelectorKind::ExplicitBindings,
                        required_bindings: vec![
                            "RUN_ID".to_owned(),
                            "ARTIFACT_BUNDLE_DIR".to_owned(),
                            "ARTIFACT_BUNDLE_RELPATH".to_owned(),
                            "PLACEMENT_PROFILE_ID".to_owned(),
                            "HARDWARE_CLASS_ID".to_owned(),
                            "MANIFEST_JSON".to_owned(),
                            "SOURCE_REVISION".to_owned(),
                            "BEADS_HASH".to_owned(),
                        ],
                    },
                    fixed_knobs: vec![
                        "no_taskset_or_numactl_binding".to_owned(),
                        "report_host_default_smt_policy".to_owned(),
                        "report_host_default_memory_policy".to_owned(),
                        "disclose_helper_lane_policy_without_relocation".to_owned(),
                    ],
                    optional_knobs: vec![
                        "exact_scheduler_chosen_cpu_set".to_owned(),
                        "extra_profiler_capture".to_owned(),
                    ],
                    claim_contract: PlacementClaimContract {
                        mandatory_for: vec![
                            "portable_baseline_claims".to_owned(),
                            "host_default_regression_checks".to_owned(),
                        ],
                        optional_for: vec!["smoke_reruns".to_owned()],
                        avoid_for: vec![
                            "transferable_many_core_win_claims".to_owned(),
                            "cross_node_sensitivity_claims".to_owned(),
                        ],
                    },
                    violation_disposition: PlacementViolationDisposition::NotComparable,
                },
            }],
            hardware_classes: vec![HardwareClass {
                id: HARDWARE_CLASS_LINUX_X86_64_ANY.to_owned(),
                id_fields: HardwareClassIdFields {
                    os_family: HardwareOsFamily::Linux,
                    cpu_arch: HardwareCpuArchitecture::X86_64,
                    topology_class: HardwareTopologyClass::Any,
                },
                min_logical_cores: 4,
                min_numa_nodes: None,
                description: "generic".to_owned(),
            }],
            retry_policies: vec![RetryPolicy {
                id: "instrumented_busy_retry_v1".to_owned(),
                max_busy_retries: 10_000,
                busy_backoff_ms: 1,
                busy_backoff_max_ms: 250,
                notes: "default".to_owned(),
            }],
            build_profiles: vec![BuildProfile {
                id: "release_perf".to_owned(),
                cargo_profile: "release-perf".to_owned(),
                cargo_args: vec!["--profile".to_owned(), "release-perf".to_owned()],
                notes: "perf".to_owned(),
            }],
            seed_policies: vec![SeedPolicy {
                id: "fixed_seed_42".to_owned(),
                kind: "fixed".to_owned(),
                base_seed: 42,
                notes: "stable".to_owned(),
            }],
            matrix_rows: vec![BeadsBenchmarkMatrixRow {
                row_id: "mixed_read_write_c4".to_owned(),
                fixtures: vec!["frankensqlite".to_owned()],
                workload: "mixed_read_write".to_owned(),
                concurrency: 4,
                modes: vec![BenchmarkMode::FsqliteMvcc],
                placement_variants: vec![PlacementVariant {
                    placement_profile_id: "baseline_unpinned".to_owned(),
                    hardware_class_id: HARDWARE_CLASS_LINUX_X86_64_ANY.to_owned(),
                    required: true,
                }],
                retry_policy_id: "instrumented_busy_retry_v1".to_owned(),
                build_profile_id: "release_perf".to_owned(),
                seed_policy_id: "fixed_seed_42".to_owned(),
            }],
            artifact_contract: BenchmarkArtifactContract {
                artifact_root_relpath: "artifacts/perf/bd-db300.1.2".to_owned(),
                bundle_dir_template:
                    "{row_id}__{workload}__c{concurrency}__{fixture_id}__{mode}__{placement_profile_id}__{build_profile_id}__run_{run_id}__rev_{source_revision}__beads_{beads_hash}"
                        .to_owned(),
                bundle_key_template:
                    "{row_id}:{fixture_id}:{workload}:c{concurrency}:{mode}:{placement_profile_id}:{build_profile_id}:run_{run_id}:rev_{source_revision}:beads_{beads_hash}"
                        .to_owned(),
                bundle_name_template:
                    "{row_id} {fixture_id} {workload} c{concurrency} {mode} {placement_profile_id} {build_profile_id} run {run_id} rev {source_revision} beads {beads_hash}"
                        .to_owned(),
                manifest_schema_version: BEADS_BENCHMARK_ARTIFACT_MANIFEST_SCHEMA_V1.to_owned(),
                result_jsonl_name: "results.jsonl".to_owned(),
                summary_md_name: "summary.md".to_owned(),
                manifest_name: "manifest.json".to_owned(),
                hardware_discovery_bundle_json_name: "hardware_discovery_bundle.json".to_owned(),
                hardware_discovery_summary_md_name: "hardware_discovery_summary.md".to_owned(),
                logs_dir_name: "logs".to_owned(),
                profiles_dir_name: "profiles".to_owned(),
                retention_policies: vec![BenchmarkArtifactRetentionPolicy {
                    class: BenchmarkArtifactRetentionClass::FailureBundle,
                    description:
                        "Failure bundle kept immutably for diagnosis and replay.".to_owned(),
                    superseded_by_newer: false,
                    immutable: true,
                    authoritative: true,
                }],
            },
        }
    }

    fn sample_benchmark_cell() -> ExpandedBenchmarkCell {
        ExpandedBenchmarkCell {
            row_id: "mixed_read_write_c4".to_owned(),
            fixture_id: "frankensqlite".to_owned(),
            workload: "mixed_read_write".to_owned(),
            concurrency: 4,
            mode: BenchmarkMode::FsqliteMvcc,
            placement_profile_id: "baseline_unpinned".to_owned(),
            hardware_class_id: HARDWARE_CLASS_LINUX_X86_64_ANY.to_owned(),
            retry_policy_id: "instrumented_busy_retry_v1".to_owned(),
            build_profile_id: "release_perf".to_owned(),
            seed_policy_id: "fixed_seed_42".to_owned(),
        }
    }

    fn sample_hot_path_manifest() -> HotPathArtifactManifest {
        HotPathArtifactManifest {
            schema_version: HOT_PATH_PROFILE_MANIFEST_SCHEMA_V1.to_owned(),
            bead_id: "bd-db300.4.1".to_owned(),
            run_id: "run-1".to_owned(),
            trace_id: "trace-1".to_owned(),
            scenario_id: "bd-db300.4.1.mixed_read_write".to_owned(),
            fixture_id: "fixture-a".to_owned(),
            workload: "mixed_read_write".to_owned(),
            seed: 42,
            scale: 50,
            concurrency: 4,
            concurrent_mode: true,
            run_integrity_check: false,
            golden_dir: Some("/tmp/golden".to_owned()),
            working_base: Some("/tmp/working".to_owned()),
            replay_command:
                "rch exec -- cargo run -p fsqlite-e2e --bin realdb-e2e -- hot-profile --db fixture-a --workload mixed_read_write --golden-dir /tmp/golden --working-base /tmp/working --concurrency 4 --seed 42 --scale 50 --output-dir /tmp/out --mvcc"
                    .to_owned(),
            counter_capture_summary: None,
            provenance: None,
            files: vec![
                HotPathArtifactFile {
                    path: "profile.json".to_owned(),
                    bytes: 1,
                    sha256: "0".repeat(64),
                    description: "report".to_owned(),
                },
                HotPathArtifactFile {
                    path: "opcode_profile.json".to_owned(),
                    bytes: 1,
                    sha256: "0".repeat(64),
                    description: "opcode pack".to_owned(),
                },
                HotPathArtifactFile {
                    path: "subsystem_profile.json".to_owned(),
                    bytes: 1,
                    sha256: "0".repeat(64),
                    description: "subsystem pack".to_owned(),
                },
                HotPathArtifactFile {
                    path: "summary.md".to_owned(),
                    bytes: 1,
                    sha256: "0".repeat(64),
                    description: "summary".to_owned(),
                },
                HotPathArtifactFile {
                    path: "actionable_ranking.json".to_owned(),
                    bytes: 1,
                    sha256: "0".repeat(64),
                    description: "ranking".to_owned(),
                },
                HotPathArtifactFile {
                    path: HOT_PATH_COMMAND_PACK_NAME.to_owned(),
                    bytes: 1,
                    sha256: "0".repeat(64),
                    description: "command pack".to_owned(),
                },
                HotPathArtifactFile {
                    path: "manifest.json".to_owned(),
                    bytes: 1,
                    sha256: "0".repeat(64),
                    description: "manifest".to_owned(),
                },
            ],
        }
    }

    fn write_sample_hot_path_artifacts(output_dir: &Path, files: &[HotPathArtifactFile]) {
        for file in files {
            if file.path == "manifest.json" {
                continue;
            }
            let path = output_dir.join(&file.path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("artifact parent should be creatable");
            }
            let body = format!("artifact fixture for {}", file.path);
            fs::write(&path, body.as_bytes()).expect("artifact fixture should be writable");
        }
    }

    fn sample_hot_path_provenance_inputs() -> HotPathArtifactProvenanceInputs {
        HotPathArtifactProvenanceInputs {
            artifact_root: "/tmp/out".to_owned(),
            workspace_root: Some("/workspace".to_owned()),
            campaign_manifest_path: Some(BEADS_BENCHMARK_CAMPAIGN_PATH_RELATIVE.to_owned()),
            source_revision: Some("0123456789abcdef0123456789abcdef01234567".to_owned()),
            beads_data_hash: Some("a".repeat(64)),
            kernel_release: "Linux 6.13.5-test".to_owned(),
            rustc_version: "rustc 1.91.0-nightly".to_owned(),
            cargo_profile: "release-perf".to_owned(),
            tool_versions: vec![
                BenchmarkArtifactToolVersion {
                    tool: "cargo".to_owned(),
                    version: "cargo 1.91.0-nightly".to_owned(),
                },
                BenchmarkArtifactToolVersion {
                    tool: "hyperfine".to_owned(),
                    version: "hyperfine 1.19.0".to_owned(),
                },
            ],
        }
    }

    #[test]
    fn resolve_hot_path_microarchitectural_context_accepts_working_copy_fixture_alias() {
        let tempdir = tempfile::tempdir().expect("tempdir should succeed");
        let manifest_path = tempdir.path().join(BEADS_BENCHMARK_CAMPAIGN_PATH_RELATIVE);
        fs::create_dir_all(
            manifest_path
                .parent()
                .expect("campaign manifest should have parent"),
        )
        .expect("campaign manifest parent should exist");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&sample_benchmark_campaign())
                .expect("sample campaign should serialize"),
        )
        .expect("sample campaign manifest should write");

        let mut report = sample_hot_path_report();
        report.fixture_id = "frankensqlite_beads".to_owned();

        let context = resolve_hot_path_microarchitectural_context(Some(tempdir.path()), &report);

        assert_eq!(context.fixture_id, "frankensqlite_beads");
        assert_eq!(context.row_id, "mixed_read_write_c4");
        assert_eq!(context.mode_id, "fsqlite_mvcc");
        assert_eq!(
            context.placement_profile_id.as_deref(),
            Some("baseline_unpinned")
        );
        assert_eq!(
            context.hardware_class_id.as_deref(),
            Some(HARDWARE_CLASS_LINUX_X86_64_ANY)
        );
        assert_eq!(
            context.hardware_signature.as_deref(),
            Some("linux:x86_64:any")
        );
    }

    #[test]
    fn test_report_has_failure_flags_integrity_failures() {
        let mut report = sample_engine_report();
        assert!(!report_has_failure(&report));

        report.correctness.integrity_check_ok = Some(false);
        assert!(report_has_failure(&report));

        report.correctness.integrity_check_ok = Some(true);
        report.error = Some("boom".to_owned());
        assert!(report_has_failure(&report));
    }

    #[test]
    fn run_mode_options_preserve_defaults_without_flags() {
        let options = RunModeOptions::from_flags(false, false, false);
        assert!(options.run_integrity_check);
        assert!(options.capture_environment_metadata);
    }

    #[test]
    fn run_mode_options_profile_only_disables_integrity_and_metadata() {
        let options = RunModeOptions::from_flags(true, false, false);
        assert!(!options.run_integrity_check);
        assert!(!options.capture_environment_metadata);
    }

    #[test]
    fn resolve_path_from_base_anchors_relative_paths() {
        assert_eq!(
            resolve_path_from_base(Path::new("/workspace"), Path::new("artifacts/out")),
            PathBuf::from("/workspace/artifacts/out")
        );
    }

    #[test]
    fn resolve_path_from_base_preserves_absolute_paths() {
        assert_eq!(
            resolve_path_from_base(Path::new("/workspace"), Path::new("/tmp/out")),
            PathBuf::from("/tmp/out")
        );
    }

    #[test]
    fn format_hot_profile_replay_command_renders_expected_flags() {
        let rendered = format_hot_profile_replay_command_with_env_overrides(
            &HotProfileReplayCommand {
                db: "fixture-a",
                workload: "hot_page_contention",
                golden_dir: Path::new("/tmp/golden dir"),
                working_base: Path::new("/tmp/working"),
                concurrency: 4,
                seed: 42,
                scale: 50,
                output_dir: Path::new("/tmp/output dir"),
                mvcc: false,
                run_integrity_check: true,
            },
            &[],
        );

        assert_eq!(
            rendered,
            "rch exec -- cargo run -p fsqlite-e2e --bin realdb-e2e -- hot-profile --db fixture-a --workload hot_page_contention --golden-dir '/tmp/golden dir' --working-base /tmp/working --concurrency 4 --seed 42 --scale 50 --output-dir '/tmp/output dir' --no-mvcc --integrity-check"
        );
    }

    #[test]
    fn format_hot_profile_replay_command_renders_env_overrides() {
        let rendered = format_hot_profile_replay_command_with_env_overrides(
            &HotProfileReplayCommand {
                db: "fixture-a",
                workload: "hot_page_contention",
                golden_dir: Path::new("/tmp/golden dir"),
                working_base: Path::new("/tmp/working"),
                concurrency: 4,
                seed: 42,
                scale: 50,
                output_dir: Path::new("/tmp/output dir"),
                mvcc: false,
                run_integrity_check: true,
            },
            &[
                (HOT_PATH_BEAD_ID_ENV, "bd-db300.1.3".to_owned()),
                (
                    HOT_PATH_WORKSPACE_ROOT_ENV,
                    "/data/projects/frankensqlite".to_owned(),
                ),
            ],
        );

        assert_eq!(
            rendered,
            "rch exec -- env FSQLITE_HOT_PATH_BEAD_ID=bd-db300.1.3 FSQLITE_HOT_PATH_WORKSPACE_ROOT=/data/projects/frankensqlite cargo run -p fsqlite-e2e --bin realdb-e2e -- hot-profile --db fixture-a --workload hot_page_contention --golden-dir '/tmp/golden dir' --working-base /tmp/working --concurrency 4 --seed 42 --scale 50 --output-dir '/tmp/output dir' --no-mvcc --integrity-check"
        );
    }

    #[test]
    fn serialize_run_output_profile_only_suppresses_environment_metadata() {
        let environment = RunModeOptions::from_flags(true, false, false).environment("release");
        let text = serialize_run_output(
            123,
            &environment,
            EngineInfo {
                name: "sqlite3".to_owned(),
                sqlite_version: Some("3.46.0".to_owned()),
                fsqlite_git: None,
            },
            "fixture-a",
            Path::new("/tmp/fixture-a.db"),
            Some("abc".to_owned()),
            "mixed_read_write",
            4,
            17,
            sample_engine_report(),
            false,
        )
        .expect("profile-only serialization should succeed");
        let value: Value = serde_json::from_str(&text).expect("profile record must parse");
        assert_eq!(
            value["schema_version"],
            fsqlite_e2e::report::RUN_RECORD_SCHEMA_V1
        );
        assert_eq!(value["environment"]["capture_mode"], "suppressed");
        assert_eq!(value["environment"]["os"], "suppressed");
    }

    #[test]
    fn serialize_run_output_default_keeps_environment_metadata() {
        let environment = RunModeOptions::default().environment("release");
        let text = serialize_run_output(
            123,
            &environment,
            EngineInfo {
                name: "sqlite3".to_owned(),
                sqlite_version: Some("3.46.0".to_owned()),
                fsqlite_git: None,
            },
            "fixture-a",
            Path::new("/tmp/fixture-a.db"),
            Some("abc".to_owned()),
            "mixed_read_write",
            4,
            17,
            sample_engine_report(),
            false,
        )
        .expect("default serialization should succeed");
        let value: Value = serde_json::from_str(&text).expect("run record must parse");
        assert_eq!(
            value["schema_version"],
            fsqlite_e2e::report::RUN_RECORD_SCHEMA_V1
        );
        assert!(value.get("environment").is_some());
        assert!(value.get("methodology").is_some());
        assert_eq!(value["environment"]["capture_mode"], "captured");
    }

    #[test]
    fn test_help_flag_exits_zero() {
        assert_eq!(run_with(&["realdb-e2e", "--help"]), 0);
        assert_eq!(run_with(&["realdb-e2e", "-h"]), 0);
    }

    #[test]
    fn test_no_args_shows_help() {
        assert_eq!(run_with(&["realdb-e2e"]), 0);
    }

    #[test]
    fn serialize_hot_path_inline_bundle_includes_expected_sections() {
        let report = sample_hot_path_report();
        let tempdir = tempfile::tempdir().expect("tempdir should succeed");
        let command_pack = build_hot_path_command_pack(
            &report,
            &HotProfileReplayCommand {
                db: "fixture-a",
                workload: "mixed_read_write",
                golden_dir: Path::new("/tmp/golden"),
                working_base: Path::new("/tmp/working"),
                concurrency: 4,
                seed: 42,
                scale: 50,
                output_dir: Path::new("/tmp/out"),
                mvcc: true,
                run_integrity_check: false,
            },
        );
        let counter_capture_summary = build_hot_path_counter_capture_summary(&command_pack);
        let mut base_manifest = sample_hot_path_manifest();
        write_sample_hot_path_artifacts(tempdir.path(), &base_manifest.files);
        base_manifest.provenance = Some(build_hot_path_artifact_provenance(
            &report,
            &command_pack,
            counter_capture_summary.as_ref(),
            sample_hot_path_provenance_inputs(),
        ));
        let manifest = finalize_hot_path_manifest(
            tempdir.path(),
            base_manifest,
            counter_capture_summary,
            vec![HotPathArtifactFile {
                path: HOT_PATH_COMMAND_PACK_NAME.to_owned(),
                bytes: 1,
                sha256: "0".repeat(64),
                description: "command pack".to_owned(),
            }],
        )
        .expect("manifest finalization should succeed");
        let microarchitectural_context = HotPathMicroarchitecturalContext {
            fixture_id: report.fixture_id.clone(),
            row_id: "mixed_read_write_c4".to_owned(),
            mode_id: "fsqlite_mvcc".to_owned(),
            placement_profile_id: Some("baseline_unpinned".to_owned()),
            hardware_class_id: Some(HARDWARE_CLASS_LINUX_X86_64_ANY.to_owned()),
            hardware_signature: Some("linux:x86_64:any".to_owned()),
        };
        let text = serialize_hot_path_inline_bundle(
            &report,
            &manifest,
            &command_pack,
            Some(&microarchitectural_context),
        )
        .expect("inline bundle serialization should succeed");
        let value: Value = serde_json::from_str(&text).expect("bundle JSON must parse");
        assert_eq!(value["schema_version"], HOT_PATH_INLINE_BUNDLE_SCHEMA_V1);
        assert_eq!(
            value["profile"]["schema_version"],
            HOT_PATH_PROFILE_SCHEMA_V1
        );
        assert_eq!(
            value["opcode_profile"]["schema_version"],
            HOT_PATH_OPCODE_PROFILE_SCHEMA_V1
        );
        assert_eq!(
            value["subsystem_profile"]["schema_version"],
            HOT_PATH_SUBSYSTEM_PROFILE_SCHEMA_V1
        );
        assert_eq!(
            value["actionable_ranking"]["schema_version"],
            HOT_PATH_PROFILE_ACTIONABLE_RANKING_SCHEMA_V3
        );
        assert!(
            value["actionable_ranking"]["baseline_reuse_ledger"]
                .as_array()
                .is_some_and(|entries| !entries.is_empty())
        );
        assert!(
            value["actionable_ranking"]["baseline_waste_ledger"]
                .as_array()
                .is_some_and(|entries| !entries.is_empty())
        );
        assert_eq!(
            value["manifest"]["schema_version"],
            HOT_PATH_PROFILE_MANIFEST_SCHEMA_V1
        );
        assert_eq!(
            value["command_pack"]["schema_version"],
            HOT_PATH_COMMAND_PACK_SCHEMA_V2
        );
        assert_eq!(value["manifest"]["fixture_id"], "fixture-a");
        assert_eq!(value["profile"]["concurrent_mode"], true);
        assert_eq!(value["profile"]["run_integrity_check"], false);
        assert_eq!(value["profile"]["golden_dir"], "/tmp/golden");
        assert_eq!(value["profile"]["working_base"], "/tmp/working");
        assert_eq!(value["manifest"]["concurrent_mode"], true);
        assert_eq!(value["manifest"]["golden_dir"], "/tmp/golden");
        assert_eq!(
            value["manifest"]["counter_capture_summary"]["host_capability_sensitive_captures"][0],
            "topdown"
        );
        assert_eq!(
            value["manifest"]["counter_capture_summary"]["topology_sensitive_captures"][0],
            "cache_to_cache"
        );
        assert_eq!(
            value["manifest"]["provenance"]["row_id"],
            "mixed_read_write_c4"
        );
        assert_eq!(value["manifest"]["provenance"]["mode_id"], "fsqlite_mvcc");
        assert_eq!(
            value["manifest"]["provenance"]["source_revision"],
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert_eq!(
            value["actionable_ranking"]["microarchitectural_signatures"][0]["fixture_id"],
            "fixture-a"
        );
        assert_eq!(
            value["actionable_ranking"]["microarchitectural_signatures"][0]["row_id"],
            "mixed_read_write_c4"
        );
        assert_eq!(
            value["actionable_ranking"]["microarchitectural_signatures"][0]["placement_profile_id"],
            "baseline_unpinned"
        );
        assert_eq!(
            value["actionable_ranking"]["microarchitectural_signatures"][0]["hardware_class_id"],
            "linux_x86_64_any"
        );
        assert_eq!(
            value["actionable_ranking"]["microarchitectural_signatures"][0]["hardware_signature"],
            "linux:x86_64:any"
        );
        assert_eq!(value["opcode_profile"]["opcodes"][0]["opcode"], "Column");
        assert_eq!(
            value["command_pack"]["commands"][0]["output_relpath"],
            "profiles/hyperfine.profiler_safe.json"
        );
        assert!(
            value["command_pack"]["commands"]
                .as_array()
                .is_some_and(|commands| commands.iter().any(|command| {
                    command["capture"] == "topdown"
                        && command["mode"] == "profiler_safe"
                        && command["counter_pack"]["primary_selection"] == "TopdownL1"
                        && command["counter_pack"]["fallback_selections"][0]
                            == "core_event_fallback"
                }))
        );
        assert!(
            value["command_pack"]["commands"]
                .as_array()
                .is_some_and(|commands| commands.iter().any(|command| {
                    command["capture"] == "cache_to_cache"
                        && command["mode"] == "full_validation"
                        && command["counter_pack"]["raw_output_relpaths"][0]
                            == "profiles/perf-c2c.full_validation.data"
                }))
        );
        assert_eq!(
            value["subsystem_profile"]["subsystem_ranking"][0]["subsystem"],
            "record_decode"
        );
        assert!(
            value["summary_markdown"]
                .as_str()
                .is_some_and(|summary| summary.contains("## Ranked Hotspots"))
        );
        assert!(
            value["summary_markdown"]
                .as_str()
                .is_some_and(|summary| summary.contains("## Microarchitectural Signatures"))
        );
        assert!(
            value["summary_markdown"]
                .as_str()
                .is_some_and(|summary| summary.contains("Runtime retry taxonomy"))
        );
        assert!(
            value["summary_markdown"]
                .as_str()
                .is_some_and(|summary| summary.contains("Page-touch classes"))
        );
    }

    #[test]
    fn build_hot_path_command_pack_emits_profiler_and_validation_commands() {
        let report = sample_hot_path_report();
        let replay_command = HotProfileReplayCommand {
            db: "fixture-a",
            workload: "mixed_read_write",
            golden_dir: Path::new("/tmp/golden"),
            working_base: Path::new("/tmp/working"),
            concurrency: 4,
            seed: 42,
            scale: 50,
            output_dir: Path::new("/tmp/out dir"),
            mvcc: true,
            run_integrity_check: false,
        };

        let pack = build_hot_path_command_pack(&report, &replay_command);
        assert_eq!(pack.schema_version, HOT_PATH_COMMAND_PACK_SCHEMA_V2);
        assert!(
            pack.profiler_safe_replay_command
                .contains("--output-dir '/tmp/out dir'")
        );
        assert!(
            pack.full_validation_replay_command
                .contains("--integrity-check")
        );
        assert_eq!(pack.commands.len(), 18);
        assert!(pack.commands.iter().any(|command| {
            command.capture == "wall_clock"
                && command.mode == "profiler_safe"
                && command.command_line.contains("hyperfine")
        }));
        assert!(pack.commands.iter().any(|command| {
            command
                .command_line
                .contains("'/tmp/out dir/profiles/hyperfine.profiler_safe.json'")
        }));
        assert!(pack.commands.iter().any(|command| {
            command.capture == "allocation"
                && command.mode == "full_validation"
                && command.output_relpath == "profiles/heaptrack.full_validation.gz"
        }));
        assert!(pack.commands.iter().any(|command| {
            command.capture == "topdown"
                && command.mode == "profiler_safe"
                && command.command_line.contains("TopdownL1")
                && command.command_line.contains("cycles,instructions")
                && command.counter_pack.as_ref().is_some_and(|counter_pack| {
                    counter_pack.host_capability_sensitive
                        && !counter_pack.topology_sensitive
                        && counter_pack.primary_selection == "TopdownL1"
                        && counter_pack
                            .fallback_selections
                            .contains(&"core_event_fallback".to_owned())
                })
        }));
        assert!(pack.commands.iter().any(|command| {
            command.capture == "cache_to_cache"
                && command.mode == "full_validation"
                && command.command_line.contains("perf c2c record")
                && command.command_line.contains("perf mem record")
                && command.output_relpath == "profiles/perf-c2c.full_validation.summary.txt"
                && command.counter_pack.as_ref().is_some_and(|counter_pack| {
                    counter_pack.topology_sensitive
                        && counter_pack.fallback_tools
                            == vec!["perf-mem".to_owned(), "perf-stat".to_owned()]
                        && counter_pack.raw_output_relpaths
                            == vec!["profiles/perf-c2c.full_validation.data".to_owned()]
                })
        }));
        assert!(pack.commands.iter().any(|command| {
            command.capture == "migration"
                && command.mode == "profiler_safe"
                && command
                    .command_line
                    .contains("cpu-migrations,context-switches")
                && command.command_line.contains("context-switches,task-clock")
                && command.counter_pack.as_ref().is_some_and(|counter_pack| {
                    counter_pack.topology_sensitive
                        && counter_pack.primary_selection == "cpu_migrations"
                        && counter_pack.fallback_event_pack
                            == vec!["context-switches".to_owned(), "task-clock".to_owned()]
                })
        }));
        assert!(pack.commands.iter().any(|command| {
            command.capture == "remote_access"
                && command.mode == "full_validation"
                && command.command_line.contains("perf mem report")
                && command
                    .command_line
                    .contains("cache-references,cache-misses,page-faults")
                && command.output_relpath
                    == "profiles/perf-mem-remote-access.full_validation.summary.txt"
                && command.counter_pack.as_ref().is_some_and(|counter_pack| {
                    counter_pack.topology_sensitive
                        && counter_pack.primary_tool == "perf-mem"
                        && counter_pack.raw_output_relpaths
                            == vec![
                                "profiles/perf-mem-remote-access.full_validation.data".to_owned(),
                            ]
                })
        }));
    }

    #[test]
    fn build_hot_path_counter_capture_summary_rolls_up_capability_and_fallback_metadata() {
        let report = sample_hot_path_report();
        let replay_command = HotProfileReplayCommand {
            db: "fixture-a",
            workload: "mixed_read_write",
            golden_dir: Path::new("/tmp/golden"),
            working_base: Path::new("/tmp/working"),
            concurrency: 4,
            seed: 42,
            scale: 50,
            output_dir: Path::new("/tmp/out dir"),
            mvcc: true,
            run_integrity_check: false,
        };

        let pack = build_hot_path_command_pack(&report, &replay_command);
        let summary = build_hot_path_counter_capture_summary(&pack)
            .expect("counter pack summary should exist");

        assert_eq!(
            summary.host_capability_sensitive_captures,
            vec![
                "topdown".to_owned(),
                "cache_to_cache".to_owned(),
                "migration".to_owned(),
                "remote_access".to_owned(),
            ]
        );
        assert_eq!(
            summary.topology_sensitive_captures,
            vec![
                "cache_to_cache".to_owned(),
                "migration".to_owned(),
                "remote_access".to_owned(),
            ]
        );
        assert!(summary.fallback_tools.iter().any(|tool| tool == "perf-mem"));
        assert!(
            summary
                .fallback_metric_pack
                .iter()
                .any(|event| event == "cache-misses")
        );
        assert!(summary.fallback_notes.iter().any(|note| {
            note == "cache_to_cache:profiler_safe: perf c2c unavailable or failed on this host"
        }));
        assert!(
            summary
                .raw_output_relpaths
                .iter()
                .any(|path| { path == "profiles/perf-c2c.profiler_safe.data" })
        );
    }

    #[test]
    fn build_hot_path_artifact_provenance_rolls_up_commands_and_context() {
        let report = sample_hot_path_report();
        let replay_command = HotProfileReplayCommand {
            db: "fixture-a",
            workload: "mixed_read_write",
            golden_dir: Path::new("/tmp/golden"),
            working_base: Path::new("/tmp/working"),
            concurrency: 4,
            seed: 42,
            scale: 50,
            output_dir: Path::new("/tmp/out"),
            mvcc: true,
            run_integrity_check: false,
        };
        let command_pack = build_hot_path_command_pack(&report, &replay_command);
        let counter_capture_summary = build_hot_path_counter_capture_summary(&command_pack);
        let provenance = build_hot_path_artifact_provenance(
            &report,
            &command_pack,
            counter_capture_summary.as_ref(),
            sample_hot_path_provenance_inputs(),
        );
        let beads_hash = "a".repeat(64);

        assert_eq!(provenance.row_id, "mixed_read_write_c4");
        assert_eq!(provenance.mode_id, "fsqlite_mvcc");
        assert_eq!(provenance.artifact_root, "/tmp/out");
        assert_eq!(provenance.workspace_root.as_deref(), Some("/workspace"));
        assert_eq!(
            provenance.campaign_manifest_path.as_deref(),
            Some(BEADS_BENCHMARK_CAMPAIGN_PATH_RELATIVE)
        );
        assert_eq!(
            provenance.source_revision.as_deref(),
            Some("0123456789abcdef0123456789abcdef01234567")
        );
        assert_eq!(
            provenance.beads_data_hash.as_deref(),
            Some(beads_hash.as_str())
        );
        assert_eq!(provenance.kernel_release, "Linux 6.13.5-test");
        assert_eq!(provenance.rustc_version, "rustc 1.91.0-nightly");
        assert_eq!(provenance.cargo_profile, "release-perf");
        assert!(
            provenance
                .commands
                .iter()
                .any(|command| command.tool == "realdb-e2e"
                    && command.command_line == report.replay_command)
        );
        assert!(provenance.commands.iter().any(
            |command| command.tool == "hyperfine" && command.command_line.contains("hyperfine")
        ));
        assert_eq!(
            provenance.fallback_notes,
            counter_capture_summary
                .expect("counter capture summary should exist")
                .fallback_notes
        );
        assert_eq!(provenance.tool_versions.len(), 2);
    }

    #[test]
    fn finalize_hot_path_manifest_rewrites_disk_manifest_with_command_pack() {
        let tempdir = tempfile::tempdir().expect("tempdir should succeed");
        let report = sample_hot_path_report();
        let replay_command = HotProfileReplayCommand {
            db: "fixture-a",
            workload: "mixed_read_write",
            golden_dir: Path::new("/tmp/golden"),
            working_base: Path::new("/tmp/working"),
            concurrency: 4,
            seed: 42,
            scale: 50,
            output_dir: Path::new("/tmp/out"),
            mvcc: true,
            run_integrity_check: false,
        };
        let command_pack = build_hot_path_command_pack(&report, &replay_command);
        let mut manifest = sample_hot_path_manifest();
        let provenance = build_hot_path_artifact_provenance(
            &report,
            &command_pack,
            None,
            sample_hot_path_provenance_inputs(),
        );
        manifest.provenance = Some(provenance.clone());
        write_sample_hot_path_artifacts(tempdir.path(), &manifest.files);
        let counter_capture_summary = HotPathCounterCaptureManifestSummary {
            host_capability_sensitive_captures: vec!["topdown".to_owned()],
            topology_sensitive_captures: vec!["cache_to_cache".to_owned()],
            fallback_tools: vec!["perf-stat".to_owned()],
            fallback_metric_pack: vec!["cache-misses".to_owned()],
            fallback_notes: vec!["cache_to_cache:profiler_safe: perf c2c unavailable".to_owned()],
            raw_output_relpaths: vec!["profiles/perf-c2c.profiler_safe.data".to_owned()],
        };
        let finalized = finalize_hot_path_manifest(
            tempdir.path(),
            manifest,
            Some(counter_capture_summary.clone()),
            vec![HotPathArtifactFile {
                path: HOT_PATH_COMMAND_PACK_NAME.to_owned(),
                bytes: 77,
                sha256: "0".repeat(64),
                description: "command pack".to_owned(),
            }],
        )
        .expect("manifest finalization should succeed");

        let disk_manifest: HotPathArtifactManifest = serde_json::from_str(
            &fs::read_to_string(tempdir.path().join("manifest.json"))
                .expect("disk manifest should be readable"),
        )
        .expect("disk manifest should parse");
        assert!(
            disk_manifest
                .files
                .iter()
                .any(|file| file.path == HOT_PATH_COMMAND_PACK_NAME)
        );
        assert!(
            !disk_manifest
                .files
                .iter()
                .any(|file| file.path == "manifest.json")
        );
        assert_eq!(
            disk_manifest.counter_capture_summary,
            Some(counter_capture_summary.clone())
        );
        assert_eq!(disk_manifest.provenance, Some(provenance.clone()));
        for file in &disk_manifest.files {
            let path = tempdir.path().join(&file.path);
            let bytes = fs::read(&path).expect("artifact file should exist on disk");
            assert_eq!(file.bytes, u64::try_from(bytes.len()).unwrap_or(u64::MAX));
            assert_eq!(file.sha256, hot_path_artifact_sha256(&bytes));
        }
        assert!(
            finalized
                .files
                .iter()
                .any(|file| file.path == "manifest.json" && file.bytes > 0)
        );
        assert!(finalized.files.iter().any(|file| {
            file.path == HOT_PATH_COMMAND_PACK_NAME
                && file.sha256
                    == hot_path_artifact_sha256(b"artifact fixture for command_pack.json")
        }));
        assert_eq!(
            finalized.counter_capture_summary,
            Some(counter_capture_summary)
        );
        assert_eq!(finalized.provenance, Some(provenance));
    }

    #[test]
    fn hot_profile_stdout_manifest_matches_disk_manifest_semantics() {
        let tempdir = tempfile::tempdir().expect("tempdir should succeed");
        let report = sample_hot_path_report();
        let replay_command = HotProfileReplayCommand {
            db: "fixture-a",
            workload: "mixed_read_write",
            golden_dir: Path::new("/tmp/golden"),
            working_base: Path::new("/tmp/working"),
            concurrency: 4,
            seed: 42,
            scale: 50,
            output_dir: Path::new("/tmp/out"),
            mvcc: true,
            run_integrity_check: false,
        };
        let command_pack = build_hot_path_command_pack(&report, &replay_command);
        let counter_capture_summary = build_hot_path_counter_capture_summary(&command_pack);
        let mut manifest = sample_hot_path_manifest();
        write_sample_hot_path_artifacts(tempdir.path(), &manifest.files);
        manifest.provenance = Some(build_hot_path_artifact_provenance(
            &report,
            &command_pack,
            counter_capture_summary.as_ref(),
            sample_hot_path_provenance_inputs(),
        ));
        let finalized = finalize_hot_path_manifest(
            tempdir.path(),
            manifest,
            counter_capture_summary,
            vec![HotPathArtifactFile {
                path: HOT_PATH_COMMAND_PACK_NAME.to_owned(),
                bytes: 77,
                sha256: "0".repeat(64),
                description: "command pack".to_owned(),
            }],
        )
        .expect("manifest finalization should succeed");

        let disk_manifest_json = fs::read_to_string(tempdir.path().join("manifest.json"))
            .expect("disk manifest should be readable");
        let disk_manifest_value: Value =
            serde_json::from_str(&disk_manifest_json).expect("disk manifest should parse");
        let stdout_manifest_json = serde_json::to_string(
            &read_hot_path_disk_manifest(tempdir.path()).expect("stdout manifest should load"),
        )
        .expect("stdout manifest should serialize");
        let stdout_manifest_value: Value =
            serde_json::from_str(&stdout_manifest_json).expect("stdout manifest should parse");
        assert_eq!(stdout_manifest_value, disk_manifest_value);
        assert_ne!(
            stdout_manifest_json,
            serde_json::to_string(&finalized).expect("in-memory manifest should serialize")
        );
    }

    #[test]
    fn finalized_hot_path_manifest_matches_tracked_json_schema() {
        assert_embedded_schema_matches_tracked_file(
            HOT_PATH_MANIFEST_SCHEMA_PATH,
            HOT_PATH_MANIFEST_SCHEMA_RAW,
        );
        let tempdir = tempfile::tempdir().expect("tempdir should succeed");
        let report = sample_hot_path_report();
        let replay_command = HotProfileReplayCommand {
            db: "fixture-a",
            workload: "mixed_read_write",
            golden_dir: Path::new("/tmp/golden"),
            working_base: Path::new("/tmp/working"),
            concurrency: 4,
            seed: 42,
            scale: 50,
            output_dir: Path::new("/tmp/out"),
            mvcc: true,
            run_integrity_check: false,
        };
        let command_pack = build_hot_path_command_pack(&report, &replay_command);
        let counter_capture_summary = HotPathCounterCaptureManifestSummary {
            host_capability_sensitive_captures: vec![
                "topdown".to_owned(),
                "cache_to_cache".to_owned(),
            ],
            topology_sensitive_captures: vec!["cache_to_cache".to_owned()],
            fallback_tools: vec!["perf-stat".to_owned(), "perf-mem".to_owned()],
            fallback_metric_pack: vec!["cache-misses".to_owned()],
            fallback_notes: vec![
                "topdown:profiler_safe: TopdownL1 unsupported on this host".to_owned(),
                "cache_to_cache:profiler_safe: perf c2c unavailable".to_owned(),
            ],
            raw_output_relpaths: vec![
                "profiles/perf-c2c.profiler_safe.data".to_owned(),
                "profiles/perf-mem-remote-access.profiler_safe.data".to_owned(),
            ],
        };
        let mut manifest = sample_hot_path_manifest();
        manifest.provenance = Some(build_hot_path_artifact_provenance(
            &report,
            &command_pack,
            Some(&counter_capture_summary),
            sample_hot_path_provenance_inputs(),
        ));
        write_sample_hot_path_artifacts(tempdir.path(), &manifest.files);
        let finalized = finalize_hot_path_manifest(
            tempdir.path(),
            manifest,
            Some(counter_capture_summary),
            vec![HotPathArtifactFile {
                path: HOT_PATH_COMMAND_PACK_NAME.to_owned(),
                bytes: 77,
                sha256: "0".repeat(64),
                description: "command pack".to_owned(),
            }],
        )
        .expect("manifest finalization should succeed");

        let manifest_raw =
            serde_json::to_string_pretty(&finalized).expect("serialize finalized manifest");
        assert_json_schema_valid(HOT_PATH_MANIFEST_SCHEMA_RAW, &manifest_raw);
    }

    #[test]
    fn hot_path_provenance_capture_matches_shared_artifact_manifest_schema() {
        assert_embedded_schema_matches_tracked_file(
            BENCHMARK_ARTIFACT_MANIFEST_SCHEMA_PATH,
            BENCHMARK_ARTIFACT_MANIFEST_SCHEMA_RAW,
        );
        let workspace_root = workspace_root();
        let campaign = sample_benchmark_campaign();
        let cell = sample_benchmark_cell();

        let report = sample_hot_path_report();
        let fixture_id = cell.fixture_id.clone();
        let replay_command = HotProfileReplayCommand {
            db: &fixture_id,
            workload: &cell.workload,
            golden_dir: Path::new("/tmp/golden"),
            working_base: Path::new("/tmp/working"),
            concurrency: cell.concurrency,
            seed: 42,
            scale: 50,
            output_dir: Path::new("/tmp/out"),
            mvcc: true,
            run_integrity_check: false,
        };
        let command_pack = build_hot_path_command_pack(&report, &replay_command);
        let counter_capture_summary = build_hot_path_counter_capture_summary(&command_pack);
        let provenance = build_hot_path_artifact_provenance(
            &report,
            &command_pack,
            counter_capture_summary.as_ref(),
            sample_hot_path_provenance_inputs(),
        );
        let manifest = build_benchmark_artifact_manifest(
            &workspace_root,
            &campaign,
            &cell,
            BenchmarkArtifactProvenanceCapture {
                run_id: "run-20260315T040100Z".to_owned(),
                retention_class: BenchmarkArtifactRetentionClass::FailureBundle,
                command_entrypoint: provenance.command_entrypoint.clone(),
                source_revision: provenance
                    .source_revision
                    .clone()
                    .expect("sample provenance should include source_revision"),
                beads_data_hash: provenance
                    .beads_data_hash
                    .clone()
                    .expect("sample provenance should include beads_data_hash"),
                kernel_release: provenance.kernel_release.clone(),
                commands: provenance.commands.clone(),
                tool_versions: provenance.tool_versions.clone(),
                fallback_notes: provenance.fallback_notes.clone(),
            },
        )
        .expect("artifact manifest should build from hot-path provenance");

        let manifest_raw =
            serde_json::to_string_pretty(&manifest).expect("serialize artifact manifest");
        assert_json_schema_valid(BENCHMARK_ARTIFACT_MANIFEST_SCHEMA_RAW, &manifest_raw);
    }

    #[test]
    fn test_unknown_subcommand_exits_two() {
        assert_eq!(run_with(&["realdb-e2e", "bogus"]), 2);
    }

    #[test]
    fn parse_u16_list_single_and_list() {
        assert_eq!(parse_u16_list("1").unwrap(), vec![1]);
        assert_eq!(parse_u16_list("1,2,4,8,16").unwrap(), vec![1, 2, 4, 8, 16]);
        assert!(parse_u16_list("0").is_err());
        assert!(parse_u16_list("1,0,2").is_err());
        assert!(parse_u16_list("").is_err());
        assert!(parse_u16_list("1,").is_err());
        assert!(parse_u16_list("nope").is_err());
    }

    #[test]
    fn test_corpus_no_action_exits_two() {
        assert_eq!(run_with(&["realdb-e2e", "corpus"]), 2);
    }

    #[test]
    fn test_corpus_help_exits_zero() {
        assert_eq!(run_with(&["realdb-e2e", "corpus", "--help"]), 0);
    }

    #[test]
    fn test_corpus_scan_help() {
        assert_eq!(run_with(&["realdb-e2e", "corpus", "scan", "--help"]), 0);
    }

    #[test]
    fn test_run_help() {
        assert_eq!(run_with(&["realdb-e2e", "run", "--help"]), 0);
    }

    #[test]
    fn test_bench_help() {
        assert_eq!(run_with(&["realdb-e2e", "bench", "--help"]), 0);
    }

    #[test]
    fn canonical_bench_defaults_match_checked_in_campaign_manifest() {
        let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = crate_dir
            .parent()
            .and_then(Path::parent)
            .expect("workspace root should exist")
            .to_path_buf();

        let defaults =
            canonical_bench_defaults(&workspace_root).expect("canonical defaults should load");

        assert_eq!(
            defaults.golden_dir,
            workspace_root.join("sample_sqlite_db_files/working/beads_bench_20260310/golden")
        );
        assert_eq!(
            defaults.fixture_ids,
            vec![
                "frankensqlite".to_owned(),
                "frankentui".to_owned(),
                "frankensearch".to_owned(),
            ]
        );
        assert_eq!(
            defaults.fixture_paths.get("frankensqlite"),
            Some(&workspace_root.join(
                "sample_sqlite_db_files/working/beads_bench_20260310/golden/frankensqlite_beads.db"
            ))
        );
        assert_eq!(
            defaults.fixture_paths.get("frankentui"),
            Some(&workspace_root.join(
                "sample_sqlite_db_files/working/beads_bench_20260310/golden/frankentui_beads.db"
            ))
        );
        assert_eq!(
            defaults.fixture_paths.get("frankensearch"),
            Some(&workspace_root.join(
                "sample_sqlite_db_files/working/beads_bench_20260310/golden/frankensearch_beads.db"
            ))
        );
        assert_eq!(defaults.presets, default_bench_presets());
        assert_eq!(defaults.concurrency, vec![1, 4, 8]);
    }

    #[test]
    fn find_bench_workspace_root_walks_up_to_campaign_manifest() {
        let dir = tempfile::tempdir().expect("tempdir should succeed");
        let nested = dir.path().join("a/b/c");
        fs::create_dir_all(&nested).expect("nested dirs should be created");
        let manifest_dir = dir.path().join("sample_sqlite_db_files/manifests");
        fs::create_dir_all(&manifest_dir).expect("manifest dir should be created");
        fs::write(manifest_dir.join("beads_benchmark_campaign.v1.json"), "{}")
            .expect("sentinel manifest should be written");

        assert_eq!(
            find_bench_workspace_root(&nested),
            Some(dir.path().to_path_buf())
        );
    }

    #[test]
    fn resolve_hot_path_workspace_root_candidate_accepts_root_or_nested_path() {
        let dir = tempfile::tempdir().expect("tempdir should succeed");
        let nested = dir.path().join("a/b/c");
        fs::create_dir_all(&nested).expect("nested dirs should be created");
        let manifest_dir = dir.path().join("sample_sqlite_db_files/manifests");
        fs::create_dir_all(&manifest_dir).expect("manifest dir should be created");
        fs::write(manifest_dir.join("beads_benchmark_campaign.v1.json"), "{}")
            .expect("sentinel manifest should be written");

        assert_eq!(
            resolve_hot_path_workspace_root_candidate(dir.path()),
            Some(dir.path().to_path_buf())
        );
        assert_eq!(
            resolve_hot_path_workspace_root_candidate(&nested),
            Some(dir.path().to_path_buf())
        );
        assert_eq!(
            resolve_hot_path_workspace_root_candidate(&dir.path().join("missing")),
            None
        );
    }

    #[test]
    fn test_corrupt_help() {
        assert_eq!(run_with(&["realdb-e2e", "corrupt", "--help"]), 0);
    }

    #[test]
    fn test_run_parses_all_options() {
        // Use a temporary on-disk database so the test is hermetic and does
        // not depend on any specific golden fixture being present.
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db_path = tmp.path().to_str().unwrap().to_owned();
        rusqlite::Connection::open(&db_path)
            .unwrap()
            .execute_batch("CREATE TABLE seed (id INTEGER PRIMARY KEY);")
            .unwrap();

        let os_args = vec![
            OsString::from("realdb-e2e"),
            OsString::from("run"),
            OsString::from("--engine"),
            OsString::from("sqlite3"),
            OsString::from("--db"),
            OsString::from(db_path),
            OsString::from("--workload"),
            OsString::from("commutative_inserts_disjoint_keys"),
            OsString::from("--concurrency"),
            OsString::from("2"),
        ];
        assert_eq!(run_cli(os_args), 0);
    }

    #[test]
    fn test_run_accepts_no_mvcc_flag() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db_path = tmp.path().to_str().unwrap().to_owned();
        rusqlite::Connection::open(&db_path)
            .unwrap()
            .execute_batch("CREATE TABLE seed (id INTEGER PRIMARY KEY);")
            .unwrap();

        let os_args = vec![
            OsString::from("realdb-e2e"),
            OsString::from("run"),
            OsString::from("--engine"),
            OsString::from("sqlite3"),
            OsString::from("--db"),
            OsString::from(db_path),
            OsString::from("--workload"),
            OsString::from("commutative_inserts_disjoint_keys"),
            OsString::from("--no-mvcc"),
        ];
        assert_eq!(run_cli(os_args), 0);
    }

    #[test]
    fn test_run_profile_only_outputs_suppressed_environment_record() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db_path = tmp.path().to_str().unwrap().to_owned();
        rusqlite::Connection::open(&db_path)
            .unwrap()
            .execute_batch("CREATE TABLE seed (id INTEGER PRIMARY KEY);")
            .unwrap();
        let output = tempfile::NamedTempFile::new().unwrap();

        let os_args = vec![
            OsString::from("realdb-e2e"),
            OsString::from("run"),
            OsString::from("--engine"),
            OsString::from("sqlite3"),
            OsString::from("--db"),
            OsString::from(db_path),
            OsString::from("--workload"),
            OsString::from("commutative_inserts_disjoint_keys"),
            OsString::from("--profile-only"),
            OsString::from("--output-jsonl"),
            OsString::from(output.path()),
        ];
        assert_eq!(run_cli(os_args), 0);

        let text = fs::read_to_string(output.path()).expect("profile output should exist");
        let first_line = text
            .lines()
            .next()
            .expect("profile output should be non-empty");
        let value: Value = serde_json::from_str(first_line).expect("profile line must be JSON");
        assert_eq!(
            value["schema_version"],
            fsqlite_e2e::report::RUN_RECORD_SCHEMA_V1
        );
        assert_eq!(value["environment"]["capture_mode"], "suppressed");
        assert_eq!(
            value["report"]["correctness"]["integrity_check_ok"],
            Value::Null
        );
    }

    #[test]
    fn test_run_granular_skip_flags_work_without_profile_only() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let db_path = tmp.path().to_str().unwrap().to_owned();
        rusqlite::Connection::open(&db_path)
            .unwrap()
            .execute_batch("CREATE TABLE seed (id INTEGER PRIMARY KEY);")
            .unwrap();
        let output = tempfile::NamedTempFile::new().unwrap();

        let os_args = vec![
            OsString::from("realdb-e2e"),
            OsString::from("run"),
            OsString::from("--engine"),
            OsString::from("sqlite3"),
            OsString::from("--db"),
            OsString::from(db_path),
            OsString::from("--workload"),
            OsString::from("commutative_inserts_disjoint_keys"),
            OsString::from("--skip-integrity-check"),
            OsString::from("--skip-environment-metadata"),
            OsString::from("--output-jsonl"),
            OsString::from(output.path()),
        ];
        assert_eq!(run_cli(os_args), 0);

        let text = fs::read_to_string(output.path()).expect("profile output should exist");
        let first_line = text
            .lines()
            .next()
            .expect("profile output should be non-empty");
        let value: Value = serde_json::from_str(first_line).expect("profile line must be JSON");
        assert_eq!(
            value["schema_version"],
            fsqlite_e2e::report::RUN_RECORD_SCHEMA_V1
        );
        assert_eq!(value["environment"]["capture_mode"], "suppressed");
        assert_eq!(
            value["report"]["correctness"]["integrity_check_ok"],
            Value::Null
        );
    }

    #[test]
    fn test_corpus_scan_runs_against_tmp() {
        // Scan an empty temp dir — should find 0 candidates.
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "scan",
                "--root",
                dir.path().to_str().unwrap(),
            ]),
            0
        );
    }

    #[test]
    fn test_corpus_scan_json_and_filters() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("not_sqlite.db"), b"nope").unwrap();

        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "scan",
                "--root",
                dir.path().to_str().unwrap(),
                "--json",
            ]),
            0
        );
        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "scan",
                "--root",
                dir.path().to_str().unwrap(),
                "--require-header-ok",
            ]),
            0
        );
        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "scan",
                "--root",
                dir.path().to_str().unwrap(),
                "--min-bytes",
                "9999999",
            ]),
            0
        );
    }

    // ── corpus verify tests ────────────────────────────────────────────

    #[test]
    fn test_verify_all_match() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        // Create a test file.
        let content = b"hello golden world";
        fs::write(golden.join("test.db"), content).unwrap();

        // Compute expected sha256.
        let expected = format!("{:x}", Sha256::digest(content));

        // Write checksums file.
        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, format!("{expected}  test.db\n")).unwrap();

        let report = verify_golden_checksums(&checksums, &golden).unwrap();
        assert_eq!(report.summary.ok, 1);
        assert_eq!(report.summary.mismatch, 0);
        assert_eq!(report.summary.missing, 0);
        assert_eq!(report.summary.error, 0);
        assert_eq!(report.summary.extra, 0);
    }

    #[test]
    fn test_verify_mismatch_detected() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        fs::write(golden.join("bad.db"), b"actual content").unwrap();

        let checksums = dir.path().join("checksums.sha256");
        let wrong_hash = "0".repeat(64);
        fs::write(&checksums, format!("{wrong_hash}  bad.db\n")).unwrap();

        let report = verify_golden_checksums(&checksums, &golden).unwrap();
        assert_eq!(report.summary.ok, 0);
        assert_eq!(report.summary.mismatch, 1);
        assert_eq!(report.summary.missing, 0);
        assert_eq!(report.summary.error, 0);
        assert_eq!(report.summary.extra, 0);
    }

    #[test]
    fn test_verify_missing_file_detected() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        let checksums = dir.path().join("checksums.sha256");
        let hash = "0".repeat(64);
        fs::write(&checksums, format!("{hash}  nonexistent.db\n")).unwrap();

        let report = verify_golden_checksums(&checksums, &golden).unwrap();
        assert_eq!(report.summary.ok, 0);
        assert_eq!(report.summary.mismatch, 0);
        assert_eq!(report.summary.missing, 1);
        assert_eq!(report.summary.error, 0);
        assert_eq!(report.summary.extra, 0);
    }

    #[test]
    fn test_verify_empty_checksums_file() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, "\n").unwrap();

        let report = verify_golden_checksums(&checksums, &golden).unwrap();
        assert_eq!(report.summary.ok, 0);
        assert_eq!(report.summary.mismatch, 0);
        assert_eq!(report.summary.missing, 0);
        assert_eq!(report.summary.error, 0);
        assert_eq!(report.summary.extra, 0);
    }

    #[test]
    fn test_verify_multiple_files() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        let a_content = b"file a";
        let b_content = b"file b";
        fs::write(golden.join("a.db"), a_content).unwrap();
        fs::write(golden.join("b.db"), b_content).unwrap();

        let a_hash = format!("{:x}", Sha256::digest(a_content));
        let b_hash = format!("{:x}", Sha256::digest(b_content));

        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, format!("{a_hash}  a.db\n{b_hash}  b.db\n")).unwrap();

        let report = verify_golden_checksums(&checksums, &golden).unwrap();
        assert_eq!(report.summary.ok, 2);
        assert_eq!(report.summary.mismatch, 0);
        assert_eq!(report.summary.missing, 0);
        assert_eq!(report.summary.error, 0);
        assert_eq!(report.summary.extra, 0);
    }

    #[test]
    fn test_verify_via_cli() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        let content = b"cli test data";
        fs::write(golden.join("x.db"), content).unwrap();

        let hash = format!("{:x}", Sha256::digest(content));
        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, format!("{hash}  x.db\n")).unwrap();

        // Test via CLI interface.
        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "verify",
                "--checksums",
                checksums.to_str().unwrap(),
                "--golden-dir",
                golden.to_str().unwrap(),
            ]),
            0
        );
    }

    #[test]
    fn test_verify_via_cli_mismatch_exits_one() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        fs::write(golden.join("y.db"), b"content").unwrap();

        let checksums = dir.path().join("checksums.sha256");
        let wrong = "f".repeat(64);
        fs::write(&checksums, format!("{wrong}  y.db\n")).unwrap();

        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "verify",
                "--checksums",
                checksums.to_str().unwrap(),
                "--golden-dir",
                golden.to_str().unwrap(),
            ]),
            1
        );
    }

    #[test]
    fn test_verify_extra_file_detected() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        // Expected file.
        let content = b"expected";
        fs::write(golden.join("a.db"), content).unwrap();
        let hash = format!("{:x}", Sha256::digest(content));

        // Extra file on disk, not in checksums.
        fs::write(golden.join("extra.db"), b"extra").unwrap();

        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, format!("{hash}  a.db\n")).unwrap();

        let report = verify_golden_checksums(&checksums, &golden).unwrap();
        assert_eq!(report.summary.ok, 1);
        assert_eq!(report.summary.mismatch, 0);
        assert_eq!(report.summary.missing, 0);
        assert_eq!(report.summary.error, 0);
        assert_eq!(report.summary.extra, 1);
        assert!(
            report.files.iter().any(|f| f.status == VerifyStatus::Extra),
            "must include at least one EXTRA result"
        );
    }

    #[test]
    fn test_verify_ignores_dotfiles_and_sqlite_sidecars_in_golden_dir() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        // Expected file.
        let content = b"expected";
        fs::write(golden.join("a.db"), content).unwrap();
        let hash = format!("{:x}", Sha256::digest(content));

        // Dotfiles and sidecars are expected to exist locally and should not break verification.
        fs::write(golden.join(".gitignore"), b"*").unwrap();
        fs::write(golden.join(".gitkeep"), b"").unwrap();
        fs::write(golden.join("a.db-wal"), b"").unwrap();
        fs::write(golden.join("a.db-shm"), b"").unwrap();
        fs::write(golden.join("a.db-journal"), b"").unwrap();

        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, format!("{hash}  a.db\n")).unwrap();

        let report = verify_golden_checksums(&checksums, &golden).unwrap();
        assert_eq!(report.summary.ok, 1);
        assert_eq!(report.summary.mismatch, 0);
        assert_eq!(report.summary.missing, 0);
        assert_eq!(report.summary.error, 0);
        assert_eq!(report.summary.extra, 0);
    }

    #[test]
    fn test_verify_via_cli_extra_exits_one() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        let content = b"expected";
        fs::write(golden.join("a.db"), content).unwrap();
        fs::write(golden.join("extra.db"), b"extra").unwrap();

        let hash = format!("{:x}", Sha256::digest(content));
        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, format!("{hash}  a.db\n")).unwrap();

        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "verify",
                "--checksums",
                checksums.to_str().unwrap(),
                "--golden-dir",
                golden.to_str().unwrap(),
            ]),
            1
        );
    }

    #[test]
    fn test_verify_via_cli_missing_checksums_exits_two() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        let missing_checksums = dir.path().join("does_not_exist.sha256");
        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "verify",
                "--checksums",
                missing_checksums.to_str().unwrap(),
                "--golden-dir",
                golden.to_str().unwrap(),
            ]),
            2
        );
    }

    #[test]
    fn test_verify_via_cli_json_flag() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        let content = b"cli json test data";
        fs::write(golden.join("x.db"), content).unwrap();

        let hash = format!("{:x}", Sha256::digest(content));
        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, format!("{hash}  x.db\n")).unwrap();

        assert_eq!(
            run_with(&[
                "realdb-e2e",
                "corpus",
                "verify",
                "--json",
                "--checksums",
                checksums.to_str().unwrap(),
                "--golden-dir",
                golden.to_str().unwrap(),
            ]),
            0
        );
    }

    #[test]
    fn test_verify_report_serializes_to_json() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        fs::create_dir(&golden).unwrap();

        let content = b"json serialize";
        fs::write(golden.join("x.db"), content).unwrap();

        let hash = format!("{:x}", Sha256::digest(content));
        let checksums = dir.path().join("checksums.sha256");
        fs::write(&checksums, format!("{hash}  x.db\n")).unwrap();

        let report = verify_golden_checksums(&checksums, &golden).unwrap();
        let text = serde_json::to_string(&report).unwrap();
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["summary"]["ok"], 1);
    }

    #[test]
    fn test_sha256_file_computes_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.bin");
        fs::write(&path, b"hello").unwrap();

        let result = sha256_file(&path).unwrap();
        // Known sha256 of "hello".
        assert_eq!(
            result,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    // ── sanitize_db_id tests ─────────────────────────────────────────────

    #[test]
    fn test_sanitize_db_id_basic() {
        assert_eq!(sanitize_db_id("beads").unwrap(), "beads");
        assert_eq!(sanitize_db_id("my-project").unwrap(), "my_project");
        assert_eq!(sanitize_db_id("MY_DB").unwrap(), "my_db");
    }

    #[test]
    fn test_sanitize_db_id_trims_underscores() {
        assert_eq!(sanitize_db_id("__foo__").unwrap(), "foo");
        assert_eq!(sanitize_db_id("  hello  ").unwrap(), "hello");
    }

    #[test]
    fn test_sanitize_db_id_rejects_empty() {
        assert!(sanitize_db_id("").is_err());
        assert!(sanitize_db_id("   ").is_err());
        assert!(sanitize_db_id("___").is_err());
    }

    // ── upsert_checksum tests ────────────────────────────────────────────

    #[test]
    fn test_upsert_checksum_creates_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let checksums = dir.path().join("checksums.sha256");
        let golden = dir.path().join("test.db");
        fs::write(&golden, b"data").unwrap();

        let hash = "a".repeat(64);
        upsert_checksum(&checksums, &golden, &hash).unwrap();

        let content = fs::read_to_string(&checksums).unwrap();
        assert!(content.contains(&format!("{hash}  test.db")));
    }

    #[test]
    fn test_upsert_checksum_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let checksums = dir.path().join("checksums.sha256");
        let golden = dir.path().join("test.db");
        fs::write(&golden, b"data").unwrap();

        let hash = "b".repeat(64);
        upsert_checksum(&checksums, &golden, &hash).unwrap();
        upsert_checksum(&checksums, &golden, &hash).unwrap();

        let content = fs::read_to_string(&checksums).unwrap();
        // Only one entry, not duplicated.
        assert_eq!(content.matches("test.db").count(), 1);
    }

    #[test]
    fn test_upsert_checksum_refuses_hash_change() {
        let dir = tempfile::tempdir().unwrap();
        let checksums = dir.path().join("checksums.sha256");
        let golden = dir.path().join("test.db");
        fs::write(&golden, b"data").unwrap();

        let hash1 = "c".repeat(64);
        let hash2 = "d".repeat(64);
        upsert_checksum(&checksums, &golden, &hash1).unwrap();
        let err = upsert_checksum(&checksums, &golden, &hash2);
        assert!(err.is_err(), "must refuse to overwrite existing hash");
        assert!(
            err.unwrap_err().contains("Refusing to overwrite"),
            "error message should mention immutability"
        );
    }

    #[test]
    fn test_upsert_checksum_maintains_sorted_order() {
        let dir = tempfile::tempdir().unwrap();
        let checksums = dir.path().join("checksums.sha256");

        let golden_b = dir.path().join("beta.db");
        let golden_a = dir.path().join("alpha.db");
        fs::write(&golden_b, b"b").unwrap();
        fs::write(&golden_a, b"a").unwrap();

        let hash_b = "b".repeat(64);
        let hash_a = "a".repeat(64);

        // Insert b first, then a.
        upsert_checksum(&checksums, &golden_b, &hash_b).unwrap();
        upsert_checksum(&checksums, &golden_a, &hash_a).unwrap();

        let content = fs::read_to_string(&checksums).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].contains("alpha.db"),
            "alpha.db must come first (sorted)"
        );
        assert!(
            lines[1].contains("beta.db"),
            "beta.db must come second (sorted)"
        );
    }

    // ── backup_sqlite_file tests ─────────────────────────────────────────

    #[test]
    fn test_backup_sqlite_file_produces_valid_copy() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("source.db");
        let dst = dir.path().join("backup.db");

        let conn = Connection::open(&src).unwrap();
        conn.execute_batch(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);
             INSERT INTO items VALUES (1, 'alpha');
             INSERT INTO items VALUES (2, 'beta');",
        )
        .unwrap();
        drop(conn);

        backup_sqlite_file(&src, &dst).unwrap();

        // The backup must be a valid SQLite database with the same data.
        let conn = Connection::open_with_flags(&dst, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM items", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 2);

        let name: String = conn
            .query_row("SELECT name FROM items WHERE id=1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(name, "alpha");
    }

    #[test]
    fn test_backup_passes_integrity_check() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("source.db");
        let dst = dir.path().join("backup.db");

        let conn = Connection::open(&src).unwrap();
        conn.execute_batch(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, data BLOB);
             INSERT INTO t VALUES (1, randomblob(1000));
             INSERT INTO t VALUES (2, randomblob(1000));",
        )
        .unwrap();
        drop(conn);

        backup_sqlite_file(&src, &dst).unwrap();
        sqlite_integrity_check(&dst).unwrap();
    }

    // ── corpus import end-to-end ─────────────────────────────────────────

    #[test]
    fn test_corpus_import_end_to_end() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        let metadata = dir.path().join("metadata");
        fs::create_dir(&golden).unwrap();
        fs::create_dir(&metadata).unwrap();

        // Create a source database.
        let src = dir.path().join("source.db");
        let conn = Connection::open(&src).unwrap();
        conn.execute_batch(
            "CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
             INSERT INTO widgets VALUES (1, 'gear');
             INSERT INTO widgets VALUES (2, 'cog');",
        )
        .unwrap();
        drop(conn);

        let checksums = dir.path().join("checksums.sha256");

        let exit_code = run_with(&[
            "realdb-e2e",
            "corpus",
            "import",
            "--db",
            src.to_str().unwrap(),
            "--id",
            "test_import",
            "--pii-risk",
            "unlikely",
            "--secrets-risk",
            "unlikely",
            "--golden-dir",
            golden.to_str().unwrap(),
            "--metadata-dir",
            metadata.to_str().unwrap(),
            "--checksums",
            checksums.to_str().unwrap(),
        ]);
        assert_eq!(exit_code, 0, "import must succeed");

        // Golden copy must exist.
        let golden_db = golden.join("test_import.db");
        assert!(golden_db.exists(), "golden DB must be created");

        // Checksums file must exist and contain the entry.
        assert!(checksums.exists(), "checksums file must be created");
        let checksums_content = fs::read_to_string(&checksums).unwrap();
        assert!(
            checksums_content.contains("test_import.db"),
            "checksums must reference the golden file"
        );

        // Metadata JSON must exist and have correct fields.
        let meta_path = metadata.join("test_import.json");
        assert!(meta_path.exists(), "metadata JSON must be created");
        let meta_json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&meta_path).unwrap()).unwrap();
        assert_eq!(
            meta_json["schema_version"].as_u64().unwrap(),
            u64::from(FIXTURE_METADATA_SCHEMA_VERSION_V1)
        );
        assert_eq!(meta_json["db_id"], "test_import");
        assert_eq!(meta_json["golden_filename"], "test_import.db");
        assert_eq!(meta_json["safety"]["pii_risk"], "unlikely");
        assert_eq!(meta_json["safety"]["secrets_risk"], "unlikely");
        assert_eq!(meta_json["safety"]["allowed_for_ci"], true);

        assert!(meta_json["sqlite_meta"]["page_size"].as_u64().unwrap() > 0);
        assert!(meta_json["size_bytes"].as_u64().unwrap() > 0);
        assert_eq!(meta_json["tables"][0]["name"], "widgets");
        assert_eq!(meta_json["tables"][0]["row_count"], 2);

        // Golden copy must pass integrity check.
        sqlite_integrity_check(&golden_db).unwrap();

        // Checksums hash must match the actual golden file.
        let actual_hash = sha256_file(&golden_db).unwrap();
        assert_eq!(meta_json["sha256_golden"], actual_hash);
        assert!(
            checksums_content.contains(&actual_hash),
            "checksums hash must match actual file"
        );
    }

    #[test]
    fn test_corpus_import_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        let metadata = dir.path().join("metadata");
        fs::create_dir(&golden).unwrap();
        fs::create_dir(&metadata).unwrap();

        let src = dir.path().join("source.db");
        let conn = Connection::open(&src).unwrap();
        conn.execute_batch("CREATE TABLE t (x INTEGER);").unwrap();
        drop(conn);

        let checksums = dir.path().join("checksums.sha256");

        let args = &[
            "realdb-e2e",
            "corpus",
            "import",
            "--db",
            src.to_str().unwrap(),
            "--id",
            "idempotent_test",
            "--golden-dir",
            golden.to_str().unwrap(),
            "--metadata-dir",
            metadata.to_str().unwrap(),
            "--checksums",
            checksums.to_str().unwrap(),
        ];

        // First import.
        assert_eq!(run_with(args), 0);

        // Second import (same fixture) should also succeed.
        assert_eq!(run_with(args), 0);

        // Only one entry in checksums.
        let content = fs::read_to_string(&checksums).unwrap();
        assert_eq!(
            content.matches("idempotent_test.db").count(),
            1,
            "idempotent re-import must not duplicate checksum"
        );
    }

    #[test]
    fn test_corpus_import_no_metadata_flag() {
        let dir = tempfile::tempdir().unwrap();
        let golden = dir.path().join("golden");
        let metadata = dir.path().join("metadata");
        fs::create_dir(&golden).unwrap();
        fs::create_dir(&metadata).unwrap();

        let src = dir.path().join("source.db");
        let conn = Connection::open(&src).unwrap();
        conn.execute_batch("CREATE TABLE t (x INTEGER);").unwrap();
        drop(conn);

        let checksums = dir.path().join("checksums.sha256");

        let exit_code = run_with(&[
            "realdb-e2e",
            "corpus",
            "import",
            "--db",
            src.to_str().unwrap(),
            "--id",
            "no_meta",
            "--golden-dir",
            golden.to_str().unwrap(),
            "--metadata-dir",
            metadata.to_str().unwrap(),
            "--checksums",
            checksums.to_str().unwrap(),
            "--no-metadata",
        ]);
        assert_eq!(exit_code, 0);

        assert!(golden.join("no_meta.db").exists());
        assert!(
            !metadata.join("no_meta.json").exists(),
            "metadata must NOT be written with --no-metadata"
        );
    }
}
