#!/usr/bin/env bash
# verify_bd_db300_4_1_hot_path_profiles.sh — inline hot-path profile campaign
#
# Captures current Track D / D1 evidence from the inline `realdb-e2e hot-profile`
# path instead of scraping archived `perf.data` files. The script runs the
# mixed_read_write hot-path profile across the pinned Beads fixture copies in
# both MVCC and forced single-writer modes, then aggregates the per-run
# `profile.json` + `actionable_ranking.json` bundles into a bead-level ledger.
# Default profile is `release` for worker survivability; override with
# `CARGO_PROFILE=release-perf` when the worker fleet has headroom for it.
#
# Outputs:
#   artifacts/perf/bd-db300.4.1/inline/
#     events.jsonl
#     run_records.jsonl
#     opcode_profile_packs.json
#     subsystem_profile_packs.json
#     command_packs.json
#     scenario_profiles.json
#     actionable_ranking.json
#     benchmark_context.json
#     report.json
#     summary.md
#     runs/<scenario>__<fixture>/{profile.json,opcode_profile.json,subsystem_profile.json,actionable_ranking.json,command_pack.json,summary.md,manifest.json}

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEAD_ID="${BEAD_ID:-bd-db300.4.1}"
SCRIPT_ENTRYPOINT="${SCRIPT_ENTRYPOINT:-scripts/verify_bd_db300_4_1_hot_path_profiles.sh}"
RUN_ID="${BEAD_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_ID_TARGET_SAFE="${RUN_ID//[^[:alnum:]]/_}"
GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
SOURCE_GOLDEN_DIR_DEFAULT="${WORKSPACE_ROOT}/sample_sqlite_db_files/working/beads_bench_20260310/golden"
SOURCE_GOLDEN_DIR="${GOLDEN_DIR:-${SOURCE_GOLDEN_DIR_DEFAULT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/artifacts/perf/${BEAD_ID}/inline}"
RUNS_DIR="${OUTPUT_DIR}/runs"
SYNC_GOLDEN_DIR="${SYNC_GOLDEN_DIR:-${OUTPUT_DIR}/golden}"
LOG_FILE="${OUTPUT_DIR}/events.jsonl"
RUN_RECORDS_JSONL="${OUTPUT_DIR}/run_records.jsonl"
OPCODE_PROFILE_PACKS_JSON="${OUTPUT_DIR}/opcode_profile_packs.json"
SUBSYSTEM_PROFILE_PACKS_JSON="${OUTPUT_DIR}/subsystem_profile_packs.json"
COMMAND_PACKS_JSON="${OUTPUT_DIR}/command_packs.json"
SCENARIO_PROFILES_JSON="${OUTPUT_DIR}/scenario_profiles.json"
ACTIONABLE_RANKING_JSON="${OUTPUT_DIR}/actionable_ranking.json"
BENCHMARK_CONTEXT_JSON="${OUTPUT_DIR}/benchmark_context.json"
REPORT_JSON="${OUTPUT_DIR}/report.json"
SUMMARY_MD="${OUTPUT_DIR}/summary.md"
WORKLOAD_ID="${WORKLOAD_ID:-mixed_read_write}"
CONCURRENCY="${CONCURRENCY:-4}"
SEED="${SEED:-42}"
SCALE="${SCALE:-50}"
CARGO_PROFILE="${CARGO_PROFILE:-release}"
BEAD_ID_TARGET_SAFE="${BEAD_ID//[^[:alnum:]]/_}"
RCH_TARGET_DIR="${RCH_TARGET_DIR:-/tmp/rch_target_${BEAD_ID_TARGET_SAFE}}"
HOT_PATH_PROFILE_SCHEMA="fsqlite-e2e.hot_path_profile.v1"
HOT_PATH_PROFILE_MANIFEST_SCHEMA="fsqlite-e2e.hot_path_profile_manifest.v1"
HOT_PATH_OPCODE_PROFILE_SCHEMA="fsqlite-e2e.hot_path_opcode_profile.v1"
HOT_PATH_SUBSYSTEM_PROFILE_SCHEMA="fsqlite-e2e.hot_path_subsystem_profile.v1"
HOT_PATH_ACTIONABLE_RANKING_SCHEMA="fsqlite-e2e.hot_path_actionable_ranking.v3"
HOT_PATH_COMMAND_PACK_SCHEMA="fsqlite-e2e.hot_path_command_pack.v2"
HOT_PATH_CAMPAIGN_RANKING_SCHEMA="fsqlite-e2e.hot_path_campaign_ranking.v2"
HOT_PATH_INLINE_BUNDLE_SCHEMA="fsqlite-e2e.hot_path_inline_bundle.v1"
HOT_PATH_INLINE_BUNDLE_PREFIX="HOT_PATH_INLINE_BUNDLE_JSON="
BEADS_DATA_PATH="${WORKSPACE_ROOT}/.beads/issues.jsonl"
FSQLITE_HOT_PATH_BEAD_ID="${FSQLITE_HOT_PATH_BEAD_ID:-${BEAD_ID}}"
FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH="${FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH:-sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.json}"
FSQLITE_HOT_PATH_CARGO_PROFILE="${FSQLITE_HOT_PATH_CARGO_PROFILE:-${CARGO_PROFILE}}"
FSQLITE_HOT_PATH_WORKSPACE_ROOT="${FSQLITE_HOT_PATH_WORKSPACE_ROOT:-${WORKSPACE_ROOT}}"
FSQLITE_HOT_PATH_SOURCE_REVISION="${FSQLITE_HOT_PATH_SOURCE_REVISION:-$(git -C "${WORKSPACE_ROOT}" rev-parse HEAD)}"
FSQLITE_HOT_PATH_BEADS_DATA_HASH="${FSQLITE_HOT_PATH_BEADS_DATA_HASH:-$(sha256sum "${BEADS_DATA_PATH}" | awk '{print $1}')}"
FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID="${FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID:-}"
FSQLITE_HOT_PATH_HARDWARE_CLASS_ID="${FSQLITE_HOT_PATH_HARDWARE_CLASS_ID:-}"
FSQLITE_HOT_PATH_HARDWARE_SIGNATURE="${FSQLITE_HOT_PATH_HARDWARE_SIGNATURE:-}"
CAMPAIGN_MANIFEST_FILE=""

mkdir -p "${RUNS_DIR}"
: > "${LOG_FILE}"
: > "${RUN_RECORDS_JSONL}"
: > "${OPCODE_PROFILE_PACKS_JSON}"
: > "${SUBSYSTEM_PROFILE_PACKS_JSON}"
: > "${COMMAND_PACKS_JSON}"
: > "${SCENARIO_PROFILES_JSON}"
: > "${ACTIONABLE_RANKING_JSON}"
: > "${BENCHMARK_CONTEXT_JSON}"
: > "${REPORT_JSON}"
: > "${SUMMARY_MD}"

log_event() {
    local level="$1"
    local stage="$2"
    local message="$3"
    printf '{"run_id":"%s","bead_id":"%s","level":"%s","stage":"%s","message":"%s","ts":"%s"}\n' \
        "${RUN_ID}" "${BEAD_ID}" "${level}" "${stage}" "${message}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "${LOG_FILE}"
}

fail() {
    local stage="$1"
    local message="$2"
    log_event "ERROR" "${stage}" "${message}"
    echo "ERROR: ${message}" >&2
    exit 1
}

require_dir() {
    local path="$1"
    [[ -d "${path}" ]] || fail "inputs" "missing required directory: ${path}"
}

require_file() {
    local path="$1"
    [[ -f "${path}" ]] || fail "inputs" "missing required file: ${path}"
}

require_nonempty_file() {
    local path="$1"
    [[ -s "${path}" ]] || fail "inputs" "missing or empty required file: ${path}"
}

resolve_campaign_manifest_file() {
    if [[ "${FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH}" = /* ]]; then
        printf '%s\n' "${FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH}"
    else
        printf '%s\n' "${WORKSPACE_ROOT}/${FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH}"
    fi
}

resolve_hot_path_microarchitectural_overrides() {
    jq -r --arg row_id "${WORKLOAD_ID}_c${CONCURRENCY}" '
        (.matrix_rows[] | select(.row_id == $row_id) | .placement_variants
            | (map(select(.placement_profile_id == "baseline_unpinned"))[0]
                // map(select(.required))[0]
                // .[0])) as $variant
        | (.hardware_classes[] | select(.id == $variant.hardware_class_id)) as $hardware
        | [
            $variant.placement_profile_id,
            $variant.hardware_class_id,
            "\($hardware.id_fields.os_family):\($hardware.id_fields.cpu_arch):\($hardware.id_fields.topology_class)"
        ]
        | @tsv
    ' "${CAMPAIGN_MANIFEST_FILE}"
}

require_json_schema() {
    local path="$1"
    local schema="$2"

    require_nonempty_file "${path}"
    jq -e --arg schema "${schema}" '
        type == "object" and .schema_version == $schema
    ' "${path}" > /dev/null \
        || fail "inputs" "invalid schema in ${path}; expected ${schema}"
}

require_topology_counter_pack_contract() {
    local path="$1"

    require_nonempty_file "${path}"
    jq -e '
        def require_plain_capture($capture; $tool):
            ([.commands[] | select(.capture == $capture)] | length == 2)
            and all(.commands[] | select(.capture == $capture);
                .tool == $tool and .counter_pack == null
            );
        def require_topdown:
            ([.commands[] | select(.capture == "topdown")] | length == 2)
            and all(.commands[] | select(.capture == "topdown");
                .tool == "perf-stat"
                and (.counter_pack != null)
                and .counter_pack.host_capability_sensitive
                and (.counter_pack.topology_sensitive == false)
                and .counter_pack.primary_tool == "perf-stat"
                and .counter_pack.primary_selection == "TopdownL1"
                and (.counter_pack.fallback_tools == ["perf-stat"])
                and (.counter_pack.fallback_selections | index("core_event_fallback") != null)
                and (.counter_pack.fallback_event_pack
                    == ["cycles","instructions","branches","branch-misses","cache-references","cache-misses"])
                and (.counter_pack.fallback_reason_hints
                    | index("TopdownL1 unsupported on this host") != null)
                and (.counter_pack.raw_output_relpaths == [])
            );
        def require_cache_to_cache:
            ([.commands[] | select(.capture == "cache_to_cache")] | length == 2)
            and all(.commands[] | select(.capture == "cache_to_cache");
                .tool == "perf-c2c"
                and (.counter_pack != null)
                and .counter_pack.host_capability_sensitive
                and .counter_pack.topology_sensitive
                and .counter_pack.primary_tool == "perf-c2c"
                and (.counter_pack.fallback_tools == ["perf-mem","perf-stat"])
                and (.counter_pack.fallback_selections == ["perf_mem_fallback","cache_event_fallback"])
                and (.counter_pack.capability_probe == "perf c2c record/report, else perf mem record/report")
                and (.counter_pack.fallback_event_pack == ["cache-references","cache-misses"])
                and (.counter_pack.fallback_reason_hints
                    == [
                        "perf c2c unavailable or failed on this host",
                        "perf c2c and perf mem unavailable or failed on this host"
                    ])
                and (.counter_pack.raw_output_relpaths
                    == ["profiles/perf-c2c." + .mode + ".data"])
            );
        def require_migration:
            ([.commands[] | select(.capture == "migration")] | length == 2)
            and all(.commands[] | select(.capture == "migration");
                .tool == "perf-stat"
                and (.counter_pack != null)
                and .counter_pack.host_capability_sensitive
                and .counter_pack.topology_sensitive
                and .counter_pack.primary_tool == "perf-stat"
                and .counter_pack.primary_selection == "cpu_migrations"
                and (.counter_pack.fallback_tools == ["perf-stat"])
                and (.counter_pack.fallback_selections == ["context_switch_fallback"])
                and (.counter_pack.capability_probe
                    == "perf stat -e cpu-migrations,context-switches -o \"$tmp\" -- true")
                and (.counter_pack.fallback_event_pack == ["context-switches","task-clock"])
                and (.counter_pack.fallback_reason_hints
                    == ["cpu-migrations unsupported on this host"])
                and (.counter_pack.raw_output_relpaths == [])
            );
        def require_remote_access:
            ([.commands[] | select(.capture == "remote_access")] | length == 2)
            and all(.commands[] | select(.capture == "remote_access");
                .tool == "perf-mem"
                and (.counter_pack != null)
                and .counter_pack.host_capability_sensitive
                and .counter_pack.topology_sensitive
                and .counter_pack.primary_tool == "perf-mem"
                and (.counter_pack.fallback_tools == ["perf-stat"])
                and (.counter_pack.fallback_selections == ["cache_event_fallback"])
                and (.counter_pack.capability_probe == "perf mem record/report")
                and (.counter_pack.fallback_event_pack
                    == ["cache-references","cache-misses","page-faults"])
                and (.counter_pack.fallback_reason_hints
                    == ["perf mem unavailable or failed on this host"])
                and (.counter_pack.raw_output_relpaths
                    == ["profiles/perf-mem-remote-access." + .mode + ".data"])
            );
        require_plain_capture("wall_clock"; "hyperfine")
        and require_plain_capture("on_cpu"; "perf-record")
        and require_plain_capture("scheduler"; "perf-sched-record")
        and require_plain_capture("syscall"; "strace")
        and require_plain_capture("allocation"; "heaptrack")
        and require_topdown
        and require_cache_to_cache
        and require_migration
        and require_remote_access
    ' "${path}" > /dev/null \
        || fail "verification" "command-pack topology counter metadata contract failed for ${path}"
}

require_manifest_counter_capture_summary() {
    local path="$1"

    require_nonempty_file "${path}"
    jq -e '
        (.counter_capture_summary != null)
        and (.counter_capture_summary.host_capability_sensitive_captures
            == ["topdown","cache_to_cache","migration","remote_access"])
        and (.counter_capture_summary.topology_sensitive_captures
            == ["cache_to_cache","migration","remote_access"])
        and (.counter_capture_summary.fallback_tools | index("perf-stat") != null)
        and (.counter_capture_summary.fallback_tools | index("perf-mem") != null)
        and (.counter_capture_summary.fallback_metric_pack | index("cache-misses") != null)
        and (.counter_capture_summary.fallback_notes
            | map(select(startswith("cache_to_cache:") or startswith("migration:") or startswith("remote_access:") or startswith("topdown:")))
            | length >= 4)
        and (.counter_capture_summary.raw_output_relpaths
            | index("profiles/perf-c2c.profiler_safe.data") != null)
        and (.counter_capture_summary.raw_output_relpaths
            | index("profiles/perf-mem-remote-access.profiler_safe.data") != null)
    ' "${path}" > /dev/null \
        || fail "verification" "manifest counter-capture summary contract failed for ${path}"
}

require_manifest_provenance() {
    local path="$1"

    require_nonempty_file "${path}"
    jq -e '
        (.provenance != null)
        and (.provenance.row_id | type == "string" and length > 0)
        and (
            .provenance.mode_id == "fsqlite_mvcc"
            or .provenance.mode_id == "fsqlite_single_writer"
        )
        and (.provenance.artifact_root | type == "string" and length > 0)
        and (.provenance.command_entrypoint | contains("hot-profile"))
        and (.provenance.workspace_root | type == "string" and length > 0)
        and (.provenance.campaign_manifest_path
            == "sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.json")
        and (.provenance.source_revision | type == "string" and length >= 12)
        and (.provenance.beads_data_hash | test("^[0-9a-f]{64}$"))
        and (.provenance.kernel_release | type == "string" and length > 0)
        and (.provenance.rustc_version | contains("rustc "))
        and (.provenance.cargo_profile | type == "string" and length > 0)
        and (.provenance.commands | length >= 19)
        and ([.provenance.commands[] | select(.tool == "realdb-e2e")] | length == 1)
        and ([.provenance.commands[] | select(.tool == "hyperfine")] | length == 2)
        and ([.provenance.commands[] | select(.tool == "perf-c2c")] | length == 2)
        and (.provenance.tool_versions | length >= 2)
        and ([.provenance.tool_versions[] | select(.tool == "cargo")] | length == 1)
        and ([.provenance.tool_versions[] | select(.tool == "hyperfine")] | length == 1)
        and (.provenance.fallback_notes
            | map(select(startswith("cache_to_cache:") or startswith("migration:") or startswith("remote_access:") or startswith("topdown:")))
            | length >= 4)
    ' "${path}" > /dev/null \
        || fail "verification" "manifest provenance contract failed for ${path}"
}

preseed_output_bundle() {
    local scenario_dir="$1"

    mkdir -p "${scenario_dir}"
    : > "${scenario_dir}/profile.json"
    : > "${scenario_dir}/opcode_profile.json"
    : > "${scenario_dir}/subsystem_profile.json"
    : > "${scenario_dir}/actionable_ranking.json"
    : > "${scenario_dir}/command_pack.json"
    : > "${scenario_dir}/summary.md"
    : > "${scenario_dir}/manifest.json"
}

materialize_output_bundle_from_log() {
    local scenario_dir="$1"
    local stderr_log="$2"
    local bundle_line

    bundle_line="$(sed -n "s/^${HOT_PATH_INLINE_BUNDLE_PREFIX}//p" "${stderr_log}" | tail -n 1)"
    [[ -n "${bundle_line}" ]] || return 0

    printf '%s\n' "${bundle_line}" \
        | jq -e --arg schema "${HOT_PATH_INLINE_BUNDLE_SCHEMA}" '
            type == "object" and .schema_version == $schema
        ' > /dev/null \
        || fail "run" "invalid inline hot-path bundle in ${stderr_log}"

    printf '%s\n' "${bundle_line}" | jq '.profile' > "${scenario_dir}/profile.json"
    printf '%s\n' "${bundle_line}" | jq '.opcode_profile' > "${scenario_dir}/opcode_profile.json"
    printf '%s\n' "${bundle_line}" | jq '.subsystem_profile' > "${scenario_dir}/subsystem_profile.json"
    printf '%s\n' "${bundle_line}" | jq '.actionable_ranking' > "${scenario_dir}/actionable_ranking.json"
    printf '%s\n' "${bundle_line}" | jq '.command_pack' > "${scenario_dir}/command_pack.json"
    printf '%s\n' "${bundle_line}" | jq -r '.summary_markdown' > "${scenario_dir}/summary.md"
    printf '%s\n' "${bundle_line}" | jq '.manifest' > "${scenario_dir}/manifest.json"
}

copy_fixture_seed() {
    local fixture_id="$1"
    local source_db="${SOURCE_GOLDEN_DIR}/${fixture_id}.db"
    local dest_db="${SYNC_GOLDEN_DIR}/${fixture_id}.db"

    require_file "${source_db}"
    mkdir -p "${SYNC_GOLDEN_DIR}"
    cp "${source_db}" "${dest_db}"
    for suffix in -wal -shm -journal; do
        if [[ -f "${source_db}${suffix}" ]]; then
            cp "${source_db}${suffix}" "${dest_db}${suffix}"
        fi
    done
}

mode_cli_flag() {
    local mode_id="$1"
    case "${mode_id}" in
        mvcc) printf '%s\n' "--mvcc" ;;
        single_writer) printf '%s\n' "--no-mvcc" ;;
        *) fail "inputs" "unsupported mode id: ${mode_id}" ;;
    esac
}

mode_engine_label() {
    local mode_id="$1"
    case "${mode_id}" in
        mvcc) printf '%s\n' "fsqlite_mvcc" ;;
        single_writer) printf '%s\n' "fsqlite_single_writer" ;;
        *) fail "inputs" "unsupported mode id: ${mode_id}" ;;
    esac
}

scenario_id_for_mode() {
    local mode_id="$1"
    printf '%s_c%s_%s\n' "${mode_id}" "${CONCURRENCY}" "${WORKLOAD_ID}"
}

discover_fixture_ids() {
    if [[ -n "${FIXTURE_IDS:-}" ]]; then
        tr ',' '\n' <<< "${FIXTURE_IDS}" | sed '/^$/d'
        return
    fi

    find "${SOURCE_GOLDEN_DIR}" -maxdepth 1 -type f -name '*.db' -printf '%f\n' \
        | sed 's/\.db$//' \
        | sort
}

discover_mode_ids() {
    if [[ -n "${MODE_IDS:-}" ]]; then
        tr ',' '\n' <<< "${MODE_IDS}" | sed '/^$/d'
        return
    fi

    printf '%s\n' mvcc single_writer
}

join_csv() {
    local IFS=','
    printf '%s' "$*"
}

capture_run_record() {
    local scenario_id="$1"
    local fixture_id="$2"
    local mode_id="$3"
    local scenario_dir="$4"
    local stdout_log="$5"
    local stderr_log="$6"

    jq -c -n \
        --arg scenario_id "${scenario_id}" \
        --arg fixture_id "${fixture_id}" \
        --arg mode_id "${mode_id}" \
        --arg workload "${WORKLOAD_ID}" \
        --arg output_dir "${scenario_dir}" \
        --arg stdout_log "${stdout_log}" \
        --arg stderr_log "${stderr_log}" \
        --arg engine_label "$(mode_engine_label "${mode_id}")" \
        --arg source_golden_dir "${SOURCE_GOLDEN_DIR}" \
        --arg sync_golden_dir "${SYNC_GOLDEN_DIR}" \
        --arg cargo_profile "${CARGO_PROFILE}" \
        --argjson concurrency "${CONCURRENCY}" \
        --argjson seed "${SEED}" \
        --argjson scale "${SCALE}" \
        --slurpfile profile "${scenario_dir}/profile.json" \
        --slurpfile opcode_profile "${scenario_dir}/opcode_profile.json" \
        --slurpfile subsystem_profile "${scenario_dir}/subsystem_profile.json" \
        --slurpfile actionable_ranking "${scenario_dir}/actionable_ranking.json" \
        --slurpfile command_pack "${scenario_dir}/command_pack.json" \
        --slurpfile manifest "${scenario_dir}/manifest.json" \
        '
        {
            scenario_id: $scenario_id,
            fixture_id: $fixture_id,
            mode_id: $mode_id,
            engine_label: $engine_label,
            workload: $workload,
            concurrency: $concurrency,
            seed: $seed,
            scale: $scale,
            cargo_profile: $cargo_profile,
            source_golden_dir: $source_golden_dir,
            sync_golden_dir: $sync_golden_dir,
            output_dir: $output_dir,
            stdout_log: $stdout_log,
            stderr_log: $stderr_log,
            profile: $profile[0],
            opcode_profile_pack: $opcode_profile[0],
            subsystem_profile_pack: $subsystem_profile[0],
            actionable_ranking: $actionable_ranking[0],
            command_pack: $command_pack[0],
            manifest: $manifest[0]
        }
        ' >> "${RUN_RECORDS_JSONL}"
}

run_hot_profile() {
    local fixture_id="$1"
    local mode_id="$2"
    local scenario_id
    scenario_id="$(scenario_id_for_mode "${mode_id}")"
    local scenario_dir="${RUNS_DIR}/${mode_id}__${WORKLOAD_ID}__c${CONCURRENCY}__${fixture_id}__${CARGO_PROFILE}__${RUN_ID_TARGET_SAFE}"
    local stdout_log="${scenario_dir}/stdout.log"
    local stderr_log="${scenario_dir}/stderr.log"
    local cli_flag
    cli_flag="$(mode_cli_flag "${mode_id}")"

    mkdir -p "${scenario_dir}"
    preseed_output_bundle "${scenario_dir}"
    log_event "INFO" "run" "starting ${scenario_id} fixture=${fixture_id} mode=${mode_id}"

    if ! rch exec -- env \
        CARGO_TARGET_DIR="${RCH_TARGET_DIR}" \
        FSQLITE_HOT_PATH_BEAD_ID="${FSQLITE_HOT_PATH_BEAD_ID}" \
        FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH="${FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH}" \
        FSQLITE_HOT_PATH_CARGO_PROFILE="${FSQLITE_HOT_PATH_CARGO_PROFILE}" \
        FSQLITE_HOT_PATH_WORKSPACE_ROOT="${FSQLITE_HOT_PATH_WORKSPACE_ROOT}" \
        FSQLITE_HOT_PATH_SOURCE_REVISION="${FSQLITE_HOT_PATH_SOURCE_REVISION}" \
        FSQLITE_HOT_PATH_BEADS_DATA_HASH="${FSQLITE_HOT_PATH_BEADS_DATA_HASH}" \
        FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID="${FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID}" \
        FSQLITE_HOT_PATH_HARDWARE_CLASS_ID="${FSQLITE_HOT_PATH_HARDWARE_CLASS_ID}" \
        FSQLITE_HOT_PATH_HARDWARE_SIGNATURE="${FSQLITE_HOT_PATH_HARDWARE_SIGNATURE}" \
        cargo run \
        -p fsqlite-e2e \
        --profile "${CARGO_PROFILE}" \
        --bin realdb-e2e \
        -- hot-profile \
        --golden-dir "${SYNC_GOLDEN_DIR}" \
        --db "${fixture_id}" \
        --concurrency "${CONCURRENCY}" \
        --seed "${SEED}" \
        --scale "${SCALE}" \
        --output-dir "${scenario_dir}" \
        --emit-inline-bundle \
        "${cli_flag}" \
        > "${stdout_log}" \
        2> "${stderr_log}"; then
        fail "run" "hot-profile failed for fixture=${fixture_id} mode=${mode_id}; see ${stderr_log}"
    fi

    materialize_output_bundle_from_log "${scenario_dir}" "${stderr_log}"
    require_json_schema "${scenario_dir}/profile.json" "${HOT_PATH_PROFILE_SCHEMA}"
    require_json_schema "${scenario_dir}/opcode_profile.json" "${HOT_PATH_OPCODE_PROFILE_SCHEMA}"
    require_json_schema "${scenario_dir}/subsystem_profile.json" "${HOT_PATH_SUBSYSTEM_PROFILE_SCHEMA}"
    require_json_schema "${scenario_dir}/actionable_ranking.json" "${HOT_PATH_ACTIONABLE_RANKING_SCHEMA}"
    require_json_schema "${scenario_dir}/command_pack.json" "${HOT_PATH_COMMAND_PACK_SCHEMA}"
    require_topology_counter_pack_contract "${scenario_dir}/command_pack.json"
    require_nonempty_file "${scenario_dir}/summary.md"
    require_json_schema "${scenario_dir}/manifest.json" "${HOT_PATH_PROFILE_MANIFEST_SCHEMA}"
    require_manifest_counter_capture_summary "${scenario_dir}/manifest.json"
    require_manifest_provenance "${scenario_dir}/manifest.json"
    capture_run_record "${scenario_id}" "${fixture_id}" "${mode_id}" "${scenario_dir}" "${stdout_log}" "${stderr_log}"
    log_event "INFO" "run" "completed ${scenario_id} fixture=${fixture_id} mode=${mode_id}"
}

build_scenario_profiles() {
    jq -s \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        --arg source_golden_dir "${SOURCE_GOLDEN_DIR}" \
        --arg sync_golden_dir "${SYNC_GOLDEN_DIR}" \
        --arg workload "${WORKLOAD_ID}" \
        '
        {
            schema_version: "fsqlite-e2e.hot_path_campaign_scenarios.v1",
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            source_golden_dir: $source_golden_dir,
            sync_golden_dir: $sync_golden_dir,
            workload: $workload,
            runs: .
        }
        ' "${RUN_RECORDS_JSONL}" > "${SCENARIO_PROFILES_JSON}"
}

build_opcode_profile_packs() {
    jq -s \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        '
        {
            schema_version: "fsqlite-e2e.hot_path_campaign_opcode_packs.v1",
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            runs: [
                .[] | {
                    scenario_id,
                    fixture_id,
                    mode_id,
                    engine_label,
                    output_dir,
                    opcode_profile: .opcode_profile_pack
                }
            ]
        }
        ' "${RUN_RECORDS_JSONL}" > "${OPCODE_PROFILE_PACKS_JSON}"
}

build_subsystem_profile_packs() {
    jq -s \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        '
        {
            schema_version: "fsqlite-e2e.hot_path_campaign_subsystem_packs.v1",
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            runs: [
                .[] | {
                    scenario_id,
                    fixture_id,
                    mode_id,
                    engine_label,
                    output_dir,
                    subsystem_profile: .subsystem_profile_pack
                }
            ]
        }
        ' "${RUN_RECORDS_JSONL}" > "${SUBSYSTEM_PROFILE_PACKS_JSON}"
}

build_command_packs() {
    jq -s \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        '
        def aggregate_counter_capture_summary:
            [ .[] | .manifest.counter_capture_summary // empty ] as $summaries
            | {
                host_capability_sensitive_captures: ($summaries | map(.host_capability_sensitive_captures // []) | add | unique),
                topology_sensitive_captures: ($summaries | map(.topology_sensitive_captures // []) | add | unique),
                fallback_tools: ($summaries | map(.fallback_tools // []) | add | unique),
                fallback_metric_pack: ($summaries | map(.fallback_metric_pack // []) | add | unique),
                fallback_notes: ($summaries | map(.fallback_notes // []) | add | unique),
                raw_output_relpaths: ($summaries | map(.raw_output_relpaths // []) | add | unique)
            };
        def aggregate_provenance_summary:
            [ .[] | .manifest.provenance // empty ] as $summaries
            | {
                row_ids: ($summaries | map(.row_id) | unique),
                mode_ids: ($summaries | map(.mode_id) | unique),
                artifact_roots: ($summaries | map(.artifact_root) | unique),
                command_entrypoints: ($summaries | map(.command_entrypoint) | unique),
                workspace_roots: ($summaries | map(.workspace_root // empty) | map(select(length > 0)) | unique),
                campaign_manifest_paths: ($summaries | map(.campaign_manifest_path // empty) | map(select(length > 0)) | unique),
                source_revisions: ($summaries | map(.source_revision // empty) | map(select(length > 0)) | unique),
                beads_data_hashes: ($summaries | map(.beads_data_hash // empty) | map(select(length > 0)) | unique),
                kernel_releases: ($summaries | map(.kernel_release) | unique),
                rustc_versions: ($summaries | map(.rustc_version) | unique),
                cargo_profiles: ($summaries | map(.cargo_profile) | unique),
                command_tools: ($summaries | map(.commands // []) | add | map(.tool) | unique),
                tool_versions: (
                    $summaries
                    | map(.tool_versions // [])
                    | add
                    | map("\(.tool)=\(.version)")
                    | unique
                ),
                fallback_notes: ($summaries | map(.fallback_notes // []) | add | unique)
            };
        {
            schema_version: "fsqlite-e2e.hot_path_campaign_command_packs.v1",
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            counter_capture_summary: aggregate_counter_capture_summary,
            provenance_summary: aggregate_provenance_summary,
            runs: [
                .[] | {
                    scenario_id,
                    fixture_id,
                    mode_id,
                    engine_label,
                    output_dir,
                    counter_capture_summary: .manifest.counter_capture_summary,
                    provenance: .manifest.provenance,
                    command_pack
                }
            ]
        }
        ' "${RUN_RECORDS_JSONL}" > "${COMMAND_PACKS_JSON}"
}

build_benchmark_context() {
    jq -s \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        --arg source_golden_dir "${SOURCE_GOLDEN_DIR}" \
        --arg sync_golden_dir "${SYNC_GOLDEN_DIR}" \
        --arg workload "${WORKLOAD_ID}" \
        --arg fixture_ids_csv "${FIXTURE_IDS_CSV}" \
        --arg mode_ids_csv "${MODE_IDS_CSV}" \
        --argjson expected_runs "${expected_runs}" \
        '
        def aggregate_counter_capture_summary:
            [ .[] | .manifest.counter_capture_summary // empty ] as $summaries
            | {
                host_capability_sensitive_captures: ($summaries | map(.host_capability_sensitive_captures // []) | add | unique),
                topology_sensitive_captures: ($summaries | map(.topology_sensitive_captures // []) | add | unique),
                fallback_tools: ($summaries | map(.fallback_tools // []) | add | unique),
                fallback_metric_pack: ($summaries | map(.fallback_metric_pack // []) | add | unique),
                fallback_notes: ($summaries | map(.fallback_notes // []) | add | unique),
                raw_output_relpaths: ($summaries | map(.raw_output_relpaths // []) | add | unique)
            };
        def aggregate_provenance_summary:
            [ .[] | .manifest.provenance // empty ] as $summaries
            | {
                row_ids: ($summaries | map(.row_id) | unique),
                mode_ids: ($summaries | map(.mode_id) | unique),
                artifact_roots: ($summaries | map(.artifact_root) | unique),
                command_entrypoints: ($summaries | map(.command_entrypoint) | unique),
                workspace_roots: ($summaries | map(.workspace_root // empty) | map(select(length > 0)) | unique),
                campaign_manifest_paths: ($summaries | map(.campaign_manifest_path // empty) | map(select(length > 0)) | unique),
                source_revisions: ($summaries | map(.source_revision // empty) | map(select(length > 0)) | unique),
                beads_data_hashes: ($summaries | map(.beads_data_hash // empty) | map(select(length > 0)) | unique),
                kernel_releases: ($summaries | map(.kernel_release) | unique),
                rustc_versions: ($summaries | map(.rustc_version) | unique),
                cargo_profiles: ($summaries | map(.cargo_profile) | unique),
                command_tools: ($summaries | map(.commands // []) | add | map(.tool) | unique),
                tool_versions: (
                    $summaries
                    | map(.tool_versions // [])
                    | add
                    | map("\(.tool)=\(.version)")
                    | unique
                ),
                fallback_notes: ($summaries | map(.fallback_notes // []) | add | unique)
            };
        {
            schema_version: "fsqlite-e2e.hot_path_campaign_context.v1",
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            source_golden_dir: $source_golden_dir,
            sync_golden_dir: $sync_golden_dir,
            workload: $workload,
            fixture_ids: ($fixture_ids_csv | split(",") | map(select(length > 0))),
            mode_ids: ($mode_ids_csv | split(",") | map(select(length > 0))),
            expected_runs: $expected_runs,
            completed_runs: length,
            counter_capture_summary: aggregate_counter_capture_summary,
            provenance_summary: aggregate_provenance_summary,
            runs: [
                .[] | {
                    scenario_id,
                    fixture_id,
                    mode_id,
                    engine_label,
                    concurrency,
                    seed,
                    scale,
                    ops_per_sec: .profile.engine_report.ops_per_sec,
                    wall_time_ms: .profile.engine_report.wall_time_ms,
                    retries: .profile.engine_report.retries,
                    aborts: .profile.engine_report.aborts,
                    error: (.profile.engine_report.error // null),
                    row_id: (.manifest.provenance.row_id // null),
                    artifact_root: (.manifest.provenance.artifact_root // null),
                    command_entrypoint: (.manifest.provenance.command_entrypoint // null),
                    source_revision: (.manifest.provenance.source_revision // null),
                    beads_data_hash: (.manifest.provenance.beads_data_hash // null),
                    output_dir
                }
            ]
        }
        ' "${RUN_RECORDS_JSONL}" > "${BENCHMARK_CONTEXT_JSON}"
}

build_actionable_ranking() {
    jq -s \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        '
        def confidence_fields($entries; $expected_runs; $expected_modes; $expected_fixtures):
            ($entries | length) as $evidence_runs
            | ($entries | map(.mode_id) | unique | length) as $mode_coverage_count
            | ($entries | map(.fixture_id) | unique | length) as $fixture_coverage_count
            | ($entries | map(.rank) | min) as $rank_min
            | ($entries | map(.rank) | max) as $rank_max
            | (if $rank_max > 1
               then (1.0 - (($rank_max - $rank_min) / ($rank_max - 1)))
               else 1.0
               end) as $rank_stability
            | (
                10000
                * (
                    0.45 * ($evidence_runs / $expected_runs)
                    + 0.20 * ($mode_coverage_count / $expected_modes)
                    + 0.20 * ($fixture_coverage_count / $expected_fixtures)
                    + 0.15 * $rank_stability
                )
                | round
              ) as $confidence_score_basis_points
            | {
                confidence_score_basis_points: $confidence_score_basis_points,
                confidence_label: (
                    if $confidence_score_basis_points >= 8500 then "high"
                    elif $confidence_score_basis_points >= 6500 then "medium"
                    else "low"
                    end
                ),
                confidence_rationale: (
                    "observed in \($evidence_runs)/\($expected_runs) runs across "
                    + "\($mode_coverage_count)/\($expected_modes) modes and "
                    + "\($fixture_coverage_count)/\($expected_fixtures) fixtures; "
                    + "rank spread \($rank_min)-\($rank_max)"
                ),
                evidence_runs: $evidence_runs,
                expected_runs: $expected_runs,
                mode_coverage_count: $mode_coverage_count,
                expected_mode_count: $expected_modes,
                fixture_coverage_count: $fixture_coverage_count,
                expected_fixture_count: $expected_fixtures,
                rank_min: $rank_min,
                rank_max: $rank_max
            };
        def named_entries:
            [ .[] as $run
              | $run.actionable_ranking.named_hotspots[]
              | . + {
                    fixture_id: $run.fixture_id,
                    mode_id: $run.mode_id,
                    scenario_id: $run.scenario_id
                }
            ];
        def allocator_entries:
            [ .[] as $run
              | $run.actionable_ranking.allocator_pressure[]
              | . + {
                    fixture_id: $run.fixture_id,
                    mode_id: $run.mode_id,
                    scenario_id: $run.scenario_id
                }
            ];
        def wall_time_entries:
            [ .[] as $run
              | $run.actionable_ranking.wall_time_components[]
              | . + {
                    fixture_id: $run.fixture_id,
                    mode_id: $run.mode_id,
                    scenario_id: $run.scenario_id
                }
            ];
        def microarchitectural_entries:
            [ .[] as $run
              | $run.actionable_ranking.microarchitectural_signatures[]
              | . + {
                    fixture_id: $run.fixture_id,
                    mode_id: $run.mode_id,
                    scenario_id: $run.scenario_id
                }
            ];
        def cost_component_entries:
            [ .[] as $run
              | $run.actionable_ranking.cost_components[]
              | . + {
                    fixture_id: $run.fixture_id,
                    mode_id: $run.mode_id,
                    scenario_id: $run.scenario_id
                }
            ];
        def opcode_entries:
            [ .[] as $run
              | $run.profile.opcode_profile[]
              | . + {
                    fixture_id: $run.fixture_id,
                    mode_id: $run.mode_id,
                    scenario_id: $run.scenario_id
                }
            ];
        (length | if . > 0 then . else 1 end) as $expected_runs
        | ((map(.mode_id) | unique | length) | if . > 0 then . else 1 end) as $expected_modes
        | ((map(.fixture_id) | unique | length) | if . > 0 then . else 1 end) as $expected_fixtures
        |
        {
            schema_version: "fsqlite-e2e.hot_path_campaign_ranking.v2",
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            scenario_count: length,
            named_hotspots: (
                named_entries
                | sort_by(.subsystem)
                | group_by(.subsystem)
                | map({
                    subsystem: .[0].subsystem,
                    metric_kind: .[0].metric_kind,
                    avg_metric_value: ((map(.metric_value) | add) / length),
                    max_metric_value: (map(.metric_value) | max),
                    run_breakdown: (
                        map({
                            fixture_id,
                            mode_id,
                            scenario_id,
                            rank,
                            metric_value,
                            rationale,
                            implication,
                            mapped_beads
                        })
                        | sort_by(.metric_value)
                        | reverse
                    )
                } + confidence_fields(.; $expected_runs; $expected_modes; $expected_fixtures))
                | sort_by(.avg_metric_value)
                | reverse
                | to_entries
                | map(.value + { rank: (.key + 1) })
            ),
            wall_time_components: (
                wall_time_entries
                | sort_by(.component)
                | group_by(.component)
                | map({
                    component: .[0].component,
                    avg_time_ns: ((map(.time_ns) | add) / length),
                    max_time_ns: (map(.time_ns) | max),
                    avg_wall_share_basis_points: ((map(.wall_share_basis_points) | add) / length),
                    run_breakdown: (
                        map({
                            fixture_id,
                            mode_id,
                            scenario_id,
                            rank,
                            time_ns,
                            wall_share_basis_points,
                            rationale,
                            implication,
                            mapped_beads
                        })
                        | sort_by(.time_ns)
                        | reverse
                    )
                } + confidence_fields(.; $expected_runs; $expected_modes; $expected_fixtures))
                | sort_by(.avg_time_ns)
                | reverse
                | to_entries
                | map(.value + { rank: (.key + 1) })
            ),
            microarchitectural_signatures: (
                microarchitectural_entries
                | sort_by([.target, .primary_signature])
                | group_by([.target, .primary_signature])
                | map({
                    target: .[0].target,
                    primary_signature: .[0].primary_signature,
                    secondary_signatures: (map(.secondary_signatures // []) | add | unique),
                    mixed_or_ambiguous: any(.mixed_or_ambiguous),
                    row_ids: (map(.row_id // empty) | map(select(length > 0)) | unique),
                    placement_profile_ids: (map(.placement_profile_id // empty) | map(select(length > 0)) | unique),
                    hardware_class_ids: (map(.hardware_class_id // empty) | map(select(length > 0)) | unique),
                    hardware_signatures: (map(.hardware_signature // empty) | map(select(length > 0)) | unique),
                    evidence_sources: (map(.evidence_sources // []) | add | unique),
                    avg_entry_confidence_score_basis_points: (
                        (map(.confidence_score_basis_points) | add) / length
                    ),
                    max_entry_confidence_score_basis_points: (
                        map(.confidence_score_basis_points) | max
                    ),
                    run_breakdown: (
                        map({
                            fixture_id,
                            mode_id,
                            scenario_id,
                            rank,
                            target,
                            primary_signature,
                            secondary_signatures,
                            confidence_label,
                            confidence_score_basis_points,
                            mixed_or_ambiguous,
                            row_id,
                            placement_profile_id,
                            hardware_class_id,
                            hardware_signature,
                            rationale,
                            implication,
                            evidence_sources,
                            mapped_beads
                        })
                        | sort_by(.confidence_score_basis_points)
                        | reverse
                    )
                } + confidence_fields(.; $expected_runs; $expected_modes; $expected_fixtures))
                | sort_by([.avg_entry_confidence_score_basis_points, .target, .primary_signature])
                | reverse
                | to_entries
                | map(.value + { rank: (.key + 1) })
            ),
            allocator_pressure: (
                allocator_entries
                | sort_by(.subsystem)
                | group_by(.subsystem)
                | map({
                    subsystem: .[0].subsystem,
                    metric_kind: .[0].metric_kind,
                    avg_metric_value: ((map(.metric_value) | add) / length),
                    max_metric_value: (map(.metric_value) | max),
                    run_breakdown: (
                        map({
                            fixture_id,
                            mode_id,
                            scenario_id,
                            rank,
                            metric_value,
                            rationale,
                            implication,
                            mapped_beads
                        })
                        | sort_by(.metric_value)
                        | reverse
                    )
                } + confidence_fields(.; $expected_runs; $expected_modes; $expected_fixtures))
                | sort_by(.avg_metric_value)
                | reverse
                | to_entries
                | map(.value + { rank: (.key + 1) })
            ),
            cost_components: (
                cost_component_entries
                | sort_by(.component)
                | group_by(.component)
                | map({
                    component: .[0].component,
                    avg_time_ns: ((map(.time_ns) | add) / length),
                    max_time_ns: (map(.time_ns) | max),
                    avg_time_share_basis_points: ((map(.time_share_basis_points) | add) / length),
                    avg_allocator_pressure_bytes: ((map(.allocator_pressure_bytes) | add) / length),
                    max_allocator_pressure_bytes: (map(.allocator_pressure_bytes) | max),
                    avg_allocator_share_basis_points: ((map(.allocator_share_basis_points) | add) / length),
                    max_allocator_share_basis_points: (map(.allocator_share_basis_points) | max),
                    max_activity_count: (map(.activity_count) | max),
                    run_breakdown: (
                        map({
                            fixture_id,
                            mode_id,
                            scenario_id,
                            rank,
                            time_ns,
                            time_share_basis_points,
                            allocator_pressure_bytes,
                            allocator_share_basis_points,
                            activity_count,
                            rationale,
                            implication,
                            mapped_beads
                        })
                        | sort_by([.time_ns, .allocator_pressure_bytes])
                        | reverse
                    )
                } + confidence_fields(.; $expected_runs; $expected_modes; $expected_fixtures))
                | sort_by([.avg_time_ns, .avg_allocator_pressure_bytes])
                | reverse
                | to_entries
                | map(.value + { rank: (.key + 1) })
            ),
            top_opcodes: (
                opcode_entries
                | sort_by(.opcode)
                | group_by(.opcode)
                | map({
                    opcode: .[0].opcode,
                    total: (map(.total) | add),
                    run_breakdown: (
                        map({
                            fixture_id,
                            mode_id,
                            scenario_id,
                            total
                        })
                        | sort_by(.total)
                        | reverse
                    )
                })
                | sort_by(.total)
                | reverse
                | .[:12]
            )
        }
        ' "${RUN_RECORDS_JSONL}" > "${ACTIONABLE_RANKING_JSON}"
}

build_summary_md() {
    local coverage_summary
    coverage_summary="$(jq -r '
        [
            "- fixture_ids: `\(.fixture_ids | join(","))`",
            "- mode_ids: `\(.mode_ids | join(","))`",
            "- expected_runs: `\(.expected_runs)`",
            "- completed_runs: `\(.completed_runs)`",
            "- provenance_row_ids: `\(.provenance_summary.row_ids | join(","))`",
            "- provenance_source_revisions: `\(.provenance_summary.source_revisions | join(","))`",
            "- provenance_beads_hashes: `\(.provenance_summary.beads_data_hashes | join(","))`",
            "- provenance_kernel_releases: `\(.provenance_summary.kernel_releases | join(","))`",
            "- provenance_rustc_versions: `\(.provenance_summary.rustc_versions | join(","))`",
            "- provenance_cargo_profiles: `\(.provenance_summary.cargo_profiles | join(","))`",
            "- provenance_command_tools: `\(.provenance_summary.command_tools | join(","))`",
            "- counter_capture_host_sensitive: `\(.counter_capture_summary.host_capability_sensitive_captures | join(","))`",
            "- counter_capture_topology_sensitive: `\(.counter_capture_summary.topology_sensitive_captures | join(","))`",
            "- counter_capture_fallback_tools: `\(.counter_capture_summary.fallback_tools | join(","))`",
            "- counter_capture_fallback_metrics: `\(.counter_capture_summary.fallback_metric_pack | join(","))`"
        ]
        | .[]
    ' "${BENCHMARK_CONTEXT_JSON}")"

    local run_summary
    run_summary="$(jq -r '
        .runs
        | map(
            "- `\(.scenario_id)` / `\(.fixture_id)` / `\(.mode_id)`: ops_per_sec=\(.ops_per_sec), wall_time_ms=\(.wall_time_ms), retries=\(.retries), aborts=\(.aborts)"
          )
        | .[]
    ' "${BENCHMARK_CONTEXT_JSON}")"

    local hotspot_summary
    hotspot_summary="$(jq -r '
        .named_hotspots[:5]
        | map(
            "- rank \(.rank): `\(.subsystem)` avg=\(.avg_metric_value) \(.metric_kind) max=\(.max_metric_value) confidence=\(.confidence_label) (\(.confidence_score_basis_points)bp) -> \(.run_breakdown[0].implication)"
          )
        | .[]
    ' "${ACTIONABLE_RANKING_JSON}")"

    local wall_time_summary
    wall_time_summary="$(jq -r '
        .wall_time_components[:6]
        | map(
            "- rank \(.rank): `\(.component)` avg_time_ns=\(.avg_time_ns) avg_wall_share_bps=\(.avg_wall_share_basis_points) confidence=\(.confidence_label) (\(.confidence_score_basis_points)bp) -> \(.run_breakdown[0].implication)"
          )
        | .[]
    ' "${ACTIONABLE_RANKING_JSON}")"

    local microarchitectural_summary
    microarchitectural_summary="$(jq -r '
        .microarchitectural_signatures[:6]
        | map(
            "- rank \(.rank): `\(.target)` primary=\(.primary_signature) secondary=\((.secondary_signatures | join(","))) coverage_confidence=\(.confidence_label) (\(.confidence_score_basis_points)bp) entry_confidence_max=\(.max_entry_confidence_score_basis_points) mixed=\(.mixed_or_ambiguous) rows=\((.row_ids | join(","))) placement=\((.placement_profile_ids | join(","))) hardware=\((.hardware_signatures | join(","))) -> \(.run_breakdown[0].implication)"
          )
        | .[]
    ' "${ACTIONABLE_RANKING_JSON}")"

    local allocator_summary
    allocator_summary="$(jq -r '
        .allocator_pressure[:3]
        | map(
            "- rank \(.rank): `\(.subsystem)` avg=\(.avg_metric_value) \(.metric_kind) max=\(.max_metric_value) confidence=\(.confidence_label) (\(.confidence_score_basis_points)bp) -> \(.run_breakdown[0].implication)"
          )
        | .[]
    ' "${ACTIONABLE_RANKING_JSON}")"

    local cost_component_summary
    cost_component_summary="$(jq -r '
        .cost_components[:3]
        | map(
            "- rank \(.rank): `\(.component)` avg_time_ns=\(.avg_time_ns) avg_time_share_bps=\(.avg_time_share_basis_points) avg_allocator_pressure_bytes=\(.avg_allocator_pressure_bytes) avg_allocator_share_bps=\(.avg_allocator_share_basis_points) max_activity_count=\(.max_activity_count) confidence=\(.confidence_label) (\(.confidence_score_basis_points)bp) -> \(.run_breakdown[0].implication)"
          )
        | .[]
    ' "${ACTIONABLE_RANKING_JSON}")"

    local opcode_summary
    opcode_summary="$(jq -r '
        .top_opcodes[:8]
        | map("- `\(.opcode)`: total=\(.total)")
        | .[]
    ' "${ACTIONABLE_RANKING_JSON}")"

    cat > "${SUMMARY_MD}" <<EOF
# ${BEAD_ID} Inline Hot-Path Campaign Summary

- run_id: \`${RUN_ID}\`
- generated_at: \`${GENERATED_AT}\`
- source_golden_dir: \`${SOURCE_GOLDEN_DIR}\`
- sync_golden_dir: \`${SYNC_GOLDEN_DIR}\`
- replay_command: \`${REPLAY_COMMAND}\`
- cargo_profile: \`${CARGO_PROFILE}\`
- workload: \`${WORKLOAD_ID}\`
- concurrency: \`${CONCURRENCY}\`
${coverage_summary}

## Run Context

${run_summary}

## Ranked Hotspots

${hotspot_summary}

## Wall-Time Decomposition

${wall_time_summary}

## Microarchitectural Signatures

${microarchitectural_summary}

## Quantified Cost Components

${cost_component_summary}

## Allocator Pressure

${allocator_summary}

## Top Opcodes

${opcode_summary}

## Artifacts

- structured_log: \`${LOG_FILE}\`
- run_records: \`${RUN_RECORDS_JSONL}\`
- opcode_profile_packs: \`${OPCODE_PROFILE_PACKS_JSON}\`
- subsystem_profile_packs: \`${SUBSYSTEM_PROFILE_PACKS_JSON}\`
- command_packs: \`${COMMAND_PACKS_JSON}\`
- scenario_profiles: \`${SCENARIO_PROFILES_JSON}\`
- actionable_ranking: \`${ACTIONABLE_RANKING_JSON}\`
- benchmark_context: \`${BENCHMARK_CONTEXT_JSON}\`
- report: \`${REPORT_JSON}\`
EOF
}

build_report_json() {
    jq -n \
        --arg schema_version "fsqlite-e2e.hot_path_campaign_report.v1" \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        --arg source_golden_dir "${SOURCE_GOLDEN_DIR}" \
        --arg sync_golden_dir "${SYNC_GOLDEN_DIR}" \
        --arg output_dir "${OUTPUT_DIR}" \
        --arg replay_command "${REPLAY_COMMAND}" \
        --arg structured_log "${LOG_FILE}" \
        --arg run_records "${RUN_RECORDS_JSONL}" \
        --arg opcode_profile_packs "${OPCODE_PROFILE_PACKS_JSON}" \
        --arg subsystem_profile_packs "${SUBSYSTEM_PROFILE_PACKS_JSON}" \
        --arg command_packs "${COMMAND_PACKS_JSON}" \
        --arg scenario_profiles "${SCENARIO_PROFILES_JSON}" \
        --arg actionable_ranking "${ACTIONABLE_RANKING_JSON}" \
        --arg benchmark_context_path "${BENCHMARK_CONTEXT_JSON}" \
        --arg summary_md "${SUMMARY_MD}" \
        --arg report_json "${REPORT_JSON}" \
        --arg workload "${WORKLOAD_ID}" \
        --arg cargo_profile "${CARGO_PROFILE}" \
        --arg fixture_ids_csv "${FIXTURE_IDS_CSV}" \
        --arg mode_ids_csv "${MODE_IDS_CSV}" \
        --argjson concurrency "${CONCURRENCY}" \
        --argjson seed "${SEED}" \
        --argjson scale "${SCALE}" \
        --argjson expected_runs "${expected_runs}" \
        --slurpfile benchmark_context_doc "${BENCHMARK_CONTEXT_JSON}" \
        '
        {
            schema_version: $schema_version,
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            workload: $workload,
            source_golden_dir: $source_golden_dir,
            sync_golden_dir: $sync_golden_dir,
            output_dir: $output_dir,
            cargo_profile: $cargo_profile,
            concurrency: $concurrency,
            seed: $seed,
            scale: $scale,
            fixture_ids: ($fixture_ids_csv | split(",") | map(select(length > 0))),
            mode_ids: ($mode_ids_csv | split(",") | map(select(length > 0))),
            expected_runs: $expected_runs,
            completed_runs: ($benchmark_context_doc[0].completed_runs // null),
            counter_capture_summary: ($benchmark_context_doc[0].counter_capture_summary // null),
            provenance_summary: ($benchmark_context_doc[0].provenance_summary // null),
            replay: {
                command: $replay_command
            },
            artifacts: {
                structured_log: $structured_log,
                run_records: $run_records,
                opcode_profile_packs: $opcode_profile_packs,
                subsystem_profile_packs: $subsystem_profile_packs,
                command_packs: $command_packs,
                scenario_profiles: $scenario_profiles,
                actionable_ranking: $actionable_ranking,
                benchmark_context: $benchmark_context_path,
                summary_md: $summary_md,
                report_json: $report_json
            },
            limitations: [
                "This workflow captures mixed_read_write inline hot-path evidence only.",
                "Results depend on the pinned Beads working-copy fixtures under the benchmark workspace golden directory."
            ]
        }
        ' > "${REPORT_JSON}"
}

log_event "INFO" "start" "starting inline hot-path campaign"
require_dir "${SOURCE_GOLDEN_DIR}"
require_file "${BEADS_DATA_PATH}"
CAMPAIGN_MANIFEST_FILE="$(resolve_campaign_manifest_file)"
require_file "${CAMPAIGN_MANIFEST_FILE}"

if [[ -z "${FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID}" || -z "${FSQLITE_HOT_PATH_HARDWARE_CLASS_ID}" || -z "${FSQLITE_HOT_PATH_HARDWARE_SIGNATURE}" ]]; then
    IFS=$'\t' read -r default_placement_profile_id default_hardware_class_id default_hardware_signature < <(resolve_hot_path_microarchitectural_overrides)
    [[ -n "${default_placement_profile_id:-}" ]] || fail "inputs" "failed to resolve placement profile for row ${WORKLOAD_ID}_c${CONCURRENCY} from ${CAMPAIGN_MANIFEST_FILE}"
    [[ -n "${default_hardware_class_id:-}" ]] || fail "inputs" "failed to resolve hardware class for row ${WORKLOAD_ID}_c${CONCURRENCY} from ${CAMPAIGN_MANIFEST_FILE}"
    [[ -n "${default_hardware_signature:-}" ]] || fail "inputs" "failed to resolve hardware signature for row ${WORKLOAD_ID}_c${CONCURRENCY} from ${CAMPAIGN_MANIFEST_FILE}"
    FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID="${FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID:-${default_placement_profile_id}}"
    FSQLITE_HOT_PATH_HARDWARE_CLASS_ID="${FSQLITE_HOT_PATH_HARDWARE_CLASS_ID:-${default_hardware_class_id}}"
    FSQLITE_HOT_PATH_HARDWARE_SIGNATURE="${FSQLITE_HOT_PATH_HARDWARE_SIGNATURE:-${default_hardware_signature}}"
fi

mapfile -t FIXTURE_IDS_ARRAY < <(discover_fixture_ids)
mapfile -t MODE_IDS_ARRAY < <(discover_mode_ids)

(( ${#FIXTURE_IDS_ARRAY[@]} > 0 )) || fail "inputs" "no fixture ids discovered under ${SOURCE_GOLDEN_DIR}"
(( ${#MODE_IDS_ARRAY[@]} > 0 )) || fail "inputs" "no mode ids configured"

expected_runs=$(( ${#FIXTURE_IDS_ARRAY[@]} * ${#MODE_IDS_ARRAY[@]} ))
FIXTURE_IDS_CSV="$(join_csv "${FIXTURE_IDS_ARRAY[@]}")"
MODE_IDS_CSV="$(join_csv "${MODE_IDS_ARRAY[@]}")"
printf -v REPLAY_COMMAND \
    'cd %q && BEAD_ID=%q SCRIPT_ENTRYPOINT=%q GOLDEN_DIR=%q SYNC_GOLDEN_DIR=%q OUTPUT_DIR=%q FIXTURE_IDS=%q MODE_IDS=%q WORKLOAD_ID=%q CONCURRENCY=%q SEED=%q SCALE=%q CARGO_PROFILE=%q RCH_TARGET_DIR=%q FSQLITE_HOT_PATH_BEAD_ID=%q FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH=%q FSQLITE_HOT_PATH_CARGO_PROFILE=%q FSQLITE_HOT_PATH_WORKSPACE_ROOT=%q FSQLITE_HOT_PATH_SOURCE_REVISION=%q FSQLITE_HOT_PATH_BEADS_DATA_HASH=%q FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID=%q FSQLITE_HOT_PATH_HARDWARE_CLASS_ID=%q FSQLITE_HOT_PATH_HARDWARE_SIGNATURE=%q bash %q' \
    "${WORKSPACE_ROOT}" "${BEAD_ID}" "${SCRIPT_ENTRYPOINT}" "${SOURCE_GOLDEN_DIR}" \
    "${SYNC_GOLDEN_DIR}" "${OUTPUT_DIR}" "${FIXTURE_IDS_CSV}" "${MODE_IDS_CSV}" "${WORKLOAD_ID}" \
    "${CONCURRENCY}" "${SEED}" "${SCALE}" "${CARGO_PROFILE}" "${RCH_TARGET_DIR}" \
    "${FSQLITE_HOT_PATH_BEAD_ID}" "${FSQLITE_HOT_PATH_CAMPAIGN_MANIFEST_PATH}" \
    "${FSQLITE_HOT_PATH_CARGO_PROFILE}" "${FSQLITE_HOT_PATH_WORKSPACE_ROOT}" \
    "${FSQLITE_HOT_PATH_SOURCE_REVISION}" "${FSQLITE_HOT_PATH_BEADS_DATA_HASH}" \
    "${FSQLITE_HOT_PATH_PLACEMENT_PROFILE_ID}" "${FSQLITE_HOT_PATH_HARDWARE_CLASS_ID}" "${FSQLITE_HOT_PATH_HARDWARE_SIGNATURE}" \
    "${SCRIPT_ENTRYPOINT}"
log_event "INFO" "plan" "fixtures=${#FIXTURE_IDS_ARRAY[@]} modes=${#MODE_IDS_ARRAY[@]} expected_runs=${expected_runs}"

for fixture_id in "${FIXTURE_IDS_ARRAY[@]}"; do
    copy_fixture_seed "${fixture_id}"
done
log_event "INFO" "inputs" "prepared synced fixture seed directory at ${SYNC_GOLDEN_DIR}"

for mode_id in "${MODE_IDS_ARRAY[@]}"; do
    for fixture_id in "${FIXTURE_IDS_ARRAY[@]}"; do
        run_hot_profile "${fixture_id}" "${mode_id}"
    done
done

build_scenario_profiles
build_opcode_profile_packs
build_subsystem_profile_packs
build_command_packs
build_benchmark_context
build_actionable_ranking
require_json_schema "${ACTIONABLE_RANKING_JSON}" "${HOT_PATH_CAMPAIGN_RANKING_SCHEMA}"
build_summary_md
build_report_json

jq -e '.runs | length >= 1' "${SCENARIO_PROFILES_JSON}" >/dev/null
jq -e '.runs | length >= 1' "${COMMAND_PACKS_JSON}" >/dev/null
jq -e '.counter_capture_summary.host_capability_sensitive_captures | length >= 1' "${COMMAND_PACKS_JSON}" >/dev/null
jq -e '.provenance_summary.row_ids | length >= 1' "${COMMAND_PACKS_JSON}" >/dev/null
jq -e '.provenance_summary.command_tools | index("realdb-e2e") != null' "${COMMAND_PACKS_JSON}" >/dev/null
jq -e '.runs | all(.provenance.row_id != null and .provenance.command_entrypoint != null)' "${COMMAND_PACKS_JSON}" >/dev/null
jq -e '.named_hotspots | length >= 1' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.wall_time_components | length >= 1' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.microarchitectural_signatures | length >= 1' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.cost_components | length >= 1' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.allocator_pressure | length >= 1' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.named_hotspots | all(has("confidence_label") and has("confidence_score_basis_points"))' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.wall_time_components | all(has("confidence_label") and has("confidence_score_basis_points"))' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.microarchitectural_signatures | all(has("confidence_label") and has("confidence_score_basis_points"))' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.microarchitectural_signatures | all((.row_ids | length) >= 1 and (.placement_profile_ids | length) >= 1 and (.hardware_class_ids | length) >= 1 and (.hardware_signatures | length) >= 1 and (.evidence_sources | length) >= 1)' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.microarchitectural_signatures | all(.run_breakdown | all(.row_id != null and .placement_profile_id != null and .hardware_class_id != null and .hardware_signature != null and (.evidence_sources | length) >= 1))' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.cost_components | all(has("confidence_label") and has("confidence_score_basis_points"))' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.allocator_pressure | all(has("confidence_label") and has("confidence_score_basis_points"))' "${ACTIONABLE_RANKING_JSON}" >/dev/null
jq -e '.runs | length == '"${expected_runs}" "${BENCHMARK_CONTEXT_JSON}" >/dev/null
jq -e '.counter_capture_summary.host_capability_sensitive_captures | length >= 1' "${BENCHMARK_CONTEXT_JSON}" >/dev/null
jq -e '.provenance_summary.source_revisions | length >= 1' "${BENCHMARK_CONTEXT_JSON}" >/dev/null
jq -e '.runs | all(.row_id != null and .artifact_root != null and .source_revision != null)' "${BENCHMARK_CONTEXT_JSON}" >/dev/null
jq -e '.counter_capture_summary.host_capability_sensitive_captures | length >= 1' "${REPORT_JSON}" >/dev/null
jq -e '.provenance_summary.beads_data_hashes | length >= 1' "${REPORT_JSON}" >/dev/null

log_event "INFO" "complete" "inline hot-path campaign completed"
echo "RUN_ID:              ${RUN_ID}"
echo "Source golden dir:   ${SOURCE_GOLDEN_DIR}"
echo "Synced golden dir:   ${SYNC_GOLDEN_DIR}"
echo "Run records:         ${RUN_RECORDS_JSONL}"
echo "Scenario profiles:   ${SCENARIO_PROFILES_JSON}"
echo "Actionable ranking:  ${ACTIONABLE_RANKING_JSON}"
echo "Benchmark context:   ${BENCHMARK_CONTEXT_JSON}"
echo "Summary:             ${SUMMARY_MD}"
echo "Report:              ${REPORT_JSON}"
