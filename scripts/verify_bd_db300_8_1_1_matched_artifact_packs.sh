#!/usr/bin/env bash
# verify_bd_db300_8_1_1_matched_artifact_packs.sh
#
# Track H matched-pack collector for SQLite vs FrankenSQLite MVCC vs forced
# single-writer mode. The script runs one canonical benchmark cell per selected
# row/fixture/placement triple, writes mode-specific benchmark artifacts, and
# produces a matched-pack manifest/report with shared provenance fields.
#
# Heavy work is always routed through `rch exec`.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEAD_ID="${BEAD_ID:-bd-db300.8.1.1}"
SCRIPT_ENTRYPOINT="${SCRIPT_ENTRYPOINT:-scripts/verify_bd_db300_8_1_1_matched_artifact_packs.sh}"
RUN_ID="${BEAD_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_ID_SAFE="${RUN_ID//[^[:alnum:]]/_}"
GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
CAMPAIGN_MANIFEST_REL="${CAMPAIGN_MANIFEST_REL:-sample_sqlite_db_files/manifests/beads_benchmark_campaign.v1.json}"
CAMPAIGN_MANIFEST_FILE="${WORKSPACE_ROOT}/${CAMPAIGN_MANIFEST_REL}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/artifacts/perf/${BEAD_ID}/${RUN_ID}}"
PACKS_DIR="${OUTPUT_DIR}/packs"
LOG_FILE="${OUTPUT_DIR}/events.jsonl"
REPORT_JSON="${OUTPUT_DIR}/report.json"
SUMMARY_MD="${OUTPUT_DIR}/summary.md"
ROW_IDS="${ROW_IDS:-mixed_read_write_c4}"
FIXTURE_IDS="${FIXTURE_IDS:-}"
PLACEMENT_PROFILE_IDS="${PLACEMENT_PROFILE_IDS:-baseline_unpinned}"
REPEAT="${REPEAT:-1}"
WARMUP="${WARMUP:-0}"
CARGO_PROFILE="${CARGO_PROFILE:-release-perf}"
RCH_TARGET_DIR="${RCH_TARGET_DIR:-/tmp/rch_target_bd_db300_8_1_1}"
RETENTION_CLASS="${RETENTION_CLASS:-quick_run}"
BEADS_DATA_PATH="${WORKSPACE_ROOT}/.beads/issues.jsonl"
SOURCE_REVISION="${SOURCE_REVISION:-$(git -C "${WORKSPACE_ROOT}" rev-parse HEAD)}"
BEADS_HASH="${BEADS_HASH:-$(sha256sum "${BEADS_DATA_PATH}" | awk '{print $1}')}"
MODES=("sqlite_reference" "fsqlite_mvcc" "fsqlite_single_writer")

mkdir -p "${PACKS_DIR}"
: > "${LOG_FILE}"

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

require_file() {
    local path="$1"
    [[ -f "${path}" ]] || fail "inputs" "missing required file: ${path}"
}

require_nonempty_file() {
    local path="$1"
    [[ -s "${path}" ]] || fail "inputs" "missing or empty required file: ${path}"
}

shell_join() {
    local rendered=""
    local arg
    for arg in "$@"; do
        rendered+="$(printf '%q' "${arg}") "
    done
    printf '%s\n' "${rendered% }"
}

csv_to_lines() {
    printf '%s\n' "$1" | tr ',' '\n' | sed '/^[[:space:]]*$/d'
}

short_hash() {
    printf '%s' "$1" | cut -c1-12
}

ensure_row_exists() {
    local row_id="$1"
    jq -e --arg row_id "${row_id}" '.matrix_rows[] | select(.row_id == $row_id)' "${CAMPAIGN_MANIFEST_FILE}" >/dev/null \
        || fail "inputs" "row_id `${row_id}` not found in ${CAMPAIGN_MANIFEST_REL}"
}

row_workload() {
    local row_id="$1"
    jq -r --arg row_id "${row_id}" '.matrix_rows[] | select(.row_id == $row_id) | .workload' "${CAMPAIGN_MANIFEST_FILE}"
}

row_concurrency() {
    local row_id="$1"
    jq -r --arg row_id "${row_id}" '.matrix_rows[] | select(.row_id == $row_id) | .concurrency' "${CAMPAIGN_MANIFEST_FILE}"
}

row_fixture_ids() {
    local row_id="$1"
    if [[ -n "${FIXTURE_IDS}" ]]; then
        csv_to_lines "${FIXTURE_IDS}"
        return
    fi
    jq -r --arg row_id "${row_id}" '.matrix_rows[] | select(.row_id == $row_id) | .fixtures[]' "${CAMPAIGN_MANIFEST_FILE}"
}

row_placement_profiles() {
    local row_id="$1"
    if [[ -n "${PLACEMENT_PROFILE_IDS}" ]]; then
        csv_to_lines "${PLACEMENT_PROFILE_IDS}"
        return
    fi
    jq -r --arg row_id "${row_id}" '
        .matrix_rows[]
        | select(.row_id == $row_id)
        | .placement_variants[]
        | .placement_profile_id
    ' "${CAMPAIGN_MANIFEST_FILE}"
}

placement_hardware_class() {
    local row_id="$1"
    local placement_profile_id="$2"
    jq -r --arg row_id "${row_id}" --arg placement_profile_id "${placement_profile_id}" '
        .matrix_rows[]
        | select(.row_id == $row_id)
        | .placement_variants[]
        | select(.placement_profile_id == $placement_profile_id)
        | .hardware_class_id
    ' "${CAMPAIGN_MANIFEST_FILE}"
}

placement_profile_json() {
    local placement_profile_id="$1"
    jq -c --arg placement_profile_id "${placement_profile_id}" '
        .placement_profiles[]
        | select(.id == $placement_profile_id)
    ' "${CAMPAIGN_MANIFEST_FILE}"
}

hardware_class_json() {
    local hardware_class_id="$1"
    jq -c --arg hardware_class_id "${hardware_class_id}" '
        .hardware_classes[]
        | select(.id == $hardware_class_id)
    ' "${CAMPAIGN_MANIFEST_FILE}"
}

placement_execution_status() {
    local placement_profile_id="$1"
    if [[ "${placement_profile_id}" == "baseline_unpinned" ]]; then
        printf 'comparable_under_scheduler_default\n'
    else
        printf 'declared_only_requires_external_placement_enforcement\n'
    fi
}

mode_engine_label() {
    case "$1" in
        sqlite_reference) printf 'sqlite3\n' ;;
        fsqlite_mvcc) printf 'fsqlite_mvcc\n' ;;
        fsqlite_single_writer) printf 'fsqlite_single_writer\n' ;;
        *) fail "inputs" "unsupported mode_id: $1" ;;
    esac
}

mode_cli_args() {
    case "$1" in
        sqlite_reference) printf '%s\n' "--engine sqlite3" ;;
        fsqlite_mvcc) printf '%s\n' "--engine fsqlite --mvcc" ;;
        fsqlite_single_writer) printf '%s\n' "--engine fsqlite --no-mvcc" ;;
        *) fail "inputs" "unsupported mode_id: $1" ;;
    esac
}

run_mode_benchmark() {
    local row_id="$1"
    local fixture_id="$2"
    local workload="$3"
    local concurrency="$4"
    local placement_profile_id="$5"
    local hardware_class_id="$6"
    local pack_dir="$7"
    local mode_id="$8"

    local mode_dir="${pack_dir}/${mode_id}"
    local results_jsonl="${mode_dir}/results.jsonl"
    local summary_md="${mode_dir}/summary.md"
    local summary_json="${mode_dir}/summary.json"
    local stdout_log="${mode_dir}/stdout.log"
    local stderr_log="${mode_dir}/stderr.log"

    mkdir -p "${mode_dir}"

    local cli_args_raw
    cli_args_raw="$(mode_cli_args "${mode_id}")"
    local -a mode_args=()
    # shellcheck disable=SC2206
    mode_args=(${cli_args_raw})

    local -a cmd=(
        env
        "CARGO_TARGET_DIR=${RCH_TARGET_DIR}"
        cargo run
        -p fsqlite-e2e
        --profile "${CARGO_PROFILE}"
        --bin realdb-e2e
        --
        bench
        "${mode_args[@]}"
        --db "${fixture_id}"
        --preset "${workload}"
        --concurrency "${concurrency}"
        --warmup "${WARMUP}"
        --repeat "${REPEAT}"
        --output-jsonl "${results_jsonl}"
        --output-md "${summary_md}"
    )

    log_event "INFO" "run" "starting ${row_id} fixture=${fixture_id} placement=${placement_profile_id} mode=${mode_id}"

    if ! rch exec -- "${cmd[@]}" >"${stdout_log}" 2>"${stderr_log}"; then
        fail "run" "benchmark failed for row=${row_id} fixture=${fixture_id} placement=${placement_profile_id} mode=${mode_id}; see ${stderr_log}"
    fi

    require_nonempty_file "${results_jsonl}"
    require_nonempty_file "${summary_md}"

    jq -c \
        --arg row_id "${row_id}" \
        --arg fixture_id "${fixture_id}" \
        --arg workload "${workload}" \
        --arg placement_profile_id "${placement_profile_id}" \
        --arg hardware_class_id "${hardware_class_id}" \
        --arg mode_id "${mode_id}" \
        --arg engine_label "$(mode_engine_label "${mode_id}")" \
        --arg results_jsonl_rel "$(realpath --relative-to="${pack_dir}" "${results_jsonl}")" \
        --arg summary_md_rel "$(realpath --relative-to="${pack_dir}" "${summary_md}")" \
        --arg stdout_log_rel "$(realpath --relative-to="${pack_dir}" "${stdout_log}")" \
        --arg stderr_log_rel "$(realpath --relative-to="${pack_dir}" "${stderr_log}")" \
        --arg rerun_command "cd ${WORKSPACE_ROOT} && BEAD_ID=${BEAD_ID} OUTPUT_DIR=${OUTPUT_DIR} ROW_IDS=${row_id} FIXTURE_IDS=${fixture_id} PLACEMENT_PROFILE_IDS=${placement_profile_id} CARGO_PROFILE=${CARGO_PROFILE} WARMUP=${WARMUP} REPEAT=${REPEAT} bash ${SCRIPT_ENTRYPOINT}" \
        '
        . as $bench
        | {
            mode_id: $mode_id,
            engine_label: $engine_label,
            row_id: $row_id,
            fixture_id: $fixture_id,
            workload: $workload,
            concurrency: $bench.concurrency,
            placement_profile_id: $placement_profile_id,
            hardware_class_id: $hardware_class_id,
            benchmark_id: $bench.benchmark_id,
            measurement_count: $bench.measurement_count,
            latency: {
                median_ms: $bench.latency.median_ms,
                p95_ms: $bench.latency.p95_ms,
                p99_ms: $bench.latency.p99_ms
            },
            throughput: {
                mean_ops_per_sec: $bench.throughput.mean_ops_per_sec,
                median_ops_per_sec: $bench.throughput.median_ops_per_sec,
                peak_ops_per_sec: $bench.throughput.peak_ops_per_sec
            },
            retries: {
                total: ($bench.iterations | map(.retries) | add // 0),
                mean_per_iteration: (
                    if ($bench.iterations | length) == 0
                    then 0
                    else (($bench.iterations | map(.retries) | add // 0) / ($bench.iterations | length))
                    end
                )
            },
            aborts: {
                total: ($bench.iterations | map(.aborts) | add // 0),
                mean_per_iteration: (
                    if ($bench.iterations | length) == 0
                    then 0
                    else (($bench.iterations | map(.aborts) | add // 0) / ($bench.iterations | length))
                    end
                )
            },
            files: {
                results_jsonl: $results_jsonl_rel,
                summary_md: $summary_md_rel,
                stdout_log: $stdout_log_rel,
                stderr_log: $stderr_log_rel
            },
            rerun_command: $rerun_command,
            benchmark_summary: $bench
        }
        ' < "${results_jsonl}" > "${summary_json}"

    log_event "INFO" "run" "completed ${row_id} fixture=${fixture_id} placement=${placement_profile_id} mode=${mode_id}"
}

build_pack_manifest() {
    local row_id="$1"
    local fixture_id="$2"
    local workload="$3"
    local concurrency="$4"
    local placement_profile_id="$5"
    local hardware_class_id="$6"
    local pack_dir="$7"

    local placement_profile
    placement_profile="$(placement_profile_json "${placement_profile_id}")"
    [[ -n "${placement_profile}" ]] || fail "inputs" "placement profile `${placement_profile_id}` not found"

    local hardware_class
    hardware_class="$(hardware_class_json "${hardware_class_id}")"
    [[ -n "${hardware_class}" ]] || fail "inputs" "hardware class `${hardware_class_id}` not found"

    jq -n \
        --arg schema_version "fsqlite-e2e.db300.matched_mode_pack.v1" \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        --arg retention_class "${RETENTION_CLASS}" \
        --arg row_id "${row_id}" \
        --arg fixture_id "${fixture_id}" \
        --arg workload "${workload}" \
        --argjson concurrency "${concurrency}" \
        --arg placement_profile_id "${placement_profile_id}" \
        --arg hardware_class_id "${hardware_class_id}" \
        --arg comparability_status "$(placement_execution_status "${placement_profile_id}")" \
        --arg source_revision "${SOURCE_REVISION}" \
        --arg beads_hash "${BEADS_HASH}" \
        --arg cargo_profile "${CARGO_PROFILE}" \
        --argjson warmup "${WARMUP}" \
        --argjson repeat "${REPEAT}" \
        --arg script_entrypoint "${SCRIPT_ENTRYPOINT}" \
        --arg pack_dir "${pack_dir}" \
        --arg pack_dir_rel "$(realpath --relative-to="${WORKSPACE_ROOT}" "${pack_dir}")" \
        --slurpfile placement_profile <(printf '%s\n' "${placement_profile}") \
        --slurpfile hardware_class <(printf '%s\n' "${hardware_class}") \
        --slurpfile sqlite "${pack_dir}/sqlite_reference/summary.json" \
        --slurpfile mvcc "${pack_dir}/fsqlite_mvcc/summary.json" \
        --slurpfile single "${pack_dir}/fsqlite_single_writer/summary.json" \
        '
        def ratio($num; $den):
            if $den == null or $den == 0 then null else ($num / $den) end;
        {
            schema_version: $schema_version,
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            retention_class: $retention_class,
            row_id: $row_id,
            fixture_id: $fixture_id,
            workload: $workload,
            concurrency: $concurrency,
            placement_profile_id: $placement_profile_id,
            hardware_class_id: $hardware_class_id,
            comparability_status: $comparability_status,
            source_revision: $source_revision,
            beads_data_hash: $beads_hash,
            cargo_profile: $cargo_profile,
            warmup_iterations: $warmup,
            measurement_iterations: $repeat,
            script_entrypoint: $script_entrypoint,
            pack_dir: $pack_dir,
            pack_dir_relpath: $pack_dir_rel,
            placement_profile: $placement_profile[0],
            hardware_class: $hardware_class[0],
            mode_results: {
                sqlite_reference: $sqlite[0],
                fsqlite_mvcc: $mvcc[0],
                fsqlite_single_writer: $single[0]
            },
            deltas: {
                mvcc_vs_sqlite_median_ops_ratio:
                    ratio($mvcc[0].throughput.median_ops_per_sec; $sqlite[0].throughput.median_ops_per_sec),
                single_writer_vs_mvcc_median_ops_ratio:
                    ratio($single[0].throughput.median_ops_per_sec; $mvcc[0].throughput.median_ops_per_sec),
                single_writer_minus_mvcc_median_latency_ms:
                    ($single[0].latency.median_ms - $mvcc[0].latency.median_ms),
                single_writer_minus_mvcc_mean_retries:
                    ($single[0].retries.mean_per_iteration - $mvcc[0].retries.mean_per_iteration),
                single_writer_minus_sqlite_mean_retries:
                    ($single[0].retries.mean_per_iteration - $sqlite[0].retries.mean_per_iteration)
            },
            notes: (
                if $placement_profile_id == "baseline_unpinned"
                then [
                    "baseline_unpinned packs are directly comparable under scheduler-default placement"
                ]
                else [
                    "non-baseline placement profiles are recorded from the canonical contract but require external CPU and memory placement enforcement outside this script",
                    "packs produced without that enforcement should be treated as declared_only rather than clean topology claims"
                ]
                end
            )
        }
        ' > "${pack_dir}/manifest.json"
}

build_pack_summary() {
    local pack_dir="$1"
    jq -r '
        [
            "# Matched Mode Pack",
            "",
            "- row_id: `\(.row_id)`",
            "- fixture_id: `\(.fixture_id)`",
            "- placement_profile_id: `\(.placement_profile_id)`",
            "- hardware_class_id: `\(.hardware_class_id)`",
            "- comparability_status: `\(.comparability_status)`",
            "- source_revision: `\(.source_revision)`",
            "- beads_data_hash: `\(.beads_data_hash)`",
            "",
            "## Mode Summary",
            "",
            "| Mode | Median ops/s | Median latency (ms) | P95 latency (ms) | Mean retries | Mean aborts |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            "| sqlite_reference | \(.mode_results.sqlite_reference.throughput.median_ops_per_sec) | \(.mode_results.sqlite_reference.latency.median_ms) | \(.mode_results.sqlite_reference.latency.p95_ms) | \(.mode_results.sqlite_reference.retries.mean_per_iteration) | \(.mode_results.sqlite_reference.aborts.mean_per_iteration) |",
            "| fsqlite_mvcc | \(.mode_results.fsqlite_mvcc.throughput.median_ops_per_sec) | \(.mode_results.fsqlite_mvcc.latency.median_ms) | \(.mode_results.fsqlite_mvcc.latency.p95_ms) | \(.mode_results.fsqlite_mvcc.retries.mean_per_iteration) | \(.mode_results.fsqlite_mvcc.aborts.mean_per_iteration) |",
            "| fsqlite_single_writer | \(.mode_results.fsqlite_single_writer.throughput.median_ops_per_sec) | \(.mode_results.fsqlite_single_writer.latency.median_ms) | \(.mode_results.fsqlite_single_writer.latency.p95_ms) | \(.mode_results.fsqlite_single_writer.retries.mean_per_iteration) | \(.mode_results.fsqlite_single_writer.aborts.mean_per_iteration) |",
            "",
            "## Deltas",
            "",
            "- mvcc_vs_sqlite_median_ops_ratio: `\(.deltas.mvcc_vs_sqlite_median_ops_ratio)`",
            "- single_writer_vs_mvcc_median_ops_ratio: `\(.deltas.single_writer_vs_mvcc_median_ops_ratio)`",
            "- single_writer_minus_mvcc_median_latency_ms: `\(.deltas.single_writer_minus_mvcc_median_latency_ms)`",
            "- single_writer_minus_mvcc_mean_retries: `\(.deltas.single_writer_minus_mvcc_mean_retries)`",
            "",
            "## Notes",
            "",
            (.notes[] | "- " + .)
        ] | join("\n")
    ' "${pack_dir}/manifest.json" > "${pack_dir}/summary.md"
}

collect_pack() {
    local row_id="$1"
    local fixture_id="$2"
    local placement_profile_id="$3"

    local workload
    workload="$(row_workload "${row_id}")"
    local concurrency
    concurrency="$(row_concurrency "${row_id}")"
    local hardware_class_id
    hardware_class_id="$(placement_hardware_class "${row_id}" "${placement_profile_id}")"

    [[ -n "${hardware_class_id}" ]] \
        || fail "inputs" "row `${row_id}` does not define placement `${placement_profile_id}`"

    local pack_dir="${PACKS_DIR}/${row_id}__${fixture_id}__${placement_profile_id}__run_${RUN_ID_SAFE}__rev_$(short_hash "${SOURCE_REVISION}")__beads_$(short_hash "${BEADS_HASH}")"
    mkdir -p "${pack_dir}"

    log_event "INFO" "pack" "collecting matched pack row=${row_id} fixture=${fixture_id} placement=${placement_profile_id}"

    local mode_id
    for mode_id in "${MODES[@]}"; do
        run_mode_benchmark \
            "${row_id}" \
            "${fixture_id}" \
            "${workload}" \
            "${concurrency}" \
            "${placement_profile_id}" \
            "${hardware_class_id}" \
            "${pack_dir}" \
            "${mode_id}"
    done

    build_pack_manifest \
        "${row_id}" \
        "${fixture_id}" \
        "${workload}" \
        "${concurrency}" \
        "${placement_profile_id}" \
        "${hardware_class_id}" \
        "${pack_dir}"
    build_pack_summary "${pack_dir}"
}

build_report() {
    local manifests=()
    while IFS= read -r path; do
        manifests+=("${path}")
    done < <(find "${PACKS_DIR}" -mindepth 2 -maxdepth 2 -name manifest.json | sort)

    ((${#manifests[@]} > 0)) || fail "report" "no pack manifests were generated"

    jq -s \
        --arg schema_version "fsqlite-e2e.db300.matched_mode_pack_report.v1" \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg generated_at "${GENERATED_AT}" \
        --arg script_entrypoint "${SCRIPT_ENTRYPOINT}" \
        --arg campaign_manifest "${CAMPAIGN_MANIFEST_REL}" \
        --arg output_dir "${OUTPUT_DIR}" \
        '
        {
            schema_version: $schema_version,
            bead_id: $bead_id,
            run_id: $run_id,
            generated_at: $generated_at,
            script_entrypoint: $script_entrypoint,
            campaign_manifest: $campaign_manifest,
            output_dir: $output_dir,
            pack_count: length,
            packs: .
        }
        ' "${manifests[@]}" > "${REPORT_JSON}"

    jq -r '
        [
            "# Track H Matched Artifact Packs",
            "",
            "- run_id: `\(.run_id)`",
            "- campaign_manifest: `\(.campaign_manifest)`",
            "- pack_count: `\(.pack_count)`",
            "",
            "| row_id | fixture_id | placement_profile_id | comparability | sqlite ops/s | mvcc ops/s | single-writer ops/s | single-writer vs mvcc ops ratio | single-writer minus mvcc retries |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            (
                .packs[]
                | "| \(.row_id) | \(.fixture_id) | \(.placement_profile_id) | \(.comparability_status) | \(.mode_results.sqlite_reference.throughput.median_ops_per_sec) | \(.mode_results.fsqlite_mvcc.throughput.median_ops_per_sec) | \(.mode_results.fsqlite_single_writer.throughput.median_ops_per_sec) | \(.deltas.single_writer_vs_mvcc_median_ops_ratio) | \(.deltas.single_writer_minus_mvcc_mean_retries) |"
            )
        ] | join("\n")
    ' "${REPORT_JSON}" > "${SUMMARY_MD}"
}

main() {
    require_file "${CAMPAIGN_MANIFEST_FILE}"
    require_nonempty_file "${BEADS_DATA_PATH}"
    log_event "INFO" "start" "starting matched artifact pack collection"

    local row_id fixture_id placement_profile_id
    while IFS= read -r row_id; do
        [[ -n "${row_id}" ]] || continue
        ensure_row_exists "${row_id}"
        while IFS= read -r fixture_id; do
            [[ -n "${fixture_id}" ]] || continue
            while IFS= read -r placement_profile_id; do
                [[ -n "${placement_profile_id}" ]] || continue
                collect_pack "${row_id}" "${fixture_id}" "${placement_profile_id}"
            done < <(row_placement_profiles "${row_id}")
        done < <(row_fixture_ids "${row_id}")
    done < <(csv_to_lines "${ROW_IDS}")

    build_report
    log_event "INFO" "complete" "matched artifact pack collection completed"

    echo "RUN_ID:      ${RUN_ID}"
    echo "OUTPUT_DIR:  ${OUTPUT_DIR}"
    echo "REPORT_JSON: ${REPORT_JSON}"
    echo "SUMMARY_MD:  ${SUMMARY_MD}"
}

main "$@"
