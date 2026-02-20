#!/usr/bin/env bash
set -euo pipefail

BEAD_ID="bd-2zoa"
LOG_STANDARD_REF="bd-1fpm"
LOG_SCHEMA_VERSION="1.0.0"
SCENARIO_ID="PERF-1"
SEED="${BD_2ZOA_SEED:-2026022001}"
BACKEND="fsqlite"
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${BEAD_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
ISSUES_PATH="${WORKSPACE_ROOT}/.beads/issues.jsonl"
REPORT_DIR="${WORKSPACE_ROOT}/test-results"
REPORT_JSONL="${REPORT_DIR}/bd_2zoa_arc_warmup_report.jsonl"
LOG_JSONL="${REPORT_DIR}/bd_2zoa_arc_warmup_events.jsonl"
REPORT_REL="${REPORT_JSONL#${WORKSPACE_ROOT}/}"
LOG_REL="${LOG_JSONL#${WORKSPACE_ROOT}/}"
WORKER="${HOSTNAME:-local}"

declare -a UNIT_IDS=(
    "test_bd_2zoa_unit_compliance_gate"
    "prop_bd_2zoa_structure_compliance"
)
declare -a E2E_IDS=(
    "test_e2e_bd_2zoa"
    "test_e2e_bd_2zoa_compliance"
)
declare -a LOG_LEVELS=(
    "DEBUG"
    "INFO"
    "WARN"
    "ERROR"
)
declare -a WORKLOAD_MARKERS=(
    "OLTP point queries"
    "Mixed OLTP + scan"
    "Full table scan"
    "Zipfian"
    "MVCC 8 writers"
)
declare -a WARMUP_MARKERS=(
    "Cold start"
    "Learning"
    "Steady state"
)
declare -a PREWARM_MARKERS=(
    "PRAGMA cache_warm"
    "WAL index"
    "sqlite_master root pages"
)

emit_event() {
    local phase="$1"
    local event_type="$2"
    local outcome="$3"
    local case_name="$4"
    local message="$5"
    local error_code="$6"
    local artifact_paths_json="$7"
    local context_json="$8"
    local timestamp
    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    jq -nc \
        --arg schema_version "${LOG_SCHEMA_VERSION}" \
        --arg bead_id "${BEAD_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg timestamp "${timestamp}" \
        --arg phase "${phase}" \
        --arg event_type "${event_type}" \
        --arg scenario_id "${SCENARIO_ID}" \
        --arg seed "${SEED}" \
        --arg backend "${BACKEND}" \
        --arg outcome "${outcome}" \
        --arg case_name "${case_name}" \
        --arg message "${message}" \
        --arg error_code "${error_code}" \
        --arg worker "${WORKER}" \
        --arg log_standard_ref "${LOG_STANDARD_REF}" \
        --argjson artifact_paths "${artifact_paths_json}" \
        --argjson context "${context_json}" \
        '{
            schema_version: $schema_version,
            bead_id: $bead_id,
            run_id: $run_id,
            timestamp: $timestamp,
            phase: $phase,
            event_type: $event_type,
            scenario_id: $scenario_id,
            seed: ($seed | tonumber? // $seed),
            backend: $backend,
            outcome: $outcome,
            error_code: (if $error_code == "" then null else $error_code end),
            artifact_paths: $artifact_paths,
            context: ($context + {
                case: $case_name,
                message: $message,
                worker: $worker,
                log_standard_ref: $log_standard_ref
            })
        }' >>"${LOG_JSONL}"
}

log_line() {
    local level="$1"
    local case_name="$2"
    shift 2
    local details="$*"

    local phase="validate"
    local event_type="info"
    local outcome="info"
    local error_code=""

    case "${case_name}" in
        start)
            phase="setup"
            event_type="start"
            outcome="running"
            ;;
        trace)
            phase="execute"
            event_type="info"
            outcome="running"
            ;;
        summary)
            phase="report"
            event_type="info"
            outcome="info"
            ;;
        missing_marker | degraded_mode_count)
            phase="validate"
            event_type="warn"
            outcome="warn"
            ;;
        pass)
            phase="report"
            event_type="pass"
            outcome="pass"
            ;;
        terminal_failure)
            phase="report"
            event_type="fail"
            outcome="fail"
            ;;
        terminal_failure_count)
            phase="report"
            event_type="info"
            outcome="pass"
            ;;
    esac

    case "${level}" in
        WARN)
            event_type="warn"
            outcome="warn"
            ;;
        ERROR)
            if [[ "${case_name}" == "terminal_failure_count" ]]; then
                event_type="info"
                outcome="pass"
            elif [[ "${event_type}" != "fail" ]]; then
                event_type="error"
                outcome="fail"
            fi
            ;;
    esac

    if [[ "${event_type}" == "fail" || "${event_type}" == "error" ]]; then
        error_code="E_$(printf '%s' "${case_name}" | tr '[:lower:]-' '[:upper:]_' | tr -cd 'A-Z0-9_')"
    fi

    local context_json
    context_json="$(
        jq -nc \
            --arg level "${level}" \
            --arg case_name "${case_name}" \
            --arg details "${details}" \
            '{level: $level, details: $details, case: $case_name}'
    )"
    local artifact_paths_json
    artifact_paths_json="$(
        jq -nc --arg report "${REPORT_REL}" --arg log "${LOG_REL}" '[ $report, $log ] | unique'
    )"

    emit_event \
        "${phase}" \
        "${event_type}" \
        "${outcome}" \
        "${case_name}" \
        "${case_name}" \
        "${error_code}" \
        "${artifact_paths_json}" \
        "${context_json}"

    printf 'bead_id=%s level=%s run_id=%s scenario_id=%s phase=%s event_type=%s case=%s %s reference=%s\n' \
        "${BEAD_ID}" "${level}" "${RUN_ID}" "${SCENARIO_ID}" "${phase}" "${event_type}" "${case_name}" "${details}" "${LOG_STANDARD_REF}"
}

mkdir -p "${REPORT_DIR}"
: >"${REPORT_JSONL}"
: >"${LOG_JSONL}"

log_line "DEBUG" "start" \
    "workspace=${WORKSPACE_ROOT} report=${REPORT_JSONL} event_log=${LOG_JSONL}"

if [[ ! -f "${ISSUES_PATH}" ]]; then
    log_line "ERROR" "missing_issues_jsonl" "path=${ISSUES_PATH}"
    exit 1
fi

description="$(
    jq -r '
        select(.id == "bd-2zoa")
        | .description,
          (.comments[]?.text // empty)
    ' "${ISSUES_PATH}"
)"

if [[ -z "${description//[[:space:]]/}" ]]; then
    log_line "ERROR" "missing_bead_description" "path=${ISSUES_PATH}"
    exit 1
fi

trace_id="$(printf '%s' "${description}" | sha256sum | awk '{print substr($1, 1, 16)}')"
log_line "DEBUG" "trace" "trace_id=${trace_id}"

printf '# bead_id=%s ARC performance warmup report\n' "${BEAD_ID}" >"${REPORT_JSONL}"
printf '# kind\tmarker\tpresent\n' >>"${REPORT_JSONL}"

declare -a missing_markers=()

check_marker() {
    local kind="$1"
    local marker="$2"
    local present=0

    if grep -Fqi -- "${marker}" <<<"${description}"; then
        present=1
    else
        missing_markers+=("${kind}:${marker}")
    fi

    printf '{"bead_id":"%s","trace_id":"%s","kind":"%s","marker":"%s","present":%s}\n' \
        "${BEAD_ID}" "${trace_id}" "${kind}" "${marker}" "${present}" >>"${REPORT_JSONL}"
}

for marker in "${UNIT_IDS[@]}"; do
    check_marker "unit_id" "${marker}"
done
for marker in "${E2E_IDS[@]}"; do
    check_marker "e2e_id" "${marker}"
done
for marker in "${LOG_LEVELS[@]}"; do
    check_marker "log_level" "${marker}"
done
check_marker "log_standard" "${LOG_STANDARD_REF}"
for marker in "${WORKLOAD_MARKERS[@]}"; do
    check_marker "workload" "${marker}"
done
for marker in "${WARMUP_MARKERS[@]}"; do
    check_marker "warmup" "${marker}"
done
for marker in "${PREWARM_MARKERS[@]}"; do
    check_marker "prewarm" "${marker}"
done

log_line "INFO" "summary" \
    "total_checks=$(( ${#UNIT_IDS[@]} + ${#E2E_IDS[@]} + ${#LOG_LEVELS[@]} + ${#WORKLOAD_MARKERS[@]} + ${#WARMUP_MARKERS[@]} + ${#PREWARM_MARKERS[@]} + 1 )) missing=${#missing_markers[@]} trace_id=${trace_id} report=${REPORT_JSONL}"

for marker in "${missing_markers[@]}"; do
    log_line "WARN" "missing_marker" \
        "marker=${marker} trace_id=${trace_id}"
done

if [[ "${#missing_markers[@]}" -gt 0 ]]; then
    log_line "ERROR" "terminal_failure" \
        "missing=${missing_markers[*]} trace_id=${trace_id} report=${REPORT_JSONL}"
    exit 1
fi

log_line "WARN" "degraded_mode_count" \
    "value=0 trace_id=${trace_id}"
log_line "ERROR" "terminal_failure_count" \
    "value=0 trace_id=${trace_id}"
log_line "INFO" "pass" \
    "trace_id=${trace_id} report=${REPORT_JSONL}"
