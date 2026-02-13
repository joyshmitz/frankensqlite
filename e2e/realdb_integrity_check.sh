#!/usr/bin/env bash
set -euo pipefail

BEAD_ID="bd-3bsz"
LOG_STANDARD_REF="bd-1fpm"
LOG_SCHEMA_VERSION="1.0.0"
SCENARIO_ID="REALDB-1"
SEED=202602130301
BACKEND="both"
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOLDEN_DIR_DEFAULT="${WORKSPACE_ROOT}/sample_sqlite_db_files/golden"
RUN_ID="${BEAD_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_ROOT="${WORKSPACE_ROOT}/test-results/bd_3bsz"
REPORT_JSONL="${REPORT_ROOT}/${RUN_ID}.jsonl"
REPORT_REL="${REPORT_JSONL#${WORKSPACE_ROOT}/}"
WORKER="${HOSTNAME:-local}"
JSON_OUTPUT=false

usage() {
  cat <<'EOF'
Usage:
  e2e/realdb_integrity_check.sh [--golden-dir PATH] [--json]

Validates that all golden SQLite DB files pass:
  - PRAGMA integrity_check == "ok"
  - PRAGMA page_count > 0
  - SELECT count(*) FROM sqlite_master > 0

Notes:
  - This script is read-only with respect to the DB files.
  - It is intended to gate the E2E RealDB suite (bead: bd-3bsz).
EOF
}

emit_event() {
  local phase="$1"
  local event_type="$2"
  local outcome="$3"
  local message="$4"
  local error_code="$5"
  local artifact_paths_json="$6"
  local context_json="$7"
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
    --argjson seed "${SEED}" \
    --arg backend "${BACKEND}" \
    --arg worker "${WORKER}" \
    --arg outcome "${outcome}" \
    --arg message "${message}" \
    --arg error_code "${error_code}" \
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
      seed: $seed,
      backend: $backend,
      outcome: $outcome,
      error_code: (if $error_code == "" then null else $error_code end),
      artifact_paths: $artifact_paths,
      context: ($context + {
        message: $message,
        worker: $worker,
        log_standard_ref: $log_standard_ref,
        artifact_paths: ($artifact_paths | join(","))
      })
    }' >>"${REPORT_JSONL}"
}

log_event() {
  local level="$1"
  local case_name="$2"
  local phase="$3"
  local event_type="$4"
  local outcome="$5"
  local error_code="$6"
  local details="$7"
  local artifact_paths_json="$8"
  local context_json="$9"

  emit_event \
    "${phase}" \
    "${event_type}" \
    "${outcome}" \
    "${case_name}" \
    "${error_code}" \
    "${artifact_paths_json}" \
    "${context_json}"

  printf 'bead_id=%s level=%s run_id=%s scenario_id=%s phase=%s event_type=%s case=%s %s reference=%s\n' \
    "${BEAD_ID}" "${level}" "${RUN_ID}" "${SCENARIO_ID}" "${phase}" "${event_type}" "${case_name}" "${details}" "${LOG_STANDARD_REF}"
}

artifact_paths_with_optional_db() {
  local db_path="${1:-}"
  if [[ -z "${db_path}" ]]; then
    jq -nc --arg report "${REPORT_REL}" '[ $report ]'
    return
  fi

  local db_rel="${db_path#${WORKSPACE_ROOT}/}"
  jq -nc --arg report "${REPORT_REL}" --arg db "${db_rel}" '[ $report, $db ] | unique'
}

golden_dir="${GOLDEN_DIR_DEFAULT}"
mkdir -p "${REPORT_ROOT}"
: >"${REPORT_JSONL}"

start_artifacts="$(artifact_paths_with_optional_db)"
start_context="$(
  jq -nc \
    --arg golden_dir "${golden_dir}" \
    --arg report "${REPORT_JSONL}" \
    '{golden_dir: $golden_dir, report: $report}'
)"
log_event "DEBUG" "start" "setup" "start" "running" "" \
  "golden_dir=${golden_dir} report=${REPORT_JSONL}" \
  "${start_artifacts}" \
  "${start_context}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --golden-dir)
      golden_dir="${2:-}"
      shift 2
      ;;
    --json)
      JSON_OUTPUT=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      arg_context="$(jq -nc --arg arg "$1" '{argument: $arg}')"
      arg_artifacts="$(artifact_paths_with_optional_db)"
      log_event "ERROR" "invalid_arg" "setup" "fail" "fail" "E_INVALID_ARG" \
        "unknown_arg=$1" \
        "${arg_artifacts}" \
        "${arg_context}"
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${golden_dir}" ]]; then
  missing_dir_context="$(jq -nc --arg golden_dir "${golden_dir}" '{golden_dir: $golden_dir}')"
  missing_dir_artifacts="$(artifact_paths_with_optional_db)"
  log_event "ERROR" "missing_golden_dir" "setup" "fail" "fail" "E_MISSING_GOLDEN_DIR" \
    "golden_dir=${golden_dir}" \
    "${missing_dir_artifacts}" \
    "${missing_dir_context}"
  echo "Golden dir does not exist: ${golden_dir}" >&2
  exit 1
fi

shopt -s nullglob
db_files=("${golden_dir}"/*.db "${golden_dir}"/*.sqlite "${golden_dir}"/*.sqlite3)
shopt -u nullglob

if [[ "${#db_files[@]}" -eq 0 ]]; then
  missing_db_context="$(jq -nc --arg golden_dir "${golden_dir}" '{golden_dir: $golden_dir}')"
  missing_db_artifacts="$(artifact_paths_with_optional_db)"
  log_event "ERROR" "no_db_files" "setup" "fail" "fail" "E_NO_DB_FILES" \
    "golden_dir=${golden_dir}" \
    "${missing_db_artifacts}" \
    "${missing_db_context}"
  echo "No DB files found under: ${golden_dir}" >&2
  exit 1
fi

failures=0
checked=0

for db in "${db_files[@]}"; do
  checked=$((checked + 1))
  db_artifacts="$(artifact_paths_with_optional_db "${db}")"
  db_rel="${db#${WORKSPACE_ROOT}/}"
  db_start_context="$(jq -nc --arg db "${db_rel}" '{db: $db}')"
  log_event "DEBUG" "db_check_start" "execute" "start" "running" "" \
    "db=${db}" \
    "${db_artifacts}" \
    "${db_start_context}"

  set +e
  integrity_raw="$(sqlite3 "${db}" "PRAGMA integrity_check;" 2>&1)"
  integrity_exit=$?
  set -e
  if [[ "${integrity_exit}" -ne 0 ]]; then
    db_fail_context="$(
      jq -nc \
        --arg db "${db_rel}" \
        --arg step "integrity_check" \
        --arg sqlite_output "${integrity_raw}" \
        '{db: $db, step: $step, sqlite_output: $sqlite_output}'
    )"
    log_event "ERROR" "db_check_failed" "validate" "fail" "fail" "E_SQLITE_COMMAND" \
      "db=${db} step=integrity_check exit_code=${integrity_exit}" \
      "${db_artifacts}" \
      "${db_fail_context}"
    failures=$((failures + 1))
    continue
  fi

  integrity="$(printf '%s\n' "${integrity_raw}" | tr -d '\r' | tail -n 1)"
  if [[ "${integrity}" != "ok" ]]; then
    db_fail_context="$(
      jq -nc \
        --arg db "${db_rel}" \
        --arg step "integrity_check" \
        --arg integrity "${integrity}" \
        '{db: $db, step: $step, integrity: $integrity}'
    )"
    log_event "ERROR" "db_check_failed" "validate" "fail" "fail" "E_INTEGRITY_CHECK" \
      "db=${db} step=integrity_check value=${integrity}" \
      "${db_artifacts}" \
      "${db_fail_context}"
    echo "FAIL integrity_check: ${db}" >&2
    printf '%s\n' "${integrity_raw}" >&2
    failures=$((failures + 1))
    continue
  fi

  set +e
  page_count_raw="$(sqlite3 "${db}" "PRAGMA page_count;" 2>&1)"
  page_count_exit=$?
  set -e
  if [[ "${page_count_exit}" -ne 0 ]]; then
    db_fail_context="$(
      jq -nc \
        --arg db "${db_rel}" \
        --arg step "page_count" \
        --arg sqlite_output "${page_count_raw}" \
        '{db: $db, step: $step, sqlite_output: $sqlite_output}'
    )"
    log_event "ERROR" "db_check_failed" "validate" "fail" "fail" "E_SQLITE_COMMAND" \
      "db=${db} step=page_count exit_code=${page_count_exit}" \
      "${db_artifacts}" \
      "${db_fail_context}"
    failures=$((failures + 1))
    continue
  fi

  page_count="$(printf '%s\n' "${page_count_raw}" | tr -d '\r' | tail -n 1)"
  if [[ ! "${page_count}" =~ ^[0-9]+$ ]] || [[ "${page_count}" -le 0 ]]; then
    db_fail_context="$(
      jq -nc \
        --arg db "${db_rel}" \
        --arg step "page_count" \
        --arg page_count "${page_count}" \
        '{db: $db, step: $step, page_count: $page_count}'
    )"
    log_event "ERROR" "db_check_failed" "validate" "fail" "fail" "E_PAGE_COUNT" \
      "db=${db} step=page_count value=${page_count:-<empty>}" \
      "${db_artifacts}" \
      "${db_fail_context}"
    echo "FAIL page_count: ${db} page_count=${page_count:-<empty>}" >&2
    failures=$((failures + 1))
    continue
  fi

  set +e
  master_count_raw="$(sqlite3 "${db}" "SELECT count(*) FROM sqlite_master;" 2>&1)"
  master_count_exit=$?
  set -e
  if [[ "${master_count_exit}" -ne 0 ]]; then
    db_fail_context="$(
      jq -nc \
        --arg db "${db_rel}" \
        --arg step "sqlite_master_count" \
        --arg sqlite_output "${master_count_raw}" \
        '{db: $db, step: $step, sqlite_output: $sqlite_output}'
    )"
    log_event "ERROR" "db_check_failed" "validate" "fail" "fail" "E_SQLITE_COMMAND" \
      "db=${db} step=sqlite_master_count exit_code=${master_count_exit}" \
      "${db_artifacts}" \
      "${db_fail_context}"
    failures=$((failures + 1))
    continue
  fi

  master_count="$(printf '%s\n' "${master_count_raw}" | tr -d '\r' | tail -n 1)"
  if [[ ! "${master_count}" =~ ^[0-9]+$ ]] || [[ "${master_count}" -le 0 ]]; then
    db_fail_context="$(
      jq -nc \
        --arg db "${db_rel}" \
        --arg step "sqlite_master_count" \
        --arg master_count "${master_count}" \
        '{db: $db, step: $step, sqlite_master_count: $master_count}'
    )"
    log_event "ERROR" "db_check_failed" "validate" "fail" "fail" "E_SQLITE_MASTER_COUNT" \
      "db=${db} step=sqlite_master_count value=${master_count:-<empty>}" \
      "${db_artifacts}" \
      "${db_fail_context}"
    echo "FAIL sqlite_master count: ${db} sqlite_master_count=${master_count:-<empty>}" >&2
    failures=$((failures + 1))
    continue
  fi

  db_pass_context="$(
    jq -nc \
      --arg db "${db_rel}" \
      --arg integrity "${integrity}" \
      --arg page_count "${page_count}" \
      --arg master_count "${master_count}" \
      '{db: $db, integrity: $integrity, page_count: $page_count, sqlite_master_count: $master_count}'
  )"
  log_event "INFO" "db_check_pass" "validate" "pass" "pass" "" \
    "db=${db} integrity=${integrity} page_count=${page_count} sqlite_master_count=${master_count}" \
    "${db_artifacts}" \
    "${db_pass_context}"
done

if [[ "${failures}" -ne 0 ]]; then
  summary_artifacts="$(artifact_paths_with_optional_db)"
  summary_context="$(
    jq -nc \
      --arg checked "${checked}" \
      --arg failures "${failures}" \
      --arg golden_dir "${golden_dir}" \
      '{checked: $checked, failures: $failures, golden_dir: $golden_dir}'
  )"
  log_event "ERROR" "summary_fail" "report" "fail" "fail" "E_INTEGRITY_GATE_FAILED" \
    "failures=${failures} checked=${checked} golden_dir=${golden_dir}" \
    "${summary_artifacts}" \
    "${summary_context}"
  summary_sha256="$(sha256sum "${REPORT_JSONL}" | awk '{print $1}')"
  echo "Integrity gate FAILED: ${failures}/${checked} database(s) failed." >&2
  if ${JSON_OUTPUT}; then
    cat <<ENDJSON
{
  "bead_id": "${BEAD_ID}",
  "run_id": "${RUN_ID}",
  "status": "fail",
  "checked": ${checked},
  "failures": ${failures},
  "report_jsonl": "${REPORT_JSONL}",
  "report_sha256": "${summary_sha256}"
}
ENDJSON
  fi
  exit 1
fi

summary_artifacts="$(artifact_paths_with_optional_db)"
summary_context="$(
  jq -nc \
    --arg checked "${checked}" \
    --arg failures "${failures}" \
    --arg golden_dir "${golden_dir}" \
    '{checked: $checked, failures: $failures, golden_dir: $golden_dir}'
)"
log_event "INFO" "summary_pass" "report" "pass" "pass" "" \
  "checked=${checked} failures=${failures} golden_dir=${golden_dir}" \
  "${summary_artifacts}" \
  "${summary_context}"

summary_sha256="$(sha256sum "${REPORT_JSONL}" | awk '{print $1}')"
echo "Integrity gate OK: ${checked} database(s) passed."

if ${JSON_OUTPUT}; then
  cat <<ENDJSON
{
  "bead_id": "${BEAD_ID}",
  "run_id": "${RUN_ID}",
  "status": "pass",
  "checked": ${checked},
  "failures": ${failures},
  "report_jsonl": "${REPORT_JSONL}",
  "report_sha256": "${summary_sha256}"
}
ENDJSON
fi
