#!/usr/bin/env bash
# verify_leapfrog_join.sh — deterministic verifier for bd-2qr3a.2
set -euo pipefail

BEAD_ID="bd-2qr3a.2"
SCHEMA_VERSION="1"
RUN_ID="${BEAD_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_DIR="test-results"
REPORT_FILE="${REPORT_DIR}/${BEAD_ID}-leapfrog-verify.json"
JSON_MODE=false
NO_RCH=false

for arg in "$@"; do
  case "$arg" in
    --json) JSON_MODE=true ;;
    --no-rch) NO_RCH=true ;;
    *)
      echo "ERROR: unknown argument '$arg'" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$REPORT_DIR"

parse_passed() {
  echo "$1" | grep 'test result:' | sed -nE 's/.* ([0-9]+) passed.*/\1/p' | tail -1 || echo "0"
}

parse_failed() {
  echo "$1" | grep 'test result:' | sed -nE 's/.* ([0-9]+) failed.*/\1/p' | tail -1 || echo "0"
}

declare -A CASE_STATUS
declare -A CASE_PASSED
declare -A CASE_FAILED
declare -A CASE_EXIT

run_case() {
  local case_id="$1"
  shift

  echo "phase=${case_id} bead_id=${BEAD_ID} run_id=${RUN_ID}"

  local output
  local exit_code
  if ! $NO_RCH && command -v rch >/dev/null 2>&1 && [ "${1:-}" = "cargo" ]; then
    if output=$(rch exec -- "$@" 2>&1); then
      exit_code=0
    else
      exit_code=$?
    fi
  else
    if output=$("$@" 2>&1); then
      exit_code=0
    else
      exit_code=$?
    fi
  fi
  local passed
  local failed
  passed=$(parse_passed "$output")
  failed=$(parse_failed "$output")

  CASE_PASSED["$case_id"]="${passed:-0}"
  CASE_FAILED["$case_id"]="${failed:-0}"
  CASE_EXIT["$case_id"]="$exit_code"

  if [ "$exit_code" -eq 0 ] && [ "${CASE_FAILED["$case_id"]}" -eq 0 ]; then
    CASE_STATUS["$case_id"]="pass"
  else
    CASE_STATUS["$case_id"]="fail"
  fi
}

run_case \
  "two_way_baseline" \
  cargo test -p fsqlite-vdbe leapfrog_join_two_way_duplicate_multiplicity -- --nocapture

run_case \
  "relation_width_4_to_6" \
  cargo test -p fsqlite-vdbe leapfrog_join_supports_four_to_six_relations -- --nocapture

run_case \
  "galloping_sublinear" \
  cargo test -p fsqlite-vdbe seek_galloping_comparisons_sublinear -- --nocapture

run_case \
  "clippy_vdbe" \
  cargo clippy -p fsqlite-vdbe --all-targets -- -D warnings

TOTAL_PASSED=0
TOTAL_FAILED=0
VERDICT="pass"
for case_id in "${!CASE_STATUS[@]}"; do
  TOTAL_PASSED=$((TOTAL_PASSED + CASE_PASSED["$case_id"]))
  TOTAL_FAILED=$((TOTAL_FAILED + CASE_FAILED["$case_id"]))
  if [ "${CASE_STATUS["$case_id"]}" != "pass" ]; then
    VERDICT="fail"
  fi
done

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REPORT_CONTENT=$(cat <<ENDJSON
{
  "schema_version": ${SCHEMA_VERSION},
  "bead_id": "${BEAD_ID}",
  "run_id": "${RUN_ID}",
  "timestamp": "${TIMESTAMP}",
  "verdict": "${VERDICT}",
  "cases": {
    "two_way_baseline": {
      "status": "${CASE_STATUS["two_way_baseline"]}",
      "exit_code": ${CASE_EXIT["two_way_baseline"]},
      "passed": ${CASE_PASSED["two_way_baseline"]},
      "failed": ${CASE_FAILED["two_way_baseline"]}
    },
    "relation_width_4_to_6": {
      "status": "${CASE_STATUS["relation_width_4_to_6"]}",
      "exit_code": ${CASE_EXIT["relation_width_4_to_6"]},
      "passed": ${CASE_PASSED["relation_width_4_to_6"]},
      "failed": ${CASE_FAILED["relation_width_4_to_6"]}
    },
    "galloping_sublinear": {
      "status": "${CASE_STATUS["galloping_sublinear"]}",
      "exit_code": ${CASE_EXIT["galloping_sublinear"]},
      "passed": ${CASE_PASSED["galloping_sublinear"]},
      "failed": ${CASE_FAILED["galloping_sublinear"]}
    },
    "clippy_vdbe": {
      "status": "${CASE_STATUS["clippy_vdbe"]}",
      "exit_code": ${CASE_EXIT["clippy_vdbe"]},
      "passed": ${CASE_PASSED["clippy_vdbe"]},
      "failed": ${CASE_FAILED["clippy_vdbe"]}
    }
  },
  "totals": {
    "passed": ${TOTAL_PASSED},
    "failed": ${TOTAL_FAILED}
  },
  "replay_command": "./scripts/verify_leapfrog_join.sh --json"
}
ENDJSON
)

echo "$REPORT_CONTENT" > "$REPORT_FILE"
REPORT_SHA=$(sha256sum "$REPORT_FILE" | cut -d' ' -f1)

if $JSON_MODE; then
  echo "$REPORT_CONTENT"
else
  echo "phase=complete bead_id=${BEAD_ID} run_id=${RUN_ID} verdict=${VERDICT}"
  echo "  two_way_baseline: ${CASE_STATUS["two_way_baseline"]}"
  echo "  relation_width_4_to_6: ${CASE_STATUS["relation_width_4_to_6"]}"
  echo "  galloping_sublinear: ${CASE_STATUS["galloping_sublinear"]}"
  echo "  clippy_vdbe: ${CASE_STATUS["clippy_vdbe"]}"
  echo "  report_path=${REPORT_FILE}"
  echo "  report_sha256=${REPORT_SHA}"
fi

if [ "$VERDICT" = "pass" ]; then
  exit 0
fi
exit 1
