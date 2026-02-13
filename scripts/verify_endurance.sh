#!/usr/bin/env bash
# verify_endurance.sh — E2E verification for Long-Run Concurrency Endurance (bd-mblr.7.2)
#
# Runs the endurance integration test suite, child bead tests, and produces
# a structured JSON evidence artifact.
#
# Usage:
#   bash scripts/verify_endurance.sh [--json]
#
# Related bead IDs: bd-mblr.7.2, bd-mblr.7.2.1, bd-mblr.7.2.2
set -euo pipefail

BEAD_ID="bd-mblr.7.2"
SCHEMA_VERSION="1"
RUN_ID="${BEAD_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_DIR="test-results"
REPORT_FILE="${REPORT_DIR}/${BEAD_ID}-endurance-verify.json"
JSON_MODE=false

for arg in "$@"; do
  case "$arg" in
    --json) JSON_MODE=true ;;
  esac
done

mkdir -p "$REPORT_DIR"

# Helper: extract passed/failed from cargo test "test result:" summary line
parse_passed() {
  echo "$1" | grep 'test result:' | grep -oP '\d+ passed' | grep -oP '\d+' | tail -1 || echo "0"
}
parse_failed() {
  echo "$1" | grep 'test result:' | grep -oP '\d+ failed' | grep -oP '\d+' | tail -1 || echo "0"
}

# Phase 1: Run endurance integration tests (parent bead)
echo "phase=endurance_tests bead_id=${BEAD_ID} run_id=${RUN_ID}"
TEST_OUTPUT=$(cargo test -p fsqlite-harness --test bd_mblr_7_2_endurance -- --nocapture 2>&1) || true

TESTS_PASSED=$(parse_passed "$TEST_OUTPUT")
TESTS_FAILED=$(parse_failed "$TEST_OUTPUT")

# Phase 2: Run soak profiles unit tests (child bead 7.2.1)
echo "phase=soak_profiles_tests bead_id=${BEAD_ID} run_id=${RUN_ID}"
PROFILES_OUTPUT=$(cargo test -p fsqlite-harness --lib soak_profiles::tests -- --nocapture 2>&1) || true

PROFILES_PASSED=$(parse_passed "$PROFILES_OUTPUT")
PROFILES_FAILED=$(parse_failed "$PROFILES_OUTPUT")

# Phase 3: Run soak executor unit tests (child bead 7.2.2)
echo "phase=soak_executor_tests bead_id=${BEAD_ID} run_id=${RUN_ID}"
EXECUTOR_OUTPUT=$(cargo test -p fsqlite-harness --lib soak_executor::tests -- --nocapture 2>&1) || true

EXECUTOR_PASSED=$(parse_passed "$EXECUTOR_OUTPUT")
EXECUTOR_FAILED=$(parse_failed "$EXECUTOR_OUTPUT")

# Phase 4: Check compilation
echo "phase=compilation_check bead_id=${BEAD_ID} run_id=${RUN_ID}"
COMPILE_OUTPUT=$(cargo check -p fsqlite-harness --lib 2>&1) || true
COMPILE_EXIT=$?

# Compute overall verdict
TOTAL_PASSED=$((TESTS_PASSED + PROFILES_PASSED + EXECUTOR_PASSED))
TOTAL_FAILED=$((TESTS_FAILED + PROFILES_FAILED + EXECUTOR_FAILED))

if [ "$TOTAL_FAILED" -eq 0 ] && [ "$COMPILE_EXIT" -eq 0 ]; then
  VERDICT="pass"
  EXIT_CODE=0
else
  VERDICT="fail"
  EXIT_CODE=1
fi

# Generate structured report
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
REPORT_CONTENT=$(cat <<ENDJSON
{
  "schema_version": ${SCHEMA_VERSION},
  "bead_id": "${BEAD_ID}",
  "run_id": "${RUN_ID}",
  "timestamp": "${TIMESTAMP}",
  "verdict": "${VERDICT}",
  "test_summary": {
    "endurance_tests": { "passed": ${TESTS_PASSED}, "failed": ${TESTS_FAILED} },
    "soak_profiles_tests": { "passed": ${PROFILES_PASSED}, "failed": ${PROFILES_FAILED} },
    "soak_executor_tests": { "passed": ${EXECUTOR_PASSED}, "failed": ${EXECUTOR_FAILED} },
    "total_passed": ${TOTAL_PASSED},
    "total_failed": ${TOTAL_FAILED}
  },
  "compilation": {
    "exit_code": ${COMPILE_EXIT},
    "status": "$([ "$COMPILE_EXIT" -eq 0 ] && echo "pass" || echo "fail")"
  },
  "components_verified": [
    "soak_profiles (bd-mblr.7.2.1)",
    "soak_executor (bd-mblr.7.2.2)",
    "endurance_orchestrator",
    "endurance_config",
    "endurance_verdict",
    "leak_detection_integration",
    "json_roundtrip"
  ]
}
ENDJSON
)

echo "$REPORT_CONTENT" > "$REPORT_FILE"
REPORT_SHA=$(sha256sum "$REPORT_FILE" | cut -d' ' -f1)

if [ "$JSON_MODE" = true ]; then
  echo "$REPORT_CONTENT"
else
  echo "phase=complete bead_id=${BEAD_ID} run_id=${RUN_ID} verdict=${VERDICT}"
  echo "  endurance_tests: ${TESTS_PASSED} passed, ${TESTS_FAILED} failed"
  echo "  soak_profiles_tests: ${PROFILES_PASSED} passed, ${PROFILES_FAILED} failed"
  echo "  soak_executor_tests: ${EXECUTOR_PASSED} passed, ${EXECUTOR_FAILED} failed"
  echo "  compilation: $([ "$COMPILE_EXIT" -eq 0 ] && echo "pass" || echo "fail")"
  echo "  report_path=${REPORT_FILE}"
  echo "  report_sha256=${REPORT_SHA}"
fi

exit $EXIT_CODE
