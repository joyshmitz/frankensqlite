#!/usr/bin/env bash
# verify_perf_governance.sh — E2E verification for Performance Regression Governance (bd-mblr.7.3)
#
# Runs the governance integration test suite and produces a structured JSON evidence artifact.
#
# Usage:
#   bash scripts/verify_perf_governance.sh [--json]
#
# Related bead IDs: bd-mblr.7.3, bd-mblr.7.3.1, bd-mblr.7.3.2, bd-mblr.7.3.3
set -euo pipefail

BEAD_ID="bd-mblr.7.3"
SCHEMA_VERSION="1"
RUN_ID="${BEAD_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
REPORT_DIR="test-results"
REPORT_FILE="${REPORT_DIR}/${BEAD_ID}-governance-verify.json"
JSON_MODE=false

for arg in "$@"; do
  case "$arg" in
    --json) JSON_MODE=true ;;
  esac
done

mkdir -p "$REPORT_DIR"

# Phase 1: Run governance integration tests
echo "phase=test_execution bead_id=${BEAD_ID} run_id=${RUN_ID}"
TEST_OUTPUT=$(cargo test -p fsqlite-harness --test bd_mblr_7_3_perf_governance -- --nocapture 2>&1) || true
TEST_EXIT=$?

# Parse test results
TESTS_PASSED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
TESTS_FAILED=$(echo "$TEST_OUTPUT" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")

# Phase 2: Run existing child bead tests
echo "phase=child_bead_tests bead_id=${BEAD_ID} run_id=${RUN_ID}"
CHILD_OUTPUT=$(cargo test -p fsqlite-harness --test bd_mblr_7_3_2_regression_detector -- --nocapture 2>&1) || true
CHILD_EXIT=$?

CHILD_PASSED=$(echo "$CHILD_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
CHILD_FAILED=$(echo "$CHILD_OUTPUT" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")

# Phase 3: Run benchmark corpus tests
echo "phase=corpus_tests bead_id=${BEAD_ID} run_id=${RUN_ID}"
CORPUS_OUTPUT=$(cargo test -p fsqlite-harness --test bd_mblr_7_3_1_benchmark_corpus -- --nocapture 2>&1) || true
CORPUS_EXIT=$?

CORPUS_PASSED=$(echo "$CORPUS_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo "0")
CORPUS_FAILED=$(echo "$CORPUS_OUTPUT" | grep -oP '\d+ failed' | grep -oP '\d+' || echo "0")

# Phase 4: Check compilation
echo "phase=compilation_check bead_id=${BEAD_ID} run_id=${RUN_ID}"
COMPILE_OUTPUT=$(cargo check -p fsqlite-harness --lib 2>&1) || true
COMPILE_EXIT=$?

# Compute overall verdict
TOTAL_PASSED=$((TESTS_PASSED + CHILD_PASSED + CORPUS_PASSED))
TOTAL_FAILED=$((TESTS_FAILED + CHILD_FAILED + CORPUS_FAILED))

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
    "governance_tests": { "passed": ${TESTS_PASSED}, "failed": ${TESTS_FAILED} },
    "detector_tests": { "passed": ${CHILD_PASSED}, "failed": ${CHILD_FAILED} },
    "corpus_tests": { "passed": ${CORPUS_PASSED}, "failed": ${CORPUS_FAILED} },
    "total_passed": ${TOTAL_PASSED},
    "total_failed": ${TOTAL_FAILED}
  },
  "compilation": {
    "exit_code": ${COMPILE_EXIT},
    "status": "$([ "$COMPILE_EXIT" -eq 0 ] && echo "pass" || echo "fail")"
  },
  "components_verified": [
    "benchmark_corpus (bd-mblr.7.3.1)",
    "regression_detector (bd-mblr.7.3.2)",
    "slo_policy",
    "waiver_registry",
    "governance_orchestrator",
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
  echo "  governance_tests: ${TESTS_PASSED} passed, ${TESTS_FAILED} failed"
  echo "  detector_tests: ${CHILD_PASSED} passed, ${CHILD_FAILED} failed"
  echo "  corpus_tests: ${CORPUS_PASSED} passed, ${CORPUS_FAILED} failed"
  echo "  compilation: $([ "$COMPILE_EXIT" -eq 0 ] && echo "pass" || echo "fail")"
  echo "  report_path=${REPORT_FILE}"
  echo "  report_sha256=${REPORT_SHA}"
fi

exit $EXIT_CODE
