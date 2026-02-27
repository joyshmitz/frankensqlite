#!/usr/bin/env bash
# Verification gate for bd-2yqp6.2.7:
# differential metadata schema contract, validator fixtures, and deterministic
# serialization checks.
#
# Deterministic replay:
#   bash scripts/verify_bd_2yqp6_2_7_differential_metadata_schema.sh

set -euo pipefail

BEAD_ID="bd-2yqp6.2.7"
SCENARIO_ID="DIFF-METADATA-SCHEMA-B7"
SEED=4242
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${BEAD_ID}-${TIMESTAMP_UTC}-${SEED}"
TRACE_ID="trace-${RUN_ID}"
ARTIFACT_DIR="artifacts/${BEAD_ID}/${RUN_ID}"
EVENTS_JSONL="${ARTIFACT_DIR}/events.jsonl"
TEST_LOG="${ARTIFACT_DIR}/test.log"
REPORT_JSON="${ARTIFACT_DIR}/report.json"

mkdir -p "${ARTIFACT_DIR}"

emit_event() {
  local phase="$1"
  local event_type="$2"
  local outcome="$3"
  local message="$4"
  printf '{"trace_id":"%s","run_id":"%s","scenario_id":"%s","seed":%d,"phase":"%s","event_type":"%s","outcome":"%s","timestamp":"%s","message":"%s"}\n' \
    "${TRACE_ID}" "${RUN_ID}" "${SCENARIO_ID}" "${SEED}" "${phase}" "${event_type}" "${outcome}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${message}" \
    >> "${EVENTS_JSONL}"
}

run_gate() {
  local label="$1"
  shift

  emit_event "${label}" "start" "running" "running: $*"
  if "$@" 2>&1 | tee -a "${TEST_LOG}"; then
    emit_event "${label}" "pass" "pass" "command passed"
    return 0
  fi
  emit_event "${label}" "fail" "fail" "command failed"
  return 1
}

echo "=== ${BEAD_ID}: differential metadata schema verification ==="
echo "run_id=${RUN_ID}"
echo "trace_id=${TRACE_ID}"
echo "scenario_id=${SCENARIO_ID}"

emit_event "bootstrap" "start" "running" "verification started"

RESULT="pass"

if ! run_gate "schema_contract_tests" \
  rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_2_7_differential_metadata_schema -- --nocapture; then
  RESULT="fail"
fi

if ! run_gate "integration_metadata_tests" \
  rch exec -- cargo test -p fsqlite-harness --test differential_v2_test differential_result_metadata_is_populated_and_strictly_valid -- --nocapture; then
  RESULT="fail"
fi

TEST_LOG_SHA256="$(sha256sum "${TEST_LOG}" | awk '{print $1}')"
EVENTS_SHA256="$(sha256sum "${EVENTS_JSONL}" | awk '{print $1}')"

cat > "${REPORT_JSON}" <<EOF_JSON
{
  "trace_id": "${TRACE_ID}",
  "run_id": "${RUN_ID}",
  "scenario_id": "${SCENARIO_ID}",
  "seed": ${SEED},
  "bead_id": "${BEAD_ID}",
  "commands": [
    "rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_2_7_differential_metadata_schema -- --nocapture",
    "rch exec -- cargo test -p fsqlite-harness --test differential_v2_test differential_result_metadata_is_populated_and_strictly_valid -- --nocapture"
  ],
  "artifacts": {
    "events_jsonl": "${EVENTS_JSONL}",
    "events_sha256": "${EVENTS_SHA256}",
    "test_log": "${TEST_LOG}",
    "test_log_sha256": "${TEST_LOG_SHA256}"
  },
  "result": "${RESULT}"
}
EOF_JSON

emit_event "finalize" "info" "${RESULT}" "report written to ${REPORT_JSON}"

if [[ "${RESULT}" != "pass" ]]; then
  echo "[GATE FAIL] ${BEAD_ID} differential metadata schema verification failed"
  exit 1
fi

echo "[GATE PASS] ${BEAD_ID} differential metadata schema verification passed"
