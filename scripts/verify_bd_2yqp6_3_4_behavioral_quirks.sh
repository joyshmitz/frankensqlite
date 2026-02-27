#!/usr/bin/env bash
# Verification gate for bd-2yqp6.3.4:
# adversarial behavioral-quirk corpus + differential oracle parity checks.
#
# Deterministic replay:
#   bash scripts/verify_bd_2yqp6_3_4_behavioral_quirks.sh
#
# Structured logging contract:
#   emits JSONL events with trace_id/run_id/scenario_id/seed/timing/outcome
#   to artifacts/bd-2yqp6.3.4/<run_id>/events.jsonl

set -euo pipefail

BEAD_ID="bd-2yqp6.3.4"
SCENARIO_ID="PARITY-CORPUS-C4"
SEED="${SEED:-3520}"
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${BEAD_ID}-${TIMESTAMP_UTC}-${SEED}"
TRACE_ID="trace-${RUN_ID}"
ARTIFACT_DIR="artifacts/${BEAD_ID}/${RUN_ID}"
EVENTS_JSONL="${ARTIFACT_DIR}/events.jsonl"
TEST_LOG="${ARTIFACT_DIR}/test.log"
REPORT_JSON="${ARTIFACT_DIR}/report.json"

mkdir -p "${ARTIFACT_DIR}"
start_ns="$(date +%s%N)"

emit_event() {
  local phase="$1"
  local event_type="$2"
  local outcome="$3"
  local message="$4"
  local now_ns elapsed_ms
  now_ns="$(date +%s%N)"
  elapsed_ms="$(( (now_ns - start_ns) / 1000000 ))"
  printf '{"trace_id":"%s","run_id":"%s","scenario_id":"%s","seed":%d,"phase":"%s","event_type":"%s","outcome":"%s","elapsed_ms":%d,"timestamp":"%s","message":"%s"}\n' \
    "${TRACE_ID}" "${RUN_ID}" "${SCENARIO_ID}" "${SEED}" "${phase}" "${event_type}" "${outcome}" "${elapsed_ms}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${message}" \
    >> "${EVENTS_JSONL}"
}

run_gate() {
  local phase="$1"
  shift
  emit_event "${phase}" "start" "running" "running: $*"
  if "$@" 2>&1 | tee -a "${TEST_LOG}"; then
    emit_event "${phase}" "pass" "pass" "command passed"
    return 0
  fi
  emit_event "${phase}" "fail" "fail" "command failed"
  return 1
}

echo "=== ${BEAD_ID}: adversarial behavioral-quirk verification ==="
echo "run_id=${RUN_ID}"
echo "trace_id=${TRACE_ID}"
echo "scenario_id=${SCENARIO_ID}"

emit_event "bootstrap" "start" "running" "verification started"

RESULT="pass"

if ! run_gate "quirk_fixture_differential_tests" \
  rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_3_4_behavioral_quirks_corpus -- --nocapture; then
  RESULT="fail"
fi

if ! run_gate "differential_v2_regression_tests" \
  rch exec -- cargo test -p fsqlite-harness --test differential_v2_test -- --nocapture; then
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
    "rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_3_4_behavioral_quirks_corpus -- --nocapture",
    "rch exec -- cargo test -p fsqlite-harness --test differential_v2_test -- --nocapture"
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
  echo "[GATE FAIL] ${BEAD_ID} adversarial behavioral-quirk verification failed"
  exit 1
fi

echo "[GATE PASS] ${BEAD_ID} adversarial behavioral-quirk verification passed"
echo "Artifacts: ${ARTIFACT_DIR}"
