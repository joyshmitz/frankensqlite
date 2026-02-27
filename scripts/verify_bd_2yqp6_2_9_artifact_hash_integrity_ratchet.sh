#!/usr/bin/env bash
# Verification gate for bd-2yqp6.2.9:
# deterministic artifact bundle hash integrity + baseline ratchet policy.
#
# Deterministic replay:
#   bash scripts/verify_bd_2yqp6_2_9_artifact_hash_integrity_ratchet.sh

set -euo pipefail

BEAD_ID="bd-2yqp6.2.9"
SCENARIO_ID="ARTIFACT-HASH-RATCHET-B9"
SEED=9209
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${BEAD_ID}-${TIMESTAMP_UTC}-${SEED}"
TRACE_ID="trace-${RUN_ID}"
ARTIFACT_DIR="artifacts/${BEAD_ID}/${RUN_ID}"
EVENTS_JSONL="${ARTIFACT_DIR}/events.jsonl"
TEST_LOG="${ARTIFACT_DIR}/test.log"
REPORT_JSON="${ARTIFACT_DIR}/report.json"
BASELINE_JSON="${ARTIFACT_DIR}/validation_manifest_hash_baseline.json"
ARTIFACT_PREFIX="${ARTIFACT_DIR}/validation-manifest"

BOOTSTRAP_OUT_DIR="${ARTIFACT_DIR}/runner-bootstrap"
REPLAY_OUT_DIR="${ARTIFACT_DIR}/runner-replay"
DRIFT_OUT_DIR="${ARTIFACT_DIR}/runner-drift"
RUNNER_BIN="${RUNNER_BIN:-}"

mkdir -p "${ARTIFACT_DIR}" "${BOOTSTRAP_OUT_DIR}" "${REPLAY_OUT_DIR}" "${DRIFT_OUT_DIR}"

emit_event() {
  local phase="$1"
  local event_type="$2"
  local outcome="$3"
  local message="$4"
  printf '{"trace_id":"%s","run_id":"%s","scenario_id":"%s","seed":%d,"phase":"%s","event_type":"%s","outcome":"%s","timestamp":"%s","message":"%s"}\n' \
    "${TRACE_ID}" "${RUN_ID}" "${SCENARIO_ID}" "${SEED}" "${phase}" "${event_type}" "${outcome}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${message}" \
    >> "${EVENTS_JSONL}"
}

resolve_runner_bin() {
  local -a candidates=()
  if [[ -n "${RUNNER_BIN}" ]]; then
    candidates+=("${RUNNER_BIN}")
  fi
  if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    candidates+=("${CARGO_TARGET_DIR}/debug/validation_manifest_runner")
  fi
  candidates+=(
    "/data/tmp/cargo-target/debug/validation_manifest_runner"
    "target/debug/validation_manifest_runner"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      RUNNER_BIN="${candidate}"
      return 0
    fi
  done

  if [[ ${#candidates[@]} -gt 0 ]]; then
    RUNNER_BIN="${candidates[0]}"
  else
    RUNNER_BIN="target/debug/validation_manifest_runner"
  fi
  return 1
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

run_gate_allow_exit_codes() {
  local label="$1"
  local allowed_codes="$2"
  shift 2

  emit_event "${label}" "start" "running" "running (allowed exits: ${allowed_codes}): $*"
  set +e
  "$@" 2>&1 | tee -a "${TEST_LOG}"
  local cmd_exit=${PIPESTATUS[0]}
  set -e

  for code in ${allowed_codes}; do
    if [[ "${cmd_exit}" -eq "${code}" ]]; then
      emit_event "${label}" "pass" "pass" "command exited with allowed code ${cmd_exit}"
      return 0
    fi
  done

  emit_event "${label}" "fail" "fail" "command exited with disallowed code ${cmd_exit}"
  return 1
}

echo "=== ${BEAD_ID}: artifact hash integrity ratchet verification ==="
echo "run_id=${RUN_ID}"
echo "trace_id=${TRACE_ID}"
echo "scenario_id=${SCENARIO_ID}"

emit_event "bootstrap" "start" "running" "verification started"

RESULT="pass"

if ! run_gate "contract_tests" \
  rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_2_9_artifact_hash_integrity_ratchet -- --nocapture; then
  RESULT="fail"
fi

if ! run_gate "runner_binary_build" \
  rch exec -- bash -lc "cargo build -p fsqlite-harness --bin validation_manifest_runner"; then
  RESULT="fail"
fi

if ! resolve_runner_bin; then
  emit_event "runner_binary_missing" "fail" "fail" "runner binary unavailable at ${RUNNER_BIN} after build attempt"
  RESULT="fail"
else
  emit_event "runner_binary_resolved" "info" "pass" "using runner binary at ${RUNNER_BIN}"
fi

COMMON_RUNNER_ARGS=(
  --workspace-root "."
  --commit-sha "bd-2yqp6.2.9-test-sha"
  --run-id "bd-2yqp6.2.9-run-fixed"
  --trace-id "trace-bd-2yqp6.2.9-fixed"
  --scenario-id "${SCENARIO_ID}"
  --root-seed "${SEED}"
  --artifact-uri-prefix "${ARTIFACT_PREFIX}"
  --artifact-hash-baseline "${BASELINE_JSON}"
)

if ! run_gate_allow_exit_codes "runner_bootstrap" "0 1" \
  rch exec -- "${RUNNER_BIN}" \
    "${COMMON_RUNNER_ARGS[@]}" \
    --generated-unix-ms 1730000000000 \
    --output-dir "${BOOTSTRAP_OUT_DIR}" \
    --output-json "${BOOTSTRAP_OUT_DIR}/validation_manifest.json" \
    --output-human "${BOOTSTRAP_OUT_DIR}/validation_manifest.md" \
    --allow-artifact-hash-bootstrap \
    --artifact-hash-update-reason initial-approved-baseline-bootstrap \
    --write-artifact-hash-baseline; then
  RESULT="fail"
fi

if ! run_gate_allow_exit_codes "runner_replay_same_hash" "0 1" \
  rch exec -- "${RUNNER_BIN}" \
    "${COMMON_RUNNER_ARGS[@]}" \
    --generated-unix-ms 1730000000000 \
    --output-dir "${REPLAY_OUT_DIR}" \
    --output-json "${REPLAY_OUT_DIR}/validation_manifest.json" \
    --output-human "${REPLAY_OUT_DIR}/validation_manifest.md"; then
  RESULT="fail"
fi

emit_event "runner_drift_expected_fail" "start" "running" "running expected ratchet failure on hash drift"
set +e
rch exec -- "${RUNNER_BIN}" \
  "${COMMON_RUNNER_ARGS[@]}" \
  --generated-unix-ms 1730000000001 \
  --output-dir "${DRIFT_OUT_DIR}" \
  --output-json "${DRIFT_OUT_DIR}/validation_manifest.json" \
  --output-human "${DRIFT_OUT_DIR}/validation_manifest.md" \
  > >(tee -a "${TEST_LOG}") 2> >(tee -a "${TEST_LOG}" >&2)
DRIFT_EXIT=$?
set -e

if [[ ${DRIFT_EXIT} -eq 0 ]]; then
  emit_event "runner_drift_expected_fail" "fail" "fail" "expected ratchet block but runner succeeded"
  RESULT="fail"
else
  emit_event "runner_drift_expected_fail" "pass" "pass" "ratchet blocked unreviewed hash drift as expected"
fi

TEST_LOG_SHA256="$(sha256sum "${TEST_LOG}" | awk '{print $1}')"
EVENTS_SHA256="$(sha256sum "${EVENTS_JSONL}" | awk '{print $1}')"
if [[ -f "${BASELINE_JSON}" ]]; then
  BASELINE_SHA256="$(sha256sum "${BASELINE_JSON}" | awk '{print $1}')"
else
  BASELINE_SHA256="missing"
  RESULT="fail"
  emit_event "baseline_artifact" "fail" "fail" "baseline file missing at ${BASELINE_JSON}"
fi

cat > "${REPORT_JSON}" <<EOF_JSON
{
  "trace_id": "${TRACE_ID}",
  "run_id": "${RUN_ID}",
  "scenario_id": "${SCENARIO_ID}",
  "seed": ${SEED},
  "bead_id": "${BEAD_ID}",
  "commands": [
    "rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_2_9_artifact_hash_integrity_ratchet -- --nocapture",
    "rch exec -- cargo run -p fsqlite-harness --bin validation_manifest_runner -- <bootstrap args>",
    "rch exec -- cargo run -p fsqlite-harness --bin validation_manifest_runner -- <replay args>",
    "rch exec -- cargo run -p fsqlite-harness --bin validation_manifest_runner -- <drift args>"
  ],
  "ratchet": {
    "baseline_file": "${BASELINE_JSON}",
    "baseline_sha256": "${BASELINE_SHA256}",
    "drift_exit_code": ${DRIFT_EXIT}
  },
  "artifacts": {
    "events_jsonl": "${EVENTS_JSONL}",
    "events_sha256": "${EVENTS_SHA256}",
    "test_log": "${TEST_LOG}",
    "test_log_sha256": "${TEST_LOG_SHA256}",
    "bootstrap_manifest": "${BOOTSTRAP_OUT_DIR}/validation_manifest.json",
    "replay_manifest": "${REPLAY_OUT_DIR}/validation_manifest.json",
    "drift_manifest": "${DRIFT_OUT_DIR}/validation_manifest.json"
  },
  "result": "${RESULT}"
}
EOF_JSON

emit_event "finalize" "info" "${RESULT}" "report written to ${REPORT_JSON}"

if [[ "${RESULT}" != "pass" ]]; then
  echo "[GATE FAIL] ${BEAD_ID} artifact hash integrity ratchet verification failed"
  exit 1
fi

echo "[GATE PASS] ${BEAD_ID} artifact hash integrity ratchet verification passed"
