#!/usr/bin/env bash
# Verification gate for bd-2yqp6.2.5:
# deterministic oracle preflight doctor classification + remediation contract.
#
# Deterministic replay:
#   bash scripts/verify_bd_2yqp6_2_5_oracle_preflight_doctor.sh

set -euo pipefail

BEAD_ID="bd-2yqp6.2.5"
SCENARIO_ID="DIFF-ORACLE-PREFLIGHT-B5"
SEED=4242
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${BEAD_ID}-${TIMESTAMP_UTC}-${SEED}"
TRACE_ID="trace-${RUN_ID}"
ARTIFACT_DIR="${REPO_ROOT}/artifacts/${BEAD_ID}/${RUN_ID}"
EVENTS_JSONL="${ARTIFACT_DIR}/events.jsonl"
TEST_LOG="${ARTIFACT_DIR}/test.log"
REPORT_JSON="${ARTIFACT_DIR}/report.json"
DOCTOR_WORKSPACE="${ARTIFACT_DIR}/doctor-workspace"
DOCTOR_FIXTURE_DIR="${DOCTOR_WORKSPACE}/crates/fsqlite-harness/conformance"
DOCTOR_MANIFEST_PATH="${DOCTOR_WORKSPACE}/corpus_manifest.toml"
DOCTOR_SQLITE_BIN="${DOCTOR_WORKSPACE}/sqlite3"
DOCTOR_GREEN_JSON="${ARTIFACT_DIR}/doctor-green.json"
DOCTOR_GREEN_MD="${ARTIFACT_DIR}/doctor-green.md"
DOCTOR_RED_JSON="${ARTIFACT_DIR}/doctor-red.json"
DOCTOR_RED_MD="${ARTIFACT_DIR}/doctor-red.md"

mkdir -p "${ARTIFACT_DIR}" "${DOCTOR_FIXTURE_DIR}"

DOCTOR_RUNNER_BIN_OVERRIDE="${DOCTOR_RUNNER_BIN:-${DIFF_DOCTOR_RUNNER_BIN:-}}"
DOCTOR_RUNNER_CMD=()
DOCTOR_RUNNER_SOURCE=""
if [[ -n "${DOCTOR_RUNNER_BIN_OVERRIDE}" ]]; then
  if [[ -x "${DOCTOR_RUNNER_BIN_OVERRIDE}" ]]; then
    DOCTOR_RUNNER_CMD=("${DOCTOR_RUNNER_BIN_OVERRIDE}")
    DOCTOR_RUNNER_SOURCE="override:${DOCTOR_RUNNER_BIN_OVERRIDE}"
  elif [[ "${DOCTOR_RUNNER_BIN_OVERRIDE}" != /* ]] && [[ -x "${REPO_ROOT}/${DOCTOR_RUNNER_BIN_OVERRIDE}" ]]; then
    DOCTOR_RUNNER_CMD=("${REPO_ROOT}/${DOCTOR_RUNNER_BIN_OVERRIDE}")
    DOCTOR_RUNNER_SOURCE="override:${REPO_ROOT}/${DOCTOR_RUNNER_BIN_OVERRIDE}"
  else
    echo "ERROR: DOCTOR_RUNNER_BIN is not executable: ${DOCTOR_RUNNER_BIN_OVERRIDE}" >&2
    exit 2
  fi
elif [[ -n "${CARGO_TARGET_DIR:-}" && -x "${CARGO_TARGET_DIR}/debug/oracle_preflight_doctor_runner" ]]; then
  DOCTOR_RUNNER_CMD=("${CARGO_TARGET_DIR}/debug/oracle_preflight_doctor_runner")
  DOCTOR_RUNNER_SOURCE="prebuilt:${CARGO_TARGET_DIR}/debug/oracle_preflight_doctor_runner"
elif [[ -x "/data/tmp/cargo-target/debug/oracle_preflight_doctor_runner" ]]; then
  DOCTOR_RUNNER_CMD=("/data/tmp/cargo-target/debug/oracle_preflight_doctor_runner")
  DOCTOR_RUNNER_SOURCE="prebuilt:/data/tmp/cargo-target/debug/oracle_preflight_doctor_runner"
elif [[ -x "${REPO_ROOT}/target/debug/oracle_preflight_doctor_runner" ]]; then
  DOCTOR_RUNNER_CMD=("${REPO_ROOT}/target/debug/oracle_preflight_doctor_runner")
  DOCTOR_RUNNER_SOURCE="prebuilt:${REPO_ROOT}/target/debug/oracle_preflight_doctor_runner"
elif command -v rch >/dev/null 2>&1; then
  DOCTOR_RUNNER_CMD=(rch exec -- cargo run -p fsqlite-harness --bin oracle_preflight_doctor_runner --)
  DOCTOR_RUNNER_SOURCE="rch-cargo-run"
else
  DOCTOR_RUNNER_CMD=(cargo run -p fsqlite-harness --bin oracle_preflight_doctor_runner --)
  DOCTOR_RUNNER_SOURCE="cargo-run"
fi

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

echo "=== ${BEAD_ID}: oracle preflight doctor verification ==="
echo "run_id=${RUN_ID}"
echo "trace_id=${TRACE_ID}"
echo "scenario_id=${SCENARIO_ID}"

emit_event "bootstrap" "start" "running" "verification started"

cat > "${DOCTOR_FIXTURE_DIR}/001_doctor_fixture.json" <<'EOF_FIXTURE'
{
  "id": "001_doctor_fixture",
  "description": "Oracle preflight doctor deterministic fixture",
  "ops": [
    { "op": "open", "path": ":memory:" },
    { "op": "exec", "sql": "CREATE TABLE t(a INTEGER, b TEXT)" },
    { "op": "exec", "sql": "INSERT INTO t VALUES(1, 'x')" },
    { "op": "query", "sql": "SELECT a, b FROM t ORDER BY a", "expect": { "columns": ["a", "b"], "rows": [["1", "x"]], "ordered": true } }
  ],
  "fsqlite_modes": ["compatibility"]
}
EOF_FIXTURE

cat > "${DOCTOR_MANIFEST_PATH}" <<'EOF_MANIFEST'
[meta]
schema_version = "1.0.0"
bead_id = "bd-2yqp6.2.5"
generated_at = "2026-02-27T00:00:00Z"
EOF_MANIFEST

cat > "${DOCTOR_SQLITE_BIN}" <<'EOF_SQLITE'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--version" ]]; then
  echo "3.52.0-test"
  exit 0
fi
echo "unsupported invocation" >&2
exit 2
EOF_SQLITE
chmod +x "${DOCTOR_SQLITE_BIN}"

RESULT="pass"

if ! run_gate "doctor_unit_tests" \
  rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_2_5_oracle_preflight_doctor -- --nocapture; then
  RESULT="fail"
fi

if ! run_gate "doctor_green_run" \
  "${DOCTOR_RUNNER_CMD[@]}" \
    --workspace-root "${DOCTOR_WORKSPACE}" \
    --fixtures-dir "${DOCTOR_FIXTURE_DIR}" \
    --fixture-manifest-path "${DOCTOR_MANIFEST_PATH}" \
    --oracle-binary "${DOCTOR_SQLITE_BIN}" \
    --expected-sqlite-version-prefix "3.52.0" \
    --min-fixture-json-files 1 \
    --min-fixture-entries 1 \
    --min-fixture-sql-statements 2 \
    --run-id "${BEAD_ID}-green-${SEED}" \
    --trace-id "trace-${BEAD_ID}-green" \
    --scenario-id "${SCENARIO_ID}-GREEN" \
    --seed "${SEED}" \
    --generated-unix-ms 1700000000000 \
    --output-json "${DOCTOR_GREEN_JSON}" \
    --output-human "${DOCTOR_GREEN_MD}"; then
  RESULT="fail"
fi

if ! run_gate "doctor_green_assertions" \
  jq -e '.outcome == "green" and .certifying == true and (.findings | length) == 0' "${DOCTOR_GREEN_JSON}"; then
  RESULT="fail"
fi

if ! run_gate_allow_exit_codes "doctor_red_missing_binary" "2" \
  "${DOCTOR_RUNNER_CMD[@]}" \
    --workspace-root "${DOCTOR_WORKSPACE}" \
    --fixtures-dir "${DOCTOR_FIXTURE_DIR}" \
    --fixture-manifest-path "${DOCTOR_MANIFEST_PATH}" \
    --oracle-binary "${DOCTOR_WORKSPACE}/sqlite3-missing" \
    --expected-sqlite-version-prefix "3.52.0" \
    --min-fixture-json-files 1 \
    --min-fixture-entries 1 \
    --min-fixture-sql-statements 2 \
    --run-id "${BEAD_ID}-red-${SEED}" \
    --trace-id "trace-${BEAD_ID}-red" \
    --scenario-id "${SCENARIO_ID}-RED" \
    --seed "${SEED}" \
    --generated-unix-ms 1700000000001 \
    --output-json "${DOCTOR_RED_JSON}" \
    --output-human "${DOCTOR_RED_MD}"; then
  RESULT="fail"
fi

if ! run_gate "doctor_red_assertions" \
  jq -e '.outcome == "red" and (.findings | map(.remediation_class) | index("missing_binary")) != null' "${DOCTOR_RED_JSON}"; then
  RESULT="fail"
fi

TEST_LOG_SHA256="$(sha256sum "${TEST_LOG}" | awk '{print $1}')"
EVENTS_SHA256="$(sha256sum "${EVENTS_JSONL}" | awk '{print $1}')"
GREEN_JSON_SHA256="$(sha256sum "${DOCTOR_GREEN_JSON}" | awk '{print $1}')"
RED_JSON_SHA256="$(sha256sum "${DOCTOR_RED_JSON}" | awk '{print $1}')"

cat > "${REPORT_JSON}" <<EOF_JSON
{
  "trace_id": "${TRACE_ID}",
  "run_id": "${RUN_ID}",
  "scenario_id": "${SCENARIO_ID}",
  "seed": ${SEED},
  "bead_id": "${BEAD_ID}",
  "commands": [
    "rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_2_5_oracle_preflight_doctor -- --nocapture",
    "${DOCTOR_RUNNER_SOURCE} -- <green args>",
    "${DOCTOR_RUNNER_SOURCE} -- <missing-binary args>"
  ],
  "artifacts": {
    "events_jsonl": "${EVENTS_JSONL}",
    "events_sha256": "${EVENTS_SHA256}",
    "test_log": "${TEST_LOG}",
    "test_log_sha256": "${TEST_LOG_SHA256}",
    "doctor_green_json": "${DOCTOR_GREEN_JSON}",
    "doctor_green_json_sha256": "${GREEN_JSON_SHA256}",
    "doctor_red_json": "${DOCTOR_RED_JSON}",
    "doctor_red_json_sha256": "${RED_JSON_SHA256}"
  },
  "result": "${RESULT}"
}
EOF_JSON

emit_event "finalize" "info" "${RESULT}" "report written to ${REPORT_JSON}"

if [[ "${RESULT}" != "pass" ]]; then
  echo "[GATE FAIL] ${BEAD_ID} oracle preflight doctor verification failed"
  exit 1
fi

echo "[GATE PASS] ${BEAD_ID} oracle preflight doctor verification passed"
