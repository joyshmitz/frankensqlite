#!/usr/bin/env bash
# Verification gate for bd-2yqp6.1.3:
# canonical SQLite target version contract + drift detection.
#
# Deterministic replay:
#   bash scripts/verify_bd_2yqp6_1_3_sqlite_version_contract.sh

set -euo pipefail

BEAD_ID="bd-2yqp6.1.3"
SCENARIO_ID="PARITY-VERSION-CONTRACT-A3"
SEED=3520
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${BEAD_ID}-${TIMESTAMP_UTC}-${SEED}"
TRACE_ID="trace-${RUN_ID}"
ARTIFACT_DIR="artifacts/${BEAD_ID}/${RUN_ID}"
EVENTS_JSONL="${ARTIFACT_DIR}/events.jsonl"
REPORT_JSON="${ARTIFACT_DIR}/report.json"
CONTRACT_FILE="sqlite_version_contract.toml"

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

echo "=== ${BEAD_ID}: sqlite version contract verification ==="
echo "run_id=${RUN_ID}"
echo "trace_id=${TRACE_ID}"
echo "scenario_id=${SCENARIO_ID}"

emit_event "bootstrap" "start" "running" "verification started"

if [[ ! -f "${CONTRACT_FILE}" ]]; then
  emit_event "contract_presence" "fail" "fail" "missing ${CONTRACT_FILE}"
  echo "missing ${CONTRACT_FILE}" >&2
  exit 1
fi
emit_event "contract_presence" "pass" "pass" "${CONTRACT_FILE} exists"

emit_event "contract_schema" "start" "running" "validating sqlite version contract + drift checks"
python3 - <<'PY'
import tomllib
from pathlib import Path

contract_path = Path("sqlite_version_contract.toml")
surface_path = Path("supported_surface_matrix.toml")
ledger_path = Path("feature_universe_ledger.toml")
runtime_path = Path("crates/fsqlite-core/src/connection.rs")
parity_module_path = Path("crates/fsqlite-harness/src/differential_v2.rs")
readme_path = Path("README.md")

doc = tomllib.loads(contract_path.read_text(encoding="utf-8"))
meta = doc.get("meta", {})
contract = doc.get("contract", {})
refs = doc.get("references", {})

for key in ("schema_version", "bead_id", "track_id", "generated_at", "contract_owner"):
    if key not in meta or not str(meta[key]).strip():
        raise SystemExit(f"missing meta.{key}")

for key in ("sqlite_target", "runtime_pragma_sqlite_version", "contract_reference_path"):
    if key not in contract or not str(contract[key]).strip():
        raise SystemExit(f"missing contract.{key}")

for key in ("runtime_source", "surface_matrix", "feature_ledger", "parity_report_module", "readme"):
    if key not in refs or not str(refs[key]).strip():
        raise SystemExit(f"missing references.{key}")

sqlite_target = str(contract["sqlite_target"])
contract_path_ref = str(contract["contract_reference_path"])

surface = tomllib.loads(surface_path.read_text(encoding="utf-8"))
surface_meta = surface.get("meta", {})
if str(surface_meta.get("sqlite_target", "")) != sqlite_target:
    raise SystemExit("supported_surface_matrix.toml sqlite_target drift detected")
if str(surface_meta.get("sqlite_version_contract", "")) != contract_path_ref:
    raise SystemExit("supported_surface_matrix.toml sqlite_version_contract drift detected")

ledger = tomllib.loads(ledger_path.read_text(encoding="utf-8"))
ledger_meta = ledger.get("meta", {})
if str(ledger_meta.get("sqlite_target", "")) != sqlite_target:
    raise SystemExit("feature_universe_ledger.toml sqlite_target drift detected")
if str(ledger_meta.get("sqlite_version_contract", "")) != contract_path_ref:
    raise SystemExit("feature_universe_ledger.toml sqlite_version_contract drift detected")

runtime_source = runtime_path.read_text(encoding="utf-8")
marker = '"sqlite_version" => SqliteValue::Text("'
if marker not in runtime_source:
    raise SystemExit("runtime sqlite_version marker missing from connection.rs")
runtime_value = runtime_source.split(marker, 1)[1].split('".to_owned())', 1)[0]
if runtime_value != str(contract["runtime_pragma_sqlite_version"]):
    raise SystemExit(
        "runtime sqlite_version drift: "
        f"expected {contract['runtime_pragma_sqlite_version']} got {runtime_value}"
    )

parity_source = parity_module_path.read_text(encoding="utf-8")
if "SQLITE_VERSION_CONTRACT_PATH" not in parity_source:
    raise SystemExit("differential_v2 parity report missing SQLITE_VERSION_CONTRACT_PATH")
if "TARGET_SQLITE_VERSION" not in parity_source:
    raise SystemExit("differential_v2 parity report missing TARGET_SQLITE_VERSION")

readme = readme_path.read_text(encoding="utf-8")
if sqlite_target not in readme:
    raise SystemExit("README missing canonical sqlite target version")
if contract_path_ref not in readme:
    raise SystemExit("README missing sqlite version contract path")
PY
emit_event "contract_schema" "pass" "pass" "contract + drift checks validated"

emit_event "harness_test" "start" "running" "running rch cargo test gate"
if rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_1_3_sqlite_version_contract -- --nocapture; then
  emit_event "harness_test" "pass" "pass" "cargo test gate passed"
  RESULT="pass"
else
  emit_event "harness_test" "fail" "fail" "cargo test gate failed"
  RESULT="fail"
fi

cat > "${REPORT_JSON}" <<EOF_JSON
{
  "trace_id": "${TRACE_ID}",
  "run_id": "${RUN_ID}",
  "scenario_id": "${SCENARIO_ID}",
  "seed": ${SEED},
  "bead_id": "${BEAD_ID}",
  "contract_file": "${CONTRACT_FILE}",
  "events_jsonl": "${EVENTS_JSONL}",
  "result": "${RESULT}"
}
EOF_JSON

emit_event "finalize" "info" "${RESULT}" "report written to ${REPORT_JSON}"

if [[ "${RESULT}" != "pass" ]]; then
  echo "[GATE FAIL] ${BEAD_ID} sqlite version contract verification failed"
  exit 1
fi

echo "[GATE PASS] ${BEAD_ID} sqlite version contract is valid"
