#!/usr/bin/env bash
# Verification gate for bd-2yqp6.1.4:
# strict parity score contract + zero-ambiguity 100% semantics.
#
# Deterministic replay:
#   bash scripts/verify_bd_2yqp6_1_4_parity_score_contract.sh

set -euo pipefail

BEAD_ID="bd-2yqp6.1.4"
SCENARIO_ID="PARITY-SCORE-CONTRACT-A4"
SEED=3520
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${BEAD_ID}-${TIMESTAMP_UTC}-${SEED}"
TRACE_ID="trace-${RUN_ID}"
ARTIFACT_DIR="artifacts/${BEAD_ID}/${RUN_ID}"
EVENTS_JSONL="${ARTIFACT_DIR}/events.jsonl"
REPORT_JSON="${ARTIFACT_DIR}/report.json"
CONTRACT_FILE="parity_score_contract.toml"

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

echo "=== ${BEAD_ID}: parity score contract verification ==="
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

emit_event "contract_schema" "start" "running" "validating parity score contract schema"
python3 - <<'PY'
import tomllib
from pathlib import Path

contract_path = Path("parity_score_contract.toml")
taxonomy_path = Path("parity_taxonomy.toml")
surface_path = Path("supported_surface_matrix.toml")
ledger_path = Path("feature_universe_ledger.toml")

doc = tomllib.loads(contract_path.read_text(encoding="utf-8"))
meta = doc.get("meta", {})
formula = doc.get("formula", {})
status_weights = doc.get("status_weights", {})
hundred = doc.get("hundred_percent", {})
claim_validation = doc.get("claim_validation", {})
refs = doc.get("references", {})

for key in ("schema_version", "bead_id", "track_id", "generated_at", "contract_owner"):
    if key not in meta or not str(meta[key]).strip():
        raise SystemExit(f"missing meta.{key}")

if meta["bead_id"] != "bd-2yqp6.1.4":
    raise SystemExit("meta.bead_id must equal bd-2yqp6.1.4")

for key in ("score_name", "numerator", "denominator", "result", "rounding_mode", "included_statuses", "excluded_statuses", "source_taxonomy"):
    if key not in formula:
        raise SystemExit(f"missing formula.{key}")

if formula["rounding_mode"] != "truncate_6dp":
    raise SystemExit("formula.rounding_mode must be truncate_6dp")
if "pass" not in formula["included_statuses"] or "partial" not in formula["included_statuses"] or "fail" not in formula["included_statuses"]:
    raise SystemExit("formula.included_statuses must contain pass/partial/fail")
if "excluded" not in formula["excluded_statuses"]:
    raise SystemExit("formula.excluded_statuses must contain excluded")

for key in ("pass", "partial", "fail"):
    if key not in status_weights:
        raise SystemExit(f"missing status_weights.{key}")

if float(status_weights["pass"]) != 1.0:
    raise SystemExit("status_weights.pass must be 1.0")
if float(status_weights["partial"]) != 0.5:
    raise SystemExit("status_weights.partial must be 0.5")
if float(status_weights["fail"]) != 0.0:
    raise SystemExit("status_weights.fail must be 0.0")

for key in ("required_score", "max_fail_features", "max_partial_features", "max_excluded_features", "max_open_divergences", "max_flaky_failures", "max_coverage_debt_items"):
    if key not in hundred:
        raise SystemExit(f"missing hundred_percent.{key}")

if float(hundred["required_score"]) != 1.0:
    raise SystemExit("hundred_percent.required_score must be 1.0")
for key in ("max_fail_features", "max_partial_features", "max_excluded_features", "max_open_divergences", "max_flaky_failures", "max_coverage_debt_items"):
    if int(hundred[key]) != 0:
        raise SystemExit(f"hundred_percent.{key} must be 0")

for key in ("disallow_inequality_operators", "disallow_approximation_terms", "forbidden_terms", "required_fields"):
    if key not in claim_validation:
        raise SystemExit(f"missing claim_validation.{key}")

if not claim_validation["disallow_inequality_operators"]:
    raise SystemExit("claim_validation.disallow_inequality_operators must be true")
if not claim_validation["disallow_approximation_terms"]:
    raise SystemExit("claim_validation.disallow_approximation_terms must be true")
if not claim_validation["forbidden_terms"]:
    raise SystemExit("claim_validation.forbidden_terms must be non-empty")
if not claim_validation["required_fields"]:
    raise SystemExit("claim_validation.required_fields must be non-empty")

for key in ("taxonomy", "surface_matrix", "feature_ledger", "verification_contract_module", "ratchet_policy_module"):
    if key not in refs or not str(refs[key]).strip():
        raise SystemExit(f"missing references.{key}")

for path in (
    formula["source_taxonomy"],
    refs["taxonomy"],
    refs["surface_matrix"],
    refs["feature_ledger"],
    refs["verification_contract_module"],
    refs["ratchet_policy_module"],
):
    if not Path(path).exists():
        raise SystemExit(f"referenced path does not exist: {path}")

if refs["taxonomy"] != formula["source_taxonomy"]:
    raise SystemExit("references.taxonomy must match formula.source_taxonomy")

# Basic deterministic recompute check from current taxonomy.
taxonomy_doc = tomllib.loads(taxonomy_path.read_text(encoding="utf-8"))
features = taxonomy_doc.get("features", [])
if not features:
    raise SystemExit("taxonomy has no features")

num = 0.0
den = 0.0
for feature in features:
    status = str(feature["status"])
    weight = float(feature["weight"])
    if status in formula["included_statuses"]:
        num += weight * float(status_weights[status])
        den += weight
    elif status in formula["excluded_statuses"]:
        continue
    else:
        raise SystemExit(f"unknown status in taxonomy: {status}")

if den <= 0.0:
    raise SystemExit("score denominator is non-positive")

score = int((num / den) * 1_000_000.0) / 1_000_000.0
if score < 0.0 or score > 1.0:
    raise SystemExit(f"score out of bounds: {score}")

# Ensure companion contracts parse.
_ = tomllib.loads(surface_path.read_text(encoding="utf-8"))
_ = tomllib.loads(ledger_path.read_text(encoding="utf-8"))
PY
emit_event "contract_schema" "pass" "pass" "contract schema validated"

emit_event "harness_test" "start" "running" "running rch cargo test gate"
if rch exec -- cargo test -p fsqlite-harness --test bd_2yqp6_1_4_parity_score_contract -- --nocapture; then
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
  echo "[GATE FAIL] ${BEAD_ID} parity score contract verification failed"
  exit 1
fi

echo "[GATE PASS] ${BEAD_ID} parity score contract is valid"
