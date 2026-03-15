#!/usr/bin/env bash
# Verification gate for bd-db300.7.3:
# explicit exit criteria for leapfrog claims against SQLite.
#
# Deterministic replay:
#   bash scripts/verify_bd_db300_7_3_leapfrog_exit_criteria.sh

set -euo pipefail

BEAD_ID="bd-db300.7.3"
SCENARIO_ID="LEAPFROG-EXIT-CRITERIA-G3"
SEED=7300
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${BEAD_ID}-${TIMESTAMP_UTC}-${SEED}"
TRACE_ID="trace-${RUN_ID}"
ARTIFACT_DIR="artifacts/${BEAD_ID}/${RUN_ID}"
EVENTS_JSONL="${ARTIFACT_DIR}/events.jsonl"
REPORT_JSON="${ARTIFACT_DIR}/manifest.json"
SUMMARY_MD="${ARTIFACT_DIR}/summary.md"
METRIC_DICTIONARY_JSON="${ARTIFACT_DIR}/metric_dictionary.json"
TMP_REPORT_JSON="${REPORT_JSON}.tmp"
TMP_METRIC_DICTIONARY_JSON="${METRIC_DICTIONARY_JSON}.tmp"
TEST_LOG="${ARTIFACT_DIR}/cargo-test.log"
CONTRACT_FILE="leapfrog_exit_criteria.toml"
TEST_FILE="crates/fsqlite-harness/tests/bd_db300_7_3_leapfrog_exit_criteria.rs"

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

echo "=== ${BEAD_ID}: leapfrog exit criteria verification ==="
echo "run_id=${RUN_ID}"
echo "trace_id=${TRACE_ID}"
echo "scenario_id=${SCENARIO_ID}"

emit_event "bootstrap" "start" "running" "verification started"

if [[ ! -f "${CONTRACT_FILE}" ]]; then
  emit_event "contract_presence" "fail" "fail" "missing ${CONTRACT_FILE}"
  echo "missing ${CONTRACT_FILE}" >&2
  exit 1
fi
if [[ ! -f "${TEST_FILE}" ]]; then
  emit_event "test_presence" "fail" "fail" "missing ${TEST_FILE}"
  echo "missing ${TEST_FILE}" >&2
  exit 1
fi

emit_event "contract_schema" "start" "running" "validating contract schema"
export METRIC_DICTIONARY_JSON
export TMP_METRIC_DICTIONARY_JSON
if ! python3 - <<'PY' > "${TMP_REPORT_JSON}"; then
import json
import os
import tomllib
from pathlib import Path

contract = tomllib.loads(Path("leapfrog_exit_criteria.toml").read_text(encoding="utf-8"))

required_meta = {
    "schema_version": "fsqlite-harness.leapfrog_exit_criteria.v1",
    "bead_id": "bd-db300.7.3",
}
required_modes = {
    "sqlite_reference",
    "fsqlite_mvcc",
    "fsqlite_single_writer",
}
required_unit_tests = {
    "test_bd_db300_7_3_contract_schema_and_links",
    "test_bd_db300_7_3_required_campaign_surface_exists",
    "test_bd_db300_7_3_cell_targets_are_monotone",
    "test_bd_db300_7_3_verification_plan_is_actionable",
    "test_bd_db300_7_3_transferability_rubric_is_actionable",
}
required_profiles = {
    "baseline_unpinned",
    "recommended_pinned",
    "adversarial_cross_node",
}
required_cell_suffixes = {"c1", "c4", "c8"}
required_scenarios = {
    "commutative_inserts_disjoint_keys_c1",
    "commutative_inserts_disjoint_keys_c4",
    "commutative_inserts_disjoint_keys_c8",
    "hot_page_contention_c1",
    "hot_page_contention_c4",
    "hot_page_contention_c8",
    "mixed_read_write_c1",
    "mixed_read_write_c4",
    "mixed_read_write_c8",
}
required_metric_families = {
    "throughput",
    "retry",
    "abort",
    "cpu_efficiency",
    "latency",
    "topology",
    "wait",
    "page_touch",
    "split_path",
    "allocator",
    "cache",
    "copy_allocation",
}
required_transferability_classes = [
    "transferable",
    "profile_specific_but_useful",
    "suspicious",
    "non_claimable",
]
required_hardware_classes = {
    "same_host",
    "same_topology_class",
    "cross_hardware_class",
}
required_downstream_beads = {"bd-db300.7.3", "bd-db300.7.4"}
required_report_labels = {
    "transferable": "transferable win",
    "profile_specific_but_useful": "lab-specific win",
    "suspicious": "topology-sensitive win",
    "non_claimable": "no-catastrophic-regression failure",
}
for key, expected in required_meta.items():
    actual = contract["meta"].get(key)
    if actual != expected:
        raise SystemExit(f"meta.{key} expected {expected!r}, found {actual!r}")

actual_modes = set(contract["campaign"].get("required_modes", []))
if actual_modes != required_modes:
    raise SystemExit(
        f"campaign.required_modes mismatch expected={sorted(required_modes)!r} actual={sorted(actual_modes)!r}"
    )
actual_profiles = set(contract["campaign"].get("required_placement_profiles", []))
if actual_profiles != required_profiles:
    raise SystemExit(
        f"campaign.required_placement_profiles mismatch expected={sorted(required_profiles)!r} actual={sorted(actual_profiles)!r}"
    )
actual_cell_suffixes = set(contract["campaign"].get("required_cell_suffixes", []))
if actual_cell_suffixes != required_cell_suffixes:
    raise SystemExit(
        f"campaign.required_cell_suffixes mismatch expected={sorted(required_cell_suffixes)!r} actual={sorted(actual_cell_suffixes)!r}"
    )

cell_gates = contract.get("cell_gates", [])
if [gate.get("cell") for gate in cell_gates] != ["c1", "c4", "c8"]:
    raise SystemExit("cell_gates must appear in c1, c4, c8 order")
actual_scenarios = set(contract["verification_plan"].get("e2e_scenarios", []))
if actual_scenarios != required_scenarios:
    raise SystemExit(
        f"verification_plan.e2e_scenarios mismatch expected={sorted(required_scenarios)!r} actual={sorted(actual_scenarios)!r}"
    )
actual_unit_tests = set(contract["verification_plan"].get("unit_tests", []))
if actual_unit_tests != required_unit_tests:
    raise SystemExit(
        f"verification_plan.unit_tests mismatch expected={sorted(required_unit_tests)!r} actual={sorted(actual_unit_tests)!r}"
    )

required_artifacts = {
    "artifacts/{bead_id}/{run_id}/events.jsonl",
    "artifacts/{bead_id}/{run_id}/manifest.json",
    "artifacts/{bead_id}/{run_id}/summary.md",
    "artifacts/{bead_id}/{run_id}/metric_dictionary.json",
    "artifacts/{bead_id}/{run_id}/cell_metrics.jsonl",
    "artifacts/{bead_id}/{run_id}/retry_report.json",
    "artifacts/{bead_id}/{run_id}/topology.json",
}
artifacts = set(contract["verification_plan"].get("logging_artifacts", []))
missing = sorted(required_artifacts - artifacts)
if missing:
    raise SystemExit(f"missing logging artifacts: {missing}")

metric_dictionary = contract.get("metric_dictionary", {})
metrics = metric_dictionary.get("metrics", [])
if not metrics:
    raise SystemExit("metric_dictionary.metrics must not be empty")
metric_ids = {metric["metric_id"] for metric in metrics}
metric_families = {metric["family"] for metric in metrics}
for field in contract["verification_plan"].get("required_log_fields", []):
    if field not in metric_ids:
        raise SystemExit(
            f"verification_plan.required_log_fields references undefined metric {field!r}"
        )
missing_metric_families = sorted(required_metric_families - metric_families)
if missing_metric_families:
    raise SystemExit(
        f"metric_dictionary.metrics missing required families {missing_metric_families!r}"
    )

rubric = contract.get("transferability_rubric", {})
if rubric.get("schema_version") != "fsqlite-harness.transferability_rubric.v1":
    raise SystemExit(
        f"transferability_rubric.schema_version mismatch actual={rubric.get('schema_version')!r}"
    )
if rubric.get("single_writer_role") != "comparison_or_fallback_only":
    raise SystemExit(
        f"transferability_rubric.single_writer_role mismatch actual={rubric.get('single_writer_role')!r}"
    )
actual_rubric_modes = set(rubric.get("required_modes", []))
if actual_rubric_modes != required_modes:
    raise SystemExit(
        f"transferability_rubric.required_modes mismatch expected={sorted(required_modes)!r} actual={sorted(actual_rubric_modes)!r}"
    )
actual_hardware_classes = set(rubric.get("required_hardware_classes", []))
if actual_hardware_classes != required_hardware_classes:
    raise SystemExit(
        f"transferability_rubric.required_hardware_classes mismatch expected={sorted(required_hardware_classes)!r} actual={sorted(actual_hardware_classes)!r}"
    )
if rubric.get("classification_order", []) != required_transferability_classes:
    raise SystemExit(
        f"transferability_rubric.classification_order mismatch expected={required_transferability_classes!r} actual={rubric.get('classification_order', [])!r}"
    )
actual_downstream_beads = set(rubric.get("downstream_beads", []))
if actual_downstream_beads != required_downstream_beads:
    raise SystemExit(
        f"transferability_rubric.downstream_beads mismatch expected={sorted(required_downstream_beads)!r} actual={sorted(actual_downstream_beads)!r}"
    )
classes = rubric.get("classes", [])
if [klass.get("classification_id") for klass in classes] != required_transferability_classes:
    raise SystemExit(
        "transferability_rubric.classes must appear in transferable/profile_specific_but_useful/suspicious/non_claimable order"
    )
covered_profiles = set()
for klass in classes:
    class_id = klass["classification_id"]
    expected_report_label = required_report_labels.get(class_id)
    if expected_report_label is None:
        raise SystemExit(f"unexpected transferability class {class_id!r}")
    if klass.get("final_report_label") != expected_report_label:
        raise SystemExit(
            f"class {class_id!r} final_report_label mismatch expected={expected_report_label!r} actual={klass.get('final_report_label')!r}"
        )
    if not klass.get("summary", "").strip():
        raise SystemExit(f"class {class_id!r} summary must not be blank")
    if not klass.get("placement_rule", "").strip():
        raise SystemExit(f"class {class_id!r} placement_rule must not be blank")
    if not klass.get("mode_rule", "").strip():
        raise SystemExit(f"class {class_id!r} mode_rule must not be blank")
    if not klass.get("hardware_rule", "").strip():
        raise SystemExit(f"class {class_id!r} hardware_rule must not be blank")
    if not klass.get("reporting_requirement", "").strip():
        raise SystemExit(f"class {class_id!r} reporting_requirement must not be blank")
    if not klass.get("example", "").strip():
        raise SystemExit(f"class {class_id!r} example must not be blank")
    example_profiles = set(klass.get("example_profiles", []))
    if not example_profiles:
        raise SystemExit(f"class {class_id!r} must reference example profiles")
    if not example_profiles.issubset(required_profiles):
        raise SystemExit(
            f"class {class_id!r} example_profiles mismatch expected subset of {sorted(required_profiles)!r} actual={sorted(example_profiles)!r}"
        )
    covered_profiles |= example_profiles
if covered_profiles != required_profiles:
    raise SystemExit(
        f"transferability_rubric example profile coverage mismatch expected={sorted(required_profiles)!r} actual={sorted(covered_profiles)!r}"
    )

metric_dictionary_path = Path(os.environ["TMP_METRIC_DICTIONARY_JSON"])
metric_dictionary_path.write_text(
    json.dumps(metric_dictionary, indent=2),
    encoding="utf-8",
)

summary = {
    "bead_id": contract["meta"]["bead_id"],
    "policy_id": contract["meta"]["policy_id"],
    "required_modes": contract["campaign"]["required_modes"],
    "required_profiles": contract["campaign"]["required_placement_profiles"],
    "required_cells": [gate["cell"] for gate in cell_gates],
    "required_scenarios": contract["verification_plan"]["e2e_scenarios"],
    "transferability_classes": rubric["classification_order"],
    "claim_language": contract["scorecard"]["claim_language"],
    "metric_dictionary_path": os.environ["METRIC_DICTIONARY_JSON"],
    "metric_ids": sorted(metric_ids),
}
print(json.dumps(summary, indent=2))
PY
  emit_event "contract_schema" "fail" "fail" "contract schema validation failed"
  exit 1
fi
if ! mv "${TMP_REPORT_JSON}" "${REPORT_JSON}"; then
  emit_event "contract_schema" "fail" "fail" "failed to publish manifest.json"
  exit 1
fi
if ! mv "${TMP_METRIC_DICTIONARY_JSON}" "${METRIC_DICTIONARY_JSON}"; then
  emit_event "contract_schema" "fail" "fail" "failed to publish metric_dictionary.json"
  exit 1
fi
emit_event "contract_schema" "pass" "pass" "contract schema validated"

cat > "${SUMMARY_MD}" <<EOF
# ${BEAD_ID} Verification Summary

- run_id: \`${RUN_ID}\`
- trace_id: \`${TRACE_ID}\`
- scenario_id: \`${SCENARIO_ID}\`
- contract: \`${CONTRACT_FILE}\`
- test: \`${TEST_FILE}\`
- report: \`${REPORT_JSON}\`
- metric_dictionary: \`${METRIC_DICTIONARY_JSON}\`
EOF

emit_event "cargo_test" "start" "running" "running harness test via rch"
if ! rch exec -- cargo test --package fsqlite-harness --test bd_db300_7_3_leapfrog_exit_criteria -- --nocapture 2>&1 | tee "${TEST_LOG}"; then
  emit_event "cargo_test" "fail" "fail" "harness test failed"
  exit 1
fi
emit_event "cargo_test" "pass" "pass" "harness test passed"

emit_event "complete" "finish" "pass" "verification completed"
echo "[GATE PASS] ${BEAD_ID} leapfrog exit criteria verification passed"
