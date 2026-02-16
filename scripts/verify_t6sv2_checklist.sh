#!/usr/bin/env bash
# verify_t6sv2_checklist.sh — deterministic E2E verifier for bd-t6sv2.16
#
# Validates:
# 1. Deterministic report generation for a fully-covered fixture.
# 2. Missing/stale evidence detection with owner attribution.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$WORKSPACE_ROOT/artifacts/t6sv2-checklist-e2e"
JSON_OUTPUT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUTPUT=true
      shift
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RUN_ROOT"

if command -v rch >/dev/null 2>&1; then
  RUNNER=(rch exec -- cargo run -p fsqlite-harness --bin t6sv2_checklist_runner --)
else
  RUNNER=(cargo run -p fsqlite-harness --bin t6sv2_checklist_runner --)
fi

FIXTURE_ROOT="$RUN_ROOT/fixture_workspace"
FIXTURE_BEADS_DIR="$FIXTURE_ROOT/.beads"
FIXTURE_SCRIPTS_DIR="$FIXTURE_ROOT/scripts"
mkdir -p "$FIXTURE_BEADS_DIR" "$FIXTURE_SCRIPTS_DIR"

PASS_ISSUES="$FIXTURE_BEADS_DIR/issues_pass.jsonl"
FAIL_ISSUES="$FIXTURE_BEADS_DIR/issues_fail.jsonl"
UNIT_OVERRIDE="$FIXTURE_ROOT/unit_matrix_override.json"
TRACE_PASS="$FIXTURE_ROOT/traceability_pass.json"
TRACE_FAIL="$FIXTURE_ROOT/traceability_fail.json"
SCRIPT_OK="$FIXTURE_SCRIPTS_DIR/verify_t6sv2_4.sh"

REPORT_PASS_A="$RUN_ROOT/report_pass_a.json"
REPORT_PASS_B="$RUN_ROOT/report_pass_b.json"
SUMMARY_PASS_A="$RUN_ROOT/report_pass_a.md"
SUMMARY_PASS_B="$RUN_ROOT/report_pass_b.md"
REPORT_FAIL="$RUN_ROOT/report_fail.json"
SUMMARY_FAIL="$RUN_ROOT/report_fail.md"
FIXED_TS="1700000000000"

cat > "$SCRIPT_OK" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
echo '{"ok":true}'
EOF
chmod +x "$SCRIPT_OK"

cat > "$PASS_ISSUES" <<'EOF'
{"id":"bd-t6sv2.4","title":"Conformance dashboard","status":"open","issue_type":"task","priority":2,"owner":"ops-a","created_by":"ops-a"}
EOF

cat > "$FAIL_ISSUES" <<'EOF'
{"id":"bd-t6sv2.4","title":"Conformance dashboard","status":"open","issue_type":"task","priority":2,"owner":"ops-a","created_by":"ops-a"}
{"id":"bd-t6sv2.6","title":"Plan hints","status":"open","issue_type":"task","priority":2,"owner":"ops-b","created_by":"ops-b"}
EOF

cat > "$UNIT_OVERRIDE" <<'EOF'
{
  "schema_version": "1.0.0",
  "bead_id": "bd-test",
  "root_seed": 42,
  "tests": [
    {
      "test_id": "UT-T6SV2-004",
      "category": "SqlGrammar",
      "crate_name": "fsqlite-harness",
      "module_path": "synthetic::tests",
      "description": "synthetic unit evidence",
      "invariants": ["synthetic_invariant"],
      "seed": 99,
      "property_based": false,
      "failure_diagnostics": {
        "dump_targets": ["rows"],
        "log_spans": ["checklist.synthetic"],
        "related_beads": ["bd-t6sv2.4"]
      }
    }
  ],
  "coverage": [
    {
      "category": "SqlGrammar",
      "test_count": 1,
      "invariant_count": 1,
      "property_test_count": 0,
      "contributing_crates": ["fsqlite-harness"],
      "missing_coverage": [],
      "fill_pct": 1.0
    }
  ]
}
EOF

cat > "$TRACE_PASS" <<'EOF'
{
  "schema_version": "1.0.0",
  "bead_id": "bd-test",
  "scripts": [
    {
      "path": "scripts/verify_t6sv2_4.sh",
      "kind": "ShellUtility",
      "bead_id": "bd-t6sv2.4",
      "description": "synthetic e2e evidence",
      "invocation": {
        "command": "bash scripts/verify_t6sv2_4.sh",
        "env_vars": [],
        "json_output": true,
        "timeout_secs": 60
      },
      "scenario_ids": ["OBS-400"],
      "storage_modes": ["InMemory"],
      "concurrency_modes": ["Sequential"],
      "artifact_paths": ["artifacts/t6sv2-4.json"],
      "log_schema_version": "1.0.0"
    }
  ],
  "gaps": []
}
EOF

cat > "$TRACE_FAIL" <<'EOF'
{
  "schema_version": "1.0.0",
  "bead_id": "bd-test",
  "scripts": [
    {
      "path": "scripts/verify_t6sv2_4.sh",
      "kind": "ShellUtility",
      "bead_id": "bd-t6sv2.4",
      "description": "synthetic e2e evidence",
      "invocation": {
        "command": "bash scripts/verify_t6sv2_4.sh",
        "env_vars": [],
        "json_output": true,
        "timeout_secs": 60
      },
      "scenario_ids": ["OBS-400"],
      "storage_modes": ["InMemory"],
      "concurrency_modes": ["Sequential"],
      "artifact_paths": ["artifacts/t6sv2-4.json"],
      "log_schema_version": "1.0.0"
    },
    {
      "path": "scripts/missing_t6sv2_6.sh",
      "kind": "ShellUtility",
      "bead_id": "bd-t6sv2.6",
      "description": "stale link fixture",
      "invocation": {
        "command": "bash scripts/missing_t6sv2_6.sh",
        "env_vars": [],
        "json_output": true,
        "timeout_secs": 60
      },
      "scenario_ids": ["OBS-600"],
      "storage_modes": ["InMemory"],
      "concurrency_modes": ["Sequential"],
      "artifact_paths": ["artifacts/t6sv2-6.json"],
      "log_schema_version": "bad"
    }
  ],
  "gaps": []
}
EOF

"${RUNNER[@]}" \
  --workspace-root "$FIXTURE_ROOT" \
  --issues-path "$PASS_ISSUES" \
  --unit-matrix-override "$UNIT_OVERRIDE" \
  --traceability-override "$TRACE_PASS" \
  --output "$REPORT_PASS_A" \
  --output-human "$SUMMARY_PASS_A" \
  --generated-unix-ms "$FIXED_TS"

"${RUNNER[@]}" \
  --workspace-root "$FIXTURE_ROOT" \
  --issues-path "$PASS_ISSUES" \
  --unit-matrix-override "$UNIT_OVERRIDE" \
  --traceability-override "$TRACE_PASS" \
  --output "$REPORT_PASS_B" \
  --output-human "$SUMMARY_PASS_B" \
  --generated-unix-ms "$FIXED_TS"

if ! diff -u "$REPORT_PASS_A" "$REPORT_PASS_B" >/dev/null; then
  echo "ERROR: deterministic pass fixture reports differ" >&2
  diff -u "$REPORT_PASS_A" "$REPORT_PASS_B" >&2 || true
  exit 1
fi

jq -e '.summary.overall_pass == true' "$REPORT_PASS_A" >/dev/null
jq -e '.summary.violation_count == 0' "$REPORT_PASS_A" >/dev/null

set +e
"${RUNNER[@]}" \
  --workspace-root "$FIXTURE_ROOT" \
  --issues-path "$FAIL_ISSUES" \
  --unit-matrix-override "$UNIT_OVERRIDE" \
  --traceability-override "$TRACE_FAIL" \
  --output "$REPORT_FAIL" \
  --output-human "$SUMMARY_FAIL" \
  --generated-unix-ms "$FIXED_TS"
FAIL_EXIT=$?
set -e

if [[ $FAIL_EXIT -eq 0 ]]; then
  echo "ERROR: expected fail fixture to return non-zero" >&2
  exit 1
fi

jq -e '.summary.overall_pass == false' "$REPORT_FAIL" >/dev/null
jq -e '.violations[] | select(.bead_id == "bd-t6sv2.6" and .kind == "missing_unit_evidence")' "$REPORT_FAIL" >/dev/null
jq -e '.violations[] | select(.bead_id == "bd-t6sv2.6" and .kind == "invalid_e2e_reference")' "$REPORT_FAIL" >/dev/null
jq -e '.violations[] | select(.bead_id == "bd-t6sv2.6" and .owner == "ops-b")' "$REPORT_FAIL" >/dev/null

if $JSON_OUTPUT; then
  cat <<ENDJSON
{
  "bead_id": "bd-t6sv2.16",
  "deterministic_pass_fixture": true,
  "stale_and_missing_detection": true,
  "pass_report": "${REPORT_PASS_A#$WORKSPACE_ROOT/}",
  "fail_report": "${REPORT_FAIL#$WORKSPACE_ROOT/}",
  "summary_pass": "${SUMMARY_PASS_A#$WORKSPACE_ROOT/}",
  "summary_fail": "${SUMMARY_FAIL#$WORKSPACE_ROOT/}"
}
ENDJSON
else
  echo "=== t6sv2 Checklist E2E ==="
  echo "Bead ID:                    bd-t6sv2.16"
  echo "Deterministic pass fixture: PASS"
  echo "Missing/stale detection:    PASS"
  echo "Pass report:                ${REPORT_PASS_A#$WORKSPACE_ROOT/}"
  echo "Fail report:                ${REPORT_FAIL#$WORKSPACE_ROOT/}"
fi
