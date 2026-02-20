#!/usr/bin/env bash
# verify_e2e_log_schema.sh — E2E validation for unified log schema
# (bd-1dp9.7.2, bd-mblr.5.3, bd-mblr.5.3.1, bd-mblr.5.5.1)
#
# Validates the E2E log schema and scenario coverage report:
# 1. Runs unit tests for the e2e_log_schema module
# 2. Verifies schema field specifications
# 3. Checks critical scenario coverage
# 4. Emits structured log output
#
# Usage: ./scripts/verify_e2e_log_schema.sh [--json] [--deterministic] [--seed <u64>] [--skip-runtime]

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_OUTPUT=false
DETERMINISTIC=false
RUN_RUNTIME=true
SEED="${E2E_LOG_SCHEMA_SEED:-424242}"
MODULE_FILE="$WORKSPACE_ROOT/crates/fsqlite-harness/src/e2e_log_schema.rs"
CONTRACT_DOC="$WORKSPACE_ROOT/docs/e2e_log_schema_contract.md"
PROFILE_DOC="$WORKSPACE_ROOT/docs/e2e_shell_script_log_profile.json"
RUNTIME_REPORT_REL="test-results/e2e_log_schema_runtime_conformance.jsonl"
RUNTIME_REPORT_PATH="$WORKSPACE_ROOT/$RUNTIME_REPORT_REL"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --deterministic)
            DETERMINISTIC=true
            shift
            ;;
        --seed)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --seed requires a value" >&2
                exit 2
            fi
            SEED="$2"
            shift 2
            ;;
        --skip-runtime)
            RUN_RUNTIME=false
            shift
            ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
done

if $DETERMINISTIC; then
    RUN_ID="e2e-log-schema-seed-${SEED}"
else
    RUN_ID="e2e-log-schema-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi

# Verify module file exists
if [[ ! -f "$MODULE_FILE" ]]; then
    echo "ERROR: $MODULE_FILE not found" >&2
    exit 1
fi
if [[ ! -f "$CONTRACT_DOC" ]]; then
    echo "ERROR: $CONTRACT_DOC not found" >&2
    exit 1
fi
if [[ ! -f "$PROFILE_DOC" ]]; then
    echo "ERROR: $PROFILE_DOC not found" >&2
    exit 1
fi

# Compute artifact hash
MODULE_HASH=$(sha256sum "$MODULE_FILE" | awk '{print $1}')
DOC_HASH=$(sha256sum "$CONTRACT_DOC" | awk '{print $1}')
PROFILE_HASH=$(sha256sum "$PROFILE_DOC" | awk '{print $1}')

# Run unit tests
ERRORS=0
TEST_RESULT="unknown"
TEST_COUNT=0
if command -v rch >/dev/null 2>&1; then
    TEST_CMD=(rch exec -- cargo test -p fsqlite-harness --lib -- e2e_log_schema)
else
    TEST_CMD=(cargo test -p fsqlite-harness --lib -- e2e_log_schema)
fi

if TEST_OUTPUT=$("${TEST_CMD[@]}" 2>&1); then
    TEST_RESULT="pass"
    TEST_COUNT=$(echo "$TEST_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo 0)
else
    TEST_RESULT="fail"
    ERRORS=$((ERRORS + 1))
fi

# Count schema elements from source
FIELD_SPEC_COUNT=$(grep -Ec '^[[:space:]]*FieldSpec \{' "$MODULE_FILE" || echo 0)
CRITICAL_SCENARIO_COUNT=$(grep -c '"[A-Z]\+-[0-9]' "$MODULE_FILE" | head -1 || echo 0)
LOG_PHASE_VARIANTS=$(grep -c '^\s*[A-Z][a-z]*,' "$MODULE_FILE" | head -1 || echo 0)

# Validate contract document alignment
CONTRACT_STATUS="pass"
for expected in "Schema version: \`1.0.0\`" "\`run_id\`" "\`timestamp\`" "\`phase\`" "\`event_type\`" "docs/e2e_shell_script_log_profile.json"; do
    if ! grep -Fq "$expected" "$CONTRACT_DOC"; then
        CONTRACT_STATUS="fail"
        ERRORS=$((ERRORS + 1))
        break
    fi
done

# Validate machine-readable shell-script profile
PROFILE_STATUS="pass"
if ! jq -e '.profile_version == "1.0.0" and .log_schema_version == "1.0.0"' "$PROFILE_DOC" >/dev/null; then
    PROFILE_STATUS="fail"
    ERRORS=$((ERRORS + 1))
fi
if ! jq -e '.required_fields == ["run_id","timestamp","phase","event_type"]' "$PROFILE_DOC" >/dev/null; then
    PROFILE_STATUS="fail"
    ERRORS=$((ERRORS + 1))
fi
if ! jq -e '.optional_fields | index("context.trace_id") != null and index("context.outcome") != null and index("context.retry_attempt") != null' "$PROFILE_DOC" >/dev/null; then
    PROFILE_STATUS="fail"
    ERRORS=$((ERRORS + 1))
fi
if ! jq -e '[.normative_examples[].event.event_type] | index("Pass") != null and index("Fail") != null' "$PROFILE_DOC" >/dev/null; then
    PROFILE_STATUS="fail"
    ERRORS=$((ERRORS + 1))
fi
if ! jq -e '.migration_aliases | map(.legacy_token + ":" + .canonical_field) | index("status:context.outcome") != null and index("scenario:scenario_id") != null' "$PROFILE_DOC" >/dev/null; then
    PROFILE_STATUS="fail"
    ERRORS=$((ERRORS + 1))
fi
if ! jq -e '.replay_instructions | length > 0' "$PROFILE_DOC" >/dev/null; then
    PROFILE_STATUS="fail"
    ERRORS=$((ERRORS + 1))
fi

REPLAY_COMMAND="$(jq -r '.replay_instructions[0] // ""' "$PROFILE_DOC")"

# Representative runtime shell-script conformance checks.
# These probes validate runtime event shape + trace continuity + artifact linkage.
RUNTIME_STATUS="pass"
RUNTIME_PROBES_TOTAL=0
RUNTIME_PROBES_PASSED=0
RUNTIME_PROBES_FAILED=0

mkdir -p "$(dirname "$RUNTIME_REPORT_PATH")"
: >"$RUNTIME_REPORT_PATH"

record_runtime_probe() {
    local probe="$1"
    local script_path="$2"
    local event_log_rel="$3"
    local result="$4"
    local detail="$5"
    jq -nc \
        --arg probe "$probe" \
        --arg script_path "$script_path" \
        --arg event_log "$event_log_rel" \
        --arg result "$result" \
        --arg detail "$detail" \
        '{probe: $probe, script_path: $script_path, event_log: $event_log, result: $result, detail: $detail}' \
        >>"$RUNTIME_REPORT_PATH"
}

run_runtime_probe() {
    local probe_name="$1"
    local script_rel="$2"
    local event_log_rel="$3"
    local scenario_id="$4"
    local script_abs="$WORKSPACE_ROOT/$script_rel"
    local event_log_abs="$WORKSPACE_ROOT/$event_log_rel"
    local probe_output="$WORKSPACE_ROOT/test-results/e2e_log_schema_runtime_${probe_name}.log"

    RUNTIME_PROBES_TOTAL=$((RUNTIME_PROBES_TOTAL + 1))

    if [[ ! -f "$script_abs" ]]; then
        ERRORS=$((ERRORS + 1))
        RUNTIME_PROBES_FAILED=$((RUNTIME_PROBES_FAILED + 1))
        RUNTIME_STATUS="fail"
        record_runtime_probe "$probe_name" "$script_rel" "$event_log_rel" "fail" "script_missing"
        return
    fi

    if ! (cd "$WORKSPACE_ROOT" && SEED="$SEED" SCENARIO_ID="$scenario_id" bash "$script_rel") >"$probe_output" 2>&1; then
        ERRORS=$((ERRORS + 1))
        RUNTIME_PROBES_FAILED=$((RUNTIME_PROBES_FAILED + 1))
        RUNTIME_STATUS="fail"
        record_runtime_probe "$probe_name" "$script_rel" "$event_log_rel" "fail" "script_execution_failed"
        return
    fi

    if [[ ! -s "$event_log_abs" ]]; then
        ERRORS=$((ERRORS + 1))
        RUNTIME_PROBES_FAILED=$((RUNTIME_PROBES_FAILED + 1))
        RUNTIME_STATUS="fail"
        record_runtime_probe "$probe_name" "$script_rel" "$event_log_rel" "fail" "missing_or_empty_event_log"
        return
    fi

    if ! jq -s -e --arg scenario_id "$scenario_id" --arg event_log "$event_log_rel" '
        length > 0
        and all(.[]; type == "object")
        and all(.[]; ((.run_id | type) == "string") and ((.run_id | length) > 0))
        and all(.[]; ((.timestamp | type) == "string") and ((.timestamp | length) > 0))
        and all(.[]; ((.phase | type) == "string") and ((.phase | length) > 0))
        and all(.[]; ((.event_type | type) == "string") and ((.event_type | length) > 0))
        and all(.[]; ((.scenario_id | type) == "string") and ((.scenario_id | length) > 0))
        and all(.[]; has("seed"))
        and all(.[]; ((.context | type) == "object"))
        and all(.[]; ((.context.log_standard_ref | type) == "string") and ((.context.log_standard_ref | length) > 0))
        and all(.[]; ((.context.schema_log_path | type) == "string") and (.context.schema_log_path | endswith($event_log)))
        and ([.[].run_id] | unique | length == 1)
        and ([.[].scenario_id] | unique == [$scenario_id])
        and ([.[].event_type] | index("start") != null)
        and (([.[].event_type] | index("pass") != null) or ([.[].event_type] | index("fail") != null))
        and all(.[]; if (.event_type == "fail" or .event_type == "error")
                      then ((.context.outcome // "") | tostring | length > 0)
                      else true end)
    ' "$event_log_abs" >/dev/null; then
        ERRORS=$((ERRORS + 1))
        RUNTIME_PROBES_FAILED=$((RUNTIME_PROBES_FAILED + 1))
        RUNTIME_STATUS="fail"
        record_runtime_probe "$probe_name" "$script_rel" "$event_log_rel" "fail" "runtime_schema_or_trace_validation_failed"
        return
    fi

    RUNTIME_PROBES_PASSED=$((RUNTIME_PROBES_PASSED + 1))
    record_runtime_probe "$probe_name" "$script_rel" "$event_log_rel" "pass" "ok"
}

if $RUN_RUNTIME; then
    run_runtime_probe "risk_register" "e2e/risk_register_report.sh" "test-results/bd_3kp_2_risk_register_events.jsonl" "DOC-1"
    run_runtime_probe "future_work" "e2e/future_work_report.sh" "test-results/bd_3kp_3_future_work_events.jsonl" "DOC-2"
    run_runtime_probe "reference_index" "e2e/reference_index_audit.sh" "test-results/reference_index_audit_events.jsonl" "IDX-1"
else
    RUNTIME_STATUS="skipped"
fi

# Output results
if $JSON_OUTPUT; then
    cat <<ENDJSON
{
  "run_id": "$RUN_ID",
  "phase": "e2e_log_schema_validation",
  "bead_id": "bd-1dp9.7.2",
  "related_bead_ids": ["bd-mblr.5.3", "bd-mblr.5.3.1", "bd-mblr.5.5.1"],
  "module_hash": "$MODULE_HASH",
  "contract_doc_hash": "$DOC_HASH",
  "profile_doc_hash": "$PROFILE_HASH",
  "deterministic": $DETERMINISTIC,
  "seed": "$SEED",
  "unit_tests": {
    "result": "$TEST_RESULT",
    "count": $TEST_COUNT
  },
  "schema_stats": {
    "field_specs": $FIELD_SPEC_COUNT,
    "log_phase_variants": $LOG_PHASE_VARIANTS
  },
  "contract_doc": {
    "path": "docs/e2e_log_schema_contract.md",
    "status": "$CONTRACT_STATUS"
  },
  "shell_script_profile": {
    "path": "docs/e2e_shell_script_log_profile.json",
    "status": "$PROFILE_STATUS"
  },
  "runtime_conformance": {
    "enabled": $RUN_RUNTIME,
    "status": "$RUNTIME_STATUS",
    "probes_total": $RUNTIME_PROBES_TOTAL,
    "probes_passed": $RUNTIME_PROBES_PASSED,
    "probes_failed": $RUNTIME_PROBES_FAILED,
    "report_path": "$RUNTIME_REPORT_REL"
  },
  "replay": {
    "command": "$REPLAY_COMMAND"
  },
  "validation_errors": $ERRORS,
  "result": "$([ $ERRORS -eq 0 ] && echo 'pass' || echo 'fail')"
}
ENDJSON
else
    echo "=== E2E Log Schema Validation ==="
    echo "Run ID:           $RUN_ID"
    echo "Module hash:      $MODULE_HASH"
    echo "Contract hash:    $DOC_HASH"
    echo "Profile hash:     $PROFILE_HASH"
    echo "Deterministic:    $DETERMINISTIC"
    echo "Seed:             $SEED"
    echo ""
    echo "--- Unit Tests ---"
    echo "Result:           $TEST_RESULT"
    echo "Tests:            $TEST_COUNT"
    echo ""
    echo "--- Schema Stats ---"
    echo "Field specs:      $FIELD_SPEC_COUNT"
    echo "Phase variants:   $LOG_PHASE_VARIANTS"
    echo "Contract doc:     $CONTRACT_STATUS"
    echo "Profile doc:      $PROFILE_STATUS"
    echo "Replay command:   $REPLAY_COMMAND"
    echo ""
    echo "--- Runtime Conformance ---"
    echo "Enabled:          $RUN_RUNTIME"
    echo "Status:           $RUNTIME_STATUS"
    echo "Probes total:     $RUNTIME_PROBES_TOTAL"
    echo "Probes passed:    $RUNTIME_PROBES_PASSED"
    echo "Probes failed:    $RUNTIME_PROBES_FAILED"
    echo "Report path:      $RUNTIME_REPORT_REL"
    echo ""
    echo "--- Validation ---"
    echo "Errors:           $ERRORS"
    echo "Result:           $([ $ERRORS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
fi

exit $ERRORS
