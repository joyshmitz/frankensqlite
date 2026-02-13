#!/usr/bin/env bash
# verify_e2e_traceability.sh — E2E validation for traceability matrix (bd-mblr.4.5.1)
#
# Validates the E2E script inventory and traceability matrix:
# 1. Runs unit tests for the traceability module
# 2. Verifies all cataloged scripts exist on disk
# 3. Checks scenario coverage and gap analysis
# 4. Emits structured log output
#
# Usage: ./scripts/verify_e2e_traceability.sh [--json]
#
# Output: structured log with run_id, coverage stats, and artifact hashes.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="e2e-traceability-$(date -u +%Y%m%dT%H%M%SZ)-$$"
JSON_OUTPUT=false
MODULE_FILE="$WORKSPACE_ROOT/crates/fsqlite-harness/src/e2e_traceability.rs"

if [[ "${1:-}" == "--json" ]]; then
    JSON_OUTPUT=true
fi

# Verify module file exists
if [[ ! -f "$MODULE_FILE" ]]; then
    echo "ERROR: $MODULE_FILE not found" >&2
    exit 1
fi

# Compute artifact hash
MODULE_HASH=$(sha256sum "$MODULE_FILE" | awk '{print $1}')

# Run unit tests
ERRORS=0
TEST_RESULT="unknown"
TEST_COUNT=0
if TEST_OUTPUT=$(cargo test -p fsqlite-harness --lib -- e2e_traceability 2>&1); then
    TEST_RESULT="pass"
    TEST_COUNT=$(echo "$TEST_OUTPUT" | grep -oP '\d+ passed' | grep -oP '\d+' || echo 0)
else
    TEST_RESULT="fail"
    ERRORS=$((ERRORS + 1))
fi

# Verify cataloged scripts exist on disk
MISSING_SCRIPTS=0
VERIFIED_SCRIPTS=0
MISSING_LIST=""

# Extract script paths from the Rust source (paths in quotes after "path:")
SCRIPT_PATHS=$(grep -oP '"\K[^"]+(?=",)' "$MODULE_FILE" | grep -E '\.(sh|rs|ts)$' | sort -u)

for script_path in $SCRIPT_PATHS; do
    full_path="$WORKSPACE_ROOT/$script_path"
    if [[ -f "$full_path" ]]; then
        VERIFIED_SCRIPTS=$((VERIFIED_SCRIPTS + 1))
    else
        MISSING_SCRIPTS=$((MISSING_SCRIPTS + 1))
        MISSING_LIST="${MISSING_LIST}${script_path}\n"
    fi
done

if [[ $MISSING_SCRIPTS -gt 0 ]]; then
    ERRORS=$((ERRORS + 1))
fi

# Count scenario and gap coverage from source
SCENARIO_COUNT=$(grep -oP '"[A-Z]+-\d+"' "$MODULE_FILE" | sort -u | wc -l)
GAP_COUNT=$(grep -c 'GapAnnotation {' "$MODULE_FILE" || echo 0)
SCRIPT_KIND_COUNT=$(grep -c 'ScriptKind::' "$MODULE_FILE" | head -1 || echo 0)

# Output results
if $JSON_OUTPUT; then
    cat <<ENDJSON
{
  "run_id": "$RUN_ID",
  "phase": "e2e_traceability_validation",
  "bead_id": "bd-mblr.4.5.1",
  "module_hash": "$MODULE_HASH",
  "unit_tests": {
    "result": "$TEST_RESULT",
    "count": $TEST_COUNT
  },
  "script_verification": {
    "verified": $VERIFIED_SCRIPTS,
    "missing": $MISSING_SCRIPTS
  },
  "coverage": {
    "unique_scenarios": $SCENARIO_COUNT,
    "gap_annotations": $GAP_COUNT
  },
  "validation_errors": $ERRORS,
  "result": "$([ $ERRORS -eq 0 ] && echo 'pass' || echo 'fail')"
}
ENDJSON
else
    echo "=== E2E Traceability Matrix Validation ==="
    echo "Run ID:           $RUN_ID"
    echo "Module hash:      $MODULE_HASH"
    echo ""
    echo "--- Unit Tests ---"
    echo "Result:           $TEST_RESULT"
    echo "Tests:            $TEST_COUNT"
    echo ""
    echo "--- Script Verification ---"
    echo "Verified:         $VERIFIED_SCRIPTS"
    echo "Missing:          $MISSING_SCRIPTS"
    if [[ $MISSING_SCRIPTS -gt 0 ]]; then
        echo "Missing scripts:"
        echo -e "$MISSING_LIST" | while read -r line; do
            [[ -n "$line" ]] && echo "  - $line"
        done
    fi
    echo ""
    echo "--- Coverage ---"
    echo "Unique scenarios: $SCENARIO_COUNT"
    echo "Gap annotations:  $GAP_COUNT"
    echo ""
    echo "--- Validation ---"
    echo "Errors:           $ERRORS"
    echo "Result:           $([ $ERRORS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
fi

exit $ERRORS
