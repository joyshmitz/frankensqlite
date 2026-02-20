#!/usr/bin/env bash
# verify_bd_3u7_3_fault_vfs_deterministic_replay.sh — bead bd-3u7.3 verification runner
#
# Usage:
#   ./scripts/verify_bd_3u7_3_fault_vfs_deterministic_replay.sh [--json] [--seed N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="bd-3u7-3-$(date -u +%Y%m%dT%H%M%SZ)-$$"
JSON_OUTPUT=false
SEED="${BD_3U7_3_SEED:-980984445}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --seed)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --seed requires value" >&2; exit 2; }
            SEED="$1"
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if ! command -v rch >/dev/null 2>&1; then
    echo "ERROR: rch is required for bd-3u7.3 verification" >&2
    exit 3
fi

TEST_LOG="$(mktemp)"
RESULT="pass"

if ! BD_3U7_3_SEED="$SEED" \
     rch exec -- cargo test -p fsqlite-harness --test bd_3u7_3_fault_vfs_deterministic_replay -- --nocapture \
     >"$TEST_LOG" 2>&1; then
    RESULT="fail"
fi

SCENARIO_ARTIFACT_PATH="$({ rg -o 'path=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^path=//'; } || true)"
RUN_ID_FROM_TEST="$({ rg -o 'run_id=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^run_id=//'; } || true)"
TRACE_ID="$({ rg -o 'trace_id=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^trace_id=//'; } || true)"
SCENARIO_ID="$({ rg -o 'scenario_id=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^scenario_id=//'; } || true)"

if [[ -n "$SCENARIO_ARTIFACT_PATH" && "$SCENARIO_ARTIFACT_PATH" != /* ]]; then
    SCENARIO_ARTIFACT_PATH="$REPO_ROOT/$SCENARIO_ARTIFACT_PATH"
fi

VERIFY_OUTPUT_DIR="$REPO_ROOT/test-results/bd_3u7_3"
mkdir -p "$VERIFY_OUTPUT_DIR"
VERIFY_ARTIFACT_PATH="$VERIFY_OUTPUT_DIR/verify-${RUN_ID}.json"

cat >"$VERIFY_ARTIFACT_PATH" <<ENDVERIFY
{
  "run_id": "$RUN_ID",
  "bead_id": "bd-3u7.3",
  "seed": $SEED,
  "result": "$RESULT",
  "test_run_id": "$RUN_ID_FROM_TEST",
  "trace_id": "$TRACE_ID",
  "scenario_id": "$SCENARIO_ID",
  "scenario_artifact_path": "$SCENARIO_ARTIFACT_PATH",
  "test_log_path": "$TEST_LOG"
}
ENDVERIFY

ARTIFACT_HASH="$(sha256sum "$VERIFY_ARTIFACT_PATH" | awk '{print $1}')"

if [[ "$JSON_OUTPUT" == "true" ]]; then
    cat <<ENDJSON
{
  "run_id": "$RUN_ID",
  "bead_id": "bd-3u7.3",
  "seed": $SEED,
  "result": "$RESULT",
  "test_run_id": "$RUN_ID_FROM_TEST",
  "trace_id": "$TRACE_ID",
  "scenario_id": "$SCENARIO_ID",
  "artifact_path": "$VERIFY_ARTIFACT_PATH",
  "scenario_artifact_path": "$SCENARIO_ARTIFACT_PATH",
  "artifact_hash": "$ARTIFACT_HASH",
  "test_log_path": "$TEST_LOG"
}
ENDJSON
else
    echo "=== bd-3u7.3 Verification ==="
    echo "Run ID:        $RUN_ID"
    echo "Result:        $RESULT"
    echo "Seed:          $SEED"
    echo "Test run_id:   $RUN_ID_FROM_TEST"
    echo "Trace ID:      $TRACE_ID"
    echo "Scenario ID:   $SCENARIO_ID"
    echo "Artifact path: $VERIFY_ARTIFACT_PATH"
    echo "Scenario path: $SCENARIO_ARTIFACT_PATH"
    echo "Artifact hash: $ARTIFACT_HASH"
    echo "Test log:      $TEST_LOG"
fi

[[ "$RESULT" == "pass" ]]
