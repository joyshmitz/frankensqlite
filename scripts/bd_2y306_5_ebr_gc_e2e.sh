#!/usr/bin/env bash
# bd_2y306_5_ebr_gc_e2e.sh — deterministic verifier for bd-2y306.5
#
# Runs:
# 1) Property-based EBR GC invariant test in fsqlite-mvcc
# 2) File-backed integrity scenario test in fsqlite-e2e
#
# Writes a machine-readable artifact with run/trace/scenario IDs, per-test
# timing, replay command, and pass/fail summary.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JSON_OUTPUT=false
NO_RCH=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      JSON_OUTPUT=true
      shift
      ;;
    --no-rch)
      NO_RCH=true
      shift
      ;;
    *)
      echo "ERROR: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

BEAD_ID="bd-2y306.5"
SCENARIO_ID="${SCENARIO_ID:-EBR-GC-1}"
SEED="${SEED:-20260218}"
TRACE_ID="${TRACE_ID:-$SEED}"
RUN_ID="${RUN_ID:-${BEAD_ID}-seed-${SEED}}"

ARTIFACT_DIR="$WORKSPACE_ROOT/artifacts/bd-2y306.5"
ARTIFACT_PATH="${FSQLITE_EBR_GC_E2E_ARTIFACT:-$ARTIFACT_DIR/ebr_gc_e2e_artifact.json}"
mkdir -p "$ARTIFACT_DIR"

TEST1_NAME="mvcc_prop_horizon_visibility"
TEST1_FILTER="prop_gc_prune_preserves_horizon_visibility_and_expected_freed"
TEST2_NAME="ebr_prop_deferred_retire_lifetime"
TEST2_FILTER="prop_deferred_retire_respects_pin_lifetime_and_eventually_reclaims"
TEST3_NAME="ebr_prop_thread_retirements"
TEST3_FILTER="prop_thread_termination_does_not_lose_deferred_retirements"
TEST4_NAME="file_backed_integrity_check"
TEST4_FILTER="integrity_check_populates_report_for_file_db"

if ! $NO_RCH && command -v rch >/dev/null 2>&1; then
  RUNNER=(rch exec --)
else
  RUNNER=()
fi

run_test() {
  local test_name="$1"
  local test_filter="$2"
  local crate="$3"
  local log_path="$ARTIFACT_DIR/${test_name}.log"
  local start_ms end_ms elapsed_ms status

  start_ms="$(date +%s%3N)"
  if "${RUNNER[@]}" cargo test -p "$crate" "$test_filter" -- --nocapture >"$log_path" 2>&1; then
    status="pass"
  else
    status="fail"
  fi
  if [[ "$status" == "pass" ]] && ! rg -q "test .*${test_filter} .* ok" "$log_path"; then
    status="fail"
  fi
  end_ms="$(date +%s%3N)"
  elapsed_ms="$((end_ms - start_ms))"

  printf '%s|%s|%s\n' "$status" "$elapsed_ms" "$log_path"
}

test1_result="$(run_test "$TEST1_NAME" "$TEST1_FILTER" "fsqlite-mvcc")"
test2_result="$(run_test "$TEST2_NAME" "$TEST2_FILTER" "fsqlite-mvcc")"
test3_result="$(run_test "$TEST3_NAME" "$TEST3_FILTER" "fsqlite-mvcc")"
test4_result="$(run_test "$TEST4_NAME" "$TEST4_FILTER" "fsqlite-e2e")"

TEST1_STATUS="${test1_result%%|*}"
test1_tail="${test1_result#*|}"
TEST1_ELAPSED_MS="${test1_tail%%|*}"
TEST1_LOG_PATH="${test1_tail#*|}"

TEST2_STATUS="${test2_result%%|*}"
test2_tail="${test2_result#*|}"
TEST2_ELAPSED_MS="${test2_tail%%|*}"
TEST2_LOG_PATH="${test2_tail#*|}"

TEST3_STATUS="${test3_result%%|*}"
test3_tail="${test3_result#*|}"
TEST3_ELAPSED_MS="${test3_tail%%|*}"
TEST3_LOG_PATH="${test3_tail#*|}"

TEST4_STATUS="${test4_result%%|*}"
test4_tail="${test4_result#*|}"
TEST4_ELAPSED_MS="${test4_tail%%|*}"
TEST4_LOG_PATH="${test4_tail#*|}"

OVERALL_STATUS="pass"
if [[ "$TEST1_STATUS" != "pass" || "$TEST2_STATUS" != "pass" || "$TEST3_STATUS" != "pass" || "$TEST4_STATUS" != "pass" ]]; then
  OVERALL_STATUS="fail"
fi

REPLAY_CMD="scripts/bd_2y306_5_ebr_gc_e2e.sh --json"

cat >"$ARTIFACT_PATH" <<EOF
{
  "bead_id": "$BEAD_ID",
  "run_id": "$RUN_ID",
  "trace_id": "$TRACE_ID",
  "scenario_id": "$SCENARIO_ID",
  "seed": $SEED,
  "overall_status": "$OVERALL_STATUS",
  "tests": [
    {
      "test_id": "$TEST1_NAME",
      "crate": "fsqlite-mvcc",
      "filter": "$TEST1_FILTER",
      "status": "$TEST1_STATUS",
      "elapsed_ms": $TEST1_ELAPSED_MS,
      "log_path": "${TEST1_LOG_PATH#$WORKSPACE_ROOT/}"
    },
    {
      "test_id": "$TEST2_NAME",
      "crate": "fsqlite-mvcc",
      "filter": "$TEST2_FILTER",
      "status": "$TEST2_STATUS",
      "elapsed_ms": $TEST2_ELAPSED_MS,
      "log_path": "${TEST2_LOG_PATH#$WORKSPACE_ROOT/}"
    },
    {
      "test_id": "$TEST3_NAME",
      "crate": "fsqlite-mvcc",
      "filter": "$TEST3_FILTER",
      "status": "$TEST3_STATUS",
      "elapsed_ms": $TEST3_ELAPSED_MS,
      "log_path": "${TEST3_LOG_PATH#$WORKSPACE_ROOT/}"
    },
    {
      "test_id": "$TEST4_NAME",
      "crate": "fsqlite-e2e",
      "filter": "$TEST4_FILTER",
      "status": "$TEST4_STATUS",
      "elapsed_ms": $TEST4_ELAPSED_MS,
      "log_path": "${TEST4_LOG_PATH#$WORKSPACE_ROOT/}"
    }
  ],
  "replay_command": "$REPLAY_CMD"
}
EOF

if $JSON_OUTPUT; then
  cat "$ARTIFACT_PATH"
else
  echo "=== bd-2y306.5 EBR GC E2E Verification ==="
  echo "Run ID:      $RUN_ID"
  echo "Trace ID:    $TRACE_ID"
  echo "Scenario ID: $SCENARIO_ID"
  echo "Seed:        $SEED"
  echo "Artifact:    ${ARTIFACT_PATH#$WORKSPACE_ROOT/}"
  echo "Test 1:      $TEST1_STATUS (${TEST1_ELAPSED_MS}ms)"
  echo "Test 2:      $TEST2_STATUS (${TEST2_ELAPSED_MS}ms)"
  echo "Test 3:      $TEST3_STATUS (${TEST3_ELAPSED_MS}ms)"
  echo "Test 4:      $TEST4_STATUS (${TEST4_ELAPSED_MS}ms)"
  echo "Overall:     $OVERALL_STATUS"
  echo "Replay:"
  echo "  $REPLAY_CMD"
fi

if [[ "$OVERALL_STATUS" != "pass" ]]; then
  exit 1
fi
