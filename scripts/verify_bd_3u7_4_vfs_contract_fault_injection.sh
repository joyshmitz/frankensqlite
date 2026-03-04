#!/usr/bin/env bash
# verify_bd_3u7_4_vfs_contract_fault_injection.sh — bead bd-3u7.4 verification runner
#
# Usage:
#   ./scripts/verify_bd_3u7_4_vfs_contract_fault_injection.sh [--json] [--seed N] [--bench-iters N] [--require-io-uring] [--max-attempts N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="bd-3u7-4-$(date -u +%Y%m%dT%H%M%SZ)-$$"
JSON_OUTPUT=false
SEED="${BD_3U7_4_SEED:-981286218}"
BENCH_ITERS="${BD_3U7_4_BENCH_ITERS:-512}"
REQUIRE_IO_URING="${BD_3U7_4_REQUIRE_IO_URING:-false}"
MAX_ATTEMPTS="${BD_3U7_4_MAX_ATTEMPTS:-2}"

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
        --bench-iters)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --bench-iters requires value" >&2; exit 2; }
            BENCH_ITERS="$1"
            shift
            ;;
        --require-io-uring)
            REQUIRE_IO_URING=true
            shift
            ;;
        --max-attempts)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --max-attempts requires value" >&2; exit 2; }
            MAX_ATTEMPTS="$1"
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if ! [[ "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --max-attempts must be a positive integer" >&2
    exit 2
fi

if ! command -v rch >/dev/null 2>&1; then
    echo "ERROR: rch is required for bd-3u7.4 verification" >&2
    exit 3
fi

RESULT="pass"
TEST_LOG=""
SCENARIO_ARTIFACT_PATH=""
RUN_ID_FROM_TEST=""
TRACE_ID=""
SCENARIO_ID=""
ATTEMPT_USED=0

for (( attempt=1; attempt<=MAX_ATTEMPTS; attempt++ )); do
    ATTEMPT_USED="$attempt"
    TEST_LOG="$(mktemp)"
    RESULT="pass"

    if ! BD_3U7_4_SEED="$SEED" \
         BD_3U7_4_BENCH_ITERS="$BENCH_ITERS" \
         BD_3U7_4_REQUIRE_IO_URING="$REQUIRE_IO_URING" \
         rch exec -- cargo test -p fsqlite-harness --test bd_3u7_4_vfs_contract_fault_injection -- --nocapture \
         >"$TEST_LOG" 2>&1; then
        RESULT="fail"
    fi

    SCENARIO_ARTIFACT_PATH="$({ rg -o 'path=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^path=//'; } || true)"
    RUN_ID_FROM_TEST="$({ rg -o 'run_id=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^run_id=//'; } || true)"
    TRACE_ID="$({ rg -o 'trace_id=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^trace_id=//'; } || true)"
    SCENARIO_ID="$({ rg -o 'scenario_id=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^scenario_id=//'; } || true)"

    if [[ "$RESULT" == "pass" ]]; then
        break
    fi

    # If the test actually ran and produced scenario metadata, treat failure as authoritative.
    if [[ -n "$RUN_ID_FROM_TEST" || -n "$SCENARIO_ARTIFACT_PATH" ]]; then
        break
    fi

    # No scenario metadata usually means compile drift / pre-test failure.
    # Retry while attempts remain to reduce false negatives under multi-agent churn.
    if (( attempt < MAX_ATTEMPTS )); then
        echo "WARN: bd-3u7.4 verification failed before scenario execution (attempt $attempt/$MAX_ATTEMPTS), retrying..." >&2
    fi
done

if [[ -n "$SCENARIO_ARTIFACT_PATH" && "$SCENARIO_ARTIFACT_PATH" != /* ]]; then
    SCENARIO_ARTIFACT_PATH="$REPO_ROOT/$SCENARIO_ARTIFACT_PATH"
fi

VERIFY_OUTPUT_DIR="$REPO_ROOT/test-results/bd_3u7_4"
mkdir -p "$VERIFY_OUTPUT_DIR"
VERIFY_ARTIFACT_PATH="$VERIFY_OUTPUT_DIR/verify-${RUN_ID}.json"

cat >"$VERIFY_ARTIFACT_PATH" <<ENDVERIFY
{
  "run_id": "$RUN_ID",
  "bead_id": "bd-3u7.4",
  "seed": $SEED,
  "bench_iters": $BENCH_ITERS,
  "attempt_used": $ATTEMPT_USED,
  "max_attempts": $MAX_ATTEMPTS,
  "require_io_uring": $REQUIRE_IO_URING,
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
  "bead_id": "bd-3u7.4",
  "seed": $SEED,
  "bench_iters": $BENCH_ITERS,
  "attempt_used": $ATTEMPT_USED,
  "max_attempts": $MAX_ATTEMPTS,
  "require_io_uring": $REQUIRE_IO_URING,
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
    echo "=== bd-3u7.4 Verification ==="
    echo "Run ID:        $RUN_ID"
    echo "Result:        $RESULT"
    echo "Seed:          $SEED"
    echo "Bench iters:   $BENCH_ITERS"
    echo "Attempt used:  $ATTEMPT_USED / $MAX_ATTEMPTS"
    echo "Require uring: $REQUIRE_IO_URING"
    echo "Test run_id:   $RUN_ID_FROM_TEST"
    echo "Trace ID:      $TRACE_ID"
    echo "Scenario ID:   $SCENARIO_ID"
    echo "Artifact path: $VERIFY_ARTIFACT_PATH"
    echo "Scenario path: $SCENARIO_ARTIFACT_PATH"
    echo "Artifact hash: $ARTIFACT_HASH"
    echo "Test log:      $TEST_LOG"
fi

[[ "$RESULT" == "pass" ]]
