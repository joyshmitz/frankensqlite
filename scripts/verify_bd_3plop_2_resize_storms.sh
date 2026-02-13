#!/usr/bin/env bash
# verify_bd_3plop_2_resize_storms.sh — bead bd-3plop.2 verification runner
#
# Usage:
#   ./scripts/verify_bd_3plop_2_resize_storms.sh [--json] [--steps N] [--query-loops N] [--writer-ops N] [--asan]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ID="bd-3plop-2-$(date -u +%Y%m%dT%H%M%SZ)-$$"
JSON_OUTPUT=false
STEPS="${BD_3PLOP2_STEPS:-80}"
QUERY_LOOPS="${BD_3PLOP2_QUERY_LOOPS:-120}"
WRITER_OPS="${BD_3PLOP2_WRITER_OPS:-80}"
ASAN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)
            JSON_OUTPUT=true
            shift
            ;;
        --steps)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --steps requires value" >&2; exit 2; }
            STEPS="$1"
            shift
            ;;
        --query-loops)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --query-loops requires value" >&2; exit 2; }
            QUERY_LOOPS="$1"
            shift
            ;;
        --writer-ops)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --writer-ops requires value" >&2; exit 2; }
            WRITER_OPS="$1"
            shift
            ;;
        --asan)
            ASAN=true
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

TEST_LOG="$(mktemp)"
RESULT="pass"

if [[ "$ASAN" == "true" ]]; then
    TEST_CMD=(
        env
        RUSTFLAGS=-Zsanitizer=address
        BD_3PLOP2_STEPS="$STEPS"
        BD_3PLOP2_QUERY_LOOPS="$QUERY_LOOPS"
        BD_3PLOP2_WRITER_OPS="$WRITER_OPS"
        cargo +nightly test -p fsqlite-harness --test bd_3plop_2_resize_storms
        --target x86_64-unknown-linux-gnu -- --nocapture
    )
else
    TEST_CMD=(
        env
        BD_3PLOP2_STEPS="$STEPS"
        BD_3PLOP2_QUERY_LOOPS="$QUERY_LOOPS"
        BD_3PLOP2_WRITER_OPS="$WRITER_OPS"
        cargo test -p fsqlite-harness --test bd_3plop_2_resize_storms -- --nocapture
    )
fi

if ! "${TEST_CMD[@]}" >"$TEST_LOG" 2>&1; then
    RESULT="fail"
fi

ARTIFACT_PATH="$({ rg -o 'path=[^ ]+' "$TEST_LOG" | tail -n1 | sed 's/^path=//'; } || true)"
if [[ -n "$ARTIFACT_PATH" && "$ARTIFACT_PATH" != /* ]]; then
    ARTIFACT_PATH="$REPO_ROOT/$ARTIFACT_PATH"
fi

if [[ -n "$ARTIFACT_PATH" && -f "$ARTIFACT_PATH" ]]; then
    ARTIFACT_HASH="$(sha256sum "$ARTIFACT_PATH" | awk '{print $1}')"
else
    ARTIFACT_HASH=""
fi

if [[ "$JSON_OUTPUT" == "true" ]]; then
    cat <<ENDJSON
{
  "run_id": "$RUN_ID",
  "bead_id": "bd-3plop.2",
  "resize_steps": $STEPS,
  "query_loops": $QUERY_LOOPS,
  "writer_ops": $WRITER_OPS,
  "asan": $ASAN,
  "result": "$RESULT",
  "artifact_path": "$ARTIFACT_PATH",
  "artifact_hash": "$ARTIFACT_HASH",
  "test_log_path": "$TEST_LOG"
}
ENDJSON
else
    echo "=== bd-3plop.2 Verification ==="
    echo "Run ID:        $RUN_ID"
    echo "Result:        $RESULT"
    echo "Resize steps:  $STEPS"
    echo "Query loops:   $QUERY_LOOPS"
    echo "Writer ops:    $WRITER_OPS"
    echo "ASAN:          $ASAN"
    echo "Artifact path: $ARTIFACT_PATH"
    echo "Artifact hash: $ARTIFACT_HASH"
    echo "Test log:      $TEST_LOG"
fi

[[ "$RESULT" == "pass" ]]
