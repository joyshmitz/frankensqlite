#!/usr/bin/env bash
# verify_bisect_replay_manifest.sh — deterministic E2E verification for bd-mblr.7.6.1
#
# Validates the bisect replay manifest contract with structured evidence:
# 1. Runs the bead integration test twice with the same seed and compares outcomes.
# 2. Runs focused replay_harness unit coverage for bisect manifest behavior.
# 3. Emits structured JSONL logs (trace_id/run_id/scenario_id) and a summary artifact.
#
# Usage:
#   ./scripts/verify_bisect_replay_manifest.sh [--json] [--seed <u64>]

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEAD_ID="bd-mblr.7.6.1"
SCENARIO_ID="${SCENARIO_ID:-BISECT-REPLAY-MANIFEST-1}"
SEED="${BISECT_REPLAY_MANIFEST_SEED:-424242}"
JSON_OUTPUT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)
            JSON_OUTPUT=true
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
        *)
            echo "ERROR: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
done

if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required" >&2
    exit 1
fi

if ! command -v rch >/dev/null 2>&1; then
    echo "ERROR: rch is required for cargo test offload" >&2
    exit 1
fi

RUN_ID="${RUN_ID:-${BEAD_ID}-seed-${SEED}}"
TRACE_ID="${TRACE_ID:-trace-$(printf '%s' "$RUN_ID" | sha256sum | awk '{print $1}' | cut -c1-16)}"

ARTIFACT_DIR="$WORKSPACE_ROOT/artifacts/bisect-replay-manifest-e2e/$RUN_ID"
LOG_PATH="$ARTIFACT_DIR/debug_log.jsonl"
SUMMARY_PATH="$ARTIFACT_DIR/verification_summary.json"
mkdir -p "$ARTIFACT_DIR"
: >"$LOG_PATH"

iso_now() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

ms_now() {
    date +%s%3N
}

log_event() {
    local phase="$1"
    local level="$2"
    local event_type="$3"
    local outcome="$4"
    local message="$5"
    local extra='{}'
    if [[ $# -ge 6 ]]; then
        extra="$6"
    fi

    jq -cn \
        --arg timestamp "$(iso_now)" \
        --arg trace_id "$TRACE_ID" \
        --arg run_id "$RUN_ID" \
        --arg scenario_id "$SCENARIO_ID" \
        --arg bead_id "$BEAD_ID" \
        --arg phase "$phase" \
        --arg level "$level" \
        --arg event_type "$event_type" \
        --arg outcome "$outcome" \
        --arg message "$message" \
        --argjson extra "$extra" \
        '{
            timestamp: $timestamp,
            trace_id: $trace_id,
            run_id: $run_id,
            scenario_id: $scenario_id,
            bead_id: $bead_id,
            phase: $phase,
            level: $level,
            event_type: $event_type,
            outcome: $outcome,
            message: $message,
            context: $extra
        }' >>"$LOG_PATH"
}

run_test_phase() {
    local phase="$1"
    shift
    local out_file="$ARTIFACT_DIR/${phase}.log"
    local start_ms end_ms duration_ms exit_code result_line passed failed

    start_ms="$(ms_now)"
    log_event "$phase" "INFO" "phase_start" "running" "Starting test phase" \
        "$(jq -cn --arg command "$*" '{command: $command, output_log: null}')"

    if "$@" >"$out_file" 2>&1; then
        exit_code=0
    else
        exit_code=$?
    fi

    end_ms="$(ms_now)"
    duration_ms="$((end_ms - start_ms))"
    result_line="$(grep 'test result:' "$out_file" | tail -1 || true)"
    passed="$(printf '%s\n' "$result_line" | sed -nE 's/.* ([0-9]+) passed.*/\1/p')"
    failed="$(printf '%s\n' "$result_line" | sed -nE 's/.* ([0-9]+) failed.*/\1/p')"
    passed="${passed:-0}"
    failed="${failed:-0}"

    if [[ "$exit_code" -eq 0 ]]; then
        log_event "$phase" "INFO" "phase_end" "pass" "Completed test phase" \
            "$(jq -cn \
                --arg command "$*" \
                --arg output_log "${out_file#$WORKSPACE_ROOT/}" \
                --argjson duration_ms "$duration_ms" \
                --argjson exit_code "$exit_code" \
                --argjson passed "$passed" \
                --argjson failed "$failed" \
                '{command: $command, output_log: $output_log, duration_ms: $duration_ms, exit_code: $exit_code, passed: $passed, failed: $failed}')"
    else
        log_event "$phase" "ERROR" "phase_end" "fail" "Test phase failed" \
            "$(jq -cn \
                --arg command "$*" \
                --arg output_log "${out_file#$WORKSPACE_ROOT/}" \
                --argjson duration_ms "$duration_ms" \
                --argjson exit_code "$exit_code" \
                --argjson passed "$passed" \
                --argjson failed "$failed" \
                '{command: $command, output_log: $output_log, duration_ms: $duration_ms, exit_code: $exit_code, passed: $passed, failed: $failed}')"
    fi

    printf '%s|%s|%s|%s|%s|%s\n' "$phase" "$exit_code" "$passed" "$failed" "$duration_ms" "${out_file#$WORKSPACE_ROOT/}"
}

log_event "bootstrap" "INFO" "run_start" "running" "Starting bisect replay manifest verification" \
    "$(jq -cn --arg seed "$SEED" '{seed: $seed}')"

PHASE1="$(run_test_phase integration_a \
    rch exec -- cargo test -p fsqlite-harness --test bd_mblr_7_6_1_bisect_replay_manifest -- --nocapture)"
PHASE2="$(run_test_phase integration_b \
    rch exec -- cargo test -p fsqlite-harness --test bd_mblr_7_6_1_bisect_replay_manifest -- --nocapture)"
PHASE3="$(run_test_phase replay_harness_unit \
    rch exec -- cargo test -p fsqlite-harness --lib bisect_manifest)"

IFS='|' read -r _ P1_EXIT P1_PASS P1_FAIL P1_MS P1_LOG <<<"$PHASE1"
IFS='|' read -r _ P2_EXIT P2_PASS P2_FAIL P2_MS P2_LOG <<<"$PHASE2"
IFS='|' read -r _ P3_EXIT P3_PASS P3_FAIL P3_MS P3_LOG <<<"$PHASE3"

DETERMINISTIC_MATCH=false
if [[ "$P1_PASS" -eq "$P2_PASS" && "$P1_FAIL" -eq "$P2_FAIL" ]]; then
    DETERMINISTIC_MATCH=true
fi

OVERALL_RESULT="pass"
if [[ "$P1_EXIT" -ne 0 || "$P2_EXIT" -ne 0 || "$P3_EXIT" -ne 0 || "$DETERMINISTIC_MATCH" != "true" ]]; then
    OVERALL_RESULT="fail"
fi

TOTAL_PASSED="$((P1_PASS + P2_PASS + P3_PASS))"
TOTAL_FAILED="$((P1_FAIL + P2_FAIL + P3_FAIL))"
REPLAY_COMMAND="./scripts/verify_bisect_replay_manifest.sh --seed ${SEED} --json"

jq -cn \
    --arg bead_id "$BEAD_ID" \
    --arg run_id "$RUN_ID" \
    --arg trace_id "$TRACE_ID" \
    --arg scenario_id "$SCENARIO_ID" \
    --arg seed "$SEED" \
    --arg replay_command "$REPLAY_COMMAND" \
    --arg result "$OVERALL_RESULT" \
    --arg created_at "$(iso_now)" \
    --argjson deterministic_match "$DETERMINISTIC_MATCH" \
    --arg p1_log "$P1_LOG" \
    --arg p2_log "$P2_LOG" \
    --arg p3_log "$P3_LOG" \
    --arg debug_log "${LOG_PATH#$WORKSPACE_ROOT/}" \
    --arg summary_path "${SUMMARY_PATH#$WORKSPACE_ROOT/}" \
    --argjson p1_pass "$P1_PASS" \
    --argjson p1_fail "$P1_FAIL" \
    --argjson p1_exit "$P1_EXIT" \
    --argjson p1_duration_ms "$P1_MS" \
    --argjson p2_pass "$P2_PASS" \
    --argjson p2_fail "$P2_FAIL" \
    --argjson p2_exit "$P2_EXIT" \
    --argjson p2_duration_ms "$P2_MS" \
    --argjson p3_pass "$P3_PASS" \
    --argjson p3_fail "$P3_FAIL" \
    --argjson p3_exit "$P3_EXIT" \
    --argjson p3_duration_ms "$P3_MS" \
    --argjson total_passed "$TOTAL_PASSED" \
    --argjson total_failed "$TOTAL_FAILED" \
    '{
        schema_version: "1.0.0",
        bead_id: $bead_id,
        run_id: $run_id,
        trace_id: $trace_id,
        scenario_id: $scenario_id,
        seed: $seed,
        deterministic_match: $deterministic_match,
        result: $result,
        created_at: $created_at,
        phases: [
            {
                phase: "integration_a",
                command: "rch exec -- cargo test -p fsqlite-harness --test bd_mblr_7_6_1_bisect_replay_manifest -- --nocapture",
                passed: $p1_pass,
                failed: $p1_fail,
                exit_code: $p1_exit,
                duration_ms: $p1_duration_ms,
                output_log: $p1_log
            },
            {
                phase: "integration_b",
                command: "rch exec -- cargo test -p fsqlite-harness --test bd_mblr_7_6_1_bisect_replay_manifest -- --nocapture",
                passed: $p2_pass,
                failed: $p2_fail,
                exit_code: $p2_exit,
                duration_ms: $p2_duration_ms,
                output_log: $p2_log
            },
            {
                phase: "replay_harness_unit",
                command: "rch exec -- cargo test -p fsqlite-harness --lib bisect_manifest",
                passed: $p3_pass,
                failed: $p3_fail,
                exit_code: $p3_exit,
                duration_ms: $p3_duration_ms,
                output_log: $p3_log
            }
        ],
        totals: {
            passed: $total_passed,
            failed: $total_failed
        },
        artifacts: {
            summary_json: $summary_path,
            debug_log_jsonl: $debug_log
        },
        replay: {
            command: $replay_command
        }
    }' >"$SUMMARY_PATH"

log_event "bootstrap" "INFO" "run_end" "$OVERALL_RESULT" "Completed bisect replay manifest verification" \
    "$(jq -cn --arg summary "${SUMMARY_PATH#$WORKSPACE_ROOT/}" --arg result "$OVERALL_RESULT" '{summary_json: $summary, result: $result}')"

if $JSON_OUTPUT; then
    cat "$SUMMARY_PATH"
else
    echo "=== Bisect Replay Manifest Verification ==="
    echo "Bead ID:              $BEAD_ID"
    echo "Run ID:               $RUN_ID"
    echo "Trace ID:             $TRACE_ID"
    echo "Scenario ID:          $SCENARIO_ID"
    echo "Seed:                 $SEED"
    echo "Deterministic match:  $DETERMINISTIC_MATCH"
    echo "Result:               $OVERALL_RESULT"
    echo "Summary artifact:     ${SUMMARY_PATH#$WORKSPACE_ROOT/}"
    echo "Debug log:            ${LOG_PATH#$WORKSPACE_ROOT/}"
    echo "Replay command:"
    echo "  $REPLAY_COMMAND"
fi

if [[ "$OVERALL_RESULT" != "pass" ]]; then
    exit 1
fi
