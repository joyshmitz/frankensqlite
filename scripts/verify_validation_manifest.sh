#!/usr/bin/env bash
# verify_validation_manifest.sh — deterministic E2E check for bd-mblr.3.5.1
#
# Validates machine-readable manifest generation from real harness gate outputs:
# 1. Generates manifest artifacts twice with fixed deterministic inputs
# 2. Verifies required schema/field contract with jq
# 3. Confirms byte-for-byte deterministic reproducibility
# 4. Emits replay command and artifact paths for operator handoff
#
# Usage:
#   ./scripts/verify_validation_manifest.sh [--json] [--seed <u64>] [--generated-unix-ms <u128>]
#
# Env:
#   VALIDATION_MANIFEST_USE_RCH=1   Use `rch exec -- cargo run ...` for the runner invocations.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSON_OUTPUT=false
ROOT_SEED="${VALIDATION_MANIFEST_SEED:-424242}"
GENERATED_UNIX_MS="${VALIDATION_MANIFEST_GENERATED_UNIX_MS:-1700000000000}"
SCENARIO_ID="QUALITY-351"

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
            ROOT_SEED="$2"
            shift 2
            ;;
        --generated-unix-ms)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --generated-unix-ms requires a value" >&2
                exit 2
            fi
            GENERATED_UNIX_MS="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
done

RUN_ROOT="$WORKSPACE_ROOT/artifacts/validation-manifest-e2e"
RUN_A="$RUN_ROOT/run-a"
RUN_B="$RUN_ROOT/run-b"
MANIFEST_A="$RUN_A/validation_manifest.json"
MANIFEST_B="$RUN_B/validation_manifest.json"
SUMMARY_A="$RUN_A/validation_manifest.md"
SUMMARY_B="$RUN_B/validation_manifest.md"
RUN_A_LOG="$RUN_A/validation_manifest_runner.log"
RUN_B_LOG="$RUN_B/validation_manifest_runner.log"
ARTIFACT_URI_PREFIX="artifacts/validation-manifest-e2e/shared"
FIXTURE_ROOT_MANIFEST="$WORKSPACE_ROOT/corpus_manifest.toml"

mkdir -p "$RUN_A" "$RUN_B"

if [[ ! -f "$FIXTURE_ROOT_MANIFEST" ]]; then
    echo "ERROR: canonical fixture-root manifest missing at $FIXTURE_ROOT_MANIFEST" >&2
    exit 1
fi
FIXTURE_ROOT_MANIFEST_SHA256="$(sha256sum "$FIXTURE_ROOT_MANIFEST" | awk '{print $1}')"

COMMIT_SHA="$(git -C "$WORKSPACE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
RUN_ID="bd-mblr.3.5.1-seed-${ROOT_SEED}"
TRACE_ID="trace-$(printf '%s' "$RUN_ID" | sha256sum | awk '{print $1}' | cut -c1-16)"

USE_RCH_RUNNER="${VALIDATION_MANIFEST_USE_RCH:-0}"
if [[ "${USE_RCH_RUNNER}" == "1" ]] && command -v rch >/dev/null 2>&1; then
    RUNNER=(rch exec -- cargo run -p fsqlite-harness --bin validation_manifest_runner --)
else
    RUNNER=(cargo run -p fsqlite-harness --bin validation_manifest_runner --)
fi

COMMON_ARGS=(
    --workspace-root "$WORKSPACE_ROOT"
    --commit-sha "$COMMIT_SHA"
    --run-id "$RUN_ID"
    --trace-id "$TRACE_ID"
    --scenario-id "$SCENARIO_ID"
    --fixture-root-manifest "$FIXTURE_ROOT_MANIFEST"
    --root-seed "$ROOT_SEED"
    --generated-unix-ms "$GENERATED_UNIX_MS"
)

run_manifest() {
    local output_dir="$1"
    local output_json="$2"
    local output_human="$3"
    local runner_log="$4"
    local run_status

    set +e
    "${RUNNER[@]}" \
        "${COMMON_ARGS[@]}" \
        --output-dir "$output_dir" \
        --output-json "$output_json" \
        --output-human "$output_human" \
        --artifact-uri-prefix "$ARTIFACT_URI_PREFIX" \
        >"$runner_log" 2>&1
    run_status=$?
    set -e

    if [[ ${run_status} -ne 0 && ${run_status} -ne 1 ]]; then
        echo "ERROR: validation_manifest_runner failed unexpectedly (exit=${run_status})" >&2
        echo "----- runner log: ${runner_log} -----" >&2
        cat "$runner_log" >&2
        exit 1
    fi

    printf '%s' "${run_status}"
}

run_a_status="$(run_manifest "$RUN_A" "$MANIFEST_A" "$SUMMARY_A" "$RUN_A_LOG")"
run_b_status="$(run_manifest "$RUN_B" "$MANIFEST_B" "$SUMMARY_B" "$RUN_B_LOG")"

if [[ "${run_a_status}" != "${run_b_status}" ]]; then
    echo "ERROR: validation manifest reruns produced different exit codes (run-a=${run_a_status}, run-b=${run_b_status})" >&2
    exit 1
fi

if [[ ! -f "$MANIFEST_A" || ! -f "$MANIFEST_B" ]]; then
    echo "ERROR: manifest output missing" >&2
    exit 1
fi

jq -e \
    --arg fixture_manifest_path "$FIXTURE_ROOT_MANIFEST" \
    --arg fixture_manifest_sha256 "$FIXTURE_ROOT_MANIFEST_SHA256" \
    '
    .schema_version == "1.0.0" and
    .bead_id == "bd-mblr.3.5.1" and
    (.commit_sha | length) > 0 and
    (.run_id | length) > 0 and
    (.trace_id | length) > 0 and
    (.scenario_id == "QUALITY-351") and
    (.fixture_root_manifest_path == $fixture_manifest_path) and
    (.fixture_root_manifest_sha256 == $fixture_manifest_sha256) and
    (.gates | length) >= 5 and
    (.artifact_uris | length) >= 6 and
    (.replay.command | length) > 0 and
    (.replay.command | contains("--fixture-root-manifest")) and
    (.logging_conformance.gate_id == "bd-mblr.5.5.1") and
    (.logging_conformance.log_validation.passed == true) and
    (.logging_conformance.shell_script_conformance.overall_pass == true)
' "$MANIFEST_A" >/dev/null

if ! diff -u "$MANIFEST_A" "$MANIFEST_B" >/dev/null; then
    echo "ERROR: deterministic replay check failed; manifests differ" >&2
    diff -u "$MANIFEST_A" "$MANIFEST_B" >&2 || true
    exit 1
fi

REPLAY_COMMAND="$(jq -r '.replay.command' "$MANIFEST_A")"
EXPECTED_REPLAY_COMMAND="cargo run -p fsqlite-harness --bin validation_manifest_runner -- --root-seed ${ROOT_SEED} --generated-unix-ms ${GENERATED_UNIX_MS} --fixture-root-manifest '${FIXTURE_ROOT_MANIFEST}' --commit-sha '${COMMIT_SHA}' --run-id '${RUN_ID}' --trace-id '${TRACE_ID}' --scenario-id '${SCENARIO_ID}' --artifact-uri-prefix '${ARTIFACT_URI_PREFIX}'"
if [[ "${REPLAY_COMMAND}" != "${EXPECTED_REPLAY_COMMAND}" ]]; then
    echo "ERROR: replay command mismatch (expected deterministic exact command)" >&2
    echo "expected: ${EXPECTED_REPLAY_COMMAND}" >&2
    echo "actual:   ${REPLAY_COMMAND}" >&2
    exit 1
fi

GATE_COUNT="$(jq -r '.gates | length' "$MANIFEST_A")"
ARTIFACT_COUNT="$(jq -r '.artifact_uris | length' "$MANIFEST_A")"
OVERALL_OUTCOME="$(jq -r '.overall_outcome' "$MANIFEST_A")"

if $JSON_OUTPUT; then
    cat <<ENDJSON
{
  "bead_id": "bd-mblr.3.5.1",
  "run_id": "$RUN_ID",
  "trace_id": "$TRACE_ID",
  "scenario_id": "$SCENARIO_ID",
  "commit_sha": "$COMMIT_SHA",
  "root_seed": "$ROOT_SEED",
  "generated_unix_ms": "$GENERATED_UNIX_MS",
  "fixture_root_manifest_path": "$FIXTURE_ROOT_MANIFEST",
  "fixture_root_manifest_sha256": "$FIXTURE_ROOT_MANIFEST_SHA256",
  "deterministic_match": true,
  "manifest_a": "${MANIFEST_A#$WORKSPACE_ROOT/}",
  "manifest_b": "${MANIFEST_B#$WORKSPACE_ROOT/}",
  "summary_a": "${SUMMARY_A#$WORKSPACE_ROOT/}",
  "summary_b": "${SUMMARY_B#$WORKSPACE_ROOT/}",
  "runner_log_a": "${RUN_A_LOG#$WORKSPACE_ROOT/}",
  "runner_log_b": "${RUN_B_LOG#$WORKSPACE_ROOT/}",
  "run_a_exit_code": ${run_a_status},
  "run_b_exit_code": ${run_b_status},
  "gate_count": $GATE_COUNT,
  "artifact_count": $ARTIFACT_COUNT,
  "overall_outcome": "$OVERALL_OUTCOME",
  "replay_command": "$REPLAY_COMMAND"
}
ENDJSON
else
    echo "=== Validation Manifest E2E Check ==="
    echo "Bead ID:            bd-mblr.3.5.1"
    echo "Run ID:             $RUN_ID"
    echo "Trace ID:           $TRACE_ID"
    echo "Scenario ID:        $SCENARIO_ID"
    echo "Commit SHA:         $COMMIT_SHA"
    echo "Root seed:          $ROOT_SEED"
    echo "Generated unix ms:  $GENERATED_UNIX_MS"
    echo "Fixture manifest:   ${FIXTURE_ROOT_MANIFEST#$WORKSPACE_ROOT/}"
    echo "Fixture sha256:     $FIXTURE_ROOT_MANIFEST_SHA256"
    echo "Gate count:         $GATE_COUNT"
    echo "Artifact count:     $ARTIFACT_COUNT"
    echo "Overall outcome:    $OVERALL_OUTCOME"
    echo "Manifest A:         ${MANIFEST_A#$WORKSPACE_ROOT/}"
    echo "Manifest B:         ${MANIFEST_B#$WORKSPACE_ROOT/}"
    echo "Runner log A:       ${RUN_A_LOG#$WORKSPACE_ROOT/}"
    echo "Runner log B:       ${RUN_B_LOG#$WORKSPACE_ROOT/}"
    echo "Run A exit code:    ${run_a_status}"
    echo "Run B exit code:    ${run_b_status}"
    echo "Deterministic:      PASS"
    echo "Replay command:"
    echo "  $REPLAY_COMMAND"
fi
