#!/usr/bin/env bash
# verify_ci_artifact_gates.sh — CI artifact gate orchestrator (bd-1lsfu.5)
#
# Gate contract:
# 1. Perf baseline artifacts are generated and validated.
# 2. p95 latency does not regress >10% versus committed baseline.
# 3. Top-3 flamegraph artifacts are generated for human review (non-blocking).
# 4. Golden checksum guard passes.
# 5. Coverage threshold gate passes (lines/functions).
# 6. Miri gate passes for selected critical crate surface.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEAD_ID="bd-1lsfu.5"
RUN_ID="${BEAD_ID}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
ARTIFACT_ROOT="${WORKSPACE_ROOT}/artifacts/ci-artifact-gates"
LOG_FILE="${ARTIFACT_ROOT}/run.jsonl"
SUMMARY_MD="${ARTIFACT_ROOT}/summary.md"
REPORT_JSON="${ARTIFACT_ROOT}/report.json"
PERF_STDOUT_LOG="${ARTIFACT_ROOT}/perf_baseline_pack.stdout.log"
P95_REPORT_JSON="${ARTIFACT_ROOT}/p95_regression_report.json"
FLAMEGRAPH_LOG="${ARTIFACT_ROOT}/flamegraph.log"
GOLDEN_LOG="${ARTIFACT_ROOT}/golden_integrity.log"
COVERAGE_LOG="${ARTIFACT_ROOT}/coverage.log"
MIRI_LOG="${ARTIFACT_ROOT}/miri.log"

PERF_BASELINE_PATH="${WORKSPACE_ROOT}/baselines/criterion/bd-1dp9.6.1-baseline.json"
PERF_BASELINE_BEFORE="${ARTIFACT_ROOT}/bd-1dp9.6.1-baseline.before.json"
PERF_BASELINE_AFTER="${WORKSPACE_ROOT}/baselines/criterion/bd-1dp9.6.1-baseline-latest.json"
FLAMEGRAPH_MANIFEST="${WORKSPACE_ROOT}/artifacts/perf/bd-1dp9.6.1/profiling/flamegraph_top3.json"

mkdir -p "${ARTIFACT_ROOT}"

log_event() {
  local level="$1"
  local phase="$2"
  local message="$3"
  printf '{"run_id":"%s","bead_id":"%s","level":"%s","phase":"%s","message":"%s","ts":"%s"}\n' \
    "${RUN_ID}" "${BEAD_ID}" "${level}" "${phase}" "${message}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> "${LOG_FILE}"
}

status="pass"
failed_phase=""
flamegraph_outcome="skipped"

log_event "INFO" "start" "starting ci artifact gate workflow"

if [[ ! -f "${PERF_BASELINE_PATH}" ]]; then
  log_event "ERROR" "perf-baseline" "missing committed perf baseline file"
  echo "ERROR: missing baseline file: ${PERF_BASELINE_PATH}" >&2
  exit 1
fi
cp "${PERF_BASELINE_PATH}" "${PERF_BASELINE_BEFORE}"

log_event "INFO" "perf-baseline" "running deterministic perf baseline pack gate"
set +e
bash "${WORKSPACE_ROOT}/scripts/verify_perf_baseline_pack.sh" > "${PERF_STDOUT_LOG}" 2>&1
perf_status=$?
set -e
if [[ ${perf_status} -ne 0 ]]; then
  log_event "ERROR" "perf-baseline" "perf baseline pack gate failed"
  status="fail"
  failed_phase="perf-baseline"
fi

if [[ "${status}" == "pass" ]]; then
  if [[ ! -f "${PERF_BASELINE_AFTER}" ]]; then
    log_event "ERROR" "perf-p95" "missing generated baseline-latest after perf run"
    echo "ERROR: missing generated baseline: ${PERF_BASELINE_AFTER}" >&2
    status="fail"
    failed_phase="perf-p95"
  else
    log_event "INFO" "perf-p95" "checking p95 regression threshold <= 10%"
    jq -n \
      --slurpfile baseline "${PERF_BASELINE_BEFORE}" \
      --slurpfile candidate "${PERF_BASELINE_AFTER}" \
      '
      def to_map(arr):
        reduce arr[] as $row ({}; .[$row.scenario_id] = $row);
      ($baseline[0] // []) as $b
      | ($candidate[0] // []) as $c
      | (to_map($b)) as $bmap
      | (to_map($c)) as $cmap
      | ($bmap | keys | sort) as $keys
      | [
          $keys[]
          | . as $scenario
          | ($bmap[$scenario]) as $base
          | ($cmap[$scenario]) as $cand
          | select($base != null and $cand != null)
          | {
              scenario_id: $scenario,
              baseline_p95_micros: ($base.p95_micros // 0),
              candidate_p95_micros: ($cand.p95_micros // 0),
              ratio: (
                if ($base.p95_micros // 0) > 0
                then (($cand.p95_micros // 0) / ($base.p95_micros // 0))
                else 1
                end
              )
            }
        ] as $comparisons
      | {
          threshold_ratio: 1.10,
          comparisons: $comparisons,
          regressions: [ $comparisons[] | select(.ratio > 1.10) ]
        }
      ' > "${P95_REPORT_JSON}"

    regression_count="$(jq -r '.regressions | length' "${P95_REPORT_JSON}")"
    if [[ "${regression_count}" != "0" ]]; then
      log_event "ERROR" "perf-p95" "p95 regression exceeded 10% threshold"
      echo "ERROR: p95 regression threshold exceeded for ${regression_count} scenario(s)." >&2
      jq -r '.regressions[] | "- \(.scenario_id): baseline=\(.baseline_p95_micros) candidate=\(.candidate_p95_micros) ratio=\(.ratio)"' "${P95_REPORT_JSON}" >&2
      status="fail"
      failed_phase="perf-p95"
    else
      log_event "INFO" "perf-p95" "p95 regression threshold check passed"
    fi
  fi
fi

if [[ "${status}" == "pass" ]]; then
  log_event "INFO" "flamegraph" "validating top-3 flamegraph artifacts (non-blocking)"
  set +e
  {
    if [[ ! -f "${FLAMEGRAPH_MANIFEST}" ]]; then
      echo "missing flamegraph manifest: ${FLAMEGRAPH_MANIFEST}" >&2
      exit 1
    fi
    if ! jq -e \
      '.schema_version == "fsqlite.perf.flamegraph-top3.v1"
      and (.entries | length >= 3)
      and (.entries | all(.rank != null and .scenario_id != null and .artifact_path != null))' \
      "${FLAMEGRAPH_MANIFEST}" >/dev/null; then
      echo "invalid flamegraph manifest payload: ${FLAMEGRAPH_MANIFEST}" >&2
      exit 1
    fi
    while IFS= read -r artifact_rel; do
      artifact_abs="${WORKSPACE_ROOT}/artifacts/perf/bd-1dp9.6.1/${artifact_rel}"
      if [[ ! -f "${artifact_abs}" ]]; then
        echo "missing flamegraph artifact from manifest: ${artifact_abs}" >&2
        exit 1
      fi
    done < <(jq -r '.entries[].artifact_path' "${FLAMEGRAPH_MANIFEST}")
    jq -r '.entries[] | "- rank=\(.rank) scenario=\(.scenario_id) artifact=\(.artifact_path)"' \
      "${FLAMEGRAPH_MANIFEST}"
  } > "${FLAMEGRAPH_LOG}" 2>&1
  flamegraph_status=$?
  set -e
  if [[ ${flamegraph_status} -ne 0 ]]; then
    log_event "WARN" "flamegraph" "top-3 flamegraph artifact validation failed (non-blocking)"
    flamegraph_outcome="warn"
  else
    log_event "INFO" "flamegraph" "top-3 flamegraph artifact validation passed"
    flamegraph_outcome="pass"
  fi
fi

if [[ "${status}" == "pass" ]]; then
  log_event "INFO" "golden" "running golden checksum immutability guard"
  set +e
  {
    cargo test -p fsqlite-e2e --test golden_integrity -- checksums_sha256 --nocapture
    cargo test -p fsqlite-e2e --test golden_integrity -- golden_checksum_file_matches_actual_hashes --nocapture
  } > "${GOLDEN_LOG}" 2>&1
  golden_status=$?
  set -e
  if [[ ${golden_status} -ne 0 ]]; then
    log_event "ERROR" "golden" "golden checksum gate failed"
    status="fail"
    failed_phase="golden"
  fi
fi

if [[ "${status}" == "pass" ]]; then
  log_event "INFO" "coverage" "running canonical coverage threshold gate"
  set +e
  bash "${WORKSPACE_ROOT}/scripts/coverage.sh" --json --fail-under-lines 70 --fail-under-functions 65 \
    > "${COVERAGE_LOG}" 2>&1
  coverage_status=$?
  set -e
  if [[ ${coverage_status} -ne 0 ]]; then
    log_event "ERROR" "coverage" "coverage threshold gate failed"
    status="fail"
    failed_phase="coverage"
  fi
fi

if [[ "${status}" == "pass" ]]; then
  log_event "INFO" "miri" "running miri gate on critical crate surface"
  set +e
  {
    cargo miri setup
    cargo miri test -p fsqlite-types --lib -- --nocapture
  } > "${MIRI_LOG}" 2>&1
  miri_status=$?
  set -e
  if [[ ${miri_status} -ne 0 ]]; then
    log_event "ERROR" "miri" "miri gate failed"
    status="fail"
    failed_phase="miri"
  fi
fi

{
  echo "# CI Artifact Gates Summary"
  echo
  echo "- run_id: \`${RUN_ID}\`"
  echo "- bead_id: \`${BEAD_ID}\`"
  echo "- status: \`${status}\`"
  echo "- flamegraph_outcome: \`${flamegraph_outcome}\`"
  if [[ -n "${failed_phase}" ]]; then
    echo "- failed_phase: \`${failed_phase}\`"
  else
    echo "- failed_phase: \`none\`"
  fi
  echo
  echo "## Artifacts"
  echo "- structured_log: \`${LOG_FILE}\`"
  echo "- perf_log: \`${PERF_STDOUT_LOG}\`"
  echo "- p95_report: \`${P95_REPORT_JSON}\`"
  echo "- flamegraph_log: \`${FLAMEGRAPH_LOG}\`"
  echo "- flamegraph_manifest: \`${FLAMEGRAPH_MANIFEST}\`"
  echo "- golden_log: \`${GOLDEN_LOG}\`"
  echo "- coverage_log: \`${COVERAGE_LOG}\`"
  echo "- miri_log: \`${MIRI_LOG}\`"
} > "${SUMMARY_MD}"

jq -n \
  --arg run_id "${RUN_ID}" \
  --arg bead_id "${BEAD_ID}" \
  --arg status "${status}" \
  --arg flamegraph_outcome "${flamegraph_outcome}" \
  --arg failed_phase "${failed_phase}" \
  --arg log_file "${LOG_FILE}" \
  --arg perf_log "${PERF_STDOUT_LOG}" \
  --arg p95_report "${P95_REPORT_JSON}" \
  --arg flamegraph_log "${FLAMEGRAPH_LOG}" \
  --arg flamegraph_manifest "${FLAMEGRAPH_MANIFEST}" \
  --arg golden_log "${GOLDEN_LOG}" \
  --arg coverage_log "${COVERAGE_LOG}" \
  --arg miri_log "${MIRI_LOG}" \
  --arg summary_md "${SUMMARY_MD}" \
  '{
    schema_version: "fsqlite.ci-artifact-gates.v1",
    run_id: $run_id,
    bead_id: $bead_id,
    status: $status,
    flamegraph_outcome: $flamegraph_outcome,
    failed_phase: (if $failed_phase == "" then null else $failed_phase end),
    artifacts: {
      structured_log: $log_file,
      perf_log: $perf_log,
      p95_report: $p95_report,
      flamegraph_log: $flamegraph_log,
      flamegraph_manifest: $flamegraph_manifest,
      golden_log: $golden_log,
      coverage_log: $coverage_log,
      miri_log: $miri_log,
      summary_md: $summary_md
    }
  }' > "${REPORT_JSON}"

if [[ "${status}" != "pass" ]]; then
  log_event "ERROR" "complete" "ci artifact gates failed"
  echo "ERROR: ci artifact gates failed (phase=${failed_phase}). See ${REPORT_JSON}" >&2
  exit 1
fi

log_event "INFO" "complete" "ci artifact gates passed"
echo "CI artifact gates passed."
echo "Report: ${REPORT_JSON}"
