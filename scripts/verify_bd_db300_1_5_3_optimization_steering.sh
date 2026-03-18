#!/usr/bin/env bash
# Verification artifact for bd-db300.1.5.3:
# publish optimization steering table from classification results.
#
# Deterministic replay:
#   bash scripts/verify_bd_db300_1_5_3_optimization_steering.sh

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEAD_ID="bd-db300.1.5.3"
SCENARIO_ID="DB300-OPTIMIZATION-STEERING-A5.3"
TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_ID="${BEAD_ID}-${TIMESTAMP_UTC}-$$"
TRACE_ID="trace-${RUN_ID}"
GENERATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REPLAY_COMMAND="bash scripts/verify_bd_db300_1_5_3_optimization_steering.sh"
ARTIFACT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/artifacts/perf/${BEAD_ID}/${RUN_ID}}"
EVENTS_JSONL="${ARTIFACT_DIR}/events.jsonl"
STEERING_JSON="${ARTIFACT_DIR}/steering_table.json"
SUMMARY_MD="${ARTIFACT_DIR}/summary.md"
MANIFEST_JSON="${ARTIFACT_DIR}/manifest.json"

mkdir -p "${ARTIFACT_DIR}"
: > "${EVENTS_JSONL}"

emit_event() {
    local phase="$1"
    local event_type="$2"
    local outcome="$3"
    local message="$4"
    jq -cn \
        --arg trace_id "${TRACE_ID}" \
        --arg run_id "${RUN_ID}" \
        --arg bead_id "${BEAD_ID}" \
        --arg scenario_id "${SCENARIO_ID}" \
        --arg phase "${phase}" \
        --arg event_type "${event_type}" \
        --arg outcome "${outcome}" \
        --arg message "${message}" \
        --arg timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        '{
            trace_id: $trace_id,
            run_id: $run_id,
            bead_id: $bead_id,
            scenario_id: $scenario_id,
            phase: $phase,
            event_type: $event_type,
            outcome: $outcome,
            timestamp: $timestamp,
            message: $message
        }' >> "${EVENTS_JSONL}"
}

fail() {
    local phase="$1"
    local message="$2"
    emit_event "${phase}" "fail" "fail" "${message}"
    echo "ERROR: ${message}" >&2
    exit 1
}

require_tool() {
    local tool_name="$1"
    command -v "${tool_name}" >/dev/null 2>&1 || fail "bootstrap" "missing required tool: ${tool_name}"
}

require_file() {
    local path="$1"
    [[ -f "${path}" ]] || fail "inputs" "missing required file: ${path}"
}

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${path}" | awk '{print $1}'
        return
    fi
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "${path}" | awk '{print $1}'
        return
    fi
    fail "bootstrap" "missing sha256 tool"
}

latest_matching_file() {
    local root="$1"
    local grep_pattern="$2"
    local file_name="$3"
    local match
    match="$(find "${root}" -type f -name "${file_name}" | rg "${grep_pattern}" | sort | tail -n 1 || true)"
    [[ -n "${match}" ]] || fail "inputs" "failed to resolve file under ${root} matching ${grep_pattern}"
    printf '%s\n' "${match}"
}

emit_event "bootstrap" "start" "running" "optimization steering synthesis started"

require_tool jq
require_tool rg
require_tool sort

A5_ROOT="${WORKSPACE_ROOT}/artifacts/perf/bd-db300.1.3/representative_hot_cell/runs"
B3_ROOT="${WORKSPACE_ROOT}/artifacts/perf/bd-db300.2.3.1"
D1_ACTIONABLE="${WORKSPACE_ROOT}/artifacts/perf/bd-db300.4.1/actionable_ranking.json"
D1_SUMMARY="${WORKSPACE_ROOT}/artifacts/perf/bd-db300.4.1/summary.md"

A5_MVCC_ACTIONABLE="$(latest_matching_file "${A5_ROOT}" '/mvcc__mixed_read_write__c4__frankensqlite_beads__release__' 'actionable_ranking.json')"
A5_SINGLE_ACTIONABLE="$(latest_matching_file "${A5_ROOT}" '/single_writer__mixed_read_write__c4__frankensqlite_beads__release__' 'actionable_ranking.json')"
B3_CLASSIFICATION="$(latest_matching_file "${B3_ROOT}" '/bd-db300\.2\.3\.1-[^/]+/classification\.json$' 'classification.json')"

A5_MVCC_SUMMARY="${A5_MVCC_ACTIONABLE%/*}/summary.md"
A5_SINGLE_SUMMARY="${A5_SINGLE_ACTIONABLE%/*}/summary.md"
B3_SUMMARY="${B3_CLASSIFICATION%/*}/summary.md"

require_file "${A5_MVCC_ACTIONABLE}"
require_file "${A5_SINGLE_ACTIONABLE}"
require_file "${B3_CLASSIFICATION}"
require_file "${D1_ACTIONABLE}"
require_file "${A5_MVCC_SUMMARY}"
require_file "${A5_SINGLE_SUMMARY}"
require_file "${B3_SUMMARY}"
require_file "${D1_SUMMARY}"

A5_MVCC_SHA="$(sha256_file "${A5_MVCC_ACTIONABLE}")"
A5_SINGLE_SHA="$(sha256_file "${A5_SINGLE_ACTIONABLE}")"
B3_SHA="$(sha256_file "${B3_CLASSIFICATION}")"
D1_SHA="$(sha256_file "${D1_ACTIONABLE}")"

emit_event "inputs" "resolved" "pass" "resolved canonical A5/B3/D1 source artifacts"

jq -n \
    --arg schema_version "frankensqlite.db300.optimization_steering.v1" \
    --arg bead_id "${BEAD_ID}" \
    --arg run_id "${RUN_ID}" \
    --arg trace_id "${TRACE_ID}" \
    --arg generated_at "${GENERATED_AT}" \
    --arg scenario_id "${SCENARIO_ID}" \
    --arg replay_command "${REPLAY_COMMAND}" \
    --arg a5_mvcc_path "${A5_MVCC_ACTIONABLE}" \
    --arg a5_mvcc_summary_path "${A5_MVCC_SUMMARY}" \
    --arg a5_mvcc_sha "${A5_MVCC_SHA}" \
    --arg a5_single_path "${A5_SINGLE_ACTIONABLE}" \
    --arg a5_single_summary_path "${A5_SINGLE_SUMMARY}" \
    --arg a5_single_sha "${A5_SINGLE_SHA}" \
    --arg b3_path "${B3_CLASSIFICATION}" \
    --arg b3_summary_path "${B3_SUMMARY}" \
    --arg b3_sha "${B3_SHA}" \
    --arg d1_path "${D1_ACTIONABLE}" \
    --arg d1_summary_path "${D1_SUMMARY}" \
    --arg d1_sha "${D1_SHA}" \
    --slurpfile a5_mvcc "${A5_MVCC_ACTIONABLE}" \
    --slurpfile a5_single "${A5_SINGLE_ACTIONABLE}" \
    --slurpfile b3 "${B3_CLASSIFICATION}" \
    --slurpfile d1 "${D1_ACTIONABLE}" \
    '
    def a5_sig($doc; $target):
      $doc[0].microarchitectural_signatures[] | select(.target == $target);
    def a5_wall($doc; $component):
      $doc[0].wall_time_components[] | select(.component == $component);
    def b3_path($id):
      $b3[0].open_hot_paths[] | select(.path_id == $id);
    def d1_cat($category):
      $d1[0].actionable_named_categories[] | select(.category == $category);
    def a5_confidence($row):
      if (($row.mixed_or_ambiguous // false) or (($row.confidence_label // "") != "high"))
      then "mixed"
      else "high"
      end;
    def d1_confidence($row):
      if (($row.implication // "") | startswith("Primary ")) then "high" else "mixed" end;
    def b3_confidence($path_id):
      if $path_id == "concurrent_registry_global_mutex" or $path_id == "next_commit_seq_global_sequencer"
      then "mixed"
      else "high"
      end;
    def fmt_pct($value):
      ((($value * 100) | round) / 100);
    def h_confidence($retry_delta; $queue_delta):
      if ($retry_delta >= 300 or $queue_delta >= 300) then "high" else "mixed" end;
    def guidance_class($confidence):
      if $confidence == "high" then "high-confidence" else "mixed-guidance" end;
    def row($steering_id; $track; $hot_cell_id; $mode; $signal; $confidence; $family; $evidence_summary; $rationale; $suggested_beads; $source_artifacts):
      {
        steering_id: $steering_id,
        track: $track,
        hot_cell_id: $hot_cell_id,
        mode: $mode,
        signal: $signal,
        confidence: $confidence,
        guidance_class: guidance_class($confidence),
        next_optimization_family: $family,
        evidence_summary: $evidence_summary,
        rationale: $rationale,
        suggested_beads: $suggested_beads,
        source_artifacts: $source_artifacts
      };
    (a5_sig($a5_mvcc; "retry")) as $mvcc_retry |
    (a5_sig($a5_mvcc; "queueing")) as $mvcc_queue |
    (a5_sig($a5_mvcc; "synchronization")) as $mvcc_sync |
    (a5_sig($a5_mvcc; "durability")) as $mvcc_durability |
    (a5_wall($a5_mvcc; "retry")) as $mvcc_retry_wall |
    (a5_wall($a5_mvcc; "queueing")) as $mvcc_queue_wall |
    (a5_wall($a5_mvcc; "synchronization")) as $mvcc_sync_wall |
    (a5_wall($a5_mvcc; "durability")) as $mvcc_durability_wall |
    (a5_wall($a5_single; "retry")) as $single_retry_wall |
    (a5_wall($a5_single; "queueing")) as $single_queue_wall |
    (a5_wall($a5_single; "synchronization")) as $single_sync_wall |
    (b3_path("inprocess_page_lock_table")) as $b3_lock_table |
    (b3_path("commit_index_sharded_rwlock")) as $b3_commit_index |
    (b3_path("commit_write_mutex_publication_gate")) as $b3_publication_gate |
    (b3_path("concurrent_registry_global_mutex")) as $b3_registry |
    (d1_cat("allocator_pressure")) as $d1_allocator |
    (d1_cat("record_decode")) as $d1_record_decode |
    (d1_cat("copy_movement")) as $d1_copy_movement |
    ([
      row(
        "track-b-lock-release-topology";
        "B";
        "inprocess_page_lock_table";
        "ownership_release";
        "The lock-table release path is an ownership-map problem, so B-lane work should target held-lock release topology instead of generic spinning or seqlock experiments.";
        b3_confidence($b3_lock_table.path_id);
        "Track B: ownership-aware handoff and release topology cleanup";
        "B3 ties the path directly to release_set or held-lock-list work instead of a blind lock-free conversion.";
        $b3_lock_table.primitive_selection_readout;
        ($b3_lock_table.best_follow_on_beads // []);
        [$b3_path, $b3_summary_path]
      ),
      row(
        "track-c-commit-index-publication";
        "C";
        "commit_index_sharded_rwlock";
        "metadata_publication";
        "CommitIndex is the cleanest publication candidate once read-mostly metadata is narrowed to finalize-bounded writes.";
        b3_confidence($b3_commit_index.path_id);
        "Track C: publish-window shrinkage and batch-friendly commit metadata";
        "B3 classified the path as the cleanest metadata publication prototype with bounded finalize writes.";
        $b3_commit_index.primitive_selection_readout;
        ($b3_commit_index.best_follow_on_beads // []);
        [$b3_path, $b3_summary_path]
      ),
      row(
        "track-c-publication-gate-shrink";
        "C";
        "commit_write_mutex_publication_gate";
        "commit_publication";
        "The commit write mutex is a true publish-window problem, not a lock-free metadata problem.";
        b3_confidence($b3_publication_gate.path_id);
        "Track C: shrink the guarded commit window to durable ordering only";
        "B3 explicitly classifies this path as out-of-family for lock-free metadata work and steers it to publish-window shrinkage.";
        $b3_publication_gate.primitive_selection_readout;
        ($b3_publication_gate.best_follow_on_beads // []);
        [$b3_path, $b3_summary_path]
      ),
      row(
        "track-d-allocator-pressure";
        "D";
        "mixed_read_write_c4";
        "mvcc+single_writer";
        "Allocator churn dominates both modes and should move before parser-focused reuse.";
        d1_confidence($d1_allocator);
        "Track D: reusable scratch / arena ownership for decode and materialization";
        ("avg_overhead_pct=\(fmt_pct($d1_allocator.avg_overhead_pct)), max_overhead_pct=\(fmt_pct($d1_allocator.max_overhead_pct))");
        $d1_allocator.implication;
        ($d1_allocator.mapped_beads // []);
        [$d1_path, $d1_summary_path]
      ),
      row(
        "track-d-record-decode";
        "D";
        "mixed_read_write_c4";
        "mvcc+single_writer";
        "Column/ResultRow decode and UTF-8 conversion remain large enough for focused decode-path work.";
        d1_confidence($d1_record_decode);
        "Track D: decode-path scratch buffers and copy reduction";
        ("avg_overhead_pct=\(fmt_pct($d1_record_decode.avg_overhead_pct)), max_overhead_pct=\(fmt_pct($d1_record_decode.max_overhead_pct))");
        $d1_record_decode.implication;
        ($d1_record_decode.mapped_beads // []);
        [$d1_path, $d1_summary_path]
      ),
      row(
        "track-d-copy-movement";
        "D";
        "mixed_read_write_c4";
        "mvcc+single_writer";
        "memmove-heavy record and cell assembly is a first-class named execution cost.";
        d1_confidence($d1_copy_movement);
        "Track D: remove unnecessary record/cell data motion";
        ("avg_overhead_pct=\(fmt_pct($d1_copy_movement.avg_overhead_pct)), max_overhead_pct=\(fmt_pct($d1_copy_movement.max_overhead_pct))");
        $d1_copy_movement.implication;
        ($d1_copy_movement.mapped_beads // []);
        [$d1_path, $d1_summary_path]
      ),
      row(
        "track-e-durability-lane";
        "E";
        "mixed_read_write_c4";
        "fsqlite_mvcc";
        "Durability is an explicit measured lane and should stay separate from generic executor service cost.";
        a5_confidence($mvcc_durability);
        "Track E: many-core architecture must isolate durable ordering from general service work";
        ("durability_wall_share_bps=\($mvcc_durability_wall.wall_share_basis_points), confidence_label=\($mvcc_durability.confidence_label)");
        $mvcc_durability.rationale;
        ((["bd-db300.5.1"] + ($mvcc_durability.mapped_beads // [])) | unique);
        [$a5_mvcc_path, $a5_mvcc_summary_path]
      ),
      row(
        "track-e-registry-structural-split";
        "E";
        "concurrent_registry_global_mutex";
        "metadata_topology";
        "The registry mutex is likely a major hotspot, but the correct next move is a structural split before choosing publication primitives.";
        b3_confidence($b3_registry.path_id);
        "Track E: split published metadata from per-session mutable witness state";
        "B3 marks the registry as structurally important but not ready for a blind primitive swap.";
        $b3_registry.primitive_selection_readout;
        ($b3_registry.best_follow_on_beads // []);
        [$b3_path, $b3_summary_path]
      ),
      row(
        "track-h-single-writer-amplification";
        "H";
        "mixed_read_write_c4";
        "fsqlite_single_writer";
        "Forced single-writer mode shows retry and queue amplification that does not appear on the matched MVCC hot cell.";
        h_confidence(
          ($single_retry_wall.wall_share_basis_points - $mvcc_retry_wall.wall_share_basis_points);
          ($single_queue_wall.wall_share_basis_points - $mvcc_queue_wall.wall_share_basis_points)
        );
        "Track H: isolate and remove single-writer-only wait amplification before using it as a comparison baseline";
        ("retry_delta_bps=\($single_retry_wall.wall_share_basis_points - $mvcc_retry_wall.wall_share_basis_points), queueing_delta_bps=\($single_queue_wall.wall_share_basis_points - $mvcc_queue_wall.wall_share_basis_points), synchronization_delta_bps=\($single_sync_wall.wall_share_basis_points - $mvcc_sync_wall.wall_share_basis_points)");
        "The matched hot-cell pack shows retry and queueing becoming meaningful single-writer-only wall time, which makes honest comparison-mode cleanup a separate lane from shared MVCC work.";
        ["bd-db300.8.1", "bd-db300.8.1.1", "bd-db300.8.1.2"];
        [$a5_mvcc_path, $a5_single_path, $a5_mvcc_summary_path, $a5_single_summary_path]
      )
    ] | sort_by(.track, .steering_id)) as $rows |
    {
      schema_version: $schema_version,
      bead_id: $bead_id,
      run_id: $run_id,
      trace_id: $trace_id,
      generated_at: $generated_at,
      scenario_id: $scenario_id,
      replay_command: $replay_command,
      source_artifacts: [
        {id: "a5_mvcc", path: $a5_mvcc_path, summary_path: $a5_mvcc_summary_path, sha256: $a5_mvcc_sha},
        {id: "a5_single_writer", path: $a5_single_path, summary_path: $a5_single_summary_path, sha256: $a5_single_sha},
        {id: "b3_metadata_classification", path: $b3_path, summary_path: $b3_summary_path, sha256: $b3_sha},
        {id: "d1_execution_hotspots", path: $d1_path, summary_path: $d1_summary_path, sha256: $d1_sha}
      ],
      steering_rows: $rows,
      track_rollup: (
        $rows
        | group_by(.track)
        | map({
            track: .[0].track,
            row_count: length,
            high_confidence_rows: map(select(.confidence == "high") | .steering_id),
            mixed_guidance_rows: map(select(.confidence == "mixed") | .steering_id),
            suggested_beads: (map(.suggested_beads[]) | unique)
          })
      )
    }
    ' > "${STEERING_JSON}"

emit_event "synthesis" "table_written" "pass" "steering table json emitted"

jq -e '
    (.steering_rows | length >= 8) and
    (([.steering_rows[].track] | unique | sort) == ["B", "C", "D", "E", "H"]) and
    (any(.steering_rows[]; .confidence == "high")) and
    (any(.steering_rows[]; .confidence == "mixed")) and
    (all(.track_rollup[]; (.suggested_beads | length) >= 1))
' "${STEERING_JSON}" >/dev/null || fail "validation" "steering table validation failed"

HIGH_ROWS="$(jq -r '
    .steering_rows[]
    | select(.confidence == "high")
    | "| `\(.track)` | `\(.hot_cell_id)` / `\(.mode)` | \(.signal | gsub("\\|"; "\\\\|")) | \(.evidence_summary | gsub("\\|"; "\\\\|")) | `\(.suggested_beads | join("`, `"))` |"
' "${STEERING_JSON}")"

MIXED_ROWS="$(jq -r '
    .steering_rows[]
    | select(.confidence == "mixed")
    | "| `\(.track)` | `\(.hot_cell_id)` / `\(.mode)` | \(.signal | gsub("\\|"; "\\\\|")) | \(.evidence_summary | gsub("\\|"; "\\\\|")) | `\(.suggested_beads | join("`, `"))` |"
' "${STEERING_JSON}")"

TRACK_ROLLUP="$(jq -r '
    def render_list($items):
      if ($items | length) == 0
      then "_none_"
      else ($items | map("`" + . + "`") | join(", "))
      end;
    .track_rollup[]
    | "- `Track \(.track)`: \(.row_count) steering row(s); high-confidence=\(render_list(.high_confidence_rows)); mixed=\(render_list(.mixed_guidance_rows)); next beads=\(render_list(.suggested_beads))"
' "${STEERING_JSON}")"

SOURCE_ROWS="$(jq -r '
    .source_artifacts[]
    | "- `\(.id)`: `\(.path)` (sha256 `\(.sha256)`)"
' "${STEERING_JSON}")"

if [[ -z "${HIGH_ROWS}" ]]; then
    HIGH_ROWS="| _none_ | _none_ | _none_ | _none_ | _none_ |"
fi

if [[ -z "${MIXED_ROWS}" ]]; then
    MIXED_ROWS="| _none_ | _none_ | _none_ | _none_ | _none_ |"
fi

cat > "${SUMMARY_MD}" <<EOF
# ${BEAD_ID} Optimization Steering Table

- run_id: \`${RUN_ID}\`
- trace_id: \`${TRACE_ID}\`
- scenario_id: \`${SCENARIO_ID}\`
- replay_command: \`${REPLAY_COMMAND}\`
- steering_json: \`${STEERING_JSON}\`
- manifest_json: \`${MANIFEST_JSON}\`

## High-Confidence Guidance

| Track | Hot cell / mode | Signal | Evidence | Suggested beads |
| --- | --- | --- | --- | --- |
${HIGH_ROWS}

## Mixed Guidance

| Track | Hot cell / mode | Signal | Evidence | Suggested beads |
| --- | --- | --- | --- | --- |
${MIXED_ROWS}

## Track Rollup

${TRACK_ROLLUP}

## Source Artifacts

${SOURCE_ROWS}
EOF

rg -n "High-Confidence Guidance" "${SUMMARY_MD}" >/dev/null || fail "validation" "summary missing high-confidence section"
rg -n "Mixed Guidance" "${SUMMARY_MD}" >/dev/null || fail "validation" "summary missing mixed-guidance section"
rg -n "manifest_json" "${SUMMARY_MD}" >/dev/null || fail "validation" "summary missing manifest reference"
if rg -n 'high-confidence=``|mixed=``|next beads=``' "${SUMMARY_MD}" >/dev/null; then
    fail "validation" "summary contains blank track-rollup list rendering"
fi

emit_event "summary" "report_written" "pass" "summary markdown emitted"
emit_event "complete" "finish" "pass" "optimization steering synthesis completed"

STEERING_SHA="$(sha256_file "${STEERING_JSON}")"
SUMMARY_SHA="$(sha256_file "${SUMMARY_MD}")"
EVENTS_SHA="$(sha256_file "${EVENTS_JSONL}")"

jq -n \
    --arg schema_version "frankensqlite.artifact_manifest.v1" \
    --arg bead_id "${BEAD_ID}" \
    --arg run_id "${RUN_ID}" \
    --arg trace_id "${TRACE_ID}" \
    --arg generated_at "${GENERATED_AT}" \
    --arg scenario_id "${SCENARIO_ID}" \
    --arg replay_command "${REPLAY_COMMAND}" \
    --arg artifact_dir "${ARTIFACT_DIR}" \
    --arg events_path "${EVENTS_JSONL}" \
    --arg events_sha "${EVENTS_SHA}" \
    --arg steering_path "${STEERING_JSON}" \
    --arg steering_sha "${STEERING_SHA}" \
    --arg summary_path "${SUMMARY_MD}" \
    --arg summary_sha "${SUMMARY_SHA}" \
    --slurpfile steering_table "${STEERING_JSON}" \
    '
    {
      schema_version: $schema_version,
      bead_id: $bead_id,
      run_id: $run_id,
      trace_id: $trace_id,
      generated_at: $generated_at,
      scenario_id: $scenario_id,
      replay_command: $replay_command,
      artifact_dir: $artifact_dir,
      artifact_files: [
        {id: "events", kind: "event_log", path: $events_path, sha256: $events_sha},
        {id: "steering_table", kind: "data", path: $steering_path, sha256: $steering_sha},
        {id: "summary", kind: "report", path: $summary_path, sha256: $summary_sha}
      ],
      source_artifacts: $steering_table[0].source_artifacts
    }
    ' > "${MANIFEST_JSON}"

jq -e '
    (.artifact_files | length == 3) and
    (.source_artifacts | length == 4)
' "${MANIFEST_JSON}" >/dev/null || fail "validation" "manifest validation failed"

echo "steering table: ${STEERING_JSON}"
echo "summary: ${SUMMARY_MD}"
echo "manifest: ${MANIFEST_JSON}"
