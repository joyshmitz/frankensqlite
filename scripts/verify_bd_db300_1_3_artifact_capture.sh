#!/usr/bin/env bash
# verify_bd_db300_1_3_artifact_capture.sh — one-command A3 evidence capture
#
# This wraps the reusable inline hot-path campaign so bd-db300.1.3 has a single
# reproducible entrypoint instead of requiring operators to reconstruct the
# child-bead workflow manually. Defaults target one representative hot cell
# (`frankensqlite_beads`, mixed_read_write, c4) while preserving env overrides
# for broader fixture or mode coverage.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEAD_ID="bd-db300.1.3"
SCRIPT_ENTRYPOINT="scripts/verify_bd_db300_1_3_artifact_capture.sh"
SOURCE_GOLDEN_DIR_DEFAULT="${WORKSPACE_ROOT}/sample_sqlite_db_files/working/beads_bench_20260310/golden"

export BEAD_ID
export SCRIPT_ENTRYPOINT
export GOLDEN_DIR="${GOLDEN_DIR:-${SOURCE_GOLDEN_DIR_DEFAULT}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_ROOT}/artifacts/perf/${BEAD_ID}/representative_hot_cell}"
export FIXTURE_IDS="${FIXTURE_IDS:-frankensqlite_beads}"
export MODE_IDS="${MODE_IDS:-mvcc,single_writer}"
export WORKLOAD_ID="${WORKLOAD_ID:-mixed_read_write}"
export CONCURRENCY="${CONCURRENCY:-4}"
export SEED="${SEED:-42}"
export SCALE="${SCALE:-50}"
export CARGO_PROFILE="${CARGO_PROFILE:-release}"
export RCH_TARGET_DIR="${RCH_TARGET_DIR:-/tmp/rch_target_bd_db300_1_3}"

bash "${WORKSPACE_ROOT}/scripts/verify_bd_db300_4_1_hot_path_profiles.sh"
