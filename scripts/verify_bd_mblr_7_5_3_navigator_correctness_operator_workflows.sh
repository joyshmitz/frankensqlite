#!/usr/bin/env bash
# CI verification gate for bd-mblr.7.5.3: Navigator Correctness Tests and
# Operator Workflows
# Validates: timeline correctness (sorting, completeness, severity classification),
# correlation accuracy (component/invariant aggregation), filter correctness
# (commit, severity, component, seed), replay command generation, operator
# triage workflows (failure-to-root-cause, divergence investigation), text
# report rendering, workflow report persistence, determinism, edge cases,
# conformance.
set -euo pipefail

echo "=== bd-mblr.7.5.3: Navigator Correctness Tests and Operator Workflows Verification ==="

# Run the harness integration tests
cargo test --package fsqlite-harness --test bd_mblr_7_5_3_navigator_correctness_operator_workflows -- --nocapture 2>&1

echo ""
echo "[GATE PASS] bd-mblr.7.5.3 Navigator Correctness Tests — all tests passed"
