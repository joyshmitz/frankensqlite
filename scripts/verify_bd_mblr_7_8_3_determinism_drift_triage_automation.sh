#!/usr/bin/env bash
# CI verification gate for bd-mblr.7.8.3: Determinism Drift Triage Automation
# Validates: drift severity classification (DivergenceClass, DriftClassification),
# actionable findings (DriftFailure with repro commands, subsystem routing),
# timing anomaly detection, watchdog report triage, drift report construction,
# JSON round-trip persistence, deterministic report generation, runner/matrix
# validation, session coverage, conformance.
set -euo pipefail

echo "=== bd-mblr.7.8.3: Determinism Drift Triage Automation Verification ==="

# Run the harness integration tests
cargo test --package fsqlite-harness --test bd_mblr_7_8_3_determinism_drift_triage_automation -- --nocapture 2>&1

echo ""
echo "[GATE PASS] bd-mblr.7.8.3 Determinism Drift Triage Automation — all tests passed"
