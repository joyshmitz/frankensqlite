#!/usr/bin/env bash
# CI verification gate for bd-2g5.2: RCU/QSBR correctness tests
# Validates zero-overhead metadata reads via QSBR grace periods.
set -euo pipefail

echo "=== bd-2g5.2: RCU/QSBR Correctness Verification ==="

# Run the harness integration tests
cargo test --package fsqlite-harness --test bd_2g5_2_rcu_correctness -- --nocapture 2>&1

echo ""
echo "[GATE PASS] bd-2g5.2 RCU/QSBR correctness — all tests passed"
