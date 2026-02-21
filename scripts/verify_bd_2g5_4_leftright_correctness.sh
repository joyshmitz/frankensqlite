#!/usr/bin/env bash
# CI verification gate for bd-2g5.4: Left-Right correctness tests
# Validates wait-free reads for metadata via dual-instance swapping.
set -euo pipefail

echo "=== bd-2g5.4: Left-Right Correctness Verification ==="

# Run the harness integration tests
cargo test --package fsqlite-harness --test bd_2g5_4_leftright_correctness -- --nocapture 2>&1

echo ""
echo "[GATE PASS] bd-2g5.4 left-right correctness — all tests passed"
