#!/usr/bin/env bash
# CI verification gate for bd-2g5.3: Seqlock correctness tests
# Validates the seqlock protocol for rarely-changing metadata.
set -euo pipefail

echo "=== bd-2g5.3: Seqlock Correctness Verification ==="

# Run the harness integration tests
cargo test --package fsqlite-harness --test bd_2g5_3_seqlock_correctness -- --nocapture 2>&1

echo ""
echo "[GATE PASS] bd-2g5.3 seqlock correctness — all tests passed"
