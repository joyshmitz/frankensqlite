#!/usr/bin/env bash
# CI verification gate for bd-2qr3a.5: golden output checksums for
# Leapfrog Triejoin vs pairwise hash-join (Q2/Q7/Q9-like scenarios).
set -euo pipefail

echo "=== bd-2qr3a.5: Leapfrog golden checksum verification ==="

rch exec -- cargo test --package fsqlite-harness --test bd_2qr3a_5_leapfrog_golden_checksums -- --nocapture 2>&1

echo ""
echo "[GATE PASS] bd-2qr3a.5 Leapfrog golden checksums — all tests passed"
