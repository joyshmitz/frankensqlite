#!/usr/bin/env bash
# bd-688.3: Latch-free MVCC Version Chain (Hekaton-style)
#
# Verification script for CI evidence collection.
set -euo pipefail

BEAD="bd-688.3"
echo "=== ${BEAD}: Latch-free MVCC Version Chain ==="

if ! command -v rch >/dev/null 2>&1; then
    echo "ERROR: rch is required for ${BEAD} verification but was not found in PATH."
    exit 1
fi

run_remote() {
    rch exec -- "$@"
}

# 1. Unit tests in fsqlite-mvcc (invariants + gc modules).
echo "--- Step 1: fsqlite-mvcc unit tests ---"
run_remote cargo test -p fsqlite-mvcc -- invariants::tests gc::tests --nocapture 2>&1
echo "PASS: fsqlite-mvcc unit tests"

# 2. Harness integration tests for this bead.
echo "--- Step 2: harness integration tests ---"
run_remote cargo test --test bd_688_3_latchfree_version_chain -- --nocapture 2>&1
echo "PASS: harness integration tests"

# 3. Clippy (workspace-wide, deny warnings).
echo "--- Step 3: clippy ---"
run_remote cargo clippy -p fsqlite-mvcc --all-targets -- -D warnings 2>&1
echo "PASS: clippy clean"

echo "=== ${BEAD}: ALL CHECKS PASSED ==="
