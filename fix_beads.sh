#!/bin/bash
set -e

# I misspelled the ids for zcdn and kdk0 and hdbd. Let's just grab the actual ones that are open.
for id in $(br list --json | jq -r '.[].id' | grep -E 'bd-2uyab|bd-6mqxu|bd-zcdn|bd-kdk0|bd-hdbd|bd-lock|bd-perf'); do
  br update "$id" --status closed > /dev/null || true
done

# List my old beads and just close them using the proper json IDs.
for id in $(br list --json | jq -r '.[].id' | grep -v '1dp9'); do
   # Filter for the ones I created.
   title=$(br show "$id" --json | jq -r '.title')
   if [[ "$title" == *"Alien Graveyard Protocol"* ]] || [[ "$title" == *"ARC Baseline and Golden Isomorphism"* ]] || [[ "$title" == *"Replace ArcCache with S3Fifo"* ]] || [[ "$title" == *"Implement Lock-Free Hit Path in S3-FIFO"* ]] || [[ "$title" == *"Run E2E Verification"* ]]; then
        br update "$id" --status closed > /dev/null || true
   fi
done

# Ensure they are closed.
# Now inject our updated dependencies into the main track 6 epic: bd-1dp9.6.7.10

lock_id=$(br create "Implement Lock-Free Hit Path in S3-FIFO (Alien Graveyard)" -t task -p 2 -d "
# Goal
Maximize concurrent MVCC read throughput by ensuring that cache hits do not require an exclusive lock.

# Context
We are migrating to S3-FIFO under Epic bd-1dp9.6.7.10. To realize the true value of S3-FIFO (SOSP 2023), it cannot use mutable \`&mut self\` references for cache hits.

# Execution Plan
1. Analyze \`fsqlite-pager/src/s3_fifo.rs\`.
2. Ensure that the access bit flag in the S3-FIFO ring buffer entries uses \`AtomicU8\` or \`AtomicBool\`.
3. Refactor the cache \`get()\` method to take a shared (\`&self\`) reference (or \`RwLockReadGuard\`) instead of a mutable/exclusive reference.
4. Verify that flipping the access bit on a cache hit does not trigger an eviction cycle on the critical path (eviction should be decoupled or lazy).
5. Ensure unit tests cover concurrent read-heavy workloads where multiple threads hit the same hot page.
" --silent)

perf_id=$(br create "Isomorphism Proof: S3-FIFO vs ARC Baseline (Extreme Optimization)" -t task -p 1 -d "
# Goal
Execute the Extreme Software Optimization non-regression constraints for the cache swap.

# Execution Plan
1. Run \`hyperfine --warmup 3 --runs 10 'cargo bench --manifest-path crates/fsqlite-pager/Cargo.toml test_e2e_bd_2zoa_arc_performance'\` on the *existing* ARC codebase to capture baseline.
2. After integrating S3-FIFO (bd-1dp9.6.7.10.2), rerun the benchmarks.
3. Assert that hit rates are mathematically comparable.
4. Assert that throughput / P99 latency is definitively improved due to the lock-free hit path.
5. Publish Golden Checksums and Isomorphism proof block into the project PR/Bead.
" --silent)

br dep add "$lock_id" "bd-1dp9.6.7.10.2" > /dev/null
br dep add "$perf_id" "bd-1dp9.6.7.10.3" > /dev/null

echo "Bead graph synchronized with existing backlog."
