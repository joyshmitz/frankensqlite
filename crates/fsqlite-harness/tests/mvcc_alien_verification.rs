use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use asupersync::lab::{DporExplorer, ExplorerConfig, LabConfig, LabRuntime};
use asupersync::runtime::yield_now;
use parking_lot::Mutex;
use proptest::prelude::*;

use fsqlite_harness::tla::{MvccStateSnapshot, MvccTlaExporter, TlaValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TxnOutcome {
    Committed,
    Aborted,
}

#[derive(Debug, Default)]
struct ModelState {
    // Global commit clock.
    commit_seq_hi: u64,
    // commit_index[page] = latest commit_seq that wrote page.
    commit_index: BTreeMap<u32, u64>,
    // Recently committed transactions' read evidence (committed pivots are possible).
    committed_readers: Vec<CommittedReaderEntry>,
    // Trace snapshots for TLA+ export/debug.
    trace: Vec<MvccStateSnapshot>,
}

#[derive(Debug, Clone)]
struct CommittedReaderEntry {
    txn_id: u64,
    begin_seq: u64,
    commit_seq: u64,
    has_in_rw: bool,
    read_pages: Vec<u32>,
}

impl ModelState {
    fn snapshot(&mut self, label: impl Into<String>, bucket: &HotBucket) {
        let mut vars = BTreeMap::new();
        vars.insert(
            "commit_seq_hi".to_string(),
            TlaValue::Nat(self.commit_seq_hi),
        );
        vars.insert(
            "hot_epoch".to_string(),
            TlaValue::Nat(u64::from(bucket.hot_epoch())),
        );
        vars.insert(
            "bucket".to_string(),
            TlaValue::Record(bucket.to_tla_record()),
        );
        vars.insert(
            "committed_readers".to_string(),
            TlaValue::Seq(
                self.committed_readers
                    .iter()
                    .map(|e| {
                        let mut r = BTreeMap::new();
                        r.insert("txn_id".to_string(), TlaValue::Nat(e.txn_id));
                        r.insert("begin_seq".to_string(), TlaValue::Nat(e.begin_seq));
                        r.insert("commit_seq".to_string(), TlaValue::Nat(e.commit_seq));
                        r.insert("has_in_rw".to_string(), TlaValue::Bool(e.has_in_rw));
                        r.insert(
                            "read_pages".to_string(),
                            TlaValue::Seq(
                                e.read_pages
                                    .iter()
                                    .copied()
                                    .map(|p| TlaValue::Nat(u64::from(p)))
                                    .collect(),
                            ),
                        );
                        TlaValue::Record(r)
                    })
                    .collect(),
            ),
        );
        self.trace.push(MvccStateSnapshot {
            label: label.into(),
            vars,
        });
    }
}

/// A minimal, SHM-shaped hot witness bucket with double-buffered epochs and a spinlock.
///
/// This is a *model* used for DPOR/injection testing. It is intentionally small:
/// one bucket entry, readers-only, 128 slots (two u64 words).
struct HotBucket {
    hot_epoch: AtomicU32,

    epoch_lock: AtomicU32,

    epoch_a: AtomicU32,
    readers_a: [AtomicU64; 2],

    epoch_b: AtomicU32,
    readers_b: [AtomicU64; 2],
}

impl HotBucket {
    fn new(initial_epoch: u32) -> Self {
        Self {
            hot_epoch: AtomicU32::new(initial_epoch),
            epoch_lock: AtomicU32::new(0),
            epoch_a: AtomicU32::new(0),
            readers_a: [AtomicU64::new(0), AtomicU64::new(0)],
            epoch_b: AtomicU32::new(0),
            readers_b: [AtomicU64::new(0), AtomicU64::new(0)],
        }
    }

    fn hot_epoch(&self) -> u32 {
        self.hot_epoch.load(Ordering::Acquire)
    }

    fn to_tla_record(&self) -> BTreeMap<String, TlaValue> {
        let mut r = BTreeMap::new();
        r.insert(
            "epoch_a".to_string(),
            TlaValue::Nat(u64::from(self.epoch_a())),
        );
        r.insert(
            "epoch_b".to_string(),
            TlaValue::Nat(u64::from(self.epoch_b())),
        );
        r.insert(
            "readers_a".to_string(),
            TlaValue::Set(
                self.readers_bits(self.epoch_a())
                    .into_iter()
                    .map(TlaValue::Nat)
                    .collect(),
            ),
        );
        r.insert(
            "readers_b".to_string(),
            TlaValue::Set(
                self.readers_bits(self.epoch_b())
                    .into_iter()
                    .map(TlaValue::Nat)
                    .collect(),
            ),
        );
        r
    }

    fn epoch_a(&self) -> u32 {
        self.epoch_a.load(Ordering::Acquire)
    }

    fn epoch_b(&self) -> u32 {
        self.epoch_b.load(Ordering::Acquire)
    }

    fn readers_bits(&self, e: u32) -> Vec<u64> {
        let mut out = Vec::new();
        let (epoch, readers) = self.match_epoch(e);
        if !epoch {
            return out;
        }
        for (word_idx, w) in readers.iter().enumerate() {
            let mut bits = w.load(Ordering::Acquire);
            while bits != 0 {
                let lsb = bits & bits.wrapping_neg();
                let bit = lsb.trailing_zeros();
                let Ok(word) = u64::try_from(word_idx) else {
                    continue;
                };
                out.push(word * 64 + u64::from(bit));
                bits ^= lsb;
            }
        }
        out
    }

    fn match_epoch(&self, target: u32) -> (bool, &[AtomicU64; 2]) {
        if self.epoch_a.load(Ordering::Acquire) == target {
            return (true, &self.readers_a);
        }
        if self.epoch_b.load(Ordering::Acquire) == target {
            return (true, &self.readers_b);
        }
        (false, &self.readers_a)
    }

    fn set_reader_bit(&self, target_epoch: u32, slot_id: u32) {
        let (ok, readers) = self.match_epoch(target_epoch);
        assert!(ok, "epoch buffer must be installed before setting bit");
        let Ok(word) = usize::try_from(slot_id / 64) else {
            return;
        };
        if word >= readers.len() {
            return;
        }
        let bit = slot_id % 64;
        let mask = 1_u64 << bit;
        readers[word].fetch_or(mask, Ordering::Relaxed);
    }

    /// Install `target_epoch` into a stale buffer (clear-then-publish) and return which buffer.
    fn install_epoch(&self, target_epoch: u32, cur: u32, prev: u32) -> Option<Buffer> {
        // Acquire spinlock.
        while self
            .epoch_lock
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }

        // Re-check now that we hold the lock.
        if self.epoch_a.load(Ordering::Acquire) == target_epoch {
            self.epoch_lock.store(0, Ordering::Release);
            return Some(Buffer::A);
        }
        if self.epoch_b.load(Ordering::Acquire) == target_epoch {
            self.epoch_lock.store(0, Ordering::Release);
            return Some(Buffer::B);
        }

        let a_epoch = self.epoch_a.load(Ordering::Acquire);
        let b_epoch = self.epoch_b.load(Ordering::Acquire);

        let chosen = if a_epoch != cur && a_epoch != prev {
            Buffer::A
        } else if b_epoch != cur && b_epoch != prev {
            Buffer::B
        } else {
            // With 2 buffers and at most {cur, prev} live, this should be unreachable.
            self.epoch_lock.store(0, Ordering::Release);
            return None;
        };

        match chosen {
            Buffer::A => {
                for w in &self.readers_a {
                    w.store(0, Ordering::Relaxed);
                }
                self.epoch_a.store(target_epoch, Ordering::Release);
            }
            Buffer::B => {
                for w in &self.readers_b {
                    w.store(0, Ordering::Relaxed);
                }
                self.epoch_b.store(target_epoch, Ordering::Release);
            }
        }

        self.epoch_lock.store(0, Ordering::Release);
        Some(chosen)
    }

    async fn register_read(&self, slot_id: u32, target_epoch: u32) {
        let cur = self.hot_epoch();
        let prev = cur.wrapping_sub(1);

        // Fast path: buffer already tagged.
        if self.epoch_a.load(Ordering::Acquire) == target_epoch {
            self.set_reader_bit(target_epoch, slot_id);
            return;
        }
        if self.epoch_b.load(Ordering::Acquire) == target_epoch {
            self.set_reader_bit(target_epoch, slot_id);
            return;
        }

        // Slow path: install + clear is serialized. Setting the bit is outside the lock.
        let installed = self.install_epoch(target_epoch, cur, prev);
        assert!(installed.is_some(), "epoch install failed unexpectedly");

        // Intentional yield point: another task may observe the installed epoch and set its bit.
        yield_now().await;

        self.set_reader_bit(target_epoch, slot_id);
    }

    fn readers_for_epoch(&self, target_epoch: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let (ok, readers) = self.match_epoch(target_epoch);
        if !ok {
            return out;
        }
        for (word_idx, w) in readers.iter().enumerate() {
            let mut bits = w.load(Ordering::Acquire);
            while bits != 0 {
                let lsb = bits & bits.wrapping_neg();
                let bit = lsb.trailing_zeros();
                let Ok(word) = u32::try_from(word_idx) else {
                    continue;
                };
                out.push(word * 64 + bit);
                bits ^= lsb;
            }
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Buffer {
    A,
    B,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum TraceAction {
    Begin { txn: u8 },
    Read { txn: u8, page: u32 },
    Write { txn: u8, page: u32 },
    Commit { txn: u8, write_pages: Vec<u32> },
}

fn actions_independent(lhs: &TraceAction, rhs: &TraceAction) -> bool {
    if lhs == rhs {
        return false;
    }

    match (lhs, rhs) {
        (TraceAction::Begin { .. }, _) | (_, TraceAction::Begin { .. }) => false,
        (TraceAction::Commit { .. }, TraceAction::Commit { .. }) => false,
        (TraceAction::Read { .. }, TraceAction::Read { .. }) => true,
        (TraceAction::Read { page: l, .. }, TraceAction::Write { page: r, .. })
        | (TraceAction::Write { page: l, .. }, TraceAction::Read { page: r, .. }) => l != r,
        (TraceAction::Write { page: l, .. }, TraceAction::Write { page: r, .. }) => l != r,
        (TraceAction::Read { page, .. }, TraceAction::Commit { write_pages, .. })
        | (TraceAction::Commit { write_pages, .. }, TraceAction::Read { page, .. }) => {
            !write_pages.contains(page)
        }
        (TraceAction::Write { page, .. }, TraceAction::Commit { write_pages, .. })
        | (TraceAction::Commit { write_pages, .. }, TraceAction::Write { page, .. }) => {
            !write_pages.contains(page)
        }
    }
}

fn foata_normal_form(word: &[TraceAction]) -> Vec<Vec<TraceAction>> {
    let n = word.len();
    let mut predecessor_counts = vec![0_usize; n];
    let mut successors = vec![Vec::<usize>::new(); n];

    for i in 0..n {
        for j in (i + 1)..n {
            if !actions_independent(&word[i], &word[j]) {
                predecessor_counts[j] += 1;
                successors[i].push(j);
            }
        }
    }

    let mut remaining = BTreeSet::from_iter(0..n);
    let mut foata = Vec::new();

    while !remaining.is_empty() {
        let mut layer_indices: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|idx| predecessor_counts[*idx] == 0)
            .collect();

        assert!(
            !layer_indices.is_empty(),
            "dependency graph must remain acyclic for linearized traces"
        );

        layer_indices.sort_by(|a, b| word[*a].cmp(&word[*b]).then_with(|| a.cmp(b)));

        let layer: Vec<TraceAction> = layer_indices.iter().map(|idx| word[*idx].clone()).collect();
        for idx in &layer_indices {
            remaining.remove(idx);
        }
        for idx in &layer_indices {
            for succ in &successors[*idx] {
                predecessor_counts[*succ] -= 1;
            }
        }
        foata.push(layer);
    }

    foata
}

fn enumerate_interleavings(per_txn: &[Vec<TraceAction>]) -> Vec<Vec<TraceAction>> {
    fn recurse(
        per_txn: &[Vec<TraceAction>],
        positions: &mut [usize],
        current: &mut Vec<TraceAction>,
        out: &mut Vec<Vec<TraceAction>>,
    ) {
        if per_txn
            .iter()
            .enumerate()
            .all(|(idx, actions)| positions[idx] == actions.len())
        {
            out.push(current.clone());
            return;
        }

        for (txn_idx, actions) in per_txn.iter().enumerate() {
            let pos = positions[txn_idx];
            if pos >= actions.len() {
                continue;
            }

            positions[txn_idx] += 1;
            current.push(actions[pos].clone());
            recurse(per_txn, positions, current, out);
            current.pop();
            positions[txn_idx] -= 1;
        }
    }

    let mut positions = vec![0_usize; per_txn.len()];
    let total_len: usize = per_txn.iter().map(Vec::len).sum();
    let mut current = Vec::with_capacity(total_len);
    let mut out = Vec::new();
    recurse(per_txn, &mut positions, &mut current, &mut out);
    out
}

fn trace_classes(
    per_txn: &[Vec<TraceAction>],
) -> BTreeMap<Vec<Vec<TraceAction>>, Vec<Vec<TraceAction>>> {
    let mut classes: BTreeMap<Vec<Vec<TraceAction>>, Vec<Vec<TraceAction>>> = BTreeMap::new();
    for interleaving in enumerate_interleavings(per_txn) {
        let key = foata_normal_form(&interleaving);
        classes.entry(key).or_default().push(interleaving);
    }
    classes
}

#[derive(Debug, Default)]
struct TraceTxnState {
    begin_seq: u64,
    read_pages: Vec<u32>,
    write_pages: Vec<u32>,
}

#[derive(Debug, Clone)]
struct CommittedTraceTxn {
    begin_seq: u64,
    commit_seq: u64,
    read_pages: Vec<u32>,
    write_pages: Vec<u32>,
}

#[derive(Debug, Default)]
struct TraceSimulation {
    commit_seq_hi: u64,
    active: BTreeMap<u8, TraceTxnState>,
    committed: Vec<CommittedTraceTxn>,
    aborted: BTreeSet<u8>,
}

impl TraceSimulation {
    fn apply(&mut self, action: &TraceAction) {
        match action {
            TraceAction::Begin { txn } => {
                self.active.insert(
                    *txn,
                    TraceTxnState {
                        begin_seq: self.commit_seq_hi,
                        ..TraceTxnState::default()
                    },
                );
            }
            TraceAction::Read { txn, page } => {
                if let Some(state) = self.active.get_mut(txn) {
                    state.read_pages.push(*page);
                }
            }
            TraceAction::Write { txn, page } => {
                if let Some(state) = self.active.get_mut(txn) {
                    state.write_pages.push(*page);
                }
            }
            TraceAction::Commit { txn, .. } => {
                let Some(state) = self.active.remove(txn) else {
                    return;
                };

                let has_out_rw = self.committed.iter().any(|committed| {
                    committed.commit_seq > state.begin_seq
                        && pages_overlap(&state.read_pages, &committed.write_pages)
                });
                let has_in_rw = self.committed.iter().any(|committed| {
                    committed.commit_seq > state.begin_seq
                        && pages_overlap(&state.write_pages, &committed.read_pages)
                });

                if has_in_rw && has_out_rw {
                    self.aborted.insert(*txn);
                    return;
                }

                self.commit_seq_hi += 1;
                self.committed.push(CommittedTraceTxn {
                    begin_seq: state.begin_seq,
                    commit_seq: self.commit_seq_hi,
                    read_pages: state.read_pages,
                    write_pages: state.write_pages,
                });
            }
        }
    }
}

#[test]
fn tla_exporter_emits_a_module() {
    let mut vars = BTreeMap::new();
    vars.insert("x".to_string(), TlaValue::Nat(1));
    let snapshots = vec![
        MvccStateSnapshot {
            label: "init".to_string(),
            vars: vars.clone(),
        },
        MvccStateSnapshot {
            label: "step".to_string(),
            vars,
        },
    ];
    let exporter = MvccTlaExporter::from_snapshots(snapshots);
    let m = exporter.export_behavior("MvccTest");
    assert!(m.source.contains("---- MODULE MvccTest ----"));
    assert!(m.source.contains("States =="));
    assert!(m.source.contains("Spec =="));
}

#[test]
fn dpor_hot_witness_epoch_install_has_no_lost_bits() {
    let mut explorer = DporExplorer::new(ExplorerConfig::new(0, 64).max_steps(50_000));

    let report = explorer.explore(|rt| {
        let root = rt
            .state
            .create_root_region(asupersync::types::Budget::INFINITE);

        let bucket = Arc::new(HotBucket::new(10));
        let state = Arc::new(Mutex::new(ModelState {
            commit_seq_hi: 10,
            ..ModelState::default()
        }));

        {
            let mut s = state.lock();
            s.snapshot("init", &bucket);
        }

        // Two concurrent registrations into the same epoch.
        let bucket_a = Arc::clone(&bucket);
        let state_a = Arc::clone(&state);
        let (t_a, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                bucket_a.register_read(1, 10).await;
                let mut s = state_a.lock();
                s.snapshot("after A", &bucket_a);
            })
            .expect("spawn A");

        let bucket_b = Arc::clone(&bucket);
        let state_b = Arc::clone(&state);
        let (t_b, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                bucket_b.register_read(2, 10).await;
                let mut s = state_b.lock();
                s.snapshot("after B", &bucket_b);
            })
            .expect("spawn B");

        let mut sched = rt
            .scheduler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        sched.schedule(t_a, 0);
        sched.schedule(t_b, 0);
        drop(sched);

        rt.run_until_quiescent();

        // Invariant: both bits are present in the installed epoch buffer.
        let bits = bucket.readers_for_epoch(10);
        assert!(bits.contains(&1), "slot 1 missing");
        assert!(bits.contains(&2), "slot 2 missing");

        // Export trace to a behavior module (debug artifact; not written to disk here).
        let exporter = {
            let s = state.lock();
            MvccTlaExporter::from_snapshots(s.trace.clone())
        };
        let m = exporter.export_behavior("HotWitnessInstall");
        assert!(m.source.contains("States =="));
    });

    assert!(
        report.violations.is_empty(),
        "DPOR exploration found violations: {:#?}",
        report.violations
    );
}

#[test]
fn dpor_outgoing_edges_cover_committed_and_freed_writers() {
    let mut explorer = DporExplorer::new(ExplorerConfig::new(7, 64).max_steps(100_000));

    let report = explorer.explore(|rt| {
        let root = rt
            .state
            .create_root_region(asupersync::types::Budget::INFINITE);

        let bucket = Arc::new(HotBucket::new(100));
        let state = Arc::new(Mutex::new(ModelState {
            commit_seq_hi: 100,
            ..ModelState::default()
        }));

        // Reader transaction T.
        let state_t = Arc::clone(&state);
        let bucket_t = Arc::clone(&bucket);
        let (t_t, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                let begin_seq = {
                    let s = state_t.lock();
                    s.commit_seq_hi
                };

                // Simulate reading page 1 under the snapshot.
                yield_now().await;
                let read_pages = vec![1_u32];

                // Commit-time outgoing edge discovery:
                yield_now().await;
                let (out_edges, w_commit_seq_at_check) = {
                    let s = state_t.lock();
                    let out_edges =
                        discover_outgoing_edges(begin_seq, &read_pages, &s.commit_index);
                    let w_commit_seq = s.commit_index.get(&1).copied();
                    drop(s);
                    (out_edges, w_commit_seq)
                };
                state_t.lock().snapshot("T commit", &bucket_t);

                // If a writer committed after our begin and touched page 1 before we committed,
                // out_edges must be non-empty (no false negatives).
                if let Some(w_commit_seq) = w_commit_seq_at_check {
                    if w_commit_seq > begin_seq {
                        assert!(
                            out_edges.contains(&w_commit_seq),
                            "missing outgoing edge to committed writer"
                        );
                    }
                }
            })
            .expect("spawn T");

        // Writer W commits page 1 and then "frees its slot" (becomes invisible to hot plane).
        let state_w = Arc::clone(&state);
        let bucket_w = Arc::clone(&bucket);
        let (t_w, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                yield_now().await;
                let commit_seq = {
                    let mut s = state_w.lock();
                    s.commit_seq_hi += 1;
                    let commit_seq = s.commit_seq_hi;
                    s.commit_index.insert(1, commit_seq);
                    commit_seq
                };
                state_w
                    .lock()
                    .snapshot(format!("W committed {commit_seq}"), &bucket_w);
            })
            .expect("spawn W");

        {
            let mut sched = rt.scheduler.lock().expect("scheduler lock");
            sched.schedule(t_t, 0);
            sched.schedule(t_w, 0);
        }
        rt.run_until_quiescent();
    });

    assert!(
        report.violations.is_empty(),
        "DPOR exploration found violations: {:#?}",
        report.violations
    );
}

#[test]
fn chaos_cancel_does_not_leak_hot_witness_epoch_lock() {
    // This is not a correctness proof: it is a liveness/safety guardrail.
    // Under cancellation injection we should never leave a per-bucket lock held.
    for seed in 0_u64..16 {
        let mut rt = LabRuntime::new(
            LabConfig::new(seed)
                .with_light_chaos()
                .worker_count(2)
                .max_steps(50_000),
        );
        let root = rt
            .state
            .create_root_region(asupersync::types::Budget::INFINITE);

        let bucket = Arc::new(HotBucket::new(10));

        let bucket_a = Arc::clone(&bucket);
        let (t_a, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                bucket_a.register_read(1, 10).await;
            })
            .expect("spawn A");

        let bucket_b = Arc::clone(&bucket);
        let (t_b, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                bucket_b.register_read(2, 10).await;
            })
            .expect("spawn B");

        let mut sched = rt
            .scheduler
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        sched.schedule(t_a, 0);
        sched.schedule(t_b, 0);
        drop(sched);

        rt.run_until_quiescent();

        assert_eq!(
            bucket.epoch_lock.load(Ordering::Acquire),
            0,
            "epoch lock leaked under chaos cancellation (seed={seed})"
        );
    }
}

#[test]
fn dpor_incoming_edges_cover_committed_pivots_via_committed_readers_index() {
    let mut explorer = DporExplorer::new(ExplorerConfig::new(99, 64).max_steps(150_000));

    let report = explorer.explore(|rt| {
        let root = rt
            .state
            .create_root_region(asupersync::types::Budget::INFINITE);

        let bucket = Arc::new(HotBucket::new(100));
        let state = Arc::new(Mutex::new(ModelState {
            commit_seq_hi: 100,
            ..ModelState::default()
        }));

        // Transaction R: reads page 1, commits, and is already known to have has_in_rw = true
        // (i.e., it has an incoming rw edge from some earlier txn X).
        let state_r = Arc::clone(&state);
        let bucket_r = Arc::clone(&bucket);
        let (t_r, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                let begin_seq = { state_r.lock().commit_seq_hi };
                let read_pages = vec![1_u32];

                yield_now().await;

                let commit_seq = {
                    let mut s = state_r.lock();
                    s.commit_seq_hi += 1;
                    s.commit_seq_hi
                };
                let entry = CommittedReaderEntry {
                    txn_id: 1,
                    begin_seq,
                    commit_seq,
                    has_in_rw: true,
                    read_pages,
                };
                let mut s = state_r.lock();
                s.committed_readers.push(entry);
                s.snapshot(format!("R committed {commit_seq}"), &bucket_r);
                drop(s);
            })
            .expect("spawn R");

        // Transaction T: begins before R commits, writes page 1, then attempts to commit.
        let state_t = Arc::clone(&state);
        let bucket_t = Arc::clone(&bucket);
        let (t_t, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                let begin_seq = { state_t.lock().commit_seq_hi };

                // Wait for R to possibly commit.
                yield_now().await;

                // Simulate a write to page 1.
                let write_pages = vec![1_u32];

                // Commit-time incoming edge discovery:
                yield_now().await;
                let (in_edges, r_committed_after_snapshot) = {
                    let s = state_t.lock();
                    let in_edges =
                        discover_incoming_edges(begin_seq, &write_pages, &s.committed_readers);
                    let r_after = s
                        .committed_readers
                        .iter()
                        .find(|e| e.txn_id == 1)
                        .is_some_and(|r| r.commit_seq > begin_seq);
                    drop(s);
                    (in_edges, r_after)
                };

                // Apply the committed-pivot T3 rule: if any committed edge source has has_in_rw,
                // this commit must abort.
                let mut outcome = TxnOutcome::Committed;
                for c in &in_edges {
                    if c.has_in_rw {
                        outcome = TxnOutcome::Aborted;
                    }
                }

                state_t
                    .lock()
                    .snapshot(format!("T outcome {outcome:?}"), &bucket_t);

                // If R committed after our snapshot and read a page we wrote, T must abort
                // because R is a committed pivot (X -rw-> R -rw-> T).
                if r_committed_after_snapshot {
                    assert!(
                        in_edges.iter().any(|c| c.txn_id == 1),
                        "missing incoming edge source txn_id=1"
                    );
                    assert_eq!(
                        outcome,
                        TxnOutcome::Aborted,
                        "committed pivot must force abort"
                    );
                }
            })
            .expect("spawn T");

        {
            let mut sched = rt.scheduler.lock().expect("scheduler lock");
            sched.schedule(t_r, 0);
            sched.schedule(t_t, 0);
        }
        rt.run_until_quiescent();
    });

    assert!(
        report.violations.is_empty(),
        "DPOR exploration found violations: {:#?}",
        report.violations
    );
}

fn sample_two_txn_three_ops() -> Vec<Vec<TraceAction>> {
    vec![
        vec![
            TraceAction::Read { txn: 1, page: 1 },
            TraceAction::Write { txn: 1, page: 2 },
            TraceAction::Commit {
                txn: 1,
                write_pages: vec![2],
            },
        ],
        vec![
            TraceAction::Read { txn: 2, page: 3 },
            TraceAction::Write { txn: 2, page: 4 },
            TraceAction::Commit {
                txn: 2,
                write_pages: vec![4],
            },
        ],
    ]
}

#[test]
fn test_read_read_different_pages_independent() {
    let lhs = TraceAction::Read { txn: 1, page: 10 };
    let rhs = TraceAction::Read { txn: 2, page: 20 };
    assert!(actions_independent(&lhs, &rhs));
}

#[test]
fn test_read_read_same_page_independent() {
    let lhs = TraceAction::Read { txn: 1, page: 10 };
    let rhs = TraceAction::Read { txn: 2, page: 10 };
    assert!(actions_independent(&lhs, &rhs));
}

#[test]
fn test_read_write_same_page_dependent() {
    let lhs = TraceAction::Read { txn: 1, page: 10 };
    let rhs = TraceAction::Write { txn: 2, page: 10 };
    assert!(!actions_independent(&lhs, &rhs));
}

#[test]
fn test_write_write_different_pages_independent() {
    let lhs = TraceAction::Write { txn: 1, page: 10 };
    let rhs = TraceAction::Write { txn: 2, page: 20 };
    assert!(actions_independent(&lhs, &rhs));
}

#[test]
fn test_write_write_same_page_dependent() {
    let lhs = TraceAction::Write { txn: 1, page: 10 };
    let rhs = TraceAction::Write { txn: 2, page: 10 };
    assert!(!actions_independent(&lhs, &rhs));
}

#[test]
fn test_commit_commit_dependent() {
    let lhs = TraceAction::Commit {
        txn: 1,
        write_pages: vec![10],
    };
    let rhs = TraceAction::Commit {
        txn: 2,
        write_pages: vec![20],
    };
    assert!(!actions_independent(&lhs, &rhs));
}

#[test]
fn test_begin_begin_dependent() {
    let lhs = TraceAction::Begin { txn: 1 };
    let rhs = TraceAction::Begin { txn: 2 };
    assert!(!actions_independent(&lhs, &rhs));
}

#[test]
fn test_read_commit_dependent_if_overlapping() {
    let read_overlapping = TraceAction::Read { txn: 1, page: 10 };
    let read_non_overlapping = TraceAction::Read { txn: 1, page: 999 };
    let commit = TraceAction::Commit {
        txn: 2,
        write_pages: vec![10, 20],
    };

    assert!(!actions_independent(&read_overlapping, &commit));
    assert!(actions_independent(&read_non_overlapping, &commit));
}

#[test]
fn test_foata_2txn_3ops_each() {
    let classes = trace_classes(&sample_two_txn_three_ops());
    assert_eq!(classes.len(), 2, "expected exactly two trace classes");
}

#[test]
fn test_foata_layers_correct() {
    let trace = vec![
        TraceAction::Read { txn: 1, page: 1 },
        TraceAction::Read { txn: 2, page: 3 },
        TraceAction::Write { txn: 1, page: 2 },
        TraceAction::Write { txn: 2, page: 4 },
        TraceAction::Commit {
            txn: 1,
            write_pages: vec![2],
        },
        TraceAction::Commit {
            txn: 2,
            write_pages: vec![4],
        },
    ];

    let expected = vec![
        vec![
            TraceAction::Read { txn: 1, page: 1 },
            TraceAction::Read { txn: 2, page: 3 },
        ],
        vec![
            TraceAction::Write { txn: 1, page: 2 },
            TraceAction::Write { txn: 2, page: 4 },
        ],
        vec![TraceAction::Commit {
            txn: 1,
            write_pages: vec![2],
        }],
        vec![TraceAction::Commit {
            txn: 2,
            write_pages: vec![4],
        }],
    ];

    assert_eq!(foata_normal_form(&trace), expected);
}

#[test]
fn test_foata_canonical_deterministic() {
    let trace = vec![
        TraceAction::Read { txn: 1, page: 1 },
        TraceAction::Write { txn: 1, page: 2 },
        TraceAction::Commit {
            txn: 1,
            write_pages: vec![2],
        },
        TraceAction::Read { txn: 2, page: 3 },
        TraceAction::Write { txn: 2, page: 4 },
        TraceAction::Commit {
            txn: 2,
            write_pages: vec![4],
        },
    ];
    let first = foata_normal_form(&trace);
    let second = foata_normal_form(&trace);
    assert_eq!(first, second);
}

#[test]
fn test_enumerate_all_classes() {
    let per_txn = vec![
        vec![
            TraceAction::Read { txn: 1, page: 1 },
            TraceAction::Write { txn: 1, page: 1 },
        ],
        vec![
            TraceAction::Read { txn: 2, page: 2 },
            TraceAction::Write { txn: 2, page: 2 },
        ],
    ];
    let interleavings = enumerate_interleavings(&per_txn);
    assert_eq!(interleavings.len(), 6, "2-way 2+2 shuffle count");

    let classes = trace_classes(&per_txn);
    assert_eq!(classes.len(), 1, "all shuffles collapse into one class");
}

#[test]
fn test_trace_reduction_ratio() {
    let per_txn = vec![
        vec![
            TraceAction::Read { txn: 1, page: 1 },
            TraceAction::Write { txn: 1, page: 1 },
            TraceAction::Commit {
                txn: 1,
                write_pages: vec![1],
            },
        ],
        vec![
            TraceAction::Read { txn: 2, page: 2 },
            TraceAction::Write { txn: 2, page: 2 },
            TraceAction::Commit {
                txn: 2,
                write_pages: vec![2],
            },
        ],
        vec![
            TraceAction::Read { txn: 3, page: 3 },
            TraceAction::Write { txn: 3, page: 3 },
            TraceAction::Commit {
                txn: 3,
                write_pages: vec![3],
            },
        ],
    ];

    let naive = enumerate_interleavings(&per_txn).len();
    let classes = trace_classes(&per_txn).len();

    assert!(classes < naive, "trace classes should reduce search space");
    assert!(classes <= 120, "expected aggressive reduction from naive");
}

#[test]
fn test_mvcc_invariants_all_classes() {
    let per_txn = vec![
        vec![
            TraceAction::Read { txn: 1, page: 10 },
            TraceAction::Read { txn: 1, page: 20 },
            TraceAction::Write { txn: 1, page: 10 },
            TraceAction::Commit {
                txn: 1,
                write_pages: vec![10],
            },
        ],
        vec![
            TraceAction::Read { txn: 2, page: 10 },
            TraceAction::Read { txn: 2, page: 20 },
            TraceAction::Write { txn: 2, page: 20 },
            TraceAction::Commit {
                txn: 2,
                write_pages: vec![20],
            },
        ],
    ];

    let classes = trace_classes(&per_txn);
    assert!(!classes.is_empty(), "must discover at least one class");

    for traces in classes.values() {
        let mut sim = TraceSimulation::default();
        sim.apply(&TraceAction::Begin { txn: 1 });
        sim.apply(&TraceAction::Begin { txn: 2 });
        for action in &traces[0] {
            sim.apply(action);
        }

        assert!(
            sim.committed.len() <= 1,
            "SSI model must prevent write-skew dual-commit"
        );
        assert_eq!(sim.committed.len() + sim.aborted.len(), 2);
        for committed in &sim.committed {
            assert!(committed.commit_seq > committed.begin_seq);
            assert!(committed.commit_seq <= sim.commit_seq_hi);
        }
    }
}

#[test]
fn test_dpor_explores_all_relevant_classes() {
    let per_txn = sample_two_txn_three_ops();
    let expected_classes = trace_classes(&per_txn).len();

    let observed_classes = Arc::new(Mutex::new(BTreeSet::<Vec<Vec<TraceAction>>>::new()));
    let explored_runs = Arc::new(AtomicU64::new(0));

    let observed_for_run = Arc::clone(&observed_classes);
    let explored_for_run = Arc::clone(&explored_runs);
    let per_txn_for_run = per_txn.clone();

    let mut explorer = DporExplorer::new(ExplorerConfig::new(211, 64).max_steps(120_000));
    let report = explorer.explore(move |rt| {
        let root = rt
            .state
            .create_root_region(asupersync::types::Budget::INFINITE);
        let trace_log = Arc::new(Mutex::new(Vec::<TraceAction>::new()));

        let log_a = Arc::clone(&trace_log);
        let actions_a = per_txn_for_run[0].clone();
        let (t_a, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                for action in actions_a {
                    log_a.lock().push(action);
                    yield_now().await;
                }
            })
            .expect("spawn A");

        let log_b = Arc::clone(&trace_log);
        let actions_b = per_txn_for_run[1].clone();
        let (t_b, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                for action in actions_b {
                    log_b.lock().push(action);
                    yield_now().await;
                }
            })
            .expect("spawn B");

        {
            let mut sched = rt.scheduler.lock().expect("scheduler lock");
            sched.schedule(t_a, 0);
            sched.schedule(t_b, 0);
        }
        rt.run_until_quiescent();

        let trace = trace_log.lock().clone();
        observed_for_run.lock().insert(foata_normal_form(&trace));
        explored_for_run.fetch_add(1, Ordering::Relaxed);
    });

    assert!(
        report.violations.is_empty(),
        "DPOR reported violations: {:#?}",
        report.violations
    );

    let observed_count = observed_classes.lock().len();
    assert_eq!(
        observed_count, expected_classes,
        "DPOR should cover each relevant class at least once"
    );
    assert!(
        explored_runs.load(Ordering::Relaxed)
            >= u64::try_from(observed_count).expect("class count fits u64"),
        "DPOR should execute at least one run per observed class"
    );
}

#[test]
fn test_dpor_finds_known_bug() {
    let mut explorer = DporExplorer::new(ExplorerConfig::new(313, 64).max_steps(40_000));
    let report = explorer.explore(|rt| {
        let root = rt
            .state
            .create_root_region(asupersync::types::Budget::INFINITE);
        let counter = Arc::new(AtomicU64::new(0));

        let c_a = Arc::clone(&counter);
        let (t_a, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                let snapshot = c_a.load(Ordering::Relaxed);
                yield_now().await;
                c_a.store(snapshot + 1, Ordering::Relaxed);
            })
            .expect("spawn A");

        let c_b = Arc::clone(&counter);
        let (t_b, _) = rt
            .state
            .create_task(root, asupersync::types::Budget::INFINITE, async move {
                let snapshot = c_b.load(Ordering::Relaxed);
                yield_now().await;
                c_b.store(snapshot + 1, Ordering::Relaxed);
            })
            .expect("spawn B");

        {
            let mut sched = rt.scheduler.lock().expect("scheduler lock");
            sched.schedule(t_a, 0);
            sched.schedule(t_b, 0);
        }
        rt.run_until_quiescent();

        // Intentionally buggy expectation for race discovery.
        assert_eq!(
            counter.load(Ordering::Acquire),
            2,
            "lost update should be discoverable by DPOR"
        );
    });

    assert!(
        !report.violations.is_empty(),
        "DPOR should find the ordering-sensitive lost-update bug"
    );
}

#[test]
fn test_e2e_exhaustive_2txn_write_skew() {
    let per_txn = vec![
        vec![
            TraceAction::Read { txn: 1, page: 10 },
            TraceAction::Read { txn: 1, page: 20 },
            TraceAction::Write { txn: 1, page: 10 },
            TraceAction::Commit {
                txn: 1,
                write_pages: vec![10],
            },
        ],
        vec![
            TraceAction::Read { txn: 2, page: 10 },
            TraceAction::Read { txn: 2, page: 20 },
            TraceAction::Write { txn: 2, page: 20 },
            TraceAction::Commit {
                txn: 2,
                write_pages: vec![20],
            },
        ],
    ];

    let classes = trace_classes(&per_txn);
    assert!(!classes.is_empty(), "must explore at least one class");

    for traces in classes.values() {
        let mut sim = TraceSimulation::default();
        sim.apply(&TraceAction::Begin { txn: 1 });
        sim.apply(&TraceAction::Begin { txn: 2 });
        for action in &traces[0] {
            sim.apply(action);
        }

        assert!(
            sim.committed.len() <= 1,
            "SSI must prevent write-skew for exhaustive 2-txn exploration"
        );
    }
}

fn discover_outgoing_edges(
    begin_seq: u64,
    read_pages: &[u32],
    commit_index: &BTreeMap<u32, u64>,
) -> Vec<u64> {
    // Outgoing edge T -rw-> W exists if W committed after T's snapshot and wrote a page T read.
    let mut out = Vec::new();
    for &p in read_pages {
        if let Some(&latest) = commit_index.get(&p) {
            if latest > begin_seq {
                out.push(latest);
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

#[derive(Debug, Clone)]
struct IncomingCandidate {
    txn_id: u64,
    has_in_rw: bool,
}

fn discover_incoming_edges(
    begin_seq: u64,
    write_pages: &[u32],
    committed_readers: &[CommittedReaderEntry],
) -> Vec<IncomingCandidate> {
    let mut out = Vec::new();
    for e in committed_readers {
        if e.commit_seq <= begin_seq {
            continue;
        }
        if !pages_overlap(write_pages, &e.read_pages) {
            continue;
        }
        out.push(IncomingCandidate {
            txn_id: e.txn_id,
            has_in_rw: e.has_in_rw,
        });
    }
    out
}

fn pages_overlap(a: &[u32], b: &[u32]) -> bool {
    // Small, deterministic overlap check (sortedness not required).
    for &x in a {
        if b.contains(&x) {
            return true;
        }
    }
    false
}

fn arb_trace_action() -> impl Strategy<Value = TraceAction> {
    prop_oneof![
        (0_u8..3).prop_map(|txn| TraceAction::Begin { txn }),
        (0_u8..3, 0_u32..8).prop_map(|(txn, page)| TraceAction::Read { txn, page }),
        (0_u8..3, 0_u32..8).prop_map(|(txn, page)| TraceAction::Write { txn, page }),
        (
            0_u8..3,
            prop::collection::vec(0_u32..8, 0..4).prop_map(|mut pages| {
                pages.sort_unstable();
                pages.dedup();
                pages
            })
        )
            .prop_map(|(txn, write_pages)| TraceAction::Commit { txn, write_pages }),
    ]
}

proptest! {
    #[test]
    fn prop_outgoing_edges_covers_committed_writer(begin_seq in 0_u64..1_000, w_commit_seq in 0_u64..1_000) {
        let mut commit_index = BTreeMap::new();
        commit_index.insert(1_u32, w_commit_seq);

        let out = discover_outgoing_edges(begin_seq, &[1_u32], &commit_index);

        if w_commit_seq > begin_seq {
            prop_assert!(out.contains(&w_commit_seq));
        } else {
            prop_assert!(!out.contains(&w_commit_seq));
        }
    }

    #[test]
    fn prop_incoming_edges_covers_committed_reader(begin_seq in 0_u64..1_000, r_commit_seq in 0_u64..1_000, has_in_rw in any::<bool>()) {
        let committed = CommittedReaderEntry {
            txn_id: 1,
            begin_seq,
            commit_seq: r_commit_seq,
            has_in_rw,
            read_pages: vec![1_u32],
        };

        let in_edges = discover_incoming_edges(begin_seq, &[1_u32], &[committed]);

        if r_commit_seq > begin_seq {
            prop_assert!(in_edges.iter().any(|c| c.txn_id == 1));
        } else {
            prop_assert!(!in_edges.iter().any(|c| c.txn_id == 1));
        }
    }

    #[test]
    fn prop_independence_symmetric(lhs in arb_trace_action(), rhs in arb_trace_action()) {
        prop_assert_eq!(
            actions_independent(&lhs, &rhs),
            actions_independent(&rhs, &lhs)
        );
    }

    #[test]
    fn prop_independence_irreflexive(action in arb_trace_action()) {
        prop_assert!(!actions_independent(&action, &action));
    }

    #[test]
    fn prop_foata_form_unique(
        prefix in prop::collection::vec(arb_trace_action(), 0..3),
        suffix in prop::collection::vec(arb_trace_action(), 0..3),
        left in arb_trace_action(),
        right in arb_trace_action(),
    ) {
        prop_assume!(left != right);
        prop_assume!(actions_independent(&left, &right));

        let mut linearization_a = prefix.clone();
        linearization_a.push(left.clone());
        linearization_a.push(right.clone());
        linearization_a.extend(suffix.clone());

        let mut linearization_b = prefix;
        linearization_b.push(right);
        linearization_b.push(left);
        linearization_b.extend(suffix);

        prop_assert_eq!(
            foata_normal_form(&linearization_a),
            foata_normal_form(&linearization_b)
        );
    }
}
