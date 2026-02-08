#![cfg(unix)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum EventKind {
    CommitDurable,
    DurableButNotRepairable,
    CommitAcked,
    RepairStarted,
    RepairCompleted,
    RepairFailed,
}

#[derive(Debug, Clone, Copy)]
struct Event {
    commit_id: u64,
    at: Instant,
    kind: EventKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepairState {
    NotScheduled,
    Pending,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Copy)]
struct CommitReceipt {
    commit_id: u64,
    durable: bool,
    repair_pending: bool,
    latency: Duration,
}

#[derive(Debug)]
struct MockCommitRepairEngine {
    repair_enabled: bool,
    repair_delay: Duration,
    fail_repair: Arc<AtomicBool>,
    next_commit_id: AtomicU64,
    repair_state: Arc<Mutex<HashMap<u64, RepairState>>>,
    generated_repair_symbols: Arc<AtomicU64>,
    events: Arc<Mutex<Vec<Event>>>,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

impl MockCommitRepairEngine {
    fn new(repair_enabled: bool, repair_delay: Duration) -> Self {
        Self {
            repair_enabled,
            repair_delay,
            fail_repair: Arc::new(AtomicBool::new(false)),
            next_commit_id: AtomicU64::new(1),
            repair_state: Arc::new(Mutex::new(HashMap::new())),
            generated_repair_symbols: Arc::new(AtomicU64::new(0)),
            events: Arc::new(Mutex::new(Vec::new())),
            handles: Mutex::new(Vec::new()),
        }
    }

    fn set_fail_repair(&self, fail: bool) {
        self.fail_repair.store(fail, Ordering::Release);
    }

    fn commit(&self, systematic_symbols: &[u8]) -> CommitReceipt {
        let started = Instant::now();
        let commit_id = self.next_commit_id.fetch_add(1, Ordering::Relaxed);

        // Critical path durability work: append+sync systematic symbols only.
        let _critical_checksum = systematic_symbols.iter().fold(0_u64, |acc, byte| {
            acc.wrapping_mul(16777619).wrapping_add(u64::from(*byte))
        });
        self.record(commit_id, EventKind::CommitDurable);

        if !self.repair_enabled {
            self.record(commit_id, EventKind::CommitAcked);
            return CommitReceipt {
                commit_id,
                durable: true,
                repair_pending: false,
                latency: started.elapsed(),
            };
        }

        {
            let mut states = self.repair_state.lock().expect("repair_state lock");
            states.insert(commit_id, RepairState::Pending);
        }
        self.record(commit_id, EventKind::DurableButNotRepairable);
        self.record(commit_id, EventKind::CommitAcked);

        let events = Arc::clone(&self.events);
        let states = Arc::clone(&self.repair_state);
        let generated_symbols = Arc::clone(&self.generated_repair_symbols);
        let fail_repair = Arc::clone(&self.fail_repair);
        let delay = self.repair_delay;
        let handle = thread::spawn(move || {
            // Background phase: generate+append+sync repair symbols.
            {
                let mut guard = events.lock().expect("events lock");
                guard.push(Event {
                    commit_id,
                    at: Instant::now(),
                    kind: EventKind::RepairStarted,
                });
            }
            thread::sleep(delay);

            if fail_repair.load(Ordering::Acquire) {
                {
                    let mut map = states.lock().expect("repair_state lock");
                    map.insert(commit_id, RepairState::Failed);
                }
                let mut guard = events.lock().expect("events lock");
                guard.push(Event {
                    commit_id,
                    at: Instant::now(),
                    kind: EventKind::RepairFailed,
                });
                return;
            }

            generated_symbols.fetch_add(8, Ordering::Release);
            {
                let mut map = states.lock().expect("repair_state lock");
                map.insert(commit_id, RepairState::Completed);
            }
            let mut guard = events.lock().expect("events lock");
            guard.push(Event {
                commit_id,
                at: Instant::now(),
                kind: EventKind::RepairCompleted,
            });
        });
        self.handles.lock().expect("handles lock").push(handle);

        CommitReceipt {
            commit_id,
            durable: true,
            repair_pending: true,
            latency: started.elapsed(),
        }
    }

    fn wait_for_background_repair(&self) {
        let mut handles = self.handles.lock().expect("handles lock");
        while let Some(handle) = handles.pop() {
            handle.join().expect("background repair thread should join");
        }
    }

    fn repair_state_for(&self, commit_id: u64) -> RepairState {
        self.repair_state
            .lock()
            .expect("repair_state lock")
            .get(&commit_id)
            .copied()
            .unwrap_or(RepairState::NotScheduled)
    }

    fn generated_repair_symbols(&self) -> u64 {
        self.generated_repair_symbols.load(Ordering::Acquire)
    }

    fn events_for_commit(&self, commit_id: u64) -> Vec<Event> {
        self.events
            .lock()
            .expect("events lock")
            .iter()
            .copied()
            .filter(|event| event.commit_id == commit_id)
            .collect()
    }

    fn record(&self, commit_id: u64, kind: EventKind) {
        let mut guard = self.events.lock().expect("events lock");
        guard.push(Event {
            commit_id,
            at: Instant::now(),
            kind,
        });
    }
}

impl Drop for MockCommitRepairEngine {
    fn drop(&mut self) {
        let mut handles = self.handles.lock().expect("handles lock");
        while let Some(handle) = handles.pop() {
            let _ = handle.join();
        }
    }
}

fn percentile(values: &[Duration], p: f64) -> Duration {
    assert!(!values.is_empty(), "percentile requires non-empty input");
    let mut sorted = values.to_vec();
    sorted.sort();
    let rank = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[rank]
}

#[test]
fn test_commit_durable_from_systematic_symbols_only() {
    let engine = MockCommitRepairEngine::new(true, Duration::from_millis(20));
    let receipt = engine.commit(&[0xAB; 4096]);

    assert!(
        receipt.durable,
        "commit must be durable after critical path"
    );
    assert!(
        receipt.repair_pending,
        "repair must be pending and off the commit critical path"
    );

    let events = engine.events_for_commit(receipt.commit_id);
    let durable_idx = events
        .iter()
        .position(|event| event.kind == EventKind::CommitDurable)
        .expect("commit durable event must exist");
    let ack_idx = events
        .iter()
        .position(|event| event.kind == EventKind::CommitAcked)
        .expect("commit ack event must exist");
    assert!(durable_idx < ack_idx, "durability must happen before ack");
}

#[test]
fn test_repair_symbols_generated_async() {
    let engine = MockCommitRepairEngine::new(true, Duration::from_millis(15));
    let receipt = engine.commit(&[0x11; 2048]);

    let early_events = engine.events_for_commit(receipt.commit_id);
    assert!(
        early_events
            .iter()
            .any(|event| event.kind == EventKind::CommitAcked),
        "commit ack must be recorded immediately"
    );
    assert!(
        !early_events
            .iter()
            .any(|event| event.kind == EventKind::RepairCompleted),
        "repair completion must not be on the commit path"
    );

    engine.wait_for_background_repair();
    let events = engine.events_for_commit(receipt.commit_id);
    assert!(
        events
            .iter()
            .any(|event| event.kind == EventKind::RepairStarted),
        "repair must start asynchronously in background"
    );
    assert!(
        events
            .iter()
            .any(|event| event.kind == EventKind::RepairCompleted),
        "repair must complete asynchronously"
    );
}

#[test]
fn test_commit_latency_unaffected_by_repair() {
    let no_repair = MockCommitRepairEngine::new(false, Duration::ZERO);
    let with_repair = MockCommitRepairEngine::new(true, Duration::from_millis(10));
    let payload = vec![0x5A; 4096];

    let mut no_repair_latencies = Vec::new();
    let mut with_repair_latencies = Vec::new();
    for _ in 0..64 {
        no_repair_latencies.push(no_repair.commit(&payload).latency);
        with_repair_latencies.push(with_repair.commit(&payload).latency);
    }
    with_repair.wait_for_background_repair();

    let p50_no_repair = percentile(&no_repair_latencies, 0.50);
    let p50_with_repair = percentile(&with_repair_latencies, 0.50);
    let p99_no_repair = percentile(&no_repair_latencies, 0.99);
    let p99_with_repair = percentile(&with_repair_latencies, 0.99);

    // Keep threshold conservative to avoid flaky timing in busy CI.
    let budget = Duration::from_millis(5);
    assert!(
        p50_with_repair.abs_diff(p50_no_repair) <= budget,
        "repair must stay off critical path (p50 drift too high): no_repair={p50_no_repair:?} with_repair={p50_with_repair:?}"
    );
    assert!(
        p99_with_repair.abs_diff(p99_no_repair) <= budget,
        "repair must stay off critical path (p99 drift too high): no_repair={p99_no_repair:?} with_repair={p99_with_repair:?}"
    );
}

#[test]
fn test_durable_but_not_repairable_state() {
    let engine = MockCommitRepairEngine::new(true, Duration::from_millis(25));
    let receipt = engine.commit(&[0x22; 1024]);

    assert!(receipt.durable, "commit must be durable immediately");
    assert_eq!(
        engine.repair_state_for(receipt.commit_id),
        RepairState::Pending,
        "repair state must be pending during the transient window"
    );

    let events = engine.events_for_commit(receipt.commit_id);
    assert!(
        events
            .iter()
            .any(|event| event.kind == EventKind::DurableButNotRepairable),
        "durable-but-not-repairable state must be explicitly logged"
    );
}

#[test]
fn test_background_repair_completes() {
    let engine = MockCommitRepairEngine::new(true, Duration::from_millis(10));
    let receipt = engine.commit(&[0x33; 1024]);
    engine.wait_for_background_repair();

    assert_eq!(
        engine.repair_state_for(receipt.commit_id),
        RepairState::Completed,
        "background repair should eventually complete"
    );
    assert!(
        engine.generated_repair_symbols() > 0,
        "repair symbols should be generated and appended"
    );
}

#[test]
fn test_repair_failure_does_not_affect_durability() {
    let engine = MockCommitRepairEngine::new(true, Duration::from_millis(10));
    engine.set_fail_repair(true);
    let receipt = engine.commit(&[0x44; 1024]);
    engine.wait_for_background_repair();

    assert!(
        receipt.durable,
        "durability must not depend on repair success"
    );
    assert_eq!(
        engine.repair_state_for(receipt.commit_id),
        RepairState::Failed,
        "repair failure should be surfaced in repair state"
    );
}

#[test]
fn test_e2e_commit_latency_not_affected_by_repair() {
    let no_repair = MockCommitRepairEngine::new(false, Duration::ZERO);
    let with_repair = MockCommitRepairEngine::new(true, Duration::from_millis(8));
    let payload = vec![0x9C; 2048];

    let mut baseline = Vec::new();
    let mut observed = Vec::new();
    let mut commit_ids = Vec::new();

    for _ in 0..128 {
        baseline.push(no_repair.commit(&payload).latency);
        let receipt = with_repair.commit(&payload);
        observed.push(receipt.latency);
        commit_ids.push(receipt.commit_id);
    }
    with_repair.wait_for_background_repair();

    let baseline_p50 = percentile(&baseline, 0.50);
    let baseline_p99 = percentile(&baseline, 0.99);
    let observed_p50 = percentile(&observed, 0.50);
    let observed_p99 = percentile(&observed, 0.99);
    let budget = Duration::from_millis(5);

    assert!(
        observed_p50.abs_diff(baseline_p50) <= budget,
        "p50 commit latency drift exceeded budget: baseline={baseline_p50:?} observed={observed_p50:?}"
    );
    assert!(
        observed_p99.abs_diff(baseline_p99) <= budget,
        "p99 commit latency drift exceeded budget: baseline={baseline_p99:?} observed={observed_p99:?}"
    );

    for commit_id in commit_ids {
        let events = with_repair.events_for_commit(commit_id);
        let ack = events
            .iter()
            .find(|event| event.kind == EventKind::CommitAcked)
            .expect("commit ack must exist");
        let repair_started = events
            .iter()
            .find(|event| event.kind == EventKind::RepairStarted)
            .expect("repair start must exist");
        let repair_completed = events
            .iter()
            .find(|event| event.kind == EventKind::RepairCompleted)
            .expect("repair completion must exist");

        assert!(
            repair_started.at >= ack.at,
            "repair must start after commit acknowledgment"
        );
        assert!(
            repair_completed.at >= repair_started.at,
            "repair completion must follow repair start"
        );
        assert!(
            repair_completed.at.duration_since(ack.at) <= Duration::from_millis(250),
            "durable-but-not-repairable window should close within bounded time"
        );
    }
}
