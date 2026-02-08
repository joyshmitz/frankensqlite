//! FrankenSQLite scheduler priority lanes (§4.20, bd-3go.13).
//!
//! Maps FrankenSQLite work items to asupersync's three-lane scheduler:
//! - **Cancel lane** (highest): cancellation, drain, finalizers, obligation
//!   completion, rollback/cleanup, coordinator cancel responses.
//! - **Timed lane** (EDF): user queries with deadlines, commit publication
//!   (marker append + response), tiered-storage reads for foreground queries.
//! - **Ready lane**: background GC, compaction, checkpointing, anti-entropy,
//!   stats updates. MUST be rate-limited/bulkheaded (§4.15).
//!
//! # Normative Rules
//!
//! - Cancel lane tasks MUST NOT be starved by background work.
//! - Timed lane tasks are scheduled in Earliest-Deadline-First order.
//! - Ready lane tasks are rate-limited and cannot degrade foreground p99.
//! - Any long-running foreground loop MUST call `cx.checkpoint()` frequently.
//! - Tasks SHOULD call `cx.set_task_type("...")` once at start.

use std::time::Duration;

use asupersync::runtime::scheduler::{DispatchLane, PriorityScheduler};
use asupersync::types::{Budget, TaskId, Time};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Bead identifier for tracing and log correlation.
const BEAD_ID: &str = "bd-3go.13";

// ---------------------------------------------------------------------------
// FrankenSQLite task classification
// ---------------------------------------------------------------------------

/// Classification of FrankenSQLite work items into scheduler lanes.
///
/// Each task class maps to exactly one of the three scheduler lanes.
/// The mapping follows §4.20 of the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FsqliteTaskClass {
    // -- Cancel lane tasks --
    /// Cancellation/drain/finalizer — highest priority.
    Cancellation,
    /// Obligation completion (permit/ack/lease commit or abort).
    ObligationCompletion,
    /// Rollback/cleanup after transaction failure.
    RollbackCleanup,
    /// Coordinator cancel response.
    CoordinatorCancelResponse,

    // -- Timed lane tasks (foreground, deadline-driven) --
    /// User query (SELECT, INSERT, UPDATE, DELETE) with deadline.
    UserQuery,
    /// Commit publication (marker append + response to client).
    CommitPublication,
    /// Tiered-storage read for a foreground query.
    ForegroundStorageRead,

    // -- Ready lane tasks (background, rate-limited) --
    /// Background garbage collection.
    BackgroundGc,
    /// Background compaction.
    Compaction,
    /// WAL checkpointing.
    Checkpointing,
    /// Anti-entropy / consistency verification.
    AntiEntropy,
    /// Statistics updates / page count refresh.
    StatsUpdate,
}

impl FsqliteTaskClass {
    /// All task classes in canonical order.
    pub const ALL: &[Self] = &[
        Self::Cancellation,
        Self::ObligationCompletion,
        Self::RollbackCleanup,
        Self::CoordinatorCancelResponse,
        Self::UserQuery,
        Self::CommitPublication,
        Self::ForegroundStorageRead,
        Self::BackgroundGc,
        Self::Compaction,
        Self::Checkpointing,
        Self::AntiEntropy,
        Self::StatsUpdate,
    ];

    /// The scheduler lane for this task class.
    #[must_use]
    pub fn lane(self) -> SchedulerLane {
        match self {
            Self::Cancellation
            | Self::ObligationCompletion
            | Self::RollbackCleanup
            | Self::CoordinatorCancelResponse => SchedulerLane::Cancel,

            Self::UserQuery | Self::CommitPublication | Self::ForegroundStorageRead => {
                SchedulerLane::Timed
            }

            Self::BackgroundGc
            | Self::Compaction
            | Self::Checkpointing
            | Self::AntiEntropy
            | Self::StatsUpdate => SchedulerLane::Ready,
        }
    }

    /// Human-readable name for logging.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cancellation => "cancellation",
            Self::ObligationCompletion => "obligation_completion",
            Self::RollbackCleanup => "rollback_cleanup",
            Self::CoordinatorCancelResponse => "coordinator_cancel_response",
            Self::UserQuery => "user_query",
            Self::CommitPublication => "commit_publication",
            Self::ForegroundStorageRead => "foreground_storage_read",
            Self::BackgroundGc => "background_gc",
            Self::Compaction => "compaction",
            Self::Checkpointing => "checkpointing",
            Self::AntiEntropy => "anti_entropy",
            Self::StatsUpdate => "stats_update",
        }
    }
}

impl std::fmt::Display for FsqliteTaskClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Scheduler lane abstraction
// ---------------------------------------------------------------------------

/// The three scheduler priority lanes (§4.20).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchedulerLane {
    /// Cancel lane: highest priority. Cancellation, drain, finalizers.
    Cancel,
    /// Timed lane: EDF ordering. User queries with deadlines.
    Timed,
    /// Ready lane: lowest priority. Background GC, compaction.
    Ready,
}

impl SchedulerLane {
    /// Map from asupersync's `DispatchLane` to `SchedulerLane`.
    #[must_use]
    pub fn from_dispatch(lane: DispatchLane) -> Self {
        match lane {
            DispatchLane::Cancel => Self::Cancel,
            DispatchLane::Timed => Self::Timed,
            DispatchLane::Ready | DispatchLane::Stolen => Self::Ready,
        }
    }
}

impl std::fmt::Display for SchedulerLane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancel => f.write_str("Cancel"),
            Self::Timed => f.write_str("Timed"),
            Self::Ready => f.write_str("Ready"),
        }
    }
}

// ---------------------------------------------------------------------------
// Lane assignment from budget
// ---------------------------------------------------------------------------

/// Determine the scheduler lane for a task based on its Cx budget.
///
/// - If the budget has a deadline → Timed lane (foreground query).
/// - Otherwise → Ready lane (background work).
///
/// Cancel lane assignment is done via `schedule_cancel()` when the
/// cancellation protocol triggers, not through budget inspection.
#[must_use]
pub fn lane_from_budget(budget: &Budget) -> SchedulerLane {
    if budget.deadline.is_some() {
        SchedulerLane::Timed
    } else {
        SchedulerLane::Ready
    }
}

// ---------------------------------------------------------------------------
// Schedule a task into the correct lane
// ---------------------------------------------------------------------------

/// Schedule a FrankenSQLite task into asupersync's three-lane scheduler.
///
/// - Cancel-lane tasks are scheduled via `schedule_cancel` with max priority.
/// - Timed-lane tasks are scheduled via `schedule_timed` with their deadline.
/// - Ready-lane tasks are scheduled via `schedule` with default priority.
pub fn schedule_task(
    scheduler: &mut PriorityScheduler,
    task_id: TaskId,
    task_class: FsqliteTaskClass,
    deadline: Option<Time>,
) {
    let lane = task_class.lane();

    debug!(
        bead_id = BEAD_ID,
        task = task_id.as_u64(),
        class = task_class.name(),
        lane = %lane,
        "scheduling task"
    );

    match lane {
        SchedulerLane::Cancel => {
            scheduler.schedule_cancel(task_id, 0); // priority 0 = highest
        }
        SchedulerLane::Timed => {
            if let Some(dl) = deadline {
                scheduler.schedule_timed(task_id, dl);
            } else {
                // Timed-lane task without deadline: assign a generous default.
                warn!(
                    bead_id = BEAD_ID,
                    task = task_id.as_u64(),
                    class = task_class.name(),
                    "timed-lane task scheduled without deadline, using default priority"
                );
                scheduler.schedule(task_id, 128);
            }
        }
        SchedulerLane::Ready => {
            scheduler.schedule(task_id, 200); // low priority for background
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint tracker for long-running loops
// ---------------------------------------------------------------------------

/// Tracks checkpoint frequency in long-running foreground loops.
///
/// Normative rule (§4.20): Any long-running foreground loop MUST call
/// `cx.checkpoint()` frequently at every yield point. This tracker
/// counts opcodes and verifies checkpoint calls happen often enough.
pub struct CheckpointTracker {
    opcodes_executed: u64,
    checkpoints_called: u64,
    max_opcodes_between_checkpoints: u64,
    current_streak: u64,
}

impl CheckpointTracker {
    /// Create a new tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            opcodes_executed: 0,
            checkpoints_called: 0,
            max_opcodes_between_checkpoints: 0,
            current_streak: 0,
        }
    }

    /// Record execution of one opcode.
    pub fn record_opcode(&mut self) {
        self.opcodes_executed += 1;
        self.current_streak += 1;
    }

    /// Record a checkpoint call. Resets the current streak.
    pub fn record_checkpoint(&mut self) {
        self.checkpoints_called += 1;
        if self.current_streak > self.max_opcodes_between_checkpoints {
            self.max_opcodes_between_checkpoints = self.current_streak;
        }
        self.current_streak = 0;
    }

    /// Total opcodes executed.
    #[must_use]
    pub fn opcodes_executed(&self) -> u64 {
        self.opcodes_executed
    }

    /// Total checkpoint calls.
    #[must_use]
    pub fn checkpoints_called(&self) -> u64 {
        self.checkpoints_called
    }

    /// Maximum opcodes between any two consecutive checkpoints.
    #[must_use]
    pub fn max_opcodes_between_checkpoints(&self) -> u64 {
        // Account for the current unfinished streak.
        self.max_opcodes_between_checkpoints
            .max(self.current_streak)
    }

    /// Whether at least one checkpoint was called during the execution.
    #[must_use]
    pub fn has_checkpointed(&self) -> bool {
        self.checkpoints_called > 0
    }
}

impl Default for CheckpointTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Dispatch recording for test verification
// ---------------------------------------------------------------------------

/// Record of a dispatched task for test assertions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchRecord {
    /// The task that was dispatched.
    pub task_id: TaskId,
    /// The lane it was dispatched from.
    pub lane: SchedulerLane,
    /// Dispatch order (0-indexed).
    pub order: usize,
}

/// Drain all tasks from the scheduler and record the dispatch order.
///
/// Useful for testing lane priority ordering.
pub fn drain_all(scheduler: &mut PriorityScheduler, rng_hint: u64) -> Vec<DispatchRecord> {
    let mut records = Vec::new();
    let mut order = 0;
    while let Some((task_id, dispatch_lane)) = scheduler.pop_with_lane(rng_hint) {
        records.push(DispatchRecord {
            task_id,
            lane: SchedulerLane::from_dispatch(dispatch_lane),
            order,
        });
        order += 1;
    }
    records
}

// ===========================================================================
// Tests (§4.20 unit test requirements, bd-3go.13)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BEAD_ID: &str = "bd-3go.13";

    fn task(n: u64) -> TaskId {
        TaskId::from_raw(n)
    }

    // -- Test 1: Cancel lane highest priority --

    #[test]
    fn test_cancel_lane_highest_priority() {
        // Enqueue tasks in all three lanes simultaneously.
        // Verify Cancel lane tasks execute before Timed and Ready.
        let mut sched = PriorityScheduler::new();
        let now = Time::from_millis(100);

        // Schedule: 1 Ready, 1 Timed (deadline at now+1s), 1 Cancel.
        schedule_task(
            &mut sched,
            task(1),
            FsqliteTaskClass::BackgroundGc,
            None,
        );
        schedule_task(
            &mut sched,
            task(2),
            FsqliteTaskClass::UserQuery,
            Some(Time::from_millis(1100)),
        );
        schedule_task(
            &mut sched,
            task(3),
            FsqliteTaskClass::Cancellation,
            None,
        );

        let records = drain_all(&mut sched, 42);
        assert_eq!(records.len(), 3, "bead_id={TEST_BEAD_ID} should have 3 tasks");

        // Cancel must be first.
        assert_eq!(
            records[0].lane,
            SchedulerLane::Cancel,
            "bead_id={TEST_BEAD_ID} cancel lane task should execute first, got {:?}",
            records[0]
        );
        assert_eq!(records[0].task_id, task(3));
    }

    // -- Test 2: Timed lane EDF ordering --

    #[test]
    fn test_timed_lane_edf_ordering() {
        // Enqueue 5 Timed-lane tasks with different deadlines.
        // Verify they execute in earliest-deadline-first order.
        let mut sched = PriorityScheduler::new();

        let deadlines = [500, 100, 300, 200, 400];
        for (i, &dl_ms) in deadlines.iter().enumerate() {
            let tid = task(i as u64 + 1);
            sched.schedule_timed(tid, Time::from_millis(dl_ms));
        }

        let records = drain_all(&mut sched, 42);
        assert_eq!(records.len(), 5, "bead_id={TEST_BEAD_ID} should have 5 timed tasks");

        // All should be from timed lane, in EDF order.
        // Expected order by deadline: 100, 200, 300, 400, 500
        // Which corresponds to tasks: 2, 4, 3, 5, 1
        let expected_task_order = [2, 4, 3, 5, 1];
        for (i, expected_tid) in expected_task_order.iter().enumerate() {
            assert_eq!(
                records[i].task_id,
                task(*expected_tid),
                "bead_id={TEST_BEAD_ID} position {i}: expected task {expected_tid}, got {:?}",
                records[i].task_id
            );
        }
    }

    // -- Test 3: Ready lane does not starve cancel --

    #[test]
    fn test_ready_lane_does_not_starve_cancel() {
        // Flood with 100 Ready-lane tasks, then enqueue 1 Cancel-lane task.
        // Verify Cancel executes within one scheduling cycle (first pop).
        let mut sched = PriorityScheduler::new();

        for i in 0..100 {
            schedule_task(
                &mut sched,
                task(i),
                FsqliteTaskClass::StatsUpdate,
                None,
            );
        }

        // Now add a cancel task.
        schedule_task(
            &mut sched,
            task(999),
            FsqliteTaskClass::Cancellation,
            None,
        );

        // Pop the first task — must be the cancel task.
        let (first_id, first_lane) = sched
            .pop_with_lane(42)
            .expect("bead_id={TEST_BEAD_ID} scheduler should have tasks");

        assert_eq!(
            SchedulerLane::from_dispatch(first_lane),
            SchedulerLane::Cancel,
            "bead_id={TEST_BEAD_ID} cancel task must not be starved by 100 ready-lane tasks"
        );
        assert_eq!(first_id, task(999));
    }

    // -- Test 4: Ready lane does not starve timed --

    #[test]
    fn test_ready_lane_does_not_starve_timed() {
        // Flood with Ready-lane background work, then add a Timed-lane user query.
        // Verify Timed is scheduled before any remaining Ready work.
        let mut sched = PriorityScheduler::new();

        for i in 0..50 {
            schedule_task(
                &mut sched,
                task(i),
                FsqliteTaskClass::Compaction,
                None,
            );
        }

        // Add a user query with a 10ms deadline.
        let deadline = Time::from_millis(10);
        schedule_task(
            &mut sched,
            task(777),
            FsqliteTaskClass::UserQuery,
            Some(deadline),
        );

        // Pop tasks and find where the timed task lands.
        let records = drain_all(&mut sched, 42);

        let timed_pos = records
            .iter()
            .position(|r| r.task_id == task(777))
            .expect("bead_id={TEST_BEAD_ID} timed task should be in dispatch records");

        assert_eq!(
            timed_pos, 0,
            "bead_id={TEST_BEAD_ID} timed-lane query should be scheduled before ready-lane work (pos={timed_pos})"
        );
    }

    // -- Test 5: Task type labeling --

    #[test]
    fn test_task_type_labeling() {
        // Verify task class names are suitable for deadline monitor bucketing.
        let query_class = FsqliteTaskClass::UserQuery;
        assert_eq!(query_class.name(), "user_query");
        assert_eq!(query_class.lane(), SchedulerLane::Timed);

        let gc_class = FsqliteTaskClass::BackgroundGc;
        assert_eq!(gc_class.name(), "background_gc");
        assert_eq!(gc_class.lane(), SchedulerLane::Ready);

        // All classes have non-empty unique names.
        let names: Vec<&str> = FsqliteTaskClass::ALL.iter().map(|c| c.name()).collect();
        for (i, name) in names.iter().enumerate() {
            assert!(!name.is_empty(), "bead_id={TEST_BEAD_ID} class name should not be empty");
            for (j, other) in names.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        name, other,
                        "bead_id={TEST_BEAD_ID} task class names must be unique"
                    );
                }
            }
        }
    }

    // -- Test 6: Checkpoint frequency in long loop --

    #[test]
    fn test_checkpoint_frequency_in_long_loop() {
        // Run a simulated VDBE instruction loop of 10000 opcodes.
        // Verify checkpoint is called at least once per boundary.
        let mut tracker = CheckpointTracker::new();
        let checkpoint_interval = 1000;

        for i in 0..10_000 {
            tracker.record_opcode();
            if (i + 1) % checkpoint_interval == 0 {
                tracker.record_checkpoint();
            }
        }

        assert_eq!(tracker.opcodes_executed(), 10_000);
        assert!(
            tracker.has_checkpointed(),
            "bead_id={TEST_BEAD_ID} checkpoint must be called at least once in 10000 opcodes"
        );
        assert!(
            tracker.checkpoints_called() >= 10,
            "bead_id={TEST_BEAD_ID} expected at least 10 checkpoints, got {}",
            tracker.checkpoints_called()
        );
        assert_eq!(
            tracker.max_opcodes_between_checkpoints(),
            checkpoint_interval,
            "bead_id={TEST_BEAD_ID} max gap between checkpoints should be {checkpoint_interval}"
        );
    }

    // -- Test 7: Lane assignment from Cx budget --

    #[test]
    fn test_lane_assignment_from_cx_budget() {
        // Budget with deadline → Timed lane.
        let timed_budget = Budget::new().with_deadline(Time::from_millis(500));
        assert_eq!(
            lane_from_budget(&timed_budget),
            SchedulerLane::Timed,
            "bead_id={TEST_BEAD_ID} budget with deadline should assign Timed lane"
        );

        // Budget without deadline → Ready lane.
        let ready_budget = Budget::new();
        assert_eq!(
            lane_from_budget(&ready_budget),
            SchedulerLane::Ready,
            "bead_id={TEST_BEAD_ID} budget without deadline should assign Ready lane"
        );

        // Unlimited budget → Ready lane (no deadline).
        let unlimited = Budget::unlimited();
        assert_eq!(
            lane_from_budget(&unlimited),
            SchedulerLane::Ready,
            "bead_id={TEST_BEAD_ID} unlimited budget should assign Ready lane"
        );
    }

    // -- All task classes mapped to a lane --

    #[test]
    fn test_all_task_classes_have_lane() {
        // Every task class maps to exactly one lane.
        for &class in FsqliteTaskClass::ALL {
            let lane = class.lane();
            assert!(
                matches!(
                    lane,
                    SchedulerLane::Cancel | SchedulerLane::Timed | SchedulerLane::Ready
                ),
                "bead_id={TEST_BEAD_ID} class {} should map to a valid lane",
                class.name()
            );
        }

        // Verify distribution: 4 cancel, 3 timed, 5 ready.
        let cancel_count = FsqliteTaskClass::ALL
            .iter()
            .filter(|c| c.lane() == SchedulerLane::Cancel)
            .count();
        let timed_count = FsqliteTaskClass::ALL
            .iter()
            .filter(|c| c.lane() == SchedulerLane::Timed)
            .count();
        let ready_count = FsqliteTaskClass::ALL
            .iter()
            .filter(|c| c.lane() == SchedulerLane::Ready)
            .count();

        assert_eq!(cancel_count, 4, "bead_id={TEST_BEAD_ID} should have 4 cancel-lane classes");
        assert_eq!(timed_count, 3, "bead_id={TEST_BEAD_ID} should have 3 timed-lane classes");
        assert_eq!(ready_count, 5, "bead_id={TEST_BEAD_ID} should have 5 ready-lane classes");
    }

    // -- Dispatch lane conversion --

    #[test]
    fn test_dispatch_lane_conversion() {
        assert_eq!(
            SchedulerLane::from_dispatch(DispatchLane::Cancel),
            SchedulerLane::Cancel
        );
        assert_eq!(
            SchedulerLane::from_dispatch(DispatchLane::Timed),
            SchedulerLane::Timed
        );
        assert_eq!(
            SchedulerLane::from_dispatch(DispatchLane::Ready),
            SchedulerLane::Ready
        );
        assert_eq!(
            SchedulerLane::from_dispatch(DispatchLane::Stolen),
            SchedulerLane::Ready
        );
    }

    // -- CheckpointTracker defaults --

    #[test]
    fn test_checkpoint_tracker_default() {
        let tracker = CheckpointTracker::default();
        assert_eq!(tracker.opcodes_executed(), 0);
        assert_eq!(tracker.checkpoints_called(), 0);
        assert_eq!(tracker.max_opcodes_between_checkpoints(), 0);
        assert!(!tracker.has_checkpointed());
    }

    // -- E2E: Mixed workload lane separation --

    #[test]
    fn test_e2e_mixed_workload_lane_separation() {
        // Concurrent: user queries (timed), background GC (ready),
        // cancellations (cancel). Verify strict lane ordering in dispatch.
        let mut sched = PriorityScheduler::new();

        // 5 background GC tasks (ready).
        for i in 0..5 {
            schedule_task(
                &mut sched,
                task(100 + i),
                FsqliteTaskClass::BackgroundGc,
                None,
            );
        }

        // 3 user queries with different deadlines (timed).
        schedule_task(
            &mut sched,
            task(201),
            FsqliteTaskClass::UserQuery,
            Some(Time::from_millis(300)),
        );
        schedule_task(
            &mut sched,
            task(202),
            FsqliteTaskClass::UserQuery,
            Some(Time::from_millis(100)),
        );
        schedule_task(
            &mut sched,
            task(203),
            FsqliteTaskClass::UserQuery,
            Some(Time::from_millis(200)),
        );

        // 2 cancellations (cancel).
        schedule_task(
            &mut sched,
            task(301),
            FsqliteTaskClass::Cancellation,
            None,
        );
        schedule_task(
            &mut sched,
            task(302),
            FsqliteTaskClass::RollbackCleanup,
            None,
        );

        let records = drain_all(&mut sched, 42);
        assert_eq!(
            records.len(),
            10,
            "bead_id={TEST_BEAD_ID} should have 10 tasks total"
        );

        // First: all cancel-lane tasks.
        let cancel_tasks: Vec<_> = records
            .iter()
            .filter(|r| r.lane == SchedulerLane::Cancel)
            .collect();
        assert_eq!(
            cancel_tasks.len(),
            2,
            "bead_id={TEST_BEAD_ID} should have 2 cancel tasks"
        );
        // Cancel tasks must be at positions 0 and 1.
        assert!(
            cancel_tasks.iter().all(|r| r.order < 2),
            "bead_id={TEST_BEAD_ID} cancel tasks must execute first"
        );

        // Then: all timed-lane tasks (positions 2-4).
        let timed_tasks: Vec<_> = records
            .iter()
            .filter(|r| r.lane == SchedulerLane::Timed)
            .collect();
        assert_eq!(
            timed_tasks.len(),
            3,
            "bead_id={TEST_BEAD_ID} should have 3 timed tasks"
        );
        // Timed tasks must come after cancel but before ready.
        for t in &timed_tasks {
            assert!(
                t.order >= 2 && t.order < 5,
                "bead_id={TEST_BEAD_ID} timed task at order {} should be in range [2,5)",
                t.order
            );
        }

        // Timed tasks should be in EDF order: deadline 100, 200, 300.
        assert_eq!(timed_tasks[0].task_id, task(202), "bead_id={TEST_BEAD_ID} earliest deadline first");
        assert_eq!(timed_tasks[1].task_id, task(203));
        assert_eq!(timed_tasks[2].task_id, task(201));

        // Finally: all ready-lane tasks (positions 5-9).
        let ready_tasks: Vec<_> = records
            .iter()
            .filter(|r| r.lane == SchedulerLane::Ready)
            .collect();
        assert_eq!(
            ready_tasks.len(),
            5,
            "bead_id={TEST_BEAD_ID} should have 5 ready tasks"
        );
        for r in &ready_tasks {
            assert!(
                r.order >= 5,
                "bead_id={TEST_BEAD_ID} ready task at order {} should be >= 5",
                r.order
            );
        }

        info!(
            bead_id = TEST_BEAD_ID,
            cancel_count = cancel_tasks.len(),
            timed_count = timed_tasks.len(),
            ready_count = ready_tasks.len(),
            "E2E mixed workload: strict lane ordering verified"
        );
    }
}
