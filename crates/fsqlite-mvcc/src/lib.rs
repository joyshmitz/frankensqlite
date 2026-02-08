//! MVCC page-level versioning for concurrent writers.
//!
//! This crate is intentionally small in early phases: it defines the core MVCC
//! primitives and the cross-process witness/lock-table coordination types.

pub mod cache_aligned;
pub mod core_types;
pub mod invariants;
pub mod lifecycle;
pub mod shm;
pub mod witness_hierarchy;

pub use cache_aligned::{
    CACHE_LINE_BYTES, CLAIMING_TIMEOUT_NO_PID_SECS, CLAIMING_TIMEOUT_SECS, CacheAligned,
    SLOT_PAYLOAD_MASK, SLOT_TAG_MASK, SLOT_TAG_SHIFT, SharedTxnSlot, TAG_CLAIMING, TAG_CLEANING,
    decode_payload, decode_tag, encode_claiming, encode_cleaning, is_sentinel,
};
pub use core_types::{
    CommitIndex, CommitLog, CommitRecord, DrainProgress, DrainResult, GcHorizonResult,
    InProcessPageLockTable, LOCK_TABLE_SHARDS, PageBuf, RebuildError, RebuildResult,
    SlotCleanupResult, Transaction, TransactionMode, TransactionState, VersionArena, VersionIdx,
    cleanup_and_raise_gc_horizon, raise_gc_horizon, try_cleanup_sentinel_slot,
};
pub use invariants::{
    SerializedWriteMutex, TxnManager, VersionStore, idx_to_version_pointer, visible,
};
pub use lifecycle::{BeginKind, CommitResponse, MvccError, Savepoint, TransactionManager};
pub use shm::{SharedMemoryLayout, ShmSnapshot};
pub use witness_hierarchy::{
    HotWitnessIndexDerivationV1, HotWitnessIndexSizingV1, WitnessHierarchyConfigV1,
    WitnessHotIndexManifestV1, WitnessSizingError,
};
