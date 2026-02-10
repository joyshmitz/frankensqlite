# Corruption + Repair Demo Runbook

This runbook walks through FrankenSQLite's corruption injection and WAL-FEC
recovery system.  It covers how sidecar encoding works, how to run each
corruption mode, what SQLite failure looks like, and how to verify repair success.

## Background: Why This Exists

SQLite handles WAL corruption by truncating at the first frame with a bad
checksum.  Any committed data after the corruption point is silently lost.
FrankenSQLite adds forward error correction (FEC) via RaptorQ repair symbols
stored in a sidecar file (`.db-wal-fec`), allowing recovery of corrupted WAL
frames without data loss.

## Sidecar Encoding

### Source Data

The FEC sidecar operates on WAL frame payloads:

- **K source symbols** = number of WAL frames (one symbol = one page payload)
- **Symbol size** = page size (4096 bytes)
- **R repair symbols** = configurable redundancy (typically 2-4)

### Chunking and Repair Symbol Generation

```
WAL File:
┌──────────┬─────────────────┬─────────────────┬─────────────────┐
│ 32B hdr  │ Frame 0 (4120B) │ Frame 1 (4120B) │ Frame N (4120B) │
└──────────┴─────────────────┴─────────────────┴─────────────────┘
                  │                   │                   │
             ┌────┘                   │                   └────┐
             ▼                        ▼                        ▼
     ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
     │ 24B frame hdr│        │ 24B frame hdr│        │ 24B frame hdr│
     │ 4096B payload│        │ 4096B payload│        │ 4096B payload│
     └──────────────┘        └──────────────┘        └──────────────┘
             │                        │                        │
             ▼                        ▼                        ▼
     Source Symbol 0          Source Symbol 1          Source Symbol K-1
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      │
                              RaptorQ Encoder
                                      │
                     ┌────────────────┼────────────────┐
                     ▼                ▼                 ▼
              Repair Symbol 0  Repair Symbol 1  Repair Symbol R-1
```

### Sidecar Metadata (`WalFecGroupMeta`)

The `.db-wal-fec` file stores:

| Field | Description |
|-------|-------------|
| `wal_salt1`, `wal_salt2` | WAL header salt values (integrity check) |
| `start_frame_no`, `end_frame_no` | Frame range covered |
| `k_source` | Number of source symbols (= frames) |
| `r_repair` | Number of repair symbols generated |
| `page_size` | SQLite page size (4096) |
| `source_page_xxh3_128` | XXHash3-128 digest per source page |
| Repair symbols | R encoded repair blocks (each 4096 bytes) |

### Recovery Capacity

With K source frames and R repair symbols, the system can recover up to R
corrupted frames.  If corruption affects more than R frames, recovery falls
back to truncation (same as stock SQLite).

## Corruption Modes

### Byte-Level Corruption

| Mode | Target | Description |
|------|--------|-------------|
| **BitFlip** | Any file | Flip a single bit at a specific byte offset + bit position (0-7) |
| **RandomOverwrite** | Any file | Overwrite a byte range with seeded random data |

### Page-Level Corruption

| Mode | Target | Description |
|------|--------|-------------|
| **PageZero** | Database | Zero out an entire page (4096 bytes) |
| **PagePartialCorrupt** | Database | Corrupt a sub-range within a single page |
| **HeaderZero** | Database | Zero the 100-byte SQLite header (page 1, bytes 0-100) |

### WAL-Level Corruption

| Mode | Target | Description |
|------|--------|-------------|
| **WalFrameCorrupt** | WAL | Overwrite frame payloads (preserves 24-byte frame headers) |

### Sidecar Corruption

| Mode | Target | Description |
|------|--------|-------------|
| **SidecarCorrupt** | `.db-wal-fec` | Damage repair symbol region |

All corruption modes accept a `seed` parameter for deterministic reproduction.

## Scenario Catalog

Eight pre-defined scenarios exercise the full range of failure modes:

### WAL Corruption Scenarios (1-4)

**Scenario 1: `wal_corrupt_within_tolerance`**
- Corrupt 2 WAL frames with R=4 repair symbols
- SQLite: truncates WAL, loses committed data
- FrankenSQLite: **full recovery** (2 corrupted < 4 repair symbols)

**Scenario 2: `wal_single_bit_flip`**
- Single-bit bitrot in frame 0
- SQLite: truncates WAL at frame 0
- FrankenSQLite: **full recovery** (trivial for FEC)

**Scenario 3: `wal_corrupt_beyond_tolerance`**
- ALL frames corrupted, only R=2 repair symbols
- SQLite: truncates WAL
- FrankenSQLite: **repair exceeds capacity** (falls back to truncation)

**Scenario 4: `wal_corrupt_recovery_disabled`**
- 2 frames corrupted, recovery toggle OFF
- SQLite: truncates WAL
- FrankenSQLite: **recovery disabled** (truncation, same as SQLite)

### Database Page Corruption (5-6)

**Scenario 5: `db_header_zeroed`**
- Zero the 100-byte SQLite header
- SQLite: **cannot open database** (header magic destroyed)
- FrankenSQLite: repair exceeds capacity (no page-level FEC yet)

**Scenario 6: `db_page_bitrot`**
- Corrupt 128 bytes in page 2
- SQLite: integrity check fails
- FrankenSQLite: repair exceeds capacity (no page-level FEC yet)

### Edge Cases (7-8)

**Scenario 7: `sidecar_damaged`**
- Corrupt the `.db-wal-fec` sidecar (offset 64, 512 bytes)
- FrankenSQLite: **sidecar damaged**, cannot decode repair symbols

**Scenario 8: `wal_corrupt_no_sidecar`**
- Corrupt WAL without any FEC sidecar present
- FrankenSQLite: **recovery disabled** (graceful degradation to truncation)

## Running the Demos

### Run All Scenarios

```bash
# Run C SQLite corruption demos (shows baseline failure behavior)
cargo test -p fsqlite-e2e --lib corruption_demo_sqlite -- --nocapture

# Run FrankenSQLite recovery demos (shows FEC repair)
cargo test -p fsqlite-e2e --lib fsqlite_recovery_demo -- --nocapture
```

### Run the CLI Demo (Once Implemented)

```bash
# Inject corruption into a working copy
cargo run -p fsqlite-e2e --bin realdb-e2e -- corrupt \
  --db asupersync \
  --strategy bitflip \
  --seed 42
```

### Run Individual Scenarios Programmatically

```rust
use fsqlite_e2e::corruption_scenarios::scenario_catalog;
use fsqlite_e2e::corruption_demo_sqlite::run_sqlite_corruption_scenario;
use fsqlite_e2e::fsqlite_recovery_demo::run_scenario;

let scenarios = scenario_catalog();

// C SQLite baseline
let sqlite_result = run_sqlite_corruption_scenario(&scenarios[0], work_dir)?;

// FrankenSQLite recovery
let fsqlite_result = run_scenario(&scenarios[0]);
```

## What SQLite Failure Looks Like

### WAL Corruption (Scenarios 1-4)

SQLite silently truncates the WAL at the first frame with a bad checksum.
The database reverts to its state before the truncated transactions:

```
Scenario: wal_corrupt_within_tolerance
  Rows inserted: 110
  Database opens: YES
  Integrity check: ok
  Rows recovered: 100    ← 10 rows LOST (WAL truncated)
```

The database passes `PRAGMA integrity_check` because the rollback journal
left the main file consistent.  Data loss is **silent** -- no error is reported.

### Header Corruption (Scenario 5)

```
Scenario: db_header_zeroed
  Database opens: NO
  Error: "file is not a database"
```

SQLite cannot recognize the file as a database when the 100-byte header
(magic bytes, page size, schema version) is zeroed.

### Page Corruption (Scenario 6)

```
Scenario: db_page_bitrot
  Database opens: YES
  Integrity check: "*** in database main ***\nPage 2: btree cell..."
  Rows recovered: 95    ← partial data loss
```

SQLite opens the database but `PRAGMA integrity_check` reports B-tree
structure corruption.  Some rows on the damaged page may be unreadable.

## FrankenSQLite Recovery Behavior

### Successful Recovery (Scenarios 1-2)

```
Scenario: wal_corrupt_within_tolerance
  Recovery enabled: true
  Sidecar present: true
  Decode attempted: true
  Required symbols: 2 (of 4 available)
  Outcome: RECOVERED
  Pages recovered: 2
  Verdict: "Recovery succeeded: 2 pages recovered via WAL-FEC"
```

All frames are restored from repair symbols.  The database contains all 110
rows with no data loss.

### Capacity Exceeded (Scenario 3)

```
Scenario: wal_corrupt_beyond_tolerance
  Recovery enabled: true
  Sidecar present: true
  Decode attempted: true
  Required symbols: 10 (of 2 available)
  Outcome: TRUNCATE_BEFORE_GROUP
  Fallback reason: RepairExceedsCapacity
  Verdict: "Corruption exceeds repair capacity; truncated WAL"
```

Too many frames corrupted for R=2 to handle.  Falls back to truncation
(same behavior as stock SQLite).

### Recovery Disabled (Scenario 4)

```
Scenario: wal_corrupt_recovery_disabled
  Recovery enabled: false
  Outcome: TRUNCATE_BEFORE_GROUP
  Fallback reason: RecoveryDisabled
  Verdict: "Recovery disabled by configuration"
```

FrankenSQLite's recovery toggle allows operators to disable FEC repair
(e.g., for performance testing or to match SQLite behavior exactly).

### Sidecar Damaged (Scenario 7)

```
Scenario: sidecar_damaged
  Recovery enabled: true
  Sidecar present: true (but damaged)
  Decode attempted: false
  Outcome: TRUNCATE_BEFORE_GROUP
  Fallback reason: SidecarDamaged
  Verdict: "Sidecar damaged; cannot decode repair symbols"
```

If the sidecar itself is corrupted, recovery gracefully falls back to
truncation rather than producing incorrect data.

## Verifying Repair Success

### Three-Tier Verification

The E2E harness uses a three-tier verification system (see `golden.rs`):

| Tier | Method | Strength |
|------|--------|----------|
| **Tier 1** | SHA-256 match | Cryptographic proof of bit-for-bit recovery |
| **Tier 2** | Logical dump comparison | Schema + all rows match (ignoring page layout) |
| **Tier 3** | Row count + integrity check | Minimum bar for partial recovery |

### Verification After Recovery

```rust
// After recovery, verify against golden copy:
let result = golden::verify_recovery(&golden_path, &repaired_path)?;

match result.highest_tier {
    Some(VerificationTier::Tier1Sha256) => println!("Bit-for-bit recovery!"),
    Some(VerificationTier::Tier2Logical) => println!("Logically equivalent"),
    Some(VerificationTier::Tier3Completeness) => println!("Data complete"),
    None => println!("Recovery incomplete"),
}
```

## Recovery Log Schema

Every recovery attempt produces a structured `WalFecRecoveryLog`:

```json
{
  "recovery_enabled": true,
  "outcome_is_recovered": true,
  "fallback_reason": null,
  "decode_attempted": true,
  "required_symbols": 2
}
```

This log is included in the E2E report for audit and regression tracking.

## Failure Mode Decision Tree

```
Corruption detected in WAL
  │
  ├─ Recovery enabled?
  │   ├─ NO → Truncate WAL (same as SQLite)
  │   └─ YES
  │       ├─ Sidecar exists?
  │       │   ├─ NO → Truncate WAL
  │       │   └─ YES
  │       │       ├─ Sidecar readable?
  │       │       │   ├─ NO → Truncate WAL (SidecarDamaged)
  │       │       │   └─ YES
  │       │       │       ├─ Corrupted frames <= R?
  │       │       │       │   ├─ NO → Truncate WAL (RepairExceedsCapacity)
  │       │       │       │   └─ YES → Decode + Recover ✓
  │       │       │       │
  │
Corruption in main database (page-level)
  └─ No FEC coverage yet → Integrity check fails
```
