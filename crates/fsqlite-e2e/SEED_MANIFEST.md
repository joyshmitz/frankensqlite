# E2E Seed Manifest and Reproducibility Contract

## Overview

FrankenSQLite E2E tests use deterministic seeding to ensure exact reproducibility
of all test scenarios. This document defines the seed schema, derivation rules,
and compatibility contract.

## Seed Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `FRANKEN_SEED` | `0x4652414E4B454E` | Default seed (ASCII "FRANKEN") |
| `SEED_MIN` | `1` | Minimum valid seed (0 = use default) |
| `SEED_MAX` | `u64::MAX` | Maximum valid seed |

### Why 0xFRANKEN?

The value `0x4652414E4B454E` spells "FRANKEN" in ASCII bytes, making it:
- Memorable and project-specific
- Unlikely to collide with common test seeds (0, 1, 42)
- Easy to identify in logs and artifacts

## Seed Derivation Schema

### Base Seed → Worker Seeds

For concurrent scenarios, each worker derives its own seed:

```
worker_seed = base_seed XOR (worker_id * 0x9E3779B97F4A7C15)
```

The multiplier is the golden ratio constant (`(sqrt(5) - 1) / 2 * 2^64`),
providing good distribution without clustering.

### Base Seed → Scenario Seeds

Each scenario can derive a unique seed from the base:

```
scenario_seed = base_seed XOR fnv1a_hash(scenario_id)
```

This allows running multiple scenarios with the same base seed while maintaining
independent RNG streams.

## RNG Specification

### Algorithm

All E2E tests use `rand::rngs::StdRng`, which is currently `ChaCha12`.

### Serialization Format (JSON)

```json
{
  "algorithm": "StdRng/ChaCha12",
  "version": "rand 0.8"
}
```

### Versioning Policy

- **Patch versions** (0.8.x → 0.8.y): Expected to be compatible
- **Minor versions** (0.8 → 0.9): May change output, requires re-baseline
- **Major versions** (0.x → 1.x): Incompatible, requires explicit migration

## OpLog Header Schema

Every operation log (JSONL) begins with a header containing seed information:

```json
{
  "fixture_id": "chinook_v1",
  "seed": 5208208757389214030,
  "rng": {
    "algorithm": "StdRng/ChaCha12",
    "version": "rand 0.8"
  },
  "concurrency": {
    "worker_count": 4,
    "transaction_size": 100,
    "commit_order": "deterministic"
  },
  "preset": "hot_page_contention"
}
```

## CLI Interface

### Seed Override

All E2E binaries accept `--seed <u64>`:

```bash
# Use specific seed
cargo run -p fsqlite-e2e --bin realdb_e2e -- --seed 12345 run

# Use default (0xFRANKEN)
cargo run -p fsqlite-e2e --bin realdb_e2e -- run

# Replay a failure (seed from failure artifact)
cargo run -p fsqlite-e2e --bin realdb_e2e -- --seed 5208208757389214030 run
```

### Environment Variable

```bash
export E2E_SEED=12345
cargo test -p fsqlite-e2e
```

Priority: CLI flag > Environment variable > Default constant

## Failure Artifacts

On failure, the seed is recorded in the artifact bundle:

```
{scenario_id}_failure.json
├── seed: 5208208757389214030
├── rng: { algorithm, version }
├── scenario_id: "CON-3"
├── worker_seeds: [derived seeds per worker]
└── replay_command: "cargo run ... --seed 5208208757389214030"
```

## Reproducibility Contract

### Guarantees

Given identical:
1. Seed value
2. RNG algorithm and version
3. Scenario ID
4. Worker count
5. FrankenSQLite version

The test MUST produce identical:
- Operation sequences
- Database row contents
- Query results
- Corruption byte patterns (COR-* scenarios)

### Non-Guarantees

The following may vary:
- Wall-clock timing
- Thread scheduling order (for `commit_order: free`)
- File system timestamps
- Memory addresses in stack traces

## Scenario-to-Seed Mapping

| Scenario | Seed Derivation | Notes |
|----------|-----------------|-------|
| SCH-* | `derive_scenario_seed(base, 0x534348)` | Schema scenarios |
| TXN-* | `derive_scenario_seed(base, 0x54584E)` | Transaction scenarios |
| CON-* | `derive_scenario_seed(base, 0x434F4E)` | Concurrency scenarios |
| COR-* | `derive_scenario_seed(base, 0x434F52)` | Corruption scenarios |
| CMP-* | `derive_scenario_seed(base, 0x434D50)` | Compatibility scenarios |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-13 | Initial schema definition |

*SwiftOwl 2026-02-13*
