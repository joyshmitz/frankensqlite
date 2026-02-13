# Test Realism Taxonomy Inventory

## Overview

This directory contains the test realism classification for FrankenSQLite's test suite.

## Realism Tiers

Tests are classified into the following tiers (from lowest to highest realism):

| Tier | Description | Count |
|------|-------------|-------|
| **unit** | Pure unit tests, no I/O or backend | 26,537 |
| **mocked** | Uses mock/fake/stub patterns | 76 |
| **in-memory** | Uses MemDatabase/MemoryVfs backends | 681 |
| **file-backed** | Uses tempfile for file I/O | 232 |
| **e2e** | End-to-end integration tests | 3,585 |

**Total: 31,111 tests**

## Mock Usage

- Tests using mocks: 968 / 31,111 (3.1%)
- Property-based tests: 2,023

## Files

- `test_inventory.csv` - Machine-readable inventory with columns:
  - `crate`: Crate name
  - `file`: Source file path
  - `test_count`: Number of `#[test]` functions
  - `realism_tier`: Classification (unit/mocked/in-memory/file-backed/e2e)
  - `uses_mock`: Whether file contains mock/fake/stub patterns
  - `uses_memory`: Whether file uses in-memory backends
  - `uses_file`: Whether file uses tempfile
  - `is_proptest`: Whether file contains property-based tests

- `summary.md` - Human-readable summary statistics

## Commands

```bash
# Generate/update inventory
./scripts/test_inventory.sh

# View summary only
./scripts/test_inventory.sh summary

# Analyze single crate
./scripts/test_inventory.sh crate fsqlite-core
```

## Interpretation

### Healthy Patterns
- High unit test count indicates good coverage of edge cases
- E2E tests exercising real storage stack
- Property-based tests for invariant verification

### Areas for Improvement
- Files classified as "mocked" should be reviewed for opportunities to use real backends
- Critical path code should have corresponding E2E tests
- In-memory tests are acceptable but file-backed tests provide higher confidence

## Updating

Re-run `./scripts/test_inventory.sh` and copy results to this directory:

```bash
./scripts/test_inventory.sh
cp target/test-inventory/test_inventory.csv docs/test-realism/
cp target/test-inventory/summary.md docs/test-realism/
```
