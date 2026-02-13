# ADR-0001: Coverage Toolchain Selection

**Status:** Accepted
**Date:** 2026-02-13
**Bead:** bd-mblr.1.1.1

## Context

FrankenSQLite requires reproducible code coverage measurement for:
- Quality gates in CI
- Coverage gap analysis for critical invariants
- Tracking test realism improvements over time

The workspace uses Rust nightly (2024 edition) with 24 crates, `#[forbid(unsafe_code)]`, and pedantic clippy lints. Coverage tooling must work reliably with this setup.

## Options Considered

### 1. cargo-llvm-cov (LLVM source-based coverage)

**Pros:**
- Native LLVM instrumentation, most accurate for Rust
- Full support for nightly toolchains
- Provides line, function, region, and branch coverage
- Fast execution (parallel test collection)
- Clean per-crate isolation with `-p <crate>` flag
- Multiple output formats: text, HTML, lcov, cobertura
- Actively maintained, version 0.6.23

**Cons:**
- Requires LLVM tools in toolchain (installed via rustup)
- HTML report generation requires disk space

**Test Results:**
```
cargo llvm-cov -p fsqlite-types --branch --summary-only

TOTAL: 12342 regions, 88.28% coverage
       773 functions, 82.41% coverage
       7451 lines, 85.18% coverage
       484 branches, 82.02% coverage

Test execution: 335 tests in 5.6s
```

### 2. cargo-tarpaulin

**Pros:**
- Pure Rust, no external dependencies
- Docker integration for isolation

**Cons:**
- Inconsistent package filtering (`-p` flag includes dependencies)
- Slower execution
- No branch coverage on nightly
- Results mix target crate with dependencies (showed 12.53% vs 85.18% for same crate)
- Known compatibility issues with newer Rust editions

**Test Results:**
```
cargo tarpaulin -p fsqlite-types --skip-clean

12.53% coverage, 842/6719 lines covered
(Includes dependency crates incorrectly - fsqlite-types alone has ~7500 lines)
```

### 3. grcov (Mozilla's coverage tool)

**Pros:**
- Supports multiple coverage formats
- Used by Firefox/Mozilla

**Cons:**
- Not installed in current environment
- Requires manual profdata generation
- More complex setup
- Less integrated with cargo workflow

## Decision

**Use cargo-llvm-cov as the primary coverage tool.**

## Rationale

1. **Accuracy:** LLVM source-based coverage provides the most accurate line, branch, and function metrics
2. **Workspace compatibility:** Clean per-crate isolation with `-p` flag works correctly
3. **Nightly support:** Full compatibility with Rust nightly 2024 edition
4. **Branch coverage:** Essential for thorough coverage analysis, works out of the box
5. **Speed:** 335 tests execute in ~5.6 seconds with coverage instrumentation
6. **CI integration:** lcov/cobertura output formats work with standard CI tools

## Commands

### Per-crate coverage (recommended for development):
```bash
cargo llvm-cov -p <crate> --branch --summary-only
```

### Workspace coverage:
```bash
cargo llvm-cov --workspace --branch --summary-only
```

### Detailed text report:
```bash
cargo llvm-cov -p <crate> --branch --text
```

### CI-compatible lcov output:
```bash
cargo llvm-cov --workspace --branch --lcov --output-path coverage.lcov
```

### HTML report (for local review):
```bash
cargo llvm-cov --workspace --branch --html --output-dir target/coverage
```

## Known Caveats

1. **Disk space:** HTML reports can be large for the full workspace; use `--summary-only` for CI
2. **Build time:** First run compiles with instrumentation; subsequent runs reuse cache
3. **Profdata location:** Coverage data stored in `$CARGO_TARGET_DIR/llvm-cov-target/`
4. **Proptest coverage:** Property-based tests execute multiple times, inflating hit counts (not a problem, just FYI)

## Consequences

- Coverage commands are standardized across the project
- CI can gate on coverage thresholds
- Branch coverage available for critical path analysis
- Per-crate coverage enables focused improvement efforts
