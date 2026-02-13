#!/usr/bin/env bash
#
# FrankenSQLite Coverage Report Generator
# Reproducible coverage report generation for CI and development
#
# Usage:
#   ./scripts/coverage.sh                    # Workspace summary (default)
#   ./scripts/coverage.sh workspace          # Full workspace coverage
#   ./scripts/coverage.sh crate <name>       # Single crate coverage
#   ./scripts/coverage.sh ci                 # CI mode with lcov output
#   ./scripts/coverage.sh report             # Generate HTML report
#
# Environment variables:
#   COVERAGE_OUTPUT_DIR  - Output directory (default: target/coverage)
#   COVERAGE_THRESHOLD   - Minimum line coverage % for CI pass (default: 70)
#   COVERAGE_NO_BRANCH   - Set to 1 to disable branch coverage
#
# Outputs:
#   target/coverage/summary.txt      - Human-readable summary
#   target/coverage/coverage.lcov    - lcov format for CI tools
#   target/coverage/html/            - HTML report (when using 'report')
#   target/coverage/<crate>.txt      - Per-crate reports
#
# Exit codes:
#   0 - Success (and coverage >= threshold in CI mode)
#   1 - Coverage below threshold
#   2 - Tool error

set -euo pipefail

# Configuration
COVERAGE_OUTPUT_DIR="${COVERAGE_OUTPUT_DIR:-target/coverage}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-70}"
BRANCH_FLAG=""
if [[ -z "${COVERAGE_NO_BRANCH:-}" ]]; then
    BRANCH_FLAG="--branch"
fi

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$COVERAGE_OUTPUT_DIR"

# Helper: print colored output
print_status() {
    local color="$1"
    local message="$2"
    echo -e "${color}${message}${NC}"
}

# Helper: extract line coverage percentage from summary
extract_line_coverage() {
    local summary="$1"
    # Extract the TOTAL line and get line coverage percentage
    echo "$summary" | grep "^TOTAL" | awk '{print $(NF-2)}' | tr -d '%'
}

# Command: workspace summary
cmd_workspace() {
    print_status "$GREEN" "Running workspace coverage..."

    local output
    output=$(cargo llvm-cov --workspace $BRANCH_FLAG --summary-only 2>&1)

    echo "$output" | tee "$COVERAGE_OUTPUT_DIR/summary.txt"

    # Extract and display key metrics
    local line_cov
    line_cov=$(extract_line_coverage "$output")

    print_status "$GREEN" ""
    print_status "$GREEN" "Coverage Summary:"
    print_status "$GREEN" "  Line coverage: ${line_cov}%"
}

# Command: single crate coverage
cmd_crate() {
    local crate_name="$1"

    if [[ -z "$crate_name" ]]; then
        print_status "$RED" "Error: crate name required"
        echo "Usage: $0 crate <crate-name>"
        exit 2
    fi

    print_status "$GREEN" "Running coverage for crate: $crate_name"

    local output
    output=$(cargo llvm-cov -p "$crate_name" $BRANCH_FLAG --summary-only 2>&1)

    echo "$output" | tee "$COVERAGE_OUTPUT_DIR/${crate_name}.txt"
}

# Command: CI mode with threshold check
cmd_ci() {
    print_status "$GREEN" "Running CI coverage check..."
    print_status "$YELLOW" "Threshold: ${COVERAGE_THRESHOLD}%"

    # Generate lcov output
    cargo llvm-cov --workspace $BRANCH_FLAG --summary-only 2>&1 | tee "$COVERAGE_OUTPUT_DIR/summary.txt"

    # Extract coverage percentage
    local line_cov
    line_cov=$(extract_line_coverage "$(cat "$COVERAGE_OUTPUT_DIR/summary.txt")")

    # Convert to integer for comparison (bash doesn't do float comparison)
    local line_cov_int
    line_cov_int=$(echo "$line_cov" | cut -d. -f1)

    print_status "$GREEN" ""
    print_status "$GREEN" "Results:"
    print_status "$GREEN" "  Line coverage: ${line_cov}%"
    print_status "$GREEN" "  Threshold:     ${COVERAGE_THRESHOLD}%"

    if [[ "$line_cov_int" -ge "$COVERAGE_THRESHOLD" ]]; then
        print_status "$GREEN" "  Status:        PASS"
        exit 0
    else
        print_status "$RED" "  Status:        FAIL"
        print_status "$RED" ""
        print_status "$RED" "Coverage ${line_cov}% is below threshold ${COVERAGE_THRESHOLD}%"
        exit 1
    fi
}

# Command: generate HTML report
cmd_report() {
    print_status "$GREEN" "Generating HTML coverage report..."

    cargo llvm-cov --workspace $BRANCH_FLAG --html --output-dir "$COVERAGE_OUTPUT_DIR/html" 2>&1

    print_status "$GREEN" ""
    print_status "$GREEN" "HTML report generated: $COVERAGE_OUTPUT_DIR/html/index.html"
}

# Command: per-crate detailed breakdown
cmd_breakdown() {
    print_status "$GREEN" "Running per-crate coverage breakdown..."

    # Get list of workspace crates
    local crates
    crates=$(cargo metadata --no-deps --format-version=1 | \
             jq -r '.packages[].name' | \
             grep '^fsqlite' | \
             sort)

    echo "Crate,Lines,LinesCovered,LineCoverage%,Functions,FunctionsCovered,FunctionCoverage%" > "$COVERAGE_OUTPUT_DIR/breakdown.csv"

    for crate in $crates; do
        print_status "$YELLOW" "  Processing: $crate"

        local output
        output=$(cargo llvm-cov -p "$crate" --summary-only 2>&1 || true)

        # Extract metrics from TOTAL line
        local total_line
        total_line=$(echo "$output" | grep "^TOTAL" || echo "")

        if [[ -n "$total_line" ]]; then
            local lines lines_missed line_pct funcs funcs_missed func_pct
            lines=$(echo "$total_line" | awk '{print $7}')
            lines_missed=$(echo "$total_line" | awk '{print $8}')
            line_pct=$(echo "$total_line" | awk '{print $9}' | tr -d '%')
            funcs=$(echo "$total_line" | awk '{print $4}')
            funcs_missed=$(echo "$total_line" | awk '{print $5}')
            func_pct=$(echo "$total_line" | awk '{print $6}' | tr -d '%')

            local lines_covered=$((lines - lines_missed))
            local funcs_covered=$((funcs - funcs_missed))

            echo "$crate,$lines,$lines_covered,$line_pct,$funcs,$funcs_covered,$func_pct" >> "$COVERAGE_OUTPUT_DIR/breakdown.csv"
        fi
    done

    print_status "$GREEN" ""
    print_status "$GREEN" "Breakdown saved to: $COVERAGE_OUTPUT_DIR/breakdown.csv"
}

# Main dispatch
main() {
    local cmd="${1:-workspace}"
    shift || true

    case "$cmd" in
        workspace)
            cmd_workspace
            ;;
        crate)
            cmd_crate "$@"
            ;;
        ci)
            cmd_ci
            ;;
        report)
            cmd_report
            ;;
        breakdown)
            cmd_breakdown
            ;;
        help|--help|-h)
            echo "FrankenSQLite Coverage Report Generator"
            echo ""
            echo "Usage: $0 <command> [args]"
            echo ""
            echo "Commands:"
            echo "  workspace   Run workspace-wide coverage (default)"
            echo "  crate NAME  Run coverage for a specific crate"
            echo "  ci          CI mode with threshold check"
            echo "  report      Generate HTML report"
            echo "  breakdown   Per-crate CSV breakdown"
            echo "  help        Show this help"
            echo ""
            echo "Environment:"
            echo "  COVERAGE_OUTPUT_DIR  Output directory (default: target/coverage)"
            echo "  COVERAGE_THRESHOLD   Min line coverage % for CI (default: 70)"
            echo "  COVERAGE_NO_BRANCH   Set to 1 to disable branch coverage"
            ;;
        *)
            print_status "$RED" "Unknown command: $cmd"
            echo "Run '$0 help' for usage"
            exit 2
            ;;
    esac
}

main "$@"
