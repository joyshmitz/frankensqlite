#!/usr/bin/env bash
#
# FrankenSQLite Test Realism Inventory
# Analyzes test patterns to classify realism levels
#
# Usage:
#   ./scripts/test_inventory.sh              # Full inventory
#   ./scripts/test_inventory.sh summary      # Summary stats only
#   ./scripts/test_inventory.sh crate NAME   # Single crate analysis

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-target/test-inventory}"
mkdir -p "$OUTPUT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() {
    echo -e "${1}${2}${NC}"
}

# Analyze a single file for test patterns
analyze_file() {
    local file="$1"
    local crate
    crate=$(echo "$file" | sed 's|crates/\([^/]*\)/.*|\1|')

    # Count test functions
    local test_count
    test_count=$(rg -c '#\[test\]' "$file" 2>/dev/null || echo "0")

    if [[ "$test_count" == "0" ]]; then
        return
    fi

    # Pattern detection
    local uses_mock=false
    local uses_memory_backend=false
    local uses_file_backend=false
    local is_e2e=false
    local is_proptest=false
    local uses_tempfile=false

    # Check for mock/fake/stub patterns
    if rg -q 'Mock|Fake|Stub|mock_|fake_|stub_' "$file" 2>/dev/null; then
        uses_mock=true
    fi

    # Check for in-memory backend
    if rg -q 'MemDatabase|MemoryVfs|:memory:|InMemory' "$file" 2>/dev/null; then
        uses_memory_backend=true
    fi

    # Check for tempfile usage (file-backed tests)
    if rg -q 'tempfile|TempDir|temp_dir|NamedTempFile' "$file" 2>/dev/null; then
        uses_tempfile=true
        uses_file_backend=true
    fi

    # Check for E2E tests
    if [[ "$file" == *"e2e"* ]] || rg -q 'test_e2e_|E2E|end.to.end' "$file" 2>/dev/null; then
        is_e2e=true
    fi

    # Check for property-based tests
    if rg -q 'proptest|prop_' "$file" 2>/dev/null; then
        is_proptest=true
    fi

    # Determine realism tier
    local tier="unknown"
    if [[ "$is_e2e" == "true" ]]; then
        tier="e2e"
    elif [[ "$uses_file_backend" == "true" ]]; then
        tier="file-backed"
    elif [[ "$uses_memory_backend" == "true" ]]; then
        tier="in-memory"
    elif [[ "$uses_mock" == "true" ]]; then
        tier="mocked"
    else
        tier="unit"
    fi

    # Output as CSV line
    echo "$crate,$file,$test_count,$tier,$uses_mock,$uses_memory_backend,$uses_file_backend,$is_proptest"
}

# Full inventory
cmd_full() {
    print_status "$GREEN" "Generating test inventory..."

    local csv_file="$OUTPUT_DIR/test_inventory.csv"
    echo "crate,file,test_count,realism_tier,uses_mock,uses_memory,uses_file,is_proptest" > "$csv_file"

    # Find all test files
    local test_files
    test_files=$(find crates -name '*.rs' -type f | sort)

    for file in $test_files; do
        local result
        result=$(analyze_file "$file" 2>/dev/null || true)
        if [[ -n "$result" ]]; then
            echo "$result" >> "$csv_file"
        fi
    done

    print_status "$GREEN" "Inventory saved to: $csv_file"

    # Generate summary
    cmd_summary
}

# Summary statistics
cmd_summary() {
    local csv_file="$OUTPUT_DIR/test_inventory.csv"

    if [[ ! -f "$csv_file" ]]; then
        print_status "$RED" "No inventory found. Run '$0' first."
        exit 1
    fi

    print_status "$CYAN" ""
    print_status "$CYAN" "=== Test Realism Inventory Summary ==="
    print_status "$CYAN" ""

    # Count by tier
    print_status "$YELLOW" "By Realism Tier:"
    awk -F, 'NR>1 {tier[$4]+=$3} END {for (t in tier) print "  " t ": " tier[t] " tests"}' "$csv_file" | sort

    print_status "$CYAN" ""
    print_status "$YELLOW" "By Crate (top 10):"
    awk -F, 'NR>1 {crate[$1]+=$3} END {for (c in crate) print crate[c] " " c}' "$csv_file" | sort -rn | head -10 | while read count crate; do
        echo "  $crate: $count tests"
    done

    print_status "$CYAN" ""
    print_status "$YELLOW" "Mock/Fake Usage:"
    local mock_count
    mock_count=$(awk -F, 'NR>1 && $5=="true" {sum+=$3} END {print sum+0}' "$csv_file")
    local total_count
    total_count=$(awk -F, 'NR>1 {sum+=$3} END {print sum+0}' "$csv_file")
    echo "  Tests using mocks: $mock_count / $total_count"

    print_status "$CYAN" ""
    print_status "$YELLOW" "Property-Based Tests:"
    local prop_count
    prop_count=$(awk -F, 'NR>1 && $8=="true" {sum+=$3} END {print sum+0}' "$csv_file")
    echo "  Proptest tests: $prop_count"

    # Save summary
    {
        echo "# Test Realism Summary"
        echo "Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo ""
        echo "## By Tier"
        awk -F, 'NR>1 {tier[$4]+=$3} END {for (t in tier) print "- " t ": " tier[t]}' "$csv_file" | sort
        echo ""
        echo "## Mock Usage"
        echo "Tests using mocks: $mock_count / $total_count"
        echo ""
        echo "## Property-Based"
        echo "Proptest tests: $prop_count"
    } > "$OUTPUT_DIR/summary.md"

    print_status "$GREEN" ""
    print_status "$GREEN" "Summary saved to: $OUTPUT_DIR/summary.md"
}

# Single crate analysis
cmd_crate() {
    local crate_name="$1"

    if [[ -z "$crate_name" ]]; then
        print_status "$RED" "Error: crate name required"
        exit 2
    fi

    print_status "$GREEN" "Analyzing crate: $crate_name"

    local crate_dir="crates/$crate_name"
    if [[ ! -d "$crate_dir" ]]; then
        print_status "$RED" "Crate directory not found: $crate_dir"
        exit 2
    fi

    echo "crate,file,test_count,realism_tier,uses_mock,uses_memory,uses_file,is_proptest"

    find "$crate_dir" -name '*.rs' -type f | sort | while read -r file; do
        local result
        result=$(analyze_file "$file" 2>/dev/null || true)
        if [[ -n "$result" ]]; then
            echo "$result"
        fi
    done
}

# Main dispatch
main() {
    local cmd="${1:-full}"
    shift || true

    case "$cmd" in
        full)
            cmd_full
            ;;
        summary)
            cmd_summary
            ;;
        crate)
            cmd_crate "$@"
            ;;
        help|--help|-h)
            echo "FrankenSQLite Test Realism Inventory"
            echo ""
            echo "Usage: $0 <command> [args]"
            echo ""
            echo "Commands:"
            echo "  full        Generate full inventory (default)"
            echo "  summary     Show summary statistics"
            echo "  crate NAME  Analyze single crate"
            echo "  help        Show this help"
            ;;
        *)
            print_status "$RED" "Unknown command: $cmd"
            exit 2
            ;;
    esac
}

main "$@"
