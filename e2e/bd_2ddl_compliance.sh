#!/usr/bin/env bash
set -euo pipefail

BEAD_ID="bd-2ddl"
LOG_STANDARD_REF="bd-1fpm"
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${WORKSPACE_ROOT}/test-results"
REPORT_JSONL="${REPORT_DIR}/bd_2ddl_compliance_report.jsonl"
LOG_DIR="${REPORT_DIR}/bd_2ddl_logs"
RUN_PER_CRATE_TESTS="${BD_2DDL_RUN_PER_CRATE_TESTS:-1}"
RUN_WORKSPACE_TEST="${BD_2DDL_RUN_WORKSPACE_TEST:-1}"

CRATES=(
    "fsqlite-types"
    "fsqlite-error"
    "fsqlite-vfs"
    "fsqlite-pager"
    "fsqlite-wal"
    "fsqlite-mvcc"
    "fsqlite-btree"
    "fsqlite-ast"
    "fsqlite-parser"
    "fsqlite-planner"
    "fsqlite-vdbe"
    "fsqlite-func"
    "fsqlite-ext-fts3"
    "fsqlite-ext-fts5"
    "fsqlite-ext-rtree"
    "fsqlite-ext-json"
    "fsqlite-ext-session"
    "fsqlite-ext-icu"
    "fsqlite-ext-misc"
    "fsqlite-core"
    "fsqlite"
    "fsqlite-cli"
    "fsqlite-harness"
)

log_line() {
    local level="$1"
    local case_name="$2"
    shift 2
    printf 'bead_id=%s level=%s case=%s %s reference=%s\n' \
        "${BEAD_ID}" "${level}" "${case_name}" "$*" "${LOG_STANDARD_REF}"
}

count_pattern() {
    local pattern="$1"
    shift

    local total=0
    local path
    local count
    for path in "$@"; do
        if [[ -d "${path}" ]]; then
            count="$(rg -n --glob '*.rs' -e "${pattern}" "${path}" 2>/dev/null | wc -l | tr -d '[:space:]')"
            total=$((total + count))
        fi
    done

    printf '%s' "${total}"
}

collect_public_fn_names() {
    local src_dir="$1"
    if [[ ! -d "${src_dir}" ]]; then
        return 0
    fi

    rg -n --glob '*.rs' -e '^[[:space:]]*pub(\([^)]*\))?[[:space:]]+fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "${src_dir}" 2>/dev/null \
        | sed -E 's/.*pub(\([^)]*\))?[[:space:]]+fn[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\2/' \
        | sort -u
}

collect_test_corpus() {
    local crate_dir="$1"

    if [[ -d "${crate_dir}/tests" ]]; then
        rg -N --glob '*.rs' -e '.' "${crate_dir}/tests" 2>/dev/null || true
    fi

    if [[ -d "${crate_dir}/src" ]]; then
        while IFS= read -r src_file; do
            awk '
                BEGIN { in_test = 0 }
                /#\[cfg\(test\)\]/ { in_test = 1 }
                { if (in_test == 1) print }
            ' "${src_file}"
        done < <(find "${crate_dir}/src" -type f -name '*.rs' | sort)
    fi
}

mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

printf '# bead_id=%s compliance report\n' "${BEAD_ID}" >"${REPORT_JSONL}"
printf '# crate\tunit\tprop\tconformance\tfuzz\tpublic_fn_total\tpublic_fn_covered\tcargo_test_exit\n' >>"${REPORT_JSONL}"

log_line "DEBUG" "start" \
    "workspace=${WORKSPACE_ROOT} report=${REPORT_JSONL} per_crate_tests=${RUN_PER_CRATE_TESTS} workspace_test=${RUN_WORKSPACE_TEST}"

if [[ ! -f "${WORKSPACE_ROOT}/Cargo.toml" ]]; then
    log_line "ERROR" "missing_workspace_manifest" "path=${WORKSPACE_ROOT}/Cargo.toml"
    exit 1
fi

declare -a failing_crates=()
declare -a zero_test_crates=()
declare -a missing_public_api_test_crates=()
declare -A missing_categories_by_crate=()

for crate in "${CRATES[@]}"; do
    crate_dir="${WORKSPACE_ROOT}/crates/${crate}"
    if [[ ! -d "${crate_dir}" ]]; then
        log_line "ERROR" "missing_crate_dir" "crate=${crate} path=${crate_dir}"
        failing_crates+=("${crate}")
        continue
    fi

    unit_count="$(count_pattern '#\[test\]' "${crate_dir}/src" "${crate_dir}/tests")"
    prop_count="$(count_pattern 'proptest!' "${crate_dir}/src" "${crate_dir}/tests")"
    conformance_count="$(count_pattern 'conformance' "${crate_dir}/src" "${crate_dir}/tests")"

    fuzz_count=0
    if [[ -d "${crate_dir}/fuzz" ]]; then
        fuzz_count="$(find "${crate_dir}/fuzz" -type f -name '*.rs' | wc -l | tr -d '[:space:]')"
    fi

    mapfile -t public_fn_names < <(collect_public_fn_names "${crate_dir}/src")
    test_corpus="$(collect_test_corpus "${crate_dir}")"

    public_fn_total="${#public_fn_names[@]}"
    public_fn_covered=0
    missing_api_preview=()
    for fn_name in "${public_fn_names[@]}"; do
        if grep -Eq "\\b${fn_name}\\b" <<<"${test_corpus}"; then
            public_fn_covered=$((public_fn_covered + 1))
        else
            if [[ "${#missing_api_preview[@]}" -lt 5 ]]; then
                missing_api_preview+=("${fn_name}")
            fi
        fi
    done

    missing_category=()
    if [[ "${unit_count}" -eq 0 ]]; then
        missing_category+=("unit")
        zero_test_crates+=("${crate}")
    fi
    if [[ "${prop_count}" -eq 0 ]]; then
        missing_category+=("prop")
    fi
    if [[ "${conformance_count}" -eq 0 ]]; then
        missing_category+=("conformance")
    fi
    if [[ "${fuzz_count}" -eq 0 ]]; then
        missing_category+=("fuzz")
    fi
    if [[ "${public_fn_total}" -gt 0 && "${public_fn_covered}" -lt "${public_fn_total}" ]]; then
        missing_public_api_test_crates+=("${crate}")
    fi

    if [[ "${#missing_category[@]}" -gt 0 ]]; then
        missing_categories_by_crate["${crate}"]="$(IFS=,; echo "${missing_category[*]}")"
        log_line "WARN" "missing_test_category" \
            "crate=${crate} missing=${missing_categories_by_crate["${crate}"]}"
    fi
    if [[ "${public_fn_total}" -gt 0 && "${public_fn_covered}" -lt "${public_fn_total}" ]]; then
        log_line "WARN" "public_api_coverage_gap" \
            "crate=${crate} covered=${public_fn_covered}/${public_fn_total} sample_missing=${missing_api_preview[*]:-none}"
    fi

    cargo_test_exit=0
    crate_log="${LOG_DIR}/${crate}.log"
    if [[ "${RUN_PER_CRATE_TESTS}" == "1" ]]; then
        set +e
        (
            cd "${WORKSPACE_ROOT}" || exit 1
            cargo test -p "${crate}"
        ) >"${crate_log}" 2>&1
        cargo_test_exit=$?
        set -e

        if [[ "${cargo_test_exit}" -ne 0 ]]; then
            failing_crates+=("${crate}")
            log_line "ERROR" "crate_test_failed" \
                "crate=${crate} exit=${cargo_test_exit} log=${crate_log}"
        fi
    fi

    log_line "INFO" "crate_matrix_summary" \
        "crate=${crate} unit=${unit_count} prop=${prop_count} conformance=${conformance_count} fuzz=${fuzz_count} public_fn_covered=${public_fn_covered}/${public_fn_total} cargo_test_exit=${cargo_test_exit}"

    jq -nc \
        --arg bead_id "${BEAD_ID}" \
        --arg crate "${crate}" \
        --argjson unit "${unit_count}" \
        --argjson prop "${prop_count}" \
        --argjson conformance "${conformance_count}" \
        --argjson fuzz "${fuzz_count}" \
        --argjson public_fn_total "${public_fn_total}" \
        --argjson public_fn_covered "${public_fn_covered}" \
        --argjson cargo_test_exit "${cargo_test_exit}" \
        '{bead_id:$bead_id,crate:$crate,unit:$unit,prop:$prop,conformance:$conformance,fuzz:$fuzz,public_fn_total:$public_fn_total,public_fn_covered:$public_fn_covered,cargo_test_exit:$cargo_test_exit}' \
        >>"${REPORT_JSONL}"
done

workspace_test_exit=0
workspace_log="${LOG_DIR}/workspace.log"
if [[ "${RUN_WORKSPACE_TEST}" == "1" ]]; then
    set +e
    (
        cd "${WORKSPACE_ROOT}" || exit 1
        cargo test --workspace
    ) >"${workspace_log}" 2>&1
    workspace_test_exit=$?
    set -e

    if [[ "${workspace_test_exit}" -ne 0 ]]; then
        log_line "ERROR" "workspace_test_failed" \
            "exit=${workspace_test_exit} log=${workspace_log}"
    fi
fi

log_line "INFO" "workspace_summary" \
    "crates=${#CRATES[@]} failing_crates=${#failing_crates[@]} zero_test_crates=${#zero_test_crates[@]} missing_public_api_test_crates=${#missing_public_api_test_crates[@]} workspace_test_exit=${workspace_test_exit} report=${REPORT_JSONL}"

if [[ "${#failing_crates[@]}" -gt 0 ]]; then
    log_line "WARN" "degraded_mode" "type=crate_failures crates=${failing_crates[*]}"
fi
if [[ "${#zero_test_crates[@]}" -gt 0 ]]; then
    log_line "WARN" "degraded_mode" "type=zero_test_crates crates=${zero_test_crates[*]}"
fi
if [[ "${#missing_public_api_test_crates[@]}" -gt 0 ]]; then
    log_line "WARN" "degraded_mode" "type=public_api_coverage_gap crates=${missing_public_api_test_crates[*]}"
fi

if [[ "${#failing_crates[@]}" -gt 0 || "${#zero_test_crates[@]}" -gt 0 || "${#missing_public_api_test_crates[@]}" -gt 0 || "${workspace_test_exit}" -ne 0 ]]; then
    log_line "ERROR" "terminal_failure" \
        "failing_crates=${failing_crates[*]:-none} zero_test_crates=${zero_test_crates[*]:-none} missing_public_api_test_crates=${missing_public_api_test_crates[*]:-none} workspace_test_exit=${workspace_test_exit} report=${REPORT_JSONL}"
    exit 1
fi

log_line "WARN" "degraded_mode_count" "count=0"
log_line "ERROR" "terminal_failure_count" "count=0"
log_line "INFO" "pass" "report=${REPORT_JSONL}"
