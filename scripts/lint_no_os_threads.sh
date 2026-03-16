#!/usr/bin/env bash
# Guardrail for bd-2jpu6.4:
# fail if production engine code reintroduces raw OS-thread spawning.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if [[ "${SCRIPT_PATH}" != /* ]]; then
  SCRIPT_PATH="$(pwd)/${SCRIPT_PATH}"
fi
REPO_ROOT="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
cd "${REPO_ROOT}"

TARGET_DIRS=(
  "crates/fsqlite/src"
  "crates/fsqlite-core/src"
  "crates/fsqlite-vfs/src"
  "crates/fsqlite-pager/src"
  "crates/fsqlite-wal/src"
  "crates/fsqlite-mvcc/src"
  "crates/fsqlite-btree/src"
  "crates/fsqlite-vdbe/src"
  "crates/fsqlite-observability/src"
  "crates/fsqlite-cli/src"
)

FORBIDDEN_PATTERN='std::thread::spawn|thread::spawn|std::thread::Builder|thread::Builder'
declare -a HITS=()

scan_file() {
  local file="$1"
  local first_test_attr
  local output
  first_test_attr="$(
    rg -n -m1 '^[[:space:]]*#\[(cfg|cfg_attr)\([^]]*\btest\b[^]]*\)\]' "${file}" \
      | cut -d: -f1 \
      || true
  )"

  if [[ -n "${first_test_attr}" ]]; then
    local production_end=$((first_test_attr - 1))
    if (( production_end <= 0 )); then
      return
    fi
    output="$(
      sed -n "1,${production_end}p" "${file}" \
      | grep -nE "${FORBIDDEN_PATTERN}" \
      | sed "s|^|${file}:|" \
      || true
    )"
  else
    output="$(rg -n "${FORBIDDEN_PATTERN}" "${file}" || true)"
  fi

  if [[ -n "${output}" ]]; then
    while IFS= read -r line; do
      HITS+=("${line}")
    done <<< "${output}"
  fi
}

while IFS= read -r -d '' file; do
  scan_file "${file}"
done < <(
  find "${TARGET_DIRS[@]}" \
    -type f \
    -name '*.rs' \
    ! -name '*_tests.rs' \
    ! -path '*/src/bin/*' \
    -print0
)

if (( ${#HITS[@]} > 0 )); then
  echo "[FAIL] raw OS-thread usage detected in production engine code:" >&2
  printf '%s\n' "${HITS[@]}" >&2
  exit 1
fi

echo "[PASS] no raw OS-thread spawn/builders found in production engine crate sources"
