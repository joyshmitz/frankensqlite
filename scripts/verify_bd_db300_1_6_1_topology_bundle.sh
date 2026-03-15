#!/usr/bin/env bash
# verify_bd_db300_1_6_1_topology_bundle.sh
#
# Capture the reusable hardware discovery bundle for bd-db300.1.6.1. The
# output is designed to attach mechanically to benchmark artifact bundles and
# to make partial hardware identity explicit instead of silently guessing.

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BEAD_ID="bd-db300.1.6.1"
SCRIPT_ENTRYPOINT="scripts/verify_bd_db300_1_6_1_topology_bundle.sh"

default_run_id() {
    date -u +%Y%m%dT%H%M%SZ
}

default_source_revision() {
    git -C "${WORKSPACE_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown'
}

default_beads_hash() {
    if [[ -f "${WORKSPACE_ROOT}/.beads/issues.jsonl" ]]; then
        sha256sum "${WORKSPACE_ROOT}/.beads/issues.jsonl" | awk '{print $1}'
    else
        printf 'unavailable'
    fi
}

RUN_ID="${RUN_ID:-$(default_run_id)}"
SOURCE_REVISION="${SOURCE_REVISION:-$(default_source_revision)}"
BEADS_HASH="${BEADS_HASH:-$(default_beads_hash)}"
CAMPAIGN_ID="${CAMPAIGN_ID:-bd-db300.1.6}"
ROW_ID="${ROW_ID:-}"
FIXTURE_ID="${FIXTURE_ID:-}"
MODE_ID="${MODE_ID:-}"
PLACEMENT_PROFILE_ID="${PLACEMENT_PROFILE_ID:-}"
HARDWARE_CLASS_ID="${HARDWARE_CLASS_ID:-}"
BUILD_PROFILE_ID="${BUILD_PROFILE_ID:-release_perf}"
MANIFEST_JSON="${MANIFEST_JSON:-}"
CPU_AFFINITY_MASK="${CPU_AFFINITY_MASK:-}"
BENCHMARK_PID="${BENCHMARK_PID:-}"
HELPER_LANE_CPU_SET="${HELPER_LANE_CPU_SET:-}"
HELPER_LANE_PID="${HELPER_LANE_PID:-}"
WORKER_CPU_SET="${WORKER_CPU_SET:-}"
MEMORY_POLICY="${MEMORY_POLICY:-}"
NUMA_BALANCING_STATE="${NUMA_BALANCING_STATE:-}"
IRQ_AFFINITY_MASK="${IRQ_AFFINITY_MASK:-}"
ARTIFACT_BUNDLE_DIR="${ARTIFACT_BUNDLE_DIR:-${WORKSPACE_ROOT}/artifacts/perf/${BEAD_ID}/run_${RUN_ID}}"

if [[ "${ARTIFACT_BUNDLE_DIR}" != /* ]]; then
    ARTIFACT_BUNDLE_DIR="${WORKSPACE_ROOT}/${ARTIFACT_BUNDLE_DIR}"
fi

ARTIFACT_BUNDLE_RELPATH="${ARTIFACT_BUNDLE_RELPATH:-${ARTIFACT_BUNDLE_DIR#"${WORKSPACE_ROOT}/"}}"
export WORKSPACE_ROOT
export BEAD_ID
export SCRIPT_ENTRYPOINT
export RUN_ID
export SOURCE_REVISION
export BEADS_HASH
export CAMPAIGN_ID
export ROW_ID
export FIXTURE_ID
export MODE_ID
export PLACEMENT_PROFILE_ID
export HARDWARE_CLASS_ID
export BUILD_PROFILE_ID
export MANIFEST_JSON
export CPU_AFFINITY_MASK
export BENCHMARK_PID
export HELPER_LANE_CPU_SET
export HELPER_LANE_PID
export WORKER_CPU_SET
export MEMORY_POLICY
export NUMA_BALANCING_STATE
export IRQ_AFFINITY_MASK
export ARTIFACT_BUNDLE_DIR
export ARTIFACT_BUNDLE_RELPATH

mkdir -p "${ARTIFACT_BUNDLE_DIR}"

python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_text(path: str | Path) -> str | None:
    candidate = Path(path)
    try:
        return candidate.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None


def run_command(command: list[str]) -> tuple[bool, str | None]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return (False, None)
    if result.returncode != 0:
        return (False, None)
    return (True, result.stdout.strip())


def available(value, source: str) -> dict[str, object]:
    return {"status": "available", "source": source, "value": value}


def inferred(value, source: str) -> dict[str, object]:
    return {"status": "inferred", "source": source, "value": value}


def unavailable(source: str | None = None, note: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"status": "unavailable", "source": source, "value": None}
    if note is not None:
        payload["note"] = note
    return payload


def normalize_scalar(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def env_field(key: str, *, normalizer=normalize_scalar) -> dict[str, object]:
    value = normalizer(os.environ.get(key) or None)
    if value is None:
        return unavailable(f"env:{key}")
    return available(value, f"env:{key}")


def parse_cpuinfo() -> dict[str, str]:
    cpuinfo = read_text("/proc/cpuinfo")
    if not cpuinfo:
        return {}
    first_block = cpuinfo.split("\n\n", 1)[0]
    values: dict[str, str] = {}
    for line in first_block.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key.strip()] = raw.strip()
    return values


def parse_lscpu() -> dict[str, str]:
    ok, stdout = run_command(["lscpu", "-J"])
    if not ok or not stdout:
        return {}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    values: dict[str, str] = {}
    for row in payload.get("lscpu", []):
        field = str(row.get("field", "")).strip().rstrip(":")
        data = str(row.get("data", "")).strip()
        if field and data:
            values[field] = data
    return values


def field_from_sources(
    *,
    lscpu: dict[str, str],
    cpuinfo: dict[str, str],
    lscpu_key: str | None = None,
    cpuinfo_key: str | None = None,
    sysfs_paths: list[str] | None = None,
) -> dict[str, object]:
    if lscpu_key and normalize_scalar(lscpu.get(lscpu_key)):
        return available(lscpu[lscpu_key], f"lscpu:{lscpu_key}")
    if cpuinfo_key and normalize_scalar(cpuinfo.get(cpuinfo_key)):
        return available(cpuinfo[cpuinfo_key], f"/proc/cpuinfo:{cpuinfo_key}")
    for sysfs_path in sysfs_paths or []:
        value = normalize_scalar(read_text(sysfs_path))
        if value is not None:
            return available(value, f"fs:{sysfs_path}")
    return unavailable(None)


def cpu_ids() -> list[int]:
    cpu_root = Path("/sys/devices/system/cpu")
    ids: list[int] = []
    for child in cpu_root.glob("cpu[0-9]*"):
        suffix = child.name.removeprefix("cpu")
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def unique_cpu_file_values(relative_path: str) -> list[dict[str, object]]:
    seen: dict[str, int] = {}
    for cpu_id in cpu_ids():
        value = normalize_scalar(read_text(f"/sys/devices/system/cpu/cpu{cpu_id}/{relative_path}"))
        if value is None or value in seen:
            continue
        seen[value] = cpu_id
    return [
        {"representative_cpu": cpu_id, "cpus": value}
        for value, cpu_id in sorted(seen.items(), key=lambda item: item[1])
    ]


def unique_llc_domains() -> list[dict[str, object]]:
    seen: dict[str, int] = {}
    for cpu_id in cpu_ids():
        cache_root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cache")
        for index_path in sorted(cache_root.glob("index*")):
            level = normalize_scalar(read_text(index_path / "level"))
            shared = normalize_scalar(read_text(index_path / "shared_cpu_list"))
            cache_type = normalize_scalar(read_text(index_path / "type"))
            if level != "3" or shared is None:
                continue
            if shared in seen:
                continue
            seen[shared] = cpu_id
            yield {
                "representative_cpu": cpu_id,
                "shared_cpu_list": shared,
                "cache_type": cache_type,
            }


def expand_cpu_list(raw: str) -> list[int]:
    cpus: list[int] = []
    normalized = raw.replace("\n", ",").replace("\t", ",").replace(" ", ",")
    for chunk in normalized.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            if start_raw.isdigit() and end_raw.isdigit():
                cpus.extend(range(int(start_raw), int(end_raw) + 1))
        elif part.isdigit():
            cpus.append(int(part))
    return sorted(set(cpus))


def expand_hex_cpumask(raw: str) -> list[int]:
    scalar = normalize_scalar(raw)
    if scalar is None:
        return []
    lowered = scalar.lower()
    if lowered.startswith("0x"):
        lowered = lowered[2:]
    if "," not in lowered and not any(char in "abcdef" for char in lowered):
        return []
    compact = lowered.replace(",", "")
    if not compact or any(char not in "0123456789abcdef" for char in compact):
        return []
    value = int(compact, 16)
    return [bit for bit in range(value.bit_length()) if value & (1 << bit)]


def compress_index_list(values: list[int]) -> str | None:
    if not values:
        return None
    deduped = sorted(set(values))
    ranges: list[str] = []
    start = deduped[0]
    previous = deduped[0]
    for value in deduped[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = value
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    return ",".join(ranges)


def normalize_index_list(raw: str | None) -> str | None:
    scalar = normalize_scalar(raw)
    if scalar is None:
        return None
    expanded = expand_cpu_list(scalar)
    if not expanded:
        expanded = expand_hex_cpumask(scalar)
    compact = compress_index_list(expanded)
    return compact or scalar


def process_affinity(pid: int | None = None) -> dict[str, object]:
    target_pid = 0 if pid is None else pid
    label = "self" if pid is None else f"pid:{pid}"
    try:
        affinity = sorted(os.sched_getaffinity(target_pid))
    except (AttributeError, OSError):
        return unavailable(f"os.sched_getaffinity:{label}")
    compact = compress_index_list(list(affinity))
    if compact is None:
        return unavailable(f"os.sched_getaffinity:{label}", "empty affinity set")
    return available(compact, f"os.sched_getaffinity:{label}")


def command_field(
    command: list[str],
    source: str,
    *,
    normalizer=normalize_scalar,
) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return unavailable(source)
    stdout = normalize_scalar(result.stdout)
    stderr = normalize_scalar(result.stderr)
    value = normalizer(stdout or stderr)
    if value is None:
        return unavailable(source)
    status = available if result.returncode == 0 else inferred
    return status(value, source)


def parse_numactl_show() -> dict[str, object]:
    ok, stdout = run_command(["numactl", "--show"])
    if not ok or not stdout:
        return unavailable("command:numactl --show")
    fields: dict[str, object] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        normalized_key = key.strip().replace(" ", "_")
        scalar = normalize_scalar(raw_value)
        if scalar is None:
            continue
        if normalized_key in {"physcpubind", "cpubind", "nodebind", "membind", "preferred"}:
            fields[normalized_key] = normalize_index_list(scalar) or scalar
        else:
            fields[normalized_key] = scalar
    if not fields:
        return unavailable("command:numactl --show", "no parseable fields")
    return available(fields, "command:numactl --show")


def irq_affinity_groups() -> dict[str, object]:
    irq_root = Path("/proc/irq")
    if not irq_root.exists():
        return unavailable("fs:/proc/irq")
    groups: dict[str, list[int]] = {}
    scanned = 0
    for irq_path in sorted(irq_root.glob("[0-9]*")):
        if not irq_path.name.isdigit():
            continue
        scanned += 1
        affinity = normalize_index_list(read_text(irq_path / "smp_affinity_list"))
        if affinity is None:
            affinity = normalize_scalar(read_text(irq_path / "smp_affinity"))
        if affinity is None:
            continue
        groups.setdefault(affinity, []).append(int(irq_path.name))
    if not groups:
        return unavailable("fs:/proc/irq/*/smp_affinity_list")
    summary = [
        {
            "affinity": affinity,
            "irq_count": len(irq_ids),
            "sample_irq_ids": irq_ids[:8],
        }
        for affinity, irq_ids in sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    ]
    return available(
        {
            "scanned_irq_count": scanned,
            "unique_affinity_groups": summary,
        },
        "fs:/proc/irq/*/smp_affinity_list",
    )


def parse_int_env(key: str) -> int | None:
    raw = normalize_scalar(os.environ.get(key) or None)
    if raw is None or not raw.isdigit():
        return None
    return int(raw)


def cpu_to_node_map(numa_layout: list[dict[str, object]]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for node in numa_layout:
        node_id = node.get("node_id")
        cpulist = node.get("cpulist")
        if not isinstance(node_id, int) or not isinstance(cpulist, str):
            continue
        for cpu_id in expand_cpu_list(cpulist):
            mapping[cpu_id] = node_id
    return mapping


def node_list_for_cpu_spec(spec: str | None, cpu_nodes: dict[int, int]) -> str | None:
    if spec is None:
        return None
    nodes = sorted({cpu_nodes[cpu] for cpu in expand_cpu_list(spec) if cpu in cpu_nodes})
    return compress_index_list(nodes)


def classify_comparability(classifications: list[str]) -> str:
    if any(classification == "suspected_drift" for classification in classifications):
        return "suspected"
    if any(classification == "declared_but_unverified" for classification in classifications):
        return "insufficient_evidence"
    if any(classification == "controlled" for classification in classifications):
        if all(classification in {"controlled", "ambient_host_default"} for classification in classifications):
            return "controlled"
    if any(classification in {"ambient_host_default", "ambient_observed"} for classification in classifications):
        return "tolerated"
    return "insufficient_evidence"


def inferred_contract_for_profile(profile_id: str | None) -> dict[str, object]:
    if profile_id == "baseline_unpinned":
        return inferred(
            {
                "cpu_affinity_policy": "scheduler_default",
                "memory_policy": "host_default",
                "helper_lane_policy": "disclose_host_default",
                "violation_disposition": "not_comparable",
                "fixed_knobs": [
                    "no_taskset_or_numactl_binding",
                    "report_host_default_memory_policy",
                    "disclose_helper_lane_policy_without_relocation",
                ],
            },
            "placement_profile_id:baseline_unpinned",
        )
    if profile_id == "recommended_pinned":
        return inferred(
            {
                "cpu_affinity_policy": "dedicated_local_one_thread_per_core",
                "memory_policy": "bind_local",
                "helper_lane_policy": "same_locality_housekeeping_core",
                "violation_disposition": "not_comparable",
                "fixed_knobs": [
                    "pin_workers_to_one_thread_per_physical_core",
                    "keep_workers_inside_one_locality_domain",
                    "bind_memory_to_worker_locality",
                    "place_helper_lane_on_housekeeping_cpu_in_same_locality",
                ],
            },
            "placement_profile_id:recommended_pinned",
        )
    if profile_id == "adversarial_cross_node":
        return inferred(
            {
                "cpu_affinity_policy": "split_across_locality_domains",
                "memory_policy": "match_cross_domain_placement",
                "helper_lane_policy": "outside_primary_worker_domains",
                "violation_disposition": "not_comparable",
                "fixed_knobs": [
                    "split_workers_across_locality_domains",
                    "avoid_smt_sibling_reuse_inside_primary_worker_set",
                    "match_memory_policy_to_cross_domain_worker_split",
                    "place_helper_lane_outside_primary_worker_domains",
                ],
            },
            "placement_profile_id:adversarial_cross_node",
        )
    return unavailable("placement_profile_id")


def contract_from_manifest_payload(manifest_payload: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(manifest_payload, dict):
        return unavailable("manifest_json")
    provenance = manifest_payload.get("provenance")
    if not isinstance(provenance, dict):
        return unavailable("manifest_json:provenance")
    placement_policy = provenance.get("placement_policy")
    if not isinstance(placement_policy, dict):
        return unavailable("manifest_json:placement_policy")
    execution_contract = placement_policy.get("execution_contract")
    if not isinstance(execution_contract, dict):
        return unavailable("manifest_json:placement_policy.execution_contract")
    contract = {
        "placement_profile_id": placement_policy.get("placement_profile_id"),
        "hardware_class_id": placement_policy.get("hardware_class_id"),
        "availability": placement_policy.get("availability"),
        "command_hint": placement_policy.get("command_hint"),
        "required": placement_policy.get("required"),
        "cpu_affinity_policy": execution_contract.get("cpu_affinity_policy"),
        "memory_policy": execution_contract.get("memory_policy"),
        "helper_lane_policy": execution_contract.get("helper_lane_policy"),
        "violation_disposition": execution_contract.get("violation_disposition"),
        "fixed_knobs": execution_contract.get("fixed_knobs"),
        "required_environment_disclosures": execution_contract.get(
            "required_environment_disclosures"
        ),
    }
    return available(contract, "manifest_json:provenance.placement_policy.execution_contract")


def assess_cpu_affinity(
    expected_policy: str | None,
    declared: dict[str, object],
    observed: dict[str, object],
) -> dict[str, object]:
    declared_value = declared.get("value")
    observed_value = observed.get("value")
    classification = "insufficient_evidence"
    reason = "no CPU affinity disclosure or observation was available"
    if isinstance(declared_value, str) and isinstance(observed_value, str):
        if declared_value == observed_value:
            classification = "controlled"
            reason = "declared CPU affinity matches the observed process affinity"
        else:
            classification = "suspected_drift"
            reason = "declared CPU affinity differs from the observed process affinity"
    elif isinstance(declared_value, str):
        classification = "declared_but_unverified"
        reason = "CPU affinity was declared but no matching process observation was available"
    elif isinstance(observed_value, str):
        if expected_policy == "scheduler_default":
            classification = "ambient_host_default"
            reason = "only host-default process affinity was observed, which is expected for baseline_unpinned"
        else:
            classification = "ambient_observed"
            reason = "process affinity was observed without an explicit disclosure binding it to the benchmark run"
    return {
        "expected_policy": expected_policy,
        "declared": declared,
        "observed": observed,
        "classification": classification,
        "reason": reason,
    }


def assess_numa_balancing(
    declared: dict[str, object],
    observed: dict[str, object],
) -> dict[str, object]:
    declared_value = declared.get("value")
    observed_value = observed.get("value")
    classification = "insufficient_evidence"
    reason = "automatic NUMA balancing state was not available"
    if isinstance(declared_value, str) and isinstance(observed_value, str):
        if declared_value == observed_value:
            classification = "controlled"
            reason = "declared NUMA balancing state matches the host setting"
        else:
            classification = "suspected_drift"
            reason = "declared NUMA balancing state differs from the host setting"
    elif isinstance(observed_value, str):
        classification = "ambient_observed"
        reason = "host NUMA balancing state was observed but not explicitly declared for the run"
    elif isinstance(declared_value, str):
        classification = "declared_but_unverified"
        reason = "NUMA balancing state was declared without a host observation"
    return {
        "declared": declared,
        "observed": observed,
        "classification": classification,
        "reason": reason,
    }


def assess_helper_lane(
    expected_policy: str | None,
    declared_cpu_set: dict[str, object],
    observed_affinity: dict[str, object],
    worker_cpu_set: str | None,
    cpu_nodes: dict[int, int],
) -> dict[str, object]:
    helper_cpu_set = None
    if isinstance(observed_affinity.get("value"), str):
        helper_cpu_set = observed_affinity["value"]
    elif isinstance(declared_cpu_set.get("value"), str):
        helper_cpu_set = declared_cpu_set["value"]
    worker_overlap = None
    worker_nodes = node_list_for_cpu_spec(worker_cpu_set, cpu_nodes)
    helper_nodes = node_list_for_cpu_spec(helper_cpu_set, cpu_nodes)
    if worker_cpu_set and helper_cpu_set:
        overlap = sorted(set(expand_cpu_list(worker_cpu_set)) & set(expand_cpu_list(helper_cpu_set)))
        worker_overlap = compress_index_list(overlap)
    classification = "insufficient_evidence"
    reason = "helper-lane placement was not disclosed or observable"
    if expected_policy == "disclose_host_default":
        if helper_cpu_set:
            classification = "ambient_host_default"
            reason = "helper-lane placement was disclosed/observed without relocation, matching the baseline policy"
    elif helper_cpu_set and worker_cpu_set:
        if worker_overlap:
            classification = "suspected_drift"
            reason = "helper-lane CPUs overlap the primary worker CPU set"
        elif expected_policy == "same_locality_housekeeping_core":
            if helper_nodes and worker_nodes and helper_nodes == worker_nodes:
                classification = "controlled"
                reason = "helper-lane CPUs stay off worker CPUs while remaining in the workers' locality domain"
            else:
                classification = "suspected_drift"
                reason = "helper-lane CPUs do not stay in the same locality domain as the workers"
        elif expected_policy == "outside_primary_worker_domains":
            if helper_nodes and worker_nodes and helper_nodes != worker_nodes:
                classification = "controlled"
                reason = "helper-lane CPUs are outside the workers' primary locality domain"
            elif helper_nodes is None or worker_nodes is None:
                classification = "declared_but_unverified"
                reason = "helper-lane and worker CPU sets are disjoint, but NUMA locality could not be confirmed"
            else:
                classification = "suspected_drift"
                reason = "helper-lane CPUs remain inside the workers' primary locality domain"
    elif helper_cpu_set:
        classification = "declared_but_unverified"
        reason = "helper-lane placement was disclosed or observed without a comparable worker CPU set"
    return {
        "expected_policy": expected_policy,
        "declared_cpu_set": declared_cpu_set,
        "observed_affinity": observed_affinity,
        "worker_cpu_set": worker_cpu_set,
        "worker_nodes": worker_nodes,
        "helper_nodes": helper_nodes,
        "overlap_with_worker_cpu_set": worker_overlap,
        "classification": classification,
        "reason": reason,
    }


def assess_memory_policy(
    expected_policy: str | None,
    declared_policy: dict[str, object],
    observed_numactl: dict[str, object],
    worker_cpu_set: str | None,
    cpu_nodes: dict[int, int],
) -> dict[str, object]:
    observed_value = observed_numactl.get("value")
    observed_policy = observed_value.get("policy") if isinstance(observed_value, dict) else None
    observed_membind = observed_value.get("membind") if isinstance(observed_value, dict) else None
    observed_interleave = observed_value.get("interleave") if isinstance(observed_value, dict) else None
    worker_nodes = node_list_for_cpu_spec(worker_cpu_set, cpu_nodes)
    classification = "insufficient_evidence"
    reason = "memory policy was not disclosed or observable"
    declared_value = declared_policy.get("value")
    if expected_policy == "host_default":
        if isinstance(observed_policy, str) and observed_policy == "default":
            classification = "ambient_host_default" if declared_value is None else "controlled"
            reason = "memory policy remained at the host default"
        elif isinstance(declared_value, str) and declared_value == "host_default":
            classification = "declared_but_unverified"
            reason = "host-default memory policy was declared without an observed numactl policy"
        elif observed_policy:
            classification = "suspected_drift"
            reason = f"observed numactl policy `{observed_policy}` is more specific than the host-default contract"
    elif expected_policy == "bind_local":
        if isinstance(observed_policy, str) and observed_policy in {"bind", "membind"}:
            if worker_nodes and isinstance(observed_membind, str) and observed_membind == worker_nodes:
                classification = "controlled"
                reason = "memory policy binds to the same NUMA nodes as the worker CPU set"
            else:
                classification = "declared_but_unverified"
                reason = "a binding policy was observed, but it could not be matched cleanly to the worker locality"
        elif isinstance(declared_value, str) and declared_value == "bind_local":
            classification = "declared_but_unverified"
            reason = "bind_local was declared without an observable numactl binding"
        elif observed_policy:
            classification = "suspected_drift"
            reason = f"observed numactl policy `{observed_policy}` does not satisfy the bind_local contract"
    elif expected_policy == "match_cross_domain_placement":
        multi_node_binding = False
        if isinstance(observed_interleave, str):
            multi_node_binding = len(expand_cpu_list(observed_interleave)) > 1
        elif isinstance(observed_membind, str):
            multi_node_binding = len(expand_cpu_list(observed_membind)) > 1
        if multi_node_binding:
            classification = "controlled"
            reason = "observed memory policy spans multiple nodes, matching a cross-domain placement contract"
        elif isinstance(declared_value, str) and declared_value == "match_cross_domain_placement":
            classification = "declared_but_unverified"
            reason = "cross-domain memory policy was declared without an observable multi-node policy"
        elif observed_policy:
            classification = "suspected_drift"
            reason = f"observed numactl policy `{observed_policy}` does not show a cross-domain memory layout"
    return {
        "expected_policy": expected_policy,
        "declared_policy": declared_policy,
        "observed_numactl_show": observed_numactl,
        "worker_nodes": worker_nodes,
        "classification": classification,
        "reason": reason,
    }


def assess_irq_placement(
    declared_affinity: dict[str, object],
    default_affinity: dict[str, object],
    groups: dict[str, object],
    irqbalance_active: dict[str, object],
    irqbalance_enabled: dict[str, object],
) -> dict[str, object]:
    declared_value = declared_affinity.get("value")
    default_value = default_affinity.get("value")
    classification = "insufficient_evidence"
    reason = "IRQ placement state was not visible"
    if isinstance(declared_value, str) and isinstance(default_value, str):
        if declared_value == default_value:
            classification = "controlled"
            reason = "declared IRQ affinity matches the host default IRQ affinity"
        else:
            classification = "suspected_drift"
            reason = "declared IRQ affinity differs from the host default IRQ affinity"
    elif isinstance(default_value, str):
        irqbalance_state = irqbalance_active.get("value")
        if irqbalance_state == "active":
            classification = "ambient_observed"
            reason = "host IRQ affinity was observed while irqbalance is active"
        else:
            classification = "ambient_host_default"
            reason = "host IRQ affinity was observed directly from /proc/irq"
    elif isinstance(declared_value, str):
        classification = "declared_but_unverified"
        reason = "IRQ affinity was declared without a host observation"
    return {
        "declared_affinity": declared_affinity,
        "default_affinity": default_affinity,
        "irq_affinity_groups": groups,
        "irqbalance_active": irqbalance_active,
        "irqbalance_enabled": irqbalance_enabled,
        "classification": classification,
        "reason": reason,
    }


def numa_nodes() -> list[dict[str, object]]:
    node_root = Path("/sys/devices/system/node")
    nodes: list[dict[str, object]] = []
    for node_path in sorted(node_root.glob("node[0-9]*")):
        node_id = node_path.name.removeprefix("node")
        if not node_id.isdigit():
            continue
        nodes.append(
            {
                "node_id": int(node_id),
                "cpulist": normalize_scalar(read_text(node_path / "cpulist")),
                "cpumap": normalize_scalar(read_text(node_path / "cpumap")),
            }
        )
    return nodes


def cpufreq_policies() -> list[dict[str, object]]:
    policy_root = Path("/sys/devices/system/cpu/cpufreq")
    policies: list[dict[str, object]] = []
    for policy_path in sorted(policy_root.glob("policy*")):
        policy_name = policy_path.name
        policy: dict[str, object] = {"policy": policy_name}
        for filename in (
            "affected_cpus",
            "related_cpus",
            "scaling_driver",
            "scaling_governor",
            "scaling_available_governors",
            "energy_performance_preference",
            "energy_performance_available_preferences",
            "cpuinfo_max_freq",
            "cpuinfo_min_freq",
            "scaling_max_freq",
            "scaling_min_freq",
        ):
            value = normalize_scalar(read_text(policy_path / filename))
            if value is not None:
                policy[filename] = value
        policies.append(policy)
    return policies


def os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    release: dict[str, str] = {}
    if not path.exists():
        return release
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, raw = line.split("=", 1)
        release[key] = raw.strip().strip('"')
    return release


def boolish(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if value in {"0", "1"}:
        return "enabled" if value == "1" else "disabled"
    return value


def path_field(path: str, *, transform=None) -> dict[str, object]:
    raw = normalize_scalar(read_text(path))
    if raw is None:
        return unavailable(f"fs:{path}")
    value = transform(raw) if transform is not None else raw
    return available(value, f"fs:{path}")


workspace_root = Path(os.environ["WORKSPACE_ROOT"])
artifact_bundle_dir = Path(os.environ["ARTIFACT_BUNDLE_DIR"])
manifest_snapshot: dict[str, object] | None = None
manifest_payload_dict: dict[str, object] | None = None
bundle_json_name = "hardware_discovery_bundle.json"
summary_md_name = "hardware_discovery_summary.md"

manifest_json = os.environ.get("MANIFEST_JSON") or None
if manifest_json:
    manifest_path = Path(manifest_json)
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest_payload = None
    if isinstance(manifest_payload, dict):
        manifest_payload_dict = manifest_payload
        manifest_artifact_names = manifest_payload.get("artifact_names")
        if isinstance(manifest_artifact_names, dict):
            manifest_bundle_name = normalize_scalar(
                manifest_artifact_names.get("hardware_discovery_bundle_json")
            )
            manifest_summary_name = normalize_scalar(
                manifest_artifact_names.get("hardware_discovery_summary_md")
            )
            if manifest_bundle_name is not None:
                bundle_json_name = manifest_bundle_name
            if manifest_summary_name is not None:
                summary_md_name = manifest_summary_name
        manifest_provenance = manifest_payload.get("provenance")
        placement_policy = (
            manifest_provenance.get("placement_policy")
            if isinstance(manifest_provenance, dict)
            else None
        )
        manifest_snapshot = {
            "run_id": manifest_payload.get("run_id"),
            "campaign_id": manifest_payload.get("campaign_id"),
            "row_id": manifest_payload.get("row_id"),
            "fixture_id": manifest_payload.get("fixture_id"),
            "mode": manifest_payload.get("mode"),
            "placement_profile_id": manifest_payload.get("placement_profile_id"),
            "hardware_class_id": manifest_payload.get("hardware_class_id"),
            "build_profile_id": manifest_payload.get("build_profile_id"),
            "artifact_bundle_dir": manifest_payload.get("artifact_bundle_dir"),
            "artifact_bundle_relpath": manifest_payload.get("artifact_bundle_relpath"),
            "source_revision": (
                manifest_payload.get("provenance", {}).get("source_revision")
                if isinstance(manifest_payload.get("provenance"), dict)
                else None
            ),
            "beads_data_hash": (
                manifest_payload.get("provenance", {}).get("beads_data_hash")
                if isinstance(manifest_payload.get("provenance"), dict)
                else None
            ),
            "artifact_names": manifest_artifact_names if isinstance(manifest_artifact_names, dict) else None,
            "placement_policy": {
                "placement_profile_id": (
                    placement_policy.get("placement_profile_id")
                    if isinstance(placement_policy, dict)
                    else None
                ),
                "hardware_class_id": (
                    placement_policy.get("hardware_class_id")
                    if isinstance(placement_policy, dict)
                    else None
                ),
                "availability": (
                    placement_policy.get("availability")
                    if isinstance(placement_policy, dict)
                    else None
                ),
                "command_hint": (
                    placement_policy.get("command_hint")
                    if isinstance(placement_policy, dict)
                    else None
                ),
                "execution_contract": (
                    placement_policy.get("execution_contract")
                    if isinstance(placement_policy, dict)
                    else None
                ),
            },
        }

bundle_json_path = artifact_bundle_dir / bundle_json_name
summary_md_path = artifact_bundle_dir / summary_md_name

lscpu = parse_lscpu()
cpuinfo = parse_cpuinfo()
sibling_sets = unique_cpu_file_values("topology/thread_siblings_list")
llc_domains = list(unique_llc_domains())
numa_layout = numa_nodes()
cpu_nodes = cpu_to_node_map(numa_layout)
cpufreq = cpufreq_policies()
placement_contract = contract_from_manifest_payload(manifest_payload_dict)
if placement_contract["status"] == "unavailable":
    placement_contract = inferred_contract_for_profile(os.environ.get("PLACEMENT_PROFILE_ID") or None)
contract_value = placement_contract.get("value")
expected_cpu_affinity_policy = (
    contract_value.get("cpu_affinity_policy") if isinstance(contract_value, dict) else None
)
expected_memory_policy = (
    contract_value.get("memory_policy") if isinstance(contract_value, dict) else None
)
expected_helper_lane_policy = (
    contract_value.get("helper_lane_policy") if isinstance(contract_value, dict) else None
)

threads_per_core = field_from_sources(
    lscpu=lscpu,
    cpuinfo=cpuinfo,
    lscpu_key="Thread(s) per core",
)
if threads_per_core["status"] == "unavailable" and sibling_sets:
    widths = sorted({len(expand_cpu_list(str(entry["cpus"]))) for entry in sibling_sets})
    if len(widths) == 1:
        threads_per_core = inferred(widths[0], "sysfs:/sys/devices/system/cpu/cpu*/topology/thread_siblings_list")

topology = {
    "sockets": field_from_sources(lscpu=lscpu, cpuinfo=cpuinfo, lscpu_key="Socket(s)"),
    "cores_per_socket": field_from_sources(
        lscpu=lscpu,
        cpuinfo=cpuinfo,
        lscpu_key="Core(s) per socket",
    ),
    "threads_per_core": threads_per_core,
    "online_cpus": path_field("/sys/devices/system/cpu/online"),
    "thread_siblings": (
        available(sibling_sets, "sysfs:/sys/devices/system/cpu/cpu*/topology/thread_siblings_list")
        if sibling_sets
        else unavailable("sysfs:/sys/devices/system/cpu/cpu*/topology/thread_siblings_list")
    ),
    "llc_domains": (
        available(llc_domains, "sysfs:/sys/devices/system/cpu/cpu*/cache/index*/shared_cpu_list")
        if llc_domains
        else unavailable("sysfs:/sys/devices/system/cpu/cpu*/cache/index*/shared_cpu_list")
    ),
}

cpu_section = {
    "model_name": field_from_sources(lscpu=lscpu, cpuinfo=cpuinfo, lscpu_key="Model name", cpuinfo_key="model name"),
    "vendor_id": field_from_sources(lscpu=lscpu, cpuinfo=cpuinfo, lscpu_key="Vendor ID", cpuinfo_key="vendor_id"),
    "cpu_family": field_from_sources(lscpu=lscpu, cpuinfo=cpuinfo, cpuinfo_key="cpu family"),
    "model": field_from_sources(lscpu=lscpu, cpuinfo=cpuinfo, cpuinfo_key="model"),
    "stepping": field_from_sources(lscpu=lscpu, cpuinfo=cpuinfo, cpuinfo_key="stepping"),
    "microcode": field_from_sources(
        lscpu=lscpu,
        cpuinfo=cpuinfo,
        cpuinfo_key="microcode",
        sysfs_paths=["/sys/devices/system/cpu/cpu0/microcode/version"],
    ),
    "bios_version": path_field("/sys/class/dmi/id/bios_version"),
    "bios_date": path_field("/sys/class/dmi/id/bios_date"),
    "topology": topology,
}

numa_section = {
    "nodes": available(numa_layout, "sysfs:/sys/devices/system/node/node*/cpulist")
    if numa_layout
    else unavailable("sysfs:/sys/devices/system/node/node*/cpulist"),
}

power_section = {
    "cpufreq_policies": available(cpufreq, "sysfs:/sys/devices/system/cpu/cpufreq/policy*")
    if cpufreq
    else unavailable("sysfs:/sys/devices/system/cpu/cpufreq/policy*"),
    "boost": path_field("/sys/devices/system/cpu/cpufreq/boost", transform=boolish),
    "intel_no_turbo": path_field("/sys/devices/system/cpu/intel_pstate/no_turbo", transform=boolish),
    "amd_pstate_status": path_field("/sys/devices/system/cpu/amd_pstate/status"),
}

memory_section = {
    "thp_enabled": path_field("/sys/kernel/mm/transparent_hugepage/enabled"),
    "thp_defrag": path_field("/sys/kernel/mm/transparent_hugepage/defrag"),
    "thp_shmem_enabled": path_field("/sys/kernel/mm/transparent_hugepage/shmem_enabled"),
    "numa_balancing": path_field("/proc/sys/kernel/numa_balancing", transform=boolish),
}

scheduler_section = {
    "sched_autogroup_enabled": path_field("/proc/sys/kernel/sched_autogroup_enabled", transform=boolish),
    "sched_schedstats": path_field("/proc/sys/kernel/sched_schedstats", transform=boolish),
    "sched_rr_timeslice_ms": path_field("/proc/sys/kernel/sched_rr_timeslice_ms"),
}

host_section = {
    "hostname": available(platform.node(), "platform.node"),
    "kernel_release": available(platform.release(), "platform.release"),
    "machine_arch": available(platform.machine(), "platform.machine"),
    "os_release": available(os_release(), "fs:/etc/os-release")
    if os_release()
    else unavailable("fs:/etc/os-release"),
}

declared_worker_affinity = env_field("CPU_AFFINITY_MASK", normalizer=normalize_index_list)
benchmark_pid = parse_int_env("BENCHMARK_PID")
ambient_worker_affinity = process_affinity()
if benchmark_pid is not None:
    observed_worker_affinity = process_affinity(benchmark_pid)
elif (
    declared_worker_affinity["status"] == "available"
    and expected_cpu_affinity_policy == "scheduler_default"
):
    observed_worker_affinity = ambient_worker_affinity
elif declared_worker_affinity["status"] == "available":
    observed_worker_affinity = unavailable(
        "env:BENCHMARK_PID",
        "CPU_AFFINITY_MASK was declared but BENCHMARK_PID was not provided for direct verification",
    )
else:
    observed_worker_affinity = ambient_worker_affinity
worker_cpu_set = None
if isinstance(declared_worker_affinity.get("value"), str):
    worker_cpu_set = declared_worker_affinity["value"]
elif isinstance(observed_worker_affinity.get("value"), str):
    worker_cpu_set = observed_worker_affinity["value"]
declared_helper_lane = env_field("HELPER_LANE_CPU_SET", normalizer=normalize_index_list)
helper_lane_pid = parse_int_env("HELPER_LANE_PID")
observed_helper_lane = process_affinity(helper_lane_pid) if helper_lane_pid is not None else unavailable("env:HELPER_LANE_PID")
declared_memory_policy = env_field("MEMORY_POLICY")
declared_numa_balancing = env_field("NUMA_BALANCING_STATE", normalizer=boolish)
declared_irq_affinity = env_field("IRQ_AFFINITY_MASK", normalizer=normalize_index_list)
irqbalance_active = command_field(
    ["systemctl", "is-active", "irqbalance.service"],
    "command:systemctl is-active irqbalance.service",
)
irqbalance_enabled = command_field(
    ["systemctl", "is-enabled", "irqbalance.service"],
    "command:systemctl is-enabled irqbalance.service",
)
irq_default_affinity = path_field("/proc/irq/default_smp_affinity_list", transform=normalize_index_list)
if irq_default_affinity["status"] == "unavailable":
    irq_default_affinity = path_field("/proc/irq/default_smp_affinity", transform=normalize_index_list)
irq_groups = irq_affinity_groups()
numactl_show = parse_numactl_show()
side_condition_cpu_affinity = assess_cpu_affinity(
    expected_cpu_affinity_policy,
    declared_worker_affinity,
    observed_worker_affinity,
)
side_condition_helper_lane = assess_helper_lane(
    expected_helper_lane_policy,
    declared_helper_lane,
    observed_helper_lane,
    normalize_index_list(os.environ.get("WORKER_CPU_SET") or worker_cpu_set),
    cpu_nodes,
)
side_condition_memory_policy = assess_memory_policy(
    expected_memory_policy,
    declared_memory_policy,
    numactl_show,
    normalize_index_list(os.environ.get("WORKER_CPU_SET") or worker_cpu_set),
    cpu_nodes,
)
side_condition_numa_balancing = assess_numa_balancing(
    declared_numa_balancing,
    memory_section["numa_balancing"],
)
side_condition_irq = assess_irq_placement(
    declared_irq_affinity,
    irq_default_affinity,
    irq_groups,
    irqbalance_active,
    irqbalance_enabled,
)
side_condition_classifications = [
    side_condition_cpu_affinity["classification"],
    side_condition_helper_lane["classification"],
    side_condition_memory_policy["classification"],
    side_condition_numa_balancing["classification"],
    side_condition_irq["classification"],
]
side_condition_comparability = classify_comparability(side_condition_classifications)
side_condition_notes: list[str] = []
if side_condition_comparability == "suspected":
    side_condition_notes.append(
        "At least one placement-sensitive side condition looks inconsistent with the declared contract; treat related benchmark claims as not_comparable until the drift is explained."
    )
if helper_lane_pid is None and normalize_scalar(os.environ.get("HELPER_LANE_CPU_SET")) is None:
    side_condition_notes.append(
        "No HELPER_LANE_PID or HELPER_LANE_CPU_SET was provided, so helper-lane placement can only be inferred from ambient host state."
    )
if benchmark_pid is None and declared_worker_affinity["status"] == "available":
    side_condition_notes.append(
        "CPU_AFFINITY_MASK was declared without BENCHMARK_PID, so worker CPU placement is treated as declared-but-unverified instead of being compared against the capture script's own affinity."
    )
if normalize_scalar(os.environ.get("MEMORY_POLICY")) is None and numactl_show["status"] == "unavailable":
    side_condition_notes.append(
        "No MEMORY_POLICY disclosure or numactl observation was available, so memory-policy effects remain underdetermined."
    )

critical_fields = {
    "cpu.model_name": cpu_section["model_name"],
    "cpu.microcode": cpu_section["microcode"],
    "cpu.topology.sockets": topology["sockets"],
    "cpu.topology.threads_per_core": topology["threads_per_core"],
    "cpu.topology.thread_siblings": topology["thread_siblings"],
    "cpu.topology.llc_domains": topology["llc_domains"],
    "numa.nodes": numa_section["nodes"],
    "power.cpufreq_policies": power_section["cpufreq_policies"],
    "power.boost": power_section["boost"],
    "memory.thp_enabled": memory_section["thp_enabled"],
    "memory.numa_balancing": memory_section["numa_balancing"],
    "scheduler.sched_autogroup_enabled": scheduler_section["sched_autogroup_enabled"],
}

available_fields = sorted(name for name, payload in critical_fields.items() if payload["status"] == "available")
inferred_fields = sorted(name for name, payload in critical_fields.items() if payload["status"] == "inferred")
missing_fields = sorted(name for name, payload in critical_fields.items() if payload["status"] == "unavailable")

notes: list[str] = []
if missing_fields:
    notes.append(
        "Some identity fields were unavailable on this host; downstream tooling should treat the hardware signature as partial."
    )
if not os.environ.get("PLACEMENT_PROFILE_ID"):
    notes.append("placement_profile_id was not provided, so the bundle is attached to a run_id but not a placement profile.")
if not os.environ.get("HARDWARE_CLASS_ID"):
    notes.append("hardware_class_id was not provided, so the bundle records discovery facts without asserting taxonomy membership.")
if not os.environ.get("MANIFEST_JSON"):
    notes.append("manifest_json was not provided, so provenance points at the artifact bundle directory rather than a manifest file.")
elif manifest_snapshot is None:
    notes.append("manifest_json was provided but could not be parsed, so the bundle could not compare run-identity fields against the manifest contract.")
else:
    for env_key, manifest_key in (
        ("RUN_ID", "run_id"),
        ("PLACEMENT_PROFILE_ID", "placement_profile_id"),
        ("HARDWARE_CLASS_ID", "hardware_class_id"),
        ("SOURCE_REVISION", "source_revision"),
        ("BEADS_HASH", "beads_data_hash"),
    ):
        env_value = os.environ.get(env_key) or None
        manifest_value = manifest_snapshot.get(manifest_key)
        if env_value and manifest_value and env_value != manifest_value:
            notes.append(
                f"{env_key.lower()}={env_value} differs from manifest {manifest_key}={manifest_value}; downstream reports should treat this as an identity mismatch."
            )
if side_condition_notes:
    notes.extend(side_condition_notes)

signature_payload = {
    "hostname": host_section["hostname"]["value"],
    "kernel_release": host_section["kernel_release"]["value"],
    "model_name": cpu_section["model_name"]["value"],
    "microcode": cpu_section["microcode"]["value"],
    "bios_version": cpu_section["bios_version"]["value"],
    "bios_date": cpu_section["bios_date"]["value"],
    "sockets": topology["sockets"]["value"],
    "cores_per_socket": topology["cores_per_socket"]["value"],
    "threads_per_core": topology["threads_per_core"]["value"],
    "online_cpus": topology["online_cpus"]["value"],
    "thread_siblings": topology["thread_siblings"]["value"],
    "llc_domains": topology["llc_domains"]["value"],
    "numa_nodes": numa_section["nodes"]["value"],
    "boost": power_section["boost"]["value"],
    "intel_no_turbo": power_section["intel_no_turbo"]["value"],
    "amd_pstate_status": power_section["amd_pstate_status"]["value"],
    "thp_enabled": memory_section["thp_enabled"]["value"],
    "thp_defrag": memory_section["thp_defrag"]["value"],
    "numa_balancing": memory_section["numa_balancing"]["value"],
    "sched_autogroup_enabled": scheduler_section["sched_autogroup_enabled"]["value"],
    "sched_schedstats": scheduler_section["sched_schedstats"]["value"],
}
hardware_signature = hashlib.sha256(
    json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

bundle = {
    "schema_version": "fsqlite-e2e.db300.hardware_discovery_bundle.v1",
    "bead_id": os.environ["BEAD_ID"],
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "command_entrypoint": os.environ["SCRIPT_ENTRYPOINT"],
    "hardware_signature": hardware_signature,
    "run_identity": {
        "run_id": os.environ["RUN_ID"],
        "campaign_id": os.environ["CAMPAIGN_ID"],
        "row_id": os.environ.get("ROW_ID") or None,
        "fixture_id": os.environ.get("FIXTURE_ID") or None,
        "mode_id": os.environ.get("MODE_ID") or None,
        "placement_profile_id": os.environ.get("PLACEMENT_PROFILE_ID") or None,
        "hardware_class_id": os.environ.get("HARDWARE_CLASS_ID") or None,
        "build_profile_id": os.environ.get("BUILD_PROFILE_ID") or None,
        "artifact_bundle_dir": str(artifact_bundle_dir),
        "artifact_bundle_relpath": os.environ.get("ARTIFACT_BUNDLE_RELPATH") or None,
        "manifest_json": os.environ.get("MANIFEST_JSON") or None,
        "manifest_snapshot": manifest_snapshot,
        "source_revision": os.environ.get("SOURCE_REVISION") or None,
        "beads_data_hash": os.environ.get("BEADS_HASH") or None,
    },
    "host": host_section,
    "cpu": cpu_section,
    "numa": numa_section,
    "power": power_section,
    "memory": memory_section,
    "scheduler": scheduler_section,
    "side_conditions": {
        "placement_contract": placement_contract,
        "comparability_assessment": side_condition_comparability,
        "disclosures": {
            "cpu_affinity_mask": side_condition_cpu_affinity,
            "memory_policy": side_condition_memory_policy,
            "helper_lane_cpu_set": side_condition_helper_lane,
            "numa_balancing_state": side_condition_numa_balancing,
        },
        "irq_placement": side_condition_irq,
        "notes": side_condition_notes,
    },
    "discovery": {
        "available_fields": available_fields,
        "inferred_fields": inferred_fields,
        "missing_fields": missing_fields,
        "notes": notes,
    },
    "artifacts": {
        "bundle_json": bundle_json_path.name,
        "summary_markdown": summary_md_path.name,
    },
}

bundle_json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

summary_lines = [
    f"# Hardware Discovery Bundle for {os.environ['BEAD_ID']}",
    "",
    f"- run_id: `{os.environ['RUN_ID']}`",
    f"- hardware_signature: `{hardware_signature}`",
    f"- artifact_bundle_dir: `{artifact_bundle_dir}`",
    f"- placement_profile_id: `{os.environ.get('PLACEMENT_PROFILE_ID') or 'unbound'}`",
    f"- hardware_class_id: `{os.environ.get('HARDWARE_CLASS_ID') or 'unbound'}`",
    f"- source_revision: `{os.environ.get('SOURCE_REVISION') or 'unknown'}`",
    "",
    "## Identity Snapshot",
    "",
    f"- cpu_model: `{cpu_section['model_name']['value'] or 'unavailable'}`",
    f"- microcode: `{cpu_section['microcode']['value'] or 'unavailable'}`",
    f"- bios_version: `{cpu_section['bios_version']['value'] or 'unavailable'}`",
    f"- sockets: `{topology['sockets']['value'] or 'unavailable'}`",
    f"- cores_per_socket: `{topology['cores_per_socket']['value'] or 'unavailable'}`",
    f"- threads_per_core: `{topology['threads_per_core']['value'] or 'unavailable'}`",
    f"- online_cpus: `{topology['online_cpus']['value'] or 'unavailable'}`",
    f"- numa_nodes: `{len(numa_layout) if numa_layout else 0}`",
    f"- boost: `{power_section['boost']['value'] or 'unavailable'}`",
    f"- intel_no_turbo: `{power_section['intel_no_turbo']['value'] or 'unavailable'}`",
    f"- amd_pstate_status: `{power_section['amd_pstate_status']['value'] or 'unavailable'}`",
    f"- thp_enabled: `{memory_section['thp_enabled']['value'] or 'unavailable'}`",
    f"- thp_defrag: `{memory_section['thp_defrag']['value'] or 'unavailable'}`",
    f"- numa_balancing: `{memory_section['numa_balancing']['value'] or 'unavailable'}`",
    f"- sched_autogroup_enabled: `{scheduler_section['sched_autogroup_enabled']['value'] or 'unavailable'}`",
    f"- sched_schedstats: `{scheduler_section['sched_schedstats']['value'] or 'unavailable'}`",
    "",
    "## Side Conditions",
    "",
    f"- comparability_assessment: `{side_condition_comparability}`",
    f"- cpu_affinity_mask: `{side_condition_cpu_affinity['classification']}` ({side_condition_cpu_affinity['reason']})",
    f"- helper_lane_cpu_set: `{side_condition_helper_lane['classification']}` ({side_condition_helper_lane['reason']})",
    f"- memory_policy: `{side_condition_memory_policy['classification']}` ({side_condition_memory_policy['reason']})",
    f"- numa_balancing_state: `{side_condition_numa_balancing['classification']}` ({side_condition_numa_balancing['reason']})",
    f"- irq_placement: `{side_condition_irq['classification']}` ({side_condition_irq['reason']})",
    "",
    "## Discovery Quality",
    "",
    f"- available_fields: `{','.join(available_fields) if available_fields else 'none'}`",
    f"- inferred_fields: `{','.join(inferred_fields) if inferred_fields else 'none'}`",
    f"- missing_fields: `{','.join(missing_fields) if missing_fields else 'none'}`",
]

if notes:
    summary_lines.extend(["", "## Notes", ""])
    summary_lines.extend(f"- {note}" for note in notes)

summary_md_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print(json.dumps({"bundle_json": str(bundle_json_path), "summary_markdown": str(summary_md_path)}))
PY
