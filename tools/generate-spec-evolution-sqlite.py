#!/usr/bin/env python3
"""
Generate a SQLite database for the FrankenSQLite spec evolution visualization.

This moves the heavy data blobs out of the HTML and into a queryable DB, using
the same "sql.js WASM loads a .sqlite3 file" approach used in /dp/beads_viewer.

Inputs:
  - spec_evolution_data_v1.json.gz (commits/base_doc/patches)
  - spec_evolution_classification_v1.json (CLASS_EARLY/MIDDLE/LATE extracted)

Outputs (defaults):
  - spec_evolution_v1.sqlite3
  - spec_evolution_v1.sqlite3.config.json  (hash for OPFS caching / cache busting)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
from typing import Any, Iterable


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compute_dataset_hash(dataset: dict[str, Any]) -> str:
    # Must match visualization's computeDatasetHash():
    # `${base_doc.length}|${commitHashes.join(",")}|${patchSizes.join(",")}`
    base_len = len(str(dataset.get("base_doc", "") or ""))
    commit_hashes = [str((c or {}).get("hash", "") or "") for c in (dataset.get("commits") or [])]
    patches = dataset.get("patches") or []
    patch_sizes = [str(len(str(p or ""))) for p in patches]
    basis = f"{base_len}|{','.join(commit_hashes)}|{','.join(patch_sizes)}"
    return sha256_hex(basis.encode("utf-8"))


def pick_primary(labels: Iterable[int]) -> int:
    s = set(int(x) for x in labels)
    # Must match visualization's pickPrimary() priority ordering.
    for k in (2, 3, 1, 4, 8, 7, 6, 5, 9):
        if k in s:
            return k
    return 10


def uniq_ints(xs: Iterable[Any]) -> list[int]:
    out: set[int] = set()
    for x in xs:
        try:
            n = int(x)
        except Exception:
            continue
        out.add(n)
    return sorted([n for n in out if isinstance(n, int)])


def derive_buckets_for_group(group: dict[str, Any], commit_subject: str) -> tuple[list[int], int]:
    # Mirrors visualization's deriveBucketsForGroup(group, commitSubject).
    categories = group.get("categories")
    if isinstance(categories, list) and all(isinstance(x, int) for x in categories):
        labels = uniq_ints(categories)
        primary_raw = group.get("primary_category")
        primary = int(primary_raw) if isinstance(primary_raw, int) else pick_primary(labels)
        return labels, primary

    labels: set[int] = set()
    evidence = group.get("evidence")
    evidence_list = evidence if isinstance(evidence, list) else []
    s = f"{commit_subject or ''} {group.get('summary') or ''} {' '.join(str(x) for x in evidence_list)}".lower()

    # Tag-driven mapping (from middle/late agents)
    tags = categories if isinstance(categories, list) else []
    for t in tags:
        tt = str(t).lower()
        if ("doc meta" in tt) or ("summary_update" in tt):
            labels.add(5)
        if "clarification" in tt:
            labels.add(9)
        if ("spec expansion" in tt) or ("addition" in tt):
            labels.add(6)
        if "architecture" in tt:
            labels.add(4)
        if "api/interface" in tt:
            labels.add(4)
        if "sql semantics" in tt:
            labels.add(2)
        if "file format" in tt:
            labels.add(2)
        if "durability" in tt:
            labels.add(7)
        if "concurrency" in tt:
            labels.add(7)
        if "performance" in tt:
            labels.add(7)
        if "math/modeling" in tt:
            labels.add(8)
        if ("correctness fix" in tt) or ("correction" in tt):
            labels.add(1)
        if "requirement_change" in tt:
            labels.add(4)

    # Content heuristics to refine buckets.
    if (
        ("sqlite" in s)
        or ("wal" in s)
        or ("wal-index" in s)
        or ("btree" in s)
        or ("vdbe" in s)
        or ("fts5" in s)
        or ("lemon" in s)
        or ("parse.y" in s)
    ):
        labels.add(2)
    if (
        ("asupersync" in s)
        or ("cx" in s)
        or ("virtualtcp" in s)
        or ("region" in s)
        or ("spawn_blocking" in s)
        or ("deadline" in s)
    ):
        labels.add(3)
    if (
        ("bocpd" in s)
        or ("conformal" in s)
        or ("e-process" in s)
        or ("evalue" in s)
        or ("vo i" in s)
        or ("gf(256)" in s)
        or ("raptorq" in s)
        or ("martingale" in s)
    ):
        labels.add(8)
    if (
        ("cache" in s)
        or ("prefetch" in s)
        or ("alignment" in s)
        or ("atomic" in s)
        or ("acquire" in s)
        or ("release" in s)
        or ("bulkhead" in s)
        or ("rate_limit" in s)
        or ("shard" in s)
        or ("cache-line" in s)
    ):
        labels.add(7)
    if (
        ("renumber" in s)
        or ("footer" in s)
        or ("document version" in s)
        or ("typo" in s)
        or ("wording tweak" in s)
    ):
        labels.add(5)
    if s.startswith("add ") or ("added " in s) or ("introduced " in s) or ("defined " in s) or ("expanded " in s):
        labels.add(6)
    if ("clarif" in s) or ("explain" in s) or ("note" in s):
        labels.add(9)
    if ("fix " in s) or ("fixed " in s) or ("correct" in s) or ("arithmetic" in s) or ("inversion" in s) or ("swapped" in s):
        labels.add(1)
    if ("rework" in s) or ("redesign" in s) or ("protocol" in s) or ("invariant" in s) or ("formal model" in s):
        labels.add(4)

    if not labels:
        labels.add(10)

    labels_arr = sorted(labels)
    primary = pick_primary(labels_arr)
    return labels_arr, primary


def normalize_classification(classification: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def add_entries(entries: Any, source: str) -> None:
        if not isinstance(entries, list):
            return
        for e in entries:
            if not isinstance(e, dict):
                continue
            commit = e.get("commit")
            if not commit:
                continue
            # Later sources override earlier (matches viz behavior).
            out[str(commit)] = {**e, "_source": source}

    add_entries(classification.get("early"), "early")
    add_entries(classification.get("middle"), "middle")
    add_entries(classification.get("late"), "late")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="spec_evolution_data_v1.json.gz")
    ap.add_argument("--classification", default="spec_evolution_classification_v1.json")
    ap.add_argument("--out-db", default="spec_evolution_v1.sqlite3")
    ap.add_argument("--out-config", default="spec_evolution_v1.sqlite3.config.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.dataset):
        print(f"[db] dataset not found: {args.dataset}", file=sys.stderr)
        return 1
    if not os.path.exists(args.classification):
        print(f"[db] classification not found: {args.classification}", file=sys.stderr)
        return 1

    with gzip.open(args.dataset, "rb") as f:
        dataset = json.loads(f.read().decode("utf-8"))

    with open(args.classification, "r", encoding="utf-8") as f:
        classification = json.load(f)

    dataset_hash = compute_dataset_hash(dataset)
    cls_norm = normalize_classification(classification)
    # Stable classification hash: only the arrays, not extraction timestamps.
    cls_hash = sha256_hex(
        json.dumps(
            {
                "early": classification.get("early") or [],
                "middle": classification.get("middle") or [],
                "late": classification.get("late") or [],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    db_hash = sha256_hex(f"{dataset_hash}|{cls_hash}".encode("utf-8"))

    commits = dataset.get("commits") or []
    patches = dataset.get("patches") or []
    base_doc = str(dataset.get("base_doc", "") or "")

    if args.dry_run:
        print("[db] Dry run:")
        print(f"  dataset_hash: {dataset_hash}")
        print(f"  classification_hash: {cls_hash}")
        print(f"  db_hash: {db_hash}")
        print(f"  commits: {len(commits)}")
        print(f"  patches: {len(patches)}")
        print(f"  base_doc chars: {len(base_doc)}")
        return 0

    # Create DB file.
    conn = sqlite3.connect(args.out_db)
    try:
        conn.execute("PRAGMA journal_mode=OFF;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE;")

        cur = conn.cursor()
        cur.executescript(
            """
            DROP TABLE IF EXISTS meta;
            DROP TABLE IF EXISTS commits;
            DROP TABLE IF EXISTS patches;
            DROP TABLE IF EXISTS base_doc;
            DROP TABLE IF EXISTS change_groups;

            CREATE TABLE meta(
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );

            CREATE TABLE commits(
              idx INTEGER PRIMARY KEY,
              hash TEXT NOT NULL,
              short TEXT NOT NULL,
              date_iso TEXT NOT NULL,
              author TEXT NOT NULL,
              subject TEXT NOT NULL,
              url TEXT NOT NULL,
              add_lines INTEGER NOT NULL,
              del_lines INTEGER NOT NULL,
              impact INTEGER NOT NULL,
              labels_json TEXT NOT NULL,
              primary_bucket INTEGER NOT NULL,
              group_count INTEGER NOT NULL,
              has_classification INTEGER NOT NULL
            );

            CREATE UNIQUE INDEX commits_hash_uq ON commits(hash);

            CREATE TABLE patches(
              idx INTEGER PRIMARY KEY,
              patch TEXT NOT NULL
            );

            CREATE TABLE base_doc(
              id INTEGER PRIMARY KEY CHECK (id=1),
              text TEXT NOT NULL
            );

            CREATE TABLE change_groups(
              commit_hash TEXT NOT NULL,
              group_idx INTEGER NOT NULL,
              summary TEXT NOT NULL,
              confidence REAL NOT NULL,
              evidence_json TEXT NOT NULL,
              changed_headings_json TEXT NOT NULL,
              labels_json TEXT NOT NULL,
              primary_bucket INTEGER NOT NULL,
              source TEXT NOT NULL,
              FOREIGN KEY(commit_hash) REFERENCES commits(hash)
            );

            CREATE INDEX change_groups_commit_hash_ix ON change_groups(commit_hash);
            """
        )

        # Meta
        meta_items = {
            "schema_version": "1",
            "dataset_hash": dataset_hash,
            "classification_hash": cls_hash,
            "db_hash": db_hash,
            "commit_count": str(len(commits)),
            "patch_count": str(len(patches)),
            "spec_path": str(dataset.get("spec_path") or ""),
            "base_commit": str(dataset.get("base_commit") or ""),
            "generated_from": str(dataset.get("generated_from") or "unknown"),
        }
        cur.executemany(
            "INSERT INTO meta(key, value) VALUES(?, ?)",
            list(meta_items.items()),
        )

        # Base doc
        cur.execute("INSERT INTO base_doc(id, text) VALUES(1, ?)", (base_doc,))

        # Patches
        cur.executemany(
            "INSERT INTO patches(idx, patch) VALUES(?, ?)",
            [(i, str(p or "")) for i, p in enumerate(patches)],
        )

        # Commits + change groups
        for idx, c in enumerate(commits):
            if not isinstance(c, dict):
                continue
            h = str(c.get("hash") or "")
            subj = str(c.get("subject") or "")
            url = f"https://github.com/Dicklesworthstone/frankensqlite/commit/{h}"

            entry = cls_norm.get(h)
            has_cls = 1 if entry else 0
            source = str((entry or {}).get("_source") or "none")
            change_groups_raw = (entry or {}).get("change_groups") or []

            groups_out = []
            labels_union: set[int] = set()

            if isinstance(change_groups_raw, list):
                for gi, g in enumerate(change_groups_raw):
                    if not isinstance(g, dict):
                        continue
                    labels, primary = derive_buckets_for_group(g, subj)
                    for lb in labels:
                        labels_union.add(int(lb))
                    summary = str(g.get("summary") or "")
                    evidence = g.get("evidence")
                    changed_headings = g.get("changed_headings")
                    evidence_list = evidence if isinstance(evidence, list) else []
                    changed_list = changed_headings if isinstance(changed_headings, list) else []
                    conf_raw = g.get("confidence")
                    conf = float(conf_raw) if isinstance(conf_raw, (int, float)) else 0.55

                    groups_out.append(
                        (
                            h,
                            gi,
                            summary,
                            conf,
                            json.dumps(evidence_list, ensure_ascii=False),
                            json.dumps(changed_list, ensure_ascii=False),
                            json.dumps(labels),
                            int(primary),
                            source,
                        )
                    )

            # Commit-level labels + primary must match viz behavior.
            commit_labels = sorted(labels_union)
            commit_primary = pick_primary(commit_labels if commit_labels else [10])

            cur.execute(
                """
                INSERT INTO commits(
                  idx, hash, short, date_iso, author, subject, url,
                  add_lines, del_lines, impact,
                  labels_json, primary_bucket, group_count, has_classification
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idx,
                    h,
                    str(c.get("short") or ""),
                    str(c.get("dateIso") or ""),
                    str(c.get("author") or ""),
                    subj,
                    url,
                    int(c.get("add") or 0),
                    int(c.get("del") or 0),
                    int(c.get("impact") or ((c.get("add") or 0) + (c.get("del") or 0))),
                    json.dumps(commit_labels),
                    int(commit_primary),
                    int(len(groups_out)),
                    int(has_cls),
                ),
            )

            if groups_out:
                cur.executemany(
                    """
                    INSERT INTO change_groups(
                      commit_hash, group_idx, summary, confidence,
                      evidence_json, changed_headings_json,
                      labels_json, primary_bucket, source
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    groups_out,
                )

        conn.commit()

    finally:
        conn.close()

    cfg = {
        "schema_version": 1,
        "hash": db_hash,
        "dataset_hash": dataset_hash,
        "classification_hash": cls_hash,
        "commit_count": len(commits),
        "patch_count": len(patches),
        "db_file": os.path.basename(args.out_db),
    }
    with open(args.out_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[db] Wrote {args.out_db}")
    print(f"[db] Wrote {args.out_config}")
    print(f"[db] db_hash={db_hash} dataset_hash={dataset_hash} classification_hash={cls_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
