# sample_sqlite_db_files

Local-only corpus of **real SQLite database files** used for end-to-end demos and benchmarking.

## Safety Rules

- NEVER run tests or demos against `/dp/...` originals.
- Always take a consistent snapshot using SQLite's backup API (`sqlite3 ... ".backup '...'"`).
- Treat `golden/` as immutable. Do work on copies in `working/`.

## What Goes Where

- `golden/`: immutable golden copies created from `/dp` sources (directory ignored by git).
- `working/`: ephemeral mutable copies created per run (directory ignored by git).
- `metadata/`: tracked JSON/markdown describing each golden DB (schema summary, stats, etc.).
- `checksums.sha256`: tracked master checksum file for golden DBs (populated by a later task).

## Snapshot Copy (Recommended)

Prefer `.backup` over `cp` because some source DBs may have active WAL/SHM files.

Example:

```bash
src="/dp/asupersync/.beads/beads.db"
dst="sample_sqlite_db_files/golden/asupersync.db"
sqlite3 "$src" ".backup '$dst'"
sqlite3 "$dst" "PRAGMA integrity_check;"
```

## Git Hygiene

This repo must never commit DB bytes from `/dp`. Only metadata + checksums are tracked.

