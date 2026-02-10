# sample_sqlite_db_files/

Fixture staging area for FrankenSQLite E2E integration tests and demos.

## Directory Layout

```
sample_sqlite_db_files/
  golden/      Read-only reference copies of real SQLite databases
  work/        Mutable working copies created at test time (gitignored)
  manifests/   JSON manifests with metadata, checksums, and run results
```

## Safety Rules

1. **Never run tests against originals in `/dp/`.** Always copy to `golden/` first via `sqlite3 <src> ".backup '<dst>'"`.
2. **Golden copies are read-only.** After placing a file in `golden/`, mark it `chmod -w`. Tests must never modify golden files.
3. **Always operate on `work/` copies.** Before any mutating operation, copy the golden file into `work/` and operate on the copy.
4. **Verify checksums before use.** Each golden file has a SHA-256 recorded in its manifest. If the checksum changes, fail fast.
5. **Database bytes are gitignored.** Only manifests (`.json`), this README, and `.gitignore` are committed. Raw `.db`/`.sqlite` files are never checked in.

## Adding a New Fixture

```bash
# 1. Back up the source database (safe, read-only operation)
sqlite3 /dp/some-project/.beads/beads.db ".backup 'sample_sqlite_db_files/golden/some-project-beads.db'"

# 2. Make it read-only
chmod -w sample_sqlite_db_files/golden/some-project-beads.db

# 3. Generate checksum
sha256sum sample_sqlite_db_files/golden/some-project-beads.db

# 4. Add manifest entry in manifests/
```

## Manifest Format

Each manifest file in `manifests/` is a JSON array:

```json
[
  {
    "name": "some-project-beads",
    "source": "/dp/some-project/.beads/beads.db",
    "golden_path": "golden/some-project-beads.db",
    "sha256": "abc123...",
    "size_bytes": 12345,
    "table_count": 3,
    "tags": ["beads", "small"]
  }
]
```
