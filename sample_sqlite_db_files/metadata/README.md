# metadata/

Tracked metadata describing each DB in `../golden/`.

Examples of useful fields (format TBD by the profiler task):
- source path (original `/dp/...` path)
- golden sha256 + size
- `PRAGMA page_size`, `PRAGMA journal_mode`, `PRAGMA user_version`
- table list + row counts
- freelist/page stats

