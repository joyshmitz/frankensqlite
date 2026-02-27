# Oracle Preflight Doctor (bd-2yqp6.2.5)

run_id: `bd-2yqp6.2.5-green-4242`
trace_id: `trace-bd-2yqp6.2.5-green`
scenario_id: `DIFF-ORACLE-PREFLIGHT-B5-GREEN`
seed: `4242`
outcome: `green`
certifying: `true`
timing_ms: `4`
oracle_binary: `/data/projects/frankensqlite/artifacts/bd-2yqp6.2.5/bd-2yqp6.2.5-20260227T011853Z-4242/doctor-workspace/sqlite3`
oracle_version: `3.52.0-test`
expected_sqlite_version_prefix: `3.52.0`
fixtures_dir: `/data/projects/frankensqlite/artifacts/bd-2yqp6.2.5/bd-2yqp6.2.5-20260227T011853Z-4242/doctor-workspace/crates/fsqlite-harness/conformance`
fixture_manifest_path: `/data/projects/frankensqlite/artifacts/bd-2yqp6.2.5/bd-2yqp6.2.5-20260227T011853Z-4242/doctor-workspace/corpus_manifest.toml`
fixture_json_files_seen: `1`
fixture_entries_ingested: `1`
fixture_sql_statements_ingested: `3`
skipped_fixture_files: `0`
first_failure: `none`

## Findings

- none

## Replay

`cargo run -p fsqlite-harness --bin oracle_preflight_doctor_runner -- --workspace-root /data/projects/frankensqlite/artifacts/bd-2yqp6.2.5/bd-2yqp6.2.5-20260227T011853Z-4242/doctor-workspace --fixtures-dir /data/projects/frankensqlite/artifacts/bd-2yqp6.2.5/bd-2yqp6.2.5-20260227T011853Z-4242/doctor-workspace/crates/fsqlite-harness/conformance --fixture-manifest-path /data/projects/frankensqlite/artifacts/bd-2yqp6.2.5/bd-2yqp6.2.5-20260227T011853Z-4242/doctor-workspace/corpus_manifest.toml --run-id bd-2yqp6.2.5-green-4242 --trace-id trace-bd-2yqp6.2.5-green --scenario-id DIFF-ORACLE-PREFLIGHT-B5-GREEN --seed 4242 --generated-unix-ms 1700000000000 --min-fixture-json-files 1 --min-fixture-entries 1 --min-fixture-sql-statements 2 --expected-sqlite-version-prefix 3.52.0 --expected-subject-identity frankensqlite --expected-reference-identity csqlite-oracle --oracle-binary /data/projects/frankensqlite/artifacts/bd-2yqp6.2.5/bd-2yqp6.2.5-20260227T011853Z-4242/doctor-workspace/sqlite3`
