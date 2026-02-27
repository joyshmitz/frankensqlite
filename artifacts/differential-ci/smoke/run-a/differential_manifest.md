# Differential Manifest (bd-mblr.7.1.2)

run_id: `bd-mblr.7.1.3-smoke-seed-424242`
trace_id: `trace-75a51ff6414e95c5`
scenario_id: `DIFF-CI-713`
commit_sha: `b908482d7257b6db8452d6d8224e92e14ff3b19e`
root_seed: `424242`
corpus_entries: `32`
fixture_json_files_seen: `22`
fixture_entries_ingested: `21`
fixture_sql_statements_ingested: `109`
min_fixture_json_files: `8`
min_fixture_entries: `8`
min_fixture_sql_statements: `40`
slt_files_seen: `0`
slt_entries_ingested: `0`
slt_sql_statements_ingested: `0`
min_slt_files: `1`
min_slt_entries: `1`
min_slt_sql_statements: `1`
total_cases: `88`
passed: `62`
diverged: `26`
overall_pass: `false`
data_hash: `3d9ef9e177d79e6a7578519d5631945f6cdbeec739d38143a97f413b6c9fceee`

## Replay

`cargo run -p fsqlite-harness --bin differential_manifest_runner -- --workspace-root /data/projects/frankensqlite --run-id bd-mblr.7.1.3-smoke-seed-424242 --trace-id trace-75a51ff6414e95c5 --scenario-id DIFF-CI-713 --root-seed 424242 --max-cases-per-entry 4 --generated-unix-ms 1700000000000 --max-entries 32 --fixtures-dir /data/projects/frankensqlite/crates/fsqlite-harness/conformance --min-fixture-json-files 8 --min-fixture-entries 8 --min-fixture-sql-statements 40 --skip-slt --output-json /data/projects/frankensqlite/artifacts/differential-ci/smoke/run-a/differential_manifest.json --output-human /data/projects/frankensqlite/artifacts/differential-ci/smoke/run-a/differential_manifest.md`
