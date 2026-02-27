# Validation Manifest

- bead_id: `bd-mblr.3.5.1`
- schema_version: `1.0.0`
- run_id: `bd-2yqp6.2.9-run-fixed`
- trace_id: `trace-bd-2yqp6.2.9-fixed`
- scenario_id: `ARTIFACT-HASH-RATCHET-B9`
- commit_sha: `bd-2yqp6.2.9-test-sha`
- generated_unix_ms: `1730000000000`
- overall_outcome: `FAIL`
- overall_pass: `false`

## Gates
- `bd-mblr.3.1.1` [coverage] outcome=`PASS_WITH_WARNINGS` artifacts=artifacts/bd-2yqp6.2.9/bd-2yqp6.2.9-20260227T005329Z-9209/validation-manifest/coverage_gate_report.json summary="Coverage gate PASS_WITH_WARNINGS: 41 tests, 124 invariants, global fill 61.5%, evidence 10000.0%, 0 blocking, 1 warnings"
- `bd-mblr.3.1.2` [invariant_drift] outcome=`PASS` artifacts=artifacts/bd-2yqp6.2.9/bd-2yqp6.2.9-20260227T005329Z-9209/validation-manifest/invariant_drift_report.json summary="invariant drift: required_gap_count=0 critical_real=57/57"
- `bd-mblr.3.2.2` [scenario_drift] outcome=`FAIL` artifacts=artifacts/bd-2yqp6.2.9/bd-2yqp6.2.9-20260227T005329Z-9209/validation-manifest/scenario_coverage_drift_report.json summary="scenario drift: required_gap_count=3 manifest_missing=3"
- `bd-mblr.3.4.1` [no_mock_critical_path] outcome=`PASS` artifacts=artifacts/bd-2yqp6.2.9/bd-2yqp6.2.9-20260227T005329Z-9209/validation-manifest/no_mock_critical_path_report.json summary="No-mock critical-path gate: PASS — 57 critical invariants, 57 with real evidence, 0 exceptions, 0 missing (0 blocking, 0 warnings)"
- `bd-mblr.5.5.1` [logging_conformance] outcome=`FAIL` artifacts=artifacts/bd-2yqp6.2.9/bd-2yqp6.2.9-20260227T005329Z-9209/validation-manifest/logging_conformance_report.json,artifacts/bd-2yqp6.2.9/bd-2yqp6.2.9-20260227T005329Z-9209/validation-manifest/validation_manifest_events.jsonl summary="logging conformance: profile_errors=0 schema_errors=6 warnings=0 shell_errors=0 shell_warnings=0"

## Replay
- command: `cargo run -p fsqlite-harness --bin validation_manifest_runner -- --root-seed 9209 --generated-unix-ms 1730000000000 --commit-sha 'bd-2yqp6.2.9-test-sha' --run-id 'bd-2yqp6.2.9-run-fixed' --trace-id 'trace-bd-2yqp6.2.9-fixed' --scenario-id 'ARTIFACT-HASH-RATCHET-B9' --artifact-uri-prefix 'artifacts/bd-2yqp6.2.9/bd-2yqp6.2.9-20260227T005329Z-9209/validation-manifest'`
