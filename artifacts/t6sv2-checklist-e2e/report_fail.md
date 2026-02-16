# t6sv2 Evidence Checklist Report

- bead_id: `bd-t6sv2.16`
- overall_pass: `false`
- child_count: `2`
- open_count: `2`
- in_progress_count: `0`
- closed_count: `0`
- missing_unit_count: `1`
- missing_e2e_count: `0`
- missing_log_count: `0`
- stale_link_count: `2`
- violation_count: `3`

## Rows

| Bead | Status | Owner | Missing | Stale Links |
|---|---|---|---|---:|
| `bd-t6sv2.4` | open | ops-a | none | 0 |
| `bd-t6sv2.6` | open | ops-b | UnitEvidence | 2 |

## Violations

- `bd-t6sv2.6` owner=ops-b kind=missing_unit_evidence detail=no linked unit/property evidence cmd=`br show bd-t6sv2.6 --json`
- `bd-t6sv2.6` owner=ops-b kind=invalid_e2e_reference detail=missing e2e script path: scripts/missing_t6sv2_6.sh cmd=`br show bd-t6sv2.6 --json`
- `bd-t6sv2.6` owner=ops-b kind=invalid_log_reference detail=invalid log schema version: bad cmd=`br show bd-t6sv2.6 --json`
