# RFC2119 Hygiene Checklist (bd-1wx.4)

This checklist operationalizes `COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md` §0.2.
Use it for spec edits, bead authoring, and implementation comments.

## Normative Keyword Policy

- [ ] `MUST` / `MUST NOT` are used only for absolute requirements/prohibitions.
- [ ] `SHOULD` / `SHOULD NOT` are used only for strong recommendations.
- [ ] `MAY` is used only for truly optional behavior.
- [ ] Normative keywords are uppercase when used in RFC2119 sense.

## MUST Boundary Rule

- [ ] Every spec `MUST`/`MUST NOT` maps to at least one bead acceptance criterion.
- [ ] Every mapped `MUST`/`MUST NOT` has at least one automated test reference (unit/property/E2E).
- [ ] If a `MUST` is intentionally deferred, the bead includes a reason and a follow-up bead ID.

## SHOULD Deviation Rule

When deviating from a `SHOULD`/`SHOULD NOT`, include this structured block in the
bead and near the implementation site:

```text
SHOULD deviation:
- Why: <why deviation is needed>
- Safety: <why this remains safe/correct>
- Tradeoff: <what we give up>
- Regression detection: <tests/metrics/logs that detect drift>
```

## CI / Local Audit Command

- [ ] Run: `cargo test -p fsqlite-harness --test rfc2119_hygiene_audit -- --nocapture`
- [ ] Confirm deterministic report exists at `target/rfc2119_hygiene_report.json`.
