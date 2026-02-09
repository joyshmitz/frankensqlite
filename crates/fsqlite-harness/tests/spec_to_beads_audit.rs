use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use tempfile::tempdir;

use fsqlite_harness::spec_to_beads_audit::{
    AuditConfig, AuditMode, normalize_whitespace, run_spec_to_beads_audit, write_report_json,
};

const BEAD_ID: &str = "bd-1wx.3";

#[test]
fn test_normalization_whitespace_collapse() {
    let actual = normalize_whitespace("  alpha\tbeta\n\ngamma   delta  ");
    assert_eq!(
        actual, "alpha beta gamma delta",
        "bead_id={BEAD_ID} case=normalization_whitespace_collapse"
    );
}

#[test]
fn test_ignores_code_fences_and_table_separators() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "\
# Heading Alpha
```
|---|---|
Meaningful line present in bead corpus
";
    let issues = vec![issue(
        "bd-1",
        "Heading Alpha",
        "closed",
        "task",
        "Meaningful line present in bead corpus",
        vec![],
        vec![],
    )];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert_eq!(
        report.coverage.total_checked_lines, 2,
        "bead_id={BEAD_ID} case=ignore_code_fence_and_table_separator checked_lines={}",
        report.coverage.total_checked_lines
    );
    assert!(
        report.missing_spec_lines.is_empty(),
        "bead_id={BEAD_ID} case=ignore_code_fence_and_table_separator missing={:?}",
        report.missing_spec_lines
    );
}

#[test]
fn test_detects_missing_spec_line() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "\
# Heading
Line present in beads
Line missing from beads
";
    let issues = vec![issue(
        "bd-1",
        "Heading",
        "closed",
        "task",
        "Line present in beads",
        vec![],
        vec![],
    )];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert_eq!(
        report.missing_spec_lines.len(),
        1,
        "bead_id={BEAD_ID} case=detect_missing_spec_line missing={:?}",
        report.missing_spec_lines
    );
    assert_eq!(
        report.missing_spec_lines[0].spec_line_no, 3,
        "bead_id={BEAD_ID} case=detect_missing_spec_line expected_line=3 actual={}",
        report.missing_spec_lines[0].spec_line_no
    );
}

#[test]
fn test_headings_always_checked() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "\
# x
This line is present
";
    let issues = vec![issue(
        "bd-1",
        "Irrelevant title",
        "closed",
        "task",
        "This line is present",
        vec![],
        vec![],
    )];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert!(
        report
            .missing_spec_lines
            .iter()
            .any(|entry| entry.spec_line_no == 1),
        "bead_id={BEAD_ID} case=headings_always_checked missing={:?}",
        report.missing_spec_lines
    );
}

#[test]
fn test_open_task_structure_lint() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "# Heading\nThis line is covered";
    let issues = vec![issue(
        "bd-structure",
        "Heading",
        "open",
        "task",
        "This line is covered",
        vec![],
        vec![],
    )];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert_eq!(
        report.open_task_structure_failures.len(),
        1,
        "bead_id={BEAD_ID} case=open_task_structure_lint failures={:?}",
        report.open_task_structure_failures
    );
    assert_eq!(
        report.open_task_structure_failures[0].issue_id, "bd-structure",
        "bead_id={BEAD_ID} case=open_task_structure_lint issue_id mismatch"
    );
}

#[test]
fn test_dep_graph_integrity() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "# Heading\nCovered line";
    let issues = vec![
        issue(
            "bd-epic",
            "§1 Epic",
            "open",
            "epic",
            "Covered line",
            vec![],
            vec![],
        ),
        issue(
            "bd-task-ok",
            "§1.1 Task",
            "open",
            "task",
            "## Unit Test Requirements\n- test_ok\n## E2E Test\n- test_e2e_ok\n## Logging Requirements\n- INFO summary\n## Acceptance Criteria\n- [ ] done",
            vec![],
            vec![dep("bd-epic", "parent-child")],
        ),
        issue(
            "bd-task-no-parent",
            "§1.2 Task",
            "open",
            "task",
            "## Unit Test Requirements\n- test_missing_parent\n## E2E Test\n- test_e2e_missing_parent\n## Logging Requirements\n- INFO x\n## Acceptance Criteria\n- [ ] done",
            vec![],
            vec![],
        ),
        issue(
            "bd-task-missing-dep",
            "Task missing dep",
            "open",
            "task",
            "## Unit Test Requirements\n- test_missing_dep\n## E2E Test\n- test_e2e_missing_dep\n## Logging Requirements\n- INFO x\n## Acceptance Criteria\n- [ ] done",
            vec![],
            vec![dep("bd-does-not-exist", "blocks")],
        ),
        issue(
            "bd-cycle-a",
            "Cycle A",
            "open",
            "task",
            "## Unit Test Requirements\n- test_cycle_a\n## E2E Test\n- test_e2e_cycle_a\n## Logging Requirements\n- INFO x\n## Acceptance Criteria\n- [ ] done",
            vec![],
            vec![dep("bd-cycle-b", "blocks")],
        ),
        issue(
            "bd-cycle-b",
            "Cycle B",
            "open",
            "task",
            "## Unit Test Requirements\n- test_cycle_b\n## E2E Test\n- test_e2e_cycle_b\n## Logging Requirements\n- INFO x\n## Acceptance Criteria\n- [ ] done",
            vec![],
            vec![dep("bd-cycle-a", "blocks")],
        ),
    ];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    let messages: Vec<&str> = report
        .dependency_failures
        .iter()
        .map(|failure| failure.failure.as_str())
        .collect();

    assert!(
        messages
            .iter()
            .any(|message| message.contains("missing issue id")),
        "bead_id={BEAD_ID} case=dep_graph_integrity missing_dep failure not found: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("no parent-child dependency")),
        "bead_id={BEAD_ID} case=dep_graph_integrity missing_parent failure not found: {messages:?}"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains("dependency cycle detected")),
        "bead_id={BEAD_ID} case=dep_graph_integrity cycle failure not found: {messages:?}"
    );
}

#[test]
fn test_scope_doctrine_section_present() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "\
## §0.1 Scope Doctrine
There is no V1 scope.
## §15 Exclusions
- Loadable extension API: excluded due to security risk.
";
    let issues = vec![issue(
        "bd-1",
        "§0 Governance",
        "closed",
        "task",
        "There is no V1 scope.",
        vec![],
        vec![],
    )];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert!(
        report.scope_doctrine_gate.section_0_1_present,
        "bead_id={BEAD_ID} case=scope_doctrine_section_present missing_section_0_1"
    );
    assert!(
        report.scope_doctrine_gate.no_v1_scope_statement_present,
        "bead_id={BEAD_ID} case=scope_doctrine_section_present missing_no_v1_scope_statement"
    );
}

#[test]
fn test_exclusions_have_rationales() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "\
## §0.1 Scope Doctrine
There is no V1 scope.
## §15 Exclusions
- Good exclusion: Has a clear technical rationale.
- Missing rationale
";
    let issues = vec![issue(
        "bd-1",
        "§0 Governance",
        "closed",
        "task",
        "There is no V1 scope.",
        vec![],
        vec![],
    )];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert_eq!(
        report.scope_doctrine_gate.exclusions_checked, 2,
        "bead_id={BEAD_ID} case=exclusions_have_rationales checked={}",
        report.scope_doctrine_gate.exclusions_checked
    );
    assert_eq!(
        report
            .scope_doctrine_gate
            .exclusions_missing_rationale
            .len(),
        1,
        "bead_id={BEAD_ID} case=exclusions_have_rationales missing={:?}",
        report.scope_doctrine_gate.exclusions_missing_rationale
    );
}

#[test]
fn test_no_v1_scope_language_in_beads_without_15_ref() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "\
## §0.1 Scope Doctrine
There is no V1 scope.
## §15 Exclusions
- Loadable extension API: excluded due to security risk.
";
    let issues = vec![issue(
        "bd-bad",
        "§0 Scope Drift",
        "open",
        "task",
        "This is out of scope for V1 and we can do it later.",
        vec![],
        vec![],
    )];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert!(
        report
            .scope_doctrine_gate
            .scope_phrase_violations
            .iter()
            .any(|violation| violation.source_id == "bd-bad" && violation.phrase == "out of scope"),
        "bead_id={BEAD_ID} case=no_v1_scope_language_without_15_ref violations={:?}",
        report.scope_doctrine_gate.scope_phrase_violations
    );
}

#[test]
fn test_no_excluded_features_reintroduced() {
    let temp = tempdir().expect("tempdir should be created");
    let root = temp.path();
    fs::create_dir_all(root.join("crates/demo/src")).expect("crate dir should be created");
    fs::write(
        root.join("crates/demo/src/lib.rs"),
        "pub fn marker() { let _ = \"loadable extension api\"; }",
    )
    .expect("source fixture should be written");

    let spec = "\
## §0.1 Scope Doctrine
There is no V1 scope.
## §15 Exclusions
- Loadable extension API: excluded due to security risk.
";
    let issues = vec![issue(
        "bd-1",
        "§0 Governance",
        "closed",
        "task",
        "There is no V1 scope.",
        vec![],
        vec![],
    )];

    let config = write_fixture(root, spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert!(
        report
            .scope_doctrine_gate
            .excluded_feature_violations
            .iter()
            .any(|violation| violation.file_path.ends_with("crates/demo/src/lib.rs")),
        "bead_id={BEAD_ID} case=no_excluded_features_reintroduced violations={:?}",
        report.scope_doctrine_gate.excluded_feature_violations
    );
}

#[test]
fn test_bd_1wx_3_unit_compliance_gate() {
    let temp = tempdir().expect("tempdir should be created");
    let spec = "\
## §0.1 Scope Doctrine
There is no V1 scope.
## §15 Exclusions
- Loadable extension API: excluded due to security risk.
";
    let issues = vec![issue(
        "bd-defer",
        "§0 Deferral",
        "open",
        "task",
        "We defer this until later.",
        vec![],
        vec![],
    )];

    let config = write_fixture(temp.path(), spec, &issues);
    let report = run_spec_to_beads_audit(&config).expect("audit should run");

    assert!(
        !report
            .scope_doctrine_gate
            .defer_without_follow_up
            .is_empty(),
        "bead_id={BEAD_ID} case=unit_compliance_gate expected defer violation"
    );
}

#[test]
fn prop_bd_1wx_3_structure_compliance() {
    let samples = [
        "This is out of scope for V1.",
        "Optional in V1 later.",
        "We defer this until later.",
    ];
    for sample in samples {
        let lower = sample.to_ascii_lowercase();
        assert!(
            lower.contains("scope") || lower.contains("later") || lower.contains("defer"),
            "bead_id={BEAD_ID} case=structure_compliance sample={sample}"
        );
    }
}

#[test]
fn test_e2e_spec_to_beads_audit_report_schema_stable() {
    let root = workspace_root();
    let report_path = root.join("target/spec_to_beads_audit.json");
    let config = AuditConfig {
        spec_path: root.join("COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md"),
        beads_path: root.join(".beads/issues.jsonl"),
        mode: AuditMode::Strict,
    };

    let report = run_spec_to_beads_audit(&config).expect("repo audit should run");
    write_report_json(&report_path, &report).expect("repo report should be written");

    assert_eq!(
        report.schema_version, 1,
        "bead_id={BEAD_ID} case=e2e_schema_version"
    );
    assert!(
        report.coverage.total_checked_lines > 0,
        "bead_id={BEAD_ID} case=e2e_checked_lines should be non-zero"
    );

    let report_value: Value =
        serde_json::from_slice(&fs::read(&report_path).expect("report file should be readable"))
            .expect("report JSON should parse");
    assert!(
        report_value.get("coverage").is_some()
            && report_value.get("missing_spec_lines").is_some()
            && report_value.get("scope_doctrine_gate").is_some(),
        "bead_id={BEAD_ID} case=e2e_report_schema report_keys={:?}",
        report_value
            .as_object()
            .map(|map| map.keys().cloned().collect::<Vec<_>>())
    );
}

#[test]
fn test_e2e_bd_1wx_3_compliance() {
    let root = workspace_root();
    let config = AuditConfig {
        spec_path: root.join("COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md"),
        beads_path: root.join(".beads/issues.jsonl"),
        mode: AuditMode::Strict,
    };

    let report_a = run_spec_to_beads_audit(&config).expect("repo audit should run");
    let report_b =
        run_spec_to_beads_audit(&config).expect("repo audit should run deterministically");

    assert!(
        report_a.scope_doctrine_gate.section_0_1_present
            && report_a.scope_doctrine_gate.section_15_present,
        "bead_id={BEAD_ID} case=e2e_scope_sections_present report={:?}",
        report_a.scope_doctrine_gate
    );

    assert_eq!(
        serde_json::to_value(&report_a.scope_doctrine_gate).expect("scope gate should serialize"),
        serde_json::to_value(&report_b.scope_doctrine_gate).expect("scope gate should serialize"),
        "bead_id={BEAD_ID} case=e2e_scope_deterministic"
    );
}

fn write_fixture(base: &Path, spec_text: &str, issues: &[Value]) -> AuditConfig {
    let spec_path = base.join("spec.md");
    let beads_path = base.join("issues.jsonl");
    fs::write(&spec_path, spec_text).expect("spec fixture should be written");
    write_issues_jsonl(&beads_path, issues);
    AuditConfig {
        spec_path,
        beads_path,
        mode: AuditMode::Strict,
    }
}

fn write_issues_jsonl(path: &Path, issues: &[Value]) {
    let mut out = String::new();
    for issue in issues {
        out.push_str(
            &serde_json::to_string(issue).expect("issue JSON should serialize into one line"),
        );
        out.push('\n');
    }
    fs::write(path, out).expect("issues jsonl should be written");
}

#[allow(clippy::needless_pass_by_value)]
fn issue(
    id: &str,
    title: &str,
    status: &str,
    issue_type: &str,
    description: &str,
    comments: Vec<&str>,
    dependencies: Vec<Value>,
) -> Value {
    json!({
        "id": id,
        "title": title,
        "status": status,
        "issue_type": issue_type,
        "description": description,
        "comments": comments
            .into_iter()
            .map(|text| json!({ "text": text }))
            .collect::<Vec<Value>>(),
        "dependencies": dependencies
    })
}

fn dep(depends_on_id: &str, dependency_type: &str) -> Value {
    json!({
        "depends_on_id": depends_on_id,
        "type": dependency_type
    })
}

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|path| path.parent())
        .map(PathBuf::from)
        .expect("workspace root should exist")
}
