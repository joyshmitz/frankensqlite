use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use proptest::prelude::proptest;
use proptest::test_runner::TestCaseError;
use serde_json::Value;

const BEAD_ID: &str = "bd-2ddl";
const ISSUES_JSONL: &str = ".beads/issues.jsonl";
const UNIT_TEST_IDS: [&str; 2] = [
    "test_bd_2ddl_unit_compliance_gate",
    "prop_bd_2ddl_structure_compliance",
];
const E2E_TEST_IDS: [&str; 2] = ["test_e2e_bd_2ddl", "test_e2e_bd_2ddl_compliance"];
const LOG_LEVEL_MARKERS: [&str; 4] = ["DEBUG", "INFO", "WARN", "ERROR"];
const LOG_STANDARD_REF: &str = "bd-1fpm";
const REQUIRED_TOKENS: [&str; 9] = [
    "test_bd_2ddl_unit_compliance_gate",
    "prop_bd_2ddl_structure_compliance",
    "test_e2e_bd_2ddl",
    "test_e2e_bd_2ddl_compliance",
    "DEBUG",
    "INFO",
    "WARN",
    "ERROR",
    "bd-1fpm",
];

#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::struct_field_names)]
struct ComplianceEvaluation {
    missing_unit_ids: Vec<&'static str>,
    missing_e2e_ids: Vec<&'static str>,
    missing_log_levels: Vec<&'static str>,
    missing_log_standard_ref: bool,
}

impl ComplianceEvaluation {
    fn is_compliant(&self) -> bool {
        self.missing_unit_ids.is_empty()
            && self.missing_e2e_ids.is_empty()
            && self.missing_log_levels.is_empty()
            && !self.missing_log_standard_ref
    }
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .map_err(|error| format!("workspace_root_canonicalize_failed: {error}"))
}

fn load_issue_description(issue_id: &str) -> Result<String, String> {
    let issues_path = workspace_root()?.join(ISSUES_JSONL);
    let raw = fs::read_to_string(&issues_path).map_err(|error| {
        format!(
            "issues_jsonl_read_failed path={} error={error}",
            issues_path.display()
        )
    })?;

    for line in raw.lines().filter(|line| !line.trim().is_empty()) {
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("issues_jsonl_parse_failed error={error} line={line}"))?;
        if value.get("id").and_then(Value::as_str) == Some(issue_id) {
            let mut canonical = value
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();

            if let Some(comments) = value.get("comments").and_then(Value::as_array) {
                for comment in comments {
                    if let Some(text) = comment.get("text").and_then(Value::as_str) {
                        canonical.push_str("\n\n");
                        canonical.push_str(text);
                    }
                }
            }

            return Ok(canonical);
        }
    }

    Err(format!("bead_id={issue_id} not_found_in={ISSUES_JSONL}"))
}

fn contains_identifier(text: &str, expected_marker: &str) -> bool {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|x| x == expected_marker)
}

fn evaluate_description(description: &str) -> ComplianceEvaluation {
    let missing_unit_ids = UNIT_TEST_IDS
        .into_iter()
        .filter(|id| !contains_identifier(description, id))
        .collect::<Vec<_>>();

    let missing_e2e_ids = E2E_TEST_IDS
        .into_iter()
        .filter(|id| !contains_identifier(description, id))
        .collect::<Vec<_>>();

    let missing_log_levels = LOG_LEVEL_MARKERS
        .into_iter()
        .filter(|level| !description.contains(level))
        .collect::<Vec<_>>();

    ComplianceEvaluation {
        missing_unit_ids,
        missing_e2e_ids,
        missing_log_levels,
        missing_log_standard_ref: !description.contains(LOG_STANDARD_REF),
    }
}

fn count_tests_in_tree(root: &Path) -> Result<usize, String> {
    if !root.exists() {
        return Ok(0);
    }

    let mut total = 0_usize;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir)
            .map_err(|error| format!("read_dir_failed path={} error={error}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|error| {
                format!("read_dir_entry_failed path={} error={error}", dir.display())
            })?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension() != Some(OsStr::new("rs")) {
                continue;
            }

            let raw = fs::read_to_string(&path).map_err(|error| {
                format!("read_file_failed path={} error={error}", path.display())
            })?;
            total += raw.matches("#[test]").count();
        }
    }

    Ok(total)
}

fn crate_test_counts() -> Result<Vec<(String, usize)>, String> {
    let workspace_crates_dir = workspace_root()?.join("crates");
    let entries = fs::read_dir(&workspace_crates_dir).map_err(|error| {
        format!(
            "crates_dir_read_failed path={} error={error}",
            workspace_crates_dir.display()
        )
    })?;

    let mut counts = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "crates_dir_entry_read_failed path={} error={error}",
                workspace_crates_dir.display()
            )
        })?;
        let package_dir = entry.path();
        if !package_dir.is_dir() || !package_dir.join("Cargo.toml").is_file() {
            continue;
        }

        let crate_name = entry.file_name().to_string_lossy().into_owned();
        let src_count = count_tests_in_tree(&package_dir.join("src"))?;
        let tests_count = count_tests_in_tree(&package_dir.join("tests"))?;
        counts.push((crate_name, src_count + tests_count));
    }
    counts.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

    Ok(counts)
}

#[test]
fn test_bd_2ddl_unit_compliance_gate() -> Result<(), String> {
    let description = load_issue_description(BEAD_ID)?;
    let evaluation = evaluate_description(&description);

    if !evaluation.missing_unit_ids.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=unit_ids_missing missing={:?}",
            evaluation.missing_unit_ids
        ));
    }
    if !evaluation.missing_e2e_ids.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=e2e_ids_missing missing={:?}",
            evaluation.missing_e2e_ids
        ));
    }
    if !evaluation.missing_log_levels.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=logging_levels_missing missing={:?}",
            evaluation.missing_log_levels
        ));
    }
    if evaluation.missing_log_standard_ref {
        return Err(format!(
            "bead_id={BEAD_ID} case=logging_standard_missing expected_ref={LOG_STANDARD_REF}"
        ));
    }

    Ok(())
}

#[test]
fn test_all_workspace_crates_have_at_least_one_unit_test() -> Result<(), String> {
    let counts = crate_test_counts()?;
    let missing = counts
        .iter()
        .filter(|(_crate_name, count)| *count == 0)
        .map(|(crate_name, _count)| crate_name.clone())
        .collect::<Vec<_>>();

    if !missing.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=zero_test_crates missing={missing:?}"
        ));
    }

    Ok(())
}

proptest! {
    #[test]
    fn prop_bd_2ddl_structure_compliance(missing_index in 0usize..REQUIRED_TOKENS.len()) {
        let mut synthetic = format!(
            "## Unit Test Requirements\n- {}\n- {}\n\n## E2E Test\n- {}\n- {}\n\n## Logging Requirements\n- DEBUG: stage progress\n- INFO: summary\n- WARN: degraded mode\n- ERROR: terminal failure\n- Reference: {}\n",
            UNIT_TEST_IDS[0],
            UNIT_TEST_IDS[1],
            E2E_TEST_IDS[0],
            E2E_TEST_IDS[1],
            LOG_STANDARD_REF,
        );

        synthetic = synthetic.replacen(REQUIRED_TOKENS[missing_index], "", 1);
        let evaluation = evaluate_description(&synthetic);
        if evaluation.is_compliant() {
            return Err(TestCaseError::fail(format!(
                "bead_id={} case=structure_compliance expected_non_compliant missing_index={} missing_marker={}",
                BEAD_ID,
                missing_index,
                REQUIRED_TOKENS[missing_index]
            )));
        }
    }
}

#[test]
fn test_e2e_bd_2ddl_compliance() -> Result<(), String> {
    let description = load_issue_description(BEAD_ID)?;
    let evaluation = evaluate_description(&description);
    let counts = crate_test_counts()?;
    let missing_crates = counts
        .iter()
        .filter(|(_crate_name, count)| *count == 0)
        .map(|(crate_name, _count)| crate_name.clone())
        .collect::<Vec<_>>();

    eprintln!(
        "DEBUG bead_id={BEAD_ID} case=test_inventory crates_total={} zero_test_crates={}",
        counts.len(),
        missing_crates.len()
    );

    eprintln!(
        "INFO bead_id={BEAD_ID} case=e2e_summary missing_unit_ids={} missing_e2e_ids={} missing_log_levels={} missing_log_standard_ref={} zero_test_crates={}",
        evaluation.missing_unit_ids.len(),
        evaluation.missing_e2e_ids.len(),
        evaluation.missing_log_levels.len(),
        evaluation.missing_log_standard_ref,
        missing_crates.len()
    );

    for id in &evaluation.missing_unit_ids {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_unit_id id={id}");
    }
    for id in &evaluation.missing_e2e_ids {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_e2e_id id={id}");
    }
    for level in &evaluation.missing_log_levels {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_log_level level={level}");
    }
    if evaluation.missing_log_standard_ref {
        eprintln!(
            "ERROR bead_id={BEAD_ID} case=missing_log_standard_ref expected={LOG_STANDARD_REF}"
        );
    }
    for crate_name in &missing_crates {
        eprintln!("WARN bead_id={BEAD_ID} case=missing_test_crate crate={crate_name}");
    }
    if !missing_crates.is_empty() {
        eprintln!("ERROR bead_id={BEAD_ID} case=zero_test_crates missing={missing_crates:?}");
    }

    if !evaluation.is_compliant() || !missing_crates.is_empty() {
        return Err(format!(
            "bead_id={BEAD_ID} case=e2e_compliance_failure evaluation={evaluation:?} missing_crates={missing_crates:?}"
        ));
    }

    Ok(())
}

#[test]
fn test_e2e_bd_2ddl() -> Result<(), String> {
    test_e2e_bd_2ddl_compliance()
}
