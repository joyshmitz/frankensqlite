use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use fsqlite_error::{FrankenError, Result};
use fsqlite_vfs::host_fs;

const STRICT_MIN_MEANINGFUL_LEN: usize = 10;
const FAST_MIN_MEANINGFUL_LEN: usize = 20;
const MAX_LINE_TEXT_LEN: usize = 200;
const SCOPE_REDUCTION_PHRASES: [&str; 4] = ["out of scope", "v1 scope", "optional in v1", "later"];
const SECTION_15_REF_TOKENS: [&str; 3] = ["§15", "section 15", "sec 15"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditMode {
    Fast,
    Strict,
}

impl AuditMode {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Strict => "strict",
        }
    }

    #[must_use]
    pub const fn min_meaningful_len(self) -> usize {
        match self {
            Self::Fast => FAST_MIN_MEANINGFUL_LEN,
            Self::Strict => STRICT_MIN_MEANINGFUL_LEN,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub spec_path: PathBuf,
    pub beads_path: PathBuf,
    pub mode: AuditMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecToBeadsAuditReport {
    pub schema_version: u32,
    pub mode: String,
    pub spec_path: String,
    pub beads_path: String,
    pub coverage: CoverageSummary,
    pub missing_spec_lines: Vec<MissingSpecLine>,
    pub open_task_structure_failures: Vec<OpenTaskStructureFailure>,
    pub dependency_failures: Vec<DependencyFailure>,
    pub closed_rollup_hygiene_warnings: Vec<ClosedRollupHygieneWarning>,
    pub scope_doctrine_gate: ScopeDoctrineGateReport,
    pub pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageSummary {
    pub total_checked_lines: usize,
    pub covered_lines: usize,
    pub missing_lines: usize,
    pub coverage_bps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingSpecLine {
    pub spec_line_no: usize,
    pub line_text: String,
    pub hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenTaskStructureFailure {
    pub issue_id: String,
    pub title: String,
    pub failures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyFailure {
    pub issue_id: String,
    pub failure: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClosedRollupHygieneWarning {
    pub issue_id: String,
    pub warning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
pub struct ScopeDoctrineGateReport {
    pub pass: bool,
    pub section_0_1_present: bool,
    pub no_v1_scope_statement_present: bool,
    pub section_15_present: bool,
    pub exclusions_checked: usize,
    pub exclusions_missing_rationale: Vec<ExclusionRationaleFailure>,
    pub scope_phrase_violations: Vec<ScopePhraseViolation>,
    pub excluded_feature_violations: Vec<ExcludedFeatureViolation>,
    pub defer_without_follow_up: Vec<DeferredDecisionViolation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExclusionRationaleFailure {
    pub line_no: usize,
    pub exclusion_name: String,
    pub line_text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopePhraseViolation {
    pub source_kind: String,
    pub source_id: String,
    pub line_no: usize,
    pub phrase: String,
    pub line_text: String,
    pub remediation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludedFeatureViolation {
    pub feature_name: String,
    pub file_path: String,
    pub line_no: usize,
    pub line_text: String,
    pub remediation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferredDecisionViolation {
    pub issue_id: String,
    pub title: String,
    pub detail: String,
}

#[derive(Debug, Clone)]
struct CheckedSpecLine {
    line_no: usize,
    text: String,
    normalized: String,
}

#[derive(Debug, Clone)]
struct Issue {
    id: String,
    title: String,
    description: String,
    acceptance_criteria: String,
    status: String,
    kind: String,
    close_reason: Option<String>,
    comments: Vec<String>,
    dependencies: Vec<IssueDependency>,
}

#[derive(Debug, Clone)]
struct IssueDependency {
    depends_on_id: String,
    dependency_type: String,
}

#[derive(Debug, Deserialize)]
struct RawIssue {
    #[serde(default)]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    acceptance_criteria: String,
    #[serde(default)]
    status: String,
    #[serde(default, rename = "issue_type")]
    kind: String,
    #[serde(default)]
    close_reason: Option<String>,
    #[serde(default)]
    comments: Vec<RawComment>,
    #[serde(default)]
    dependencies: Vec<RawDependency>,
}

#[derive(Debug, Deserialize)]
struct RawComment {
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct RawDependency {
    #[serde(default)]
    depends_on_id: String,
    #[serde(rename = "type", default)]
    dependency_type: String,
}

#[derive(Debug, Clone, Copy)]
struct SectionBounds {
    start: usize,
    end: usize,
}

pub fn run_spec_to_beads_audit(config: &AuditConfig) -> Result<SpecToBeadsAuditReport> {
    let spec_text = host_fs::read_to_string(&config.spec_path).map_err(|err| {
        internal_error(format!(
            "failed reading spec file '{}': {err}",
            config.spec_path.display()
        ))
    })?;
    let issues = load_issues(&config.beads_path)?;
    let bead_corpus = build_bead_corpus(&issues);
    let checked_spec_lines = collect_checked_spec_lines(&spec_text, config.mode);
    let mut missing_spec_lines = Vec::new();

    for line in &checked_spec_lines {
        if !bead_corpus.contains(&line.normalized) {
            let missing = MissingSpecLine {
                spec_line_no: line.line_no,
                line_text: truncate_line(&line.text, MAX_LINE_TEXT_LEN),
                hint: keyword_hint(&line.text),
            };
            warn!(
                spec_line_no = missing.spec_line_no,
                line_text = %missing.line_text,
                hint = %missing.hint,
                "spec line missing from bead corpus"
            );
            missing_spec_lines.push(missing);
        }
    }

    let mut open_task_structure_failures = lint_open_task_structure(&issues);
    let mut dependency_failures = lint_dependency_integrity(&issues);
    let mut closed_rollup_hygiene_warnings = lint_closed_rollup_hygiene(&issues);
    let scope_doctrine_gate = lint_scope_doctrine_gate(config, &spec_text, &issues);

    missing_spec_lines.sort_by_key(|entry| entry.spec_line_no);
    open_task_structure_failures.sort_by(|a, b| a.issue_id.cmp(&b.issue_id));
    dependency_failures.sort_by(|a, b| {
        a.issue_id
            .cmp(&b.issue_id)
            .then_with(|| a.failure.cmp(&b.failure))
    });
    closed_rollup_hygiene_warnings.sort_by(|a, b| {
        a.issue_id
            .cmp(&b.issue_id)
            .then_with(|| a.warning.cmp(&b.warning))
    });

    let total_checked_lines = checked_spec_lines.len();
    let missing_lines = missing_spec_lines.len();
    let covered_lines = total_checked_lines.saturating_sub(missing_lines);
    let coverage_bps = coverage_basis_points(covered_lines, total_checked_lines);
    let pass = missing_spec_lines.is_empty()
        && open_task_structure_failures.is_empty()
        && dependency_failures.is_empty()
        && scope_doctrine_gate.pass;

    info!(
        mode = config.mode.as_str(),
        total_checked_lines,
        covered_lines,
        missing_lines,
        open_task_structure_failures = open_task_structure_failures.len(),
        dependency_failures = dependency_failures.len(),
        closed_rollup_hygiene_warnings = closed_rollup_hygiene_warnings.len(),
        scope_phrase_violations = scope_doctrine_gate.scope_phrase_violations.len(),
        excluded_feature_violations = scope_doctrine_gate.excluded_feature_violations.len(),
        defer_without_follow_up = scope_doctrine_gate.defer_without_follow_up.len(),
        pass,
        "spec-to-beads audit completed"
    );

    if !pass {
        error!(
            missing_lines,
            open_task_structure_failures = open_task_structure_failures.len(),
            dependency_failures = dependency_failures.len(),
            scope_phrase_violations = scope_doctrine_gate.scope_phrase_violations.len(),
            excluded_feature_violations = scope_doctrine_gate.excluded_feature_violations.len(),
            defer_without_follow_up = scope_doctrine_gate.defer_without_follow_up.len(),
            "spec-to-beads audit failed"
        );
    }

    Ok(SpecToBeadsAuditReport {
        schema_version: 1,
        mode: config.mode.as_str().to_string(),
        spec_path: config.spec_path.display().to_string(),
        beads_path: config.beads_path.display().to_string(),
        coverage: CoverageSummary {
            total_checked_lines,
            covered_lines,
            missing_lines,
            coverage_bps,
        },
        missing_spec_lines,
        open_task_structure_failures,
        dependency_failures,
        closed_rollup_hygiene_warnings,
        scope_doctrine_gate,
        pass,
    })
}

pub fn write_report_json(path: &Path, report: &SpecToBeadsAuditReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        host_fs::create_dir_all(parent).map_err(|err| {
            internal_error(format!(
                "failed creating report directory '{}': {err}",
                parent.display()
            ))
        })?;
    }

    let encoded = serde_json::to_vec_pretty(report)
        .map_err(|err| internal_error(format!("failed to encode audit report JSON: {err}")))?;
    host_fs::write(path, encoded).map_err(|err| {
        internal_error(format!(
            "failed writing report file '{}': {err}",
            path.display()
        ))
    })?;
    Ok(())
}

pub fn normalize_whitespace(input: &str) -> String {
    let mut out = String::new();
    let mut in_ws = false;

    for ch in input.chars() {
        if ch.is_whitespace() {
            in_ws = true;
            continue;
        }

        if in_ws && !out.is_empty() {
            out.push(' ');
        }
        in_ws = false;
        out.push(ch);
    }

    out.trim().to_string()
}

fn load_issues(path: &Path) -> Result<Vec<Issue>> {
    let contents = host_fs::read_to_string(path).map_err(|err| {
        internal_error(format!(
            "failed reading issues JSONL '{}': {err}",
            path.display()
        ))
    })?;

    let mut issues = Vec::new();
    for (line_idx, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        let parsed: RawIssue = serde_json::from_str(line).map_err(|err| {
            internal_error(format!(
                "issues JSONL parse error at line {}: {err}",
                line_idx + 1
            ))
        })?;
        issues.push(Issue {
            id: parsed.id,
            title: parsed.title,
            description: parsed.description,
            acceptance_criteria: parsed.acceptance_criteria,
            status: parsed.status,
            kind: parsed.kind,
            close_reason: parsed.close_reason,
            comments: parsed.comments.into_iter().map(|c| c.text).collect(),
            dependencies: parsed
                .dependencies
                .into_iter()
                .map(|dep| IssueDependency {
                    depends_on_id: dep.depends_on_id,
                    dependency_type: dep.dependency_type,
                })
                .collect(),
        });
    }

    Ok(issues)
}

fn build_bead_corpus(issues: &[Issue]) -> String {
    let mut corpus = String::new();
    for issue in issues {
        append_normalized_segment(&mut corpus, &issue.title);
        append_normalized_segment(&mut corpus, &issue.description);
        for comment in &issue.comments {
            append_normalized_segment(&mut corpus, comment);
        }
    }
    corpus
}

fn append_normalized_segment(corpus: &mut String, text: &str) {
    let normalized = normalize_whitespace(text);
    if normalized.is_empty() {
        return;
    }
    if !corpus.is_empty() {
        corpus.push(' ');
    }
    corpus.push_str(&normalized);
}

fn collect_checked_spec_lines(spec_text: &str, mode: AuditMode) -> Vec<CheckedSpecLine> {
    let min_len = mode.min_meaningful_len();
    let mut out = Vec::new();

    for (idx, line) in spec_text.lines().enumerate() {
        let line_no = idx + 1;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if is_code_fence_delimiter(trimmed) || is_table_separator(trimmed) {
            continue;
        }

        let heading = is_markdown_heading(trimmed);
        let meaningful =
            trimmed.chars().any(char::is_alphanumeric) && trimmed.chars().count() >= min_len;

        if !heading && !meaningful {
            continue;
        }

        let normalized = if heading {
            normalize_whitespace(strip_markdown_heading(trimmed))
        } else {
            normalize_whitespace(trimmed)
        };
        if normalized.is_empty() {
            continue;
        }

        out.push(CheckedSpecLine {
            line_no,
            text: trimmed.to_string(),
            normalized,
        });
    }

    out
}

fn lint_open_task_structure(issues: &[Issue]) -> Vec<OpenTaskStructureFailure> {
    let mut failures = Vec::new();

    for issue in issues {
        if !issue.status.eq_ignore_ascii_case("open") {
            continue;
        }
        if !issue.kind.eq_ignore_ascii_case("task") {
            continue;
        }

        let full_text = issue.full_text();
        let lines: Vec<&str> = full_text.lines().collect();
        let mut issue_failures = Vec::new();

        let unit_section = find_section(
            &lines,
            &["unit test requirements", "unit tests", "test requirements"],
        );
        match unit_section {
            None => issue_failures.push(
                "missing required unit-test section header (Unit Test Requirements / Unit Tests / Test Requirements)"
                    .to_string(),
            ),
            Some(section) => {
                if !contains_prefixed_identifier(section.slice(&lines), &["test_", "prop_"]) {
                    issue_failures.push(
                        "unit-test section missing concrete identifier (test_* or prop_)"
                            .to_string(),
                    );
                }
            }
        }

        let e2e_section = find_section(&lines, &["e2e test", "e2e tests", "e2e benchmark"]);
        match e2e_section {
            None => issue_failures.push(
                "missing required E2E section header (E2E Test / E2E Tests / E2E Benchmark)"
                    .to_string(),
            ),
            Some(section) => {
                if !contains_prefixed_identifier(section.slice(&lines), &["test_e2e_", "e2e_"]) {
                    issue_failures.push(
                        "E2E section missing concrete identifier (test_e2e_* or e2e_*)".to_string(),
                    );
                }
            }
        }

        let logging_section = find_section(
            &lines,
            &["logging requirements", "observability requirements"],
        );
        match logging_section {
            None => issue_failures
                .push("missing required logging/observability section header".to_string()),
            Some(section) => {
                let section_lines = section.slice(&lines);
                let references_bundle_standard = full_text.to_ascii_lowercase().contains("bd-1fpm");
                if !has_log_level_marker(section_lines) && !references_bundle_standard {
                    issue_failures.push(
                        "logging section missing log-level markers (DEBUG/INFO/WARN/ERROR) or bd-1fpm reference"
                            .to_string(),
                    );
                }
            }
        }

        let acceptance_section = find_section(&lines, &["acceptance criteria"]);
        match acceptance_section {
            None => issue_failures.push("missing required Acceptance Criteria section".to_string()),
            Some(section) => {
                if !contains_bullet(section.slice(&lines)) {
                    issue_failures.push(
                        "Acceptance Criteria section missing at least one checkbox/bullet"
                            .to_string(),
                    );
                }
            }
        }

        if !issue_failures.is_empty() {
            failures.push(OpenTaskStructureFailure {
                issue_id: issue.id.clone(),
                title: issue.title.clone(),
                failures: issue_failures,
            });
        }
    }

    failures
}

fn lint_dependency_integrity(issues: &[Issue]) -> Vec<DependencyFailure> {
    let mut failures = Vec::new();
    let index_by_id: HashMap<String, usize> = issues
        .iter()
        .enumerate()
        .map(|(idx, issue)| (issue.id.clone(), idx))
        .collect();

    for issue in issues {
        for dep in &issue.dependencies {
            if !index_by_id.contains_key(dep.depends_on_id.as_str()) {
                failures.push(DependencyFailure {
                    issue_id: issue.id.clone(),
                    failure: format!(
                        "dependency references missing issue id '{}'",
                        dep.depends_on_id
                    ),
                });
            }
        }
    }

    for issue in issues {
        if !issue.kind.eq_ignore_ascii_case("task") || !issue.title.contains('§') {
            continue;
        }

        let has_parent = issue.dependencies.iter().any(|dep| {
            if !dep.dependency_type.eq_ignore_ascii_case("parent-child") {
                return false;
            }

            let Some(parent_idx) = index_by_id.get(dep.depends_on_id.as_str()) else {
                return false;
            };
            let parent = &issues[*parent_idx];
            parent.kind.eq_ignore_ascii_case("epic") && parent.title.contains('§')
        });

        if !has_parent {
            failures.push(DependencyFailure {
                issue_id: issue.id.clone(),
                failure: "spec-derived task has no parent-child dependency to a spec epic"
                    .to_string(),
            });
        }
    }

    for cycle in find_cycles(issues, &index_by_id) {
        failures.push(DependencyFailure {
            issue_id: cycle
                .first()
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
            failure: format!("dependency cycle detected: {}", cycle.join(" -> ")),
        });
    }

    failures
}

#[allow(clippy::items_after_statements)]
fn find_cycles(issues: &[Issue], index_by_id: &HashMap<String, usize>) -> Vec<Vec<String>> {
    let mut state = vec![0_u8; issues.len()];
    let mut stack: Vec<usize> = Vec::new();
    let mut cycles: Vec<Vec<String>> = Vec::new();
    let mut seen = BTreeSet::new();

    fn dfs(
        node: usize,
        issues: &[Issue],
        index_by_id: &HashMap<String, usize>,
        state: &mut [u8],
        stack: &mut Vec<usize>,
        cycles: &mut Vec<Vec<String>>,
        seen: &mut BTreeSet<String>,
    ) {
        state[node] = 1;
        stack.push(node);

        for dep in &issues[node].dependencies {
            let Some(next) = index_by_id.get(dep.depends_on_id.as_str()) else {
                continue;
            };

            match state[*next] {
                0 => dfs(*next, issues, index_by_id, state, stack, cycles, seen),
                1 => {
                    if let Some(pos) = stack.iter().position(|idx| idx == next) {
                        let mut cycle_nodes: Vec<String> = stack[pos..]
                            .iter()
                            .map(|idx| issues[*idx].id.clone())
                            .collect();
                        cycle_nodes.push(issues[*next].id.clone());

                        let mut canonical = cycle_nodes.clone();
                        canonical.sort();
                        canonical.dedup();
                        let key = canonical.join("|");
                        if seen.insert(key) {
                            cycles.push(cycle_nodes);
                        }
                    }
                }
                _ => {}
            }
        }

        stack.pop();
        state[node] = 2;
    }

    for idx in 0..issues.len() {
        if state[idx] == 0 {
            dfs(
                idx,
                issues,
                index_by_id,
                &mut state,
                &mut stack,
                &mut cycles,
                &mut seen,
            );
        }
    }

    cycles
}

fn lint_closed_rollup_hygiene(issues: &[Issue]) -> Vec<ClosedRollupHygieneWarning> {
    let mut warnings = Vec::new();
    let index_by_id: HashMap<String, usize> = issues
        .iter()
        .enumerate()
        .map(|(idx, issue)| (issue.id.clone(), idx))
        .collect();

    for issue in issues {
        if !issue.status.eq_ignore_ascii_case("closed") {
            continue;
        }
        if !issue.kind.eq_ignore_ascii_case("task") {
            continue;
        }
        if !issue.title.contains('§') {
            continue;
        }

        let reason = issue
            .close_reason
            .as_deref()
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();
        if !is_superseded_close_reason(&reason) {
            continue;
        }

        let description_lower = issue.description.to_ascii_lowercase();
        if !description_lower.contains("status: closed (superseded rollup)") {
            warnings.push(ClosedRollupHygieneWarning {
                issue_id: issue.id.clone(),
                warning: "superseded close_reason is missing explicit marker 'STATUS: CLOSED (superseded rollup)'"
                    .to_string(),
            });
        }

        if !description_lower.contains("replacement beads") {
            warnings.push(ClosedRollupHygieneWarning {
                issue_id: issue.id.clone(),
                warning: "superseded close_reason should include 'replacement beads' section"
                    .to_string(),
            });
        }

        let replacement_ids = extract_bead_ids(&issue.description);
        if replacement_ids.is_empty() {
            warnings.push(ClosedRollupHygieneWarning {
                issue_id: issue.id.clone(),
                warning: "superseded close_reason should list at least one replacement bead id"
                    .to_string(),
            });
            continue;
        }

        let mut has_open_replacement = false;
        for replacement_id in replacement_ids {
            let Some(repl_pos) = index_by_id.get(replacement_id.as_str()) else {
                warnings.push(ClosedRollupHygieneWarning {
                    issue_id: issue.id.clone(),
                    warning: format!(
                        "replacement bead id '{}' does not exist in issues corpus",
                        replacement_id
                    ),
                });
                continue;
            };

            if issues[*repl_pos].status.eq_ignore_ascii_case("open") {
                has_open_replacement = true;
            }
        }

        if !has_open_replacement {
            warnings.push(ClosedRollupHygieneWarning {
                issue_id: issue.id.clone(),
                warning: "superseded rollup should reference at least one open replacement bead"
                    .to_string(),
            });
        }
    }

    warnings
}

#[derive(Debug, Clone)]
struct ExclusionItem {
    line_no: usize,
    name: String,
    rationale: String,
    line_text: String,
}

#[allow(clippy::too_many_lines)]
fn lint_scope_doctrine_gate(
    config: &AuditConfig,
    spec_text: &str,
    issues: &[Issue],
) -> ScopeDoctrineGateReport {
    let spec_lines: Vec<&str> = spec_text.lines().collect();
    let section_0_1 = find_spec_section_bounds(&spec_lines, &["§0.1", "0.1"]);
    let section_15 = find_spec_section_bounds(&spec_lines, &["§15", "15", "exclusions"]);

    let section_0_1_present = section_0_1.is_some();
    let section_15_present = section_15.is_some();

    let no_v1_scope_statement_present = section_0_1.is_some_and(|bounds| {
        bounds
            .slice(&spec_lines)
            .iter()
            .map(|line| line.to_ascii_lowercase())
            .any(|line| {
                line.contains("no v1 scope")
                    || line.contains("there is no v1 scope")
                    || line.contains("there is no \"v1 scope\"")
            })
    });

    let exclusions = section_15
        .map(|bounds| parse_exclusion_items(&spec_lines, bounds))
        .unwrap_or_default();
    let exclusions_checked = exclusions.len();

    let mut exclusions_missing_rationale: Vec<ExclusionRationaleFailure> = exclusions
        .iter()
        .filter(|item| item.rationale.chars().count() < STRICT_MIN_MEANINGFUL_LEN)
        .map(|item| ExclusionRationaleFailure {
            line_no: item.line_no,
            exclusion_name: item.name.clone(),
            line_text: item.line_text.clone(),
        })
        .collect();
    exclusions_missing_rationale.sort_by(|a, b| {
        a.line_no
            .cmp(&b.line_no)
            .then_with(|| a.exclusion_name.cmp(&b.exclusion_name))
    });

    let mut scope_phrase_violations = lint_scope_phrase_violations(config, issues);
    scope_phrase_violations.sort_by(|a, b| {
        a.source_kind
            .cmp(&b.source_kind)
            .then_with(|| a.source_id.cmp(&b.source_id))
            .then_with(|| a.line_no.cmp(&b.line_no))
            .then_with(|| a.phrase.cmp(&b.phrase))
    });

    let mut excluded_feature_violations = lint_excluded_feature_reintroduction(config, &exclusions);
    excluded_feature_violations.sort_by(|a, b| {
        a.feature_name
            .cmp(&b.feature_name)
            .then_with(|| a.file_path.cmp(&b.file_path))
            .then_with(|| a.line_no.cmp(&b.line_no))
    });

    let mut defer_without_follow_up = lint_deferred_decisions(issues);
    defer_without_follow_up.sort_by(|a, b| {
        a.issue_id
            .cmp(&b.issue_id)
            .then_with(|| a.title.cmp(&b.title))
            .then_with(|| a.detail.cmp(&b.detail))
    });

    if !section_0_1_present {
        error!("scope doctrine gate: missing §0.1 section in spec");
    }
    if !no_v1_scope_statement_present {
        error!("scope doctrine gate: missing explicit 'no V1 scope' statement in §0.1");
    }
    if !section_15_present {
        error!("scope doctrine gate: missing §15 exclusions section in spec");
    }

    for failure in &exclusions_missing_rationale {
        warn!(
            line_no = failure.line_no,
            exclusion = %failure.exclusion_name,
            "scope doctrine gate: exclusion missing technical rationale"
        );
    }
    for violation in &scope_phrase_violations {
        warn!(
            source_kind = %violation.source_kind,
            source_id = %violation.source_id,
            line_no = violation.line_no,
            phrase = %violation.phrase,
            "scope doctrine gate: found scope-reduction language without §15 reference"
        );
    }
    for violation in &excluded_feature_violations {
        error!(
            feature = %violation.feature_name,
            file_path = %violation.file_path,
            line_no = violation.line_no,
            "scope doctrine gate: excluded feature appears in implementation sources"
        );
    }
    for violation in &defer_without_follow_up {
        warn!(
            issue_id = %violation.issue_id,
            detail = %violation.detail,
            "scope doctrine gate: deferred decision missing reason or follow-up bead"
        );
    }

    let pass = section_0_1_present
        && no_v1_scope_statement_present
        && section_15_present
        && exclusions_missing_rationale.is_empty()
        && scope_phrase_violations.is_empty()
        && excluded_feature_violations.is_empty()
        && defer_without_follow_up.is_empty();

    info!(
        section_0_1_present,
        no_v1_scope_statement_present,
        section_15_present,
        exclusions_checked,
        exclusions_missing_rationale = exclusions_missing_rationale.len(),
        scope_phrase_violations = scope_phrase_violations.len(),
        excluded_feature_violations = excluded_feature_violations.len(),
        defer_without_follow_up = defer_without_follow_up.len(),
        pass,
        "scope doctrine gate completed"
    );

    ScopeDoctrineGateReport {
        pass,
        section_0_1_present,
        no_v1_scope_statement_present,
        section_15_present,
        exclusions_checked,
        exclusions_missing_rationale,
        scope_phrase_violations,
        excluded_feature_violations,
        defer_without_follow_up,
    }
}

fn lint_scope_phrase_violations(
    config: &AuditConfig,
    issues: &[Issue],
) -> Vec<ScopePhraseViolation> {
    let mut violations = Vec::new();

    for issue in issues {
        let full_text = issue.full_text();
        let has_section_15_ref = references_section_15(&full_text);
        for (line_idx, line) in full_text.lines().enumerate() {
            let line_lower = line.to_ascii_lowercase();
            for phrase in SCOPE_REDUCTION_PHRASES {
                if !contains_scope_phrase(&line_lower, phrase) {
                    continue;
                }
                if has_section_15_ref || references_section_15(line) {
                    continue;
                }

                violations.push(ScopePhraseViolation {
                    source_kind: "bead".to_string(),
                    source_id: issue.id.clone(),
                    line_no: line_idx + 1,
                    phrase: phrase.to_string(),
                    line_text: truncate_line(line.trim(), MAX_LINE_TEXT_LEN),
                    remediation: "Reference §15 explicitly or replace with scheduling language."
                        .to_string(),
                });
            }
        }
    }

    let workspace_root = config
        .spec_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    for path in collect_markdown_files(&workspace_root) {
        if path == config.spec_path {
            continue;
        }
        let Ok(contents) = host_fs::read_to_string(&path) else {
            continue;
        };
        let has_section_15_ref = references_section_15(&contents);
        for (line_idx, line) in contents.lines().enumerate() {
            let line_lower = line.to_ascii_lowercase();
            for phrase in SCOPE_REDUCTION_PHRASES {
                if !contains_scope_phrase(&line_lower, phrase) {
                    continue;
                }
                if has_section_15_ref || references_section_15(line) {
                    continue;
                }

                violations.push(ScopePhraseViolation {
                    source_kind: "doc".to_string(),
                    source_id: relativize_path(&workspace_root, &path),
                    line_no: line_idx + 1,
                    phrase: phrase.to_string(),
                    line_text: truncate_line(line.trim(), MAX_LINE_TEXT_LEN),
                    remediation: "Add a §15 reference or remove scope-reduction wording."
                        .to_string(),
                });
            }
        }
    }

    violations
}

fn lint_excluded_feature_reintroduction(
    config: &AuditConfig,
    exclusions: &[ExclusionItem],
) -> Vec<ExcludedFeatureViolation> {
    let workspace_root = config
        .spec_path
        .parent()
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    let mut violations = Vec::new();

    let excluded_names: BTreeSet<String> = exclusions
        .iter()
        .map(|item| item.name.to_ascii_lowercase())
        .map(|name| normalize_whitespace(&name))
        .filter(|name| name.chars().count() >= 8)
        .filter(|name| name.split_whitespace().count() >= 2)
        .collect();

    if excluded_names.is_empty() {
        return violations;
    }

    for path in collect_implementation_source_files(&workspace_root) {
        let Ok(contents) = host_fs::read_to_string(&path) else {
            continue;
        };
        let contents_lower = contents.to_ascii_lowercase();
        for excluded_name in &excluded_names {
            if !contents_lower.contains(excluded_name) {
                continue;
            }

            if contents.contains("§15") || contents.to_ascii_lowercase().contains("section 15") {
                continue;
            }

            for (line_idx, line) in contents.lines().enumerate() {
                let line_lower = line.to_ascii_lowercase();
                if !line_lower.contains(excluded_name) {
                    continue;
                }
                violations.push(ExcludedFeatureViolation {
                    feature_name: excluded_name.clone(),
                    file_path: relativize_path(&workspace_root, &path),
                    line_no: line_idx + 1,
                    line_text: truncate_line(line.trim(), MAX_LINE_TEXT_LEN),
                    remediation:
                        "Either remove this implementation mention or explicitly re-scope via spec update."
                            .to_string(),
                });
            }
        }
    }

    violations
}

fn lint_deferred_decisions(issues: &[Issue]) -> Vec<DeferredDecisionViolation> {
    let mut violations = Vec::new();

    for issue in issues {
        let full_text = issue.full_text();
        let lower = full_text.to_ascii_lowercase();
        let has_defer = contains_word(&lower, "defer")
            || contains_word(&lower, "deferred")
            || contains_word(&lower, "postpone")
            || lower.contains("defer ");
        if !has_defer {
            continue;
        }

        let has_reason = ["because", "due to", "reason", "rationale"]
            .iter()
            .any(|needle| lower.contains(needle));
        let has_follow_up = extract_bead_ids(&full_text)
            .into_iter()
            .any(|id| id != issue.id.to_ascii_lowercase());

        if has_reason && has_follow_up {
            continue;
        }

        let mut detail = String::new();
        if !has_reason {
            detail.push_str("missing explicit reason");
        }
        if !has_reason && !has_follow_up {
            detail.push_str("; ");
        }
        if !has_follow_up {
            detail.push_str("missing follow-up bead id (bd-...)");
        }

        violations.push(DeferredDecisionViolation {
            issue_id: issue.id.clone(),
            title: issue.title.clone(),
            detail,
        });
    }

    violations
}

fn find_spec_section_bounds(spec_lines: &[&str], markers: &[&str]) -> Option<SectionBounds> {
    let start_idx = spec_lines.iter().position(|line| {
        let trimmed = line.trim();
        if !is_markdown_heading(trimmed) {
            return false;
        }

        let heading_lower = strip_markdown_heading(trimmed).to_ascii_lowercase();
        markers
            .iter()
            .map(|marker| marker.to_ascii_lowercase())
            .any(|marker| heading_lower.contains(&marker))
    })?;

    let start = start_idx + 1;
    let mut end = spec_lines.len();
    for (idx, line) in spec_lines.iter().enumerate().skip(start) {
        if is_markdown_heading(line.trim()) {
            end = idx;
            break;
        }
    }
    Some(SectionBounds { start, end })
}

fn parse_exclusion_items(spec_lines: &[&str], section: SectionBounds) -> Vec<ExclusionItem> {
    let mut items = Vec::new();
    for (line_idx, line) in spec_lines
        .iter()
        .enumerate()
        .take(section.end)
        .skip(section.start)
    {
        let raw = line.trim();
        let Some(body) = strip_bullet_prefix(raw) else {
            continue;
        };

        let clean = clean_markdown_inline(body);
        if clean.is_empty() {
            continue;
        }
        let (name, rationale) = split_name_and_rationale(&clean);
        items.push(ExclusionItem {
            line_no: line_idx + 1,
            name,
            rationale,
            line_text: truncate_line(raw, MAX_LINE_TEXT_LEN),
        });
    }
    items
}

fn split_name_and_rationale(input: &str) -> (String, String) {
    let delimiters = [" — ", " – ", ": ", " - "];
    for delimiter in delimiters {
        if let Some((name, rationale)) = input.split_once(delimiter) {
            return (
                normalize_whitespace(name),
                normalize_whitespace(rationale.trim()),
            );
        }
    }

    let lower = input.to_ascii_lowercase();
    if let Some(pos) = lower.find(" because ") {
        let (name, rationale_with_keyword) = input.split_at(pos);
        let rationale = rationale_with_keyword.trim_start_matches(" because ");
        return (normalize_whitespace(name), normalize_whitespace(rationale));
    }

    (normalize_whitespace(input), String::new())
}

fn clean_markdown_inline(input: &str) -> String {
    let stripped = input
        .replace("**", "")
        .replace(['*', '`', '[', ']'], "")
        .replace(['(', ')'], " ");
    normalize_whitespace(&stripped)
}

fn strip_bullet_prefix(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        return Some(trimmed[2..].trim());
    }

    let digit_count = trimmed.chars().take_while(char::is_ascii_digit).count();
    if digit_count > 0 {
        let suffix = &trimmed[digit_count..];
        if let Some(stripped) = suffix.strip_prefix(". ") {
            return Some(stripped.trim());
        }
    }

    None
}

fn references_section_15(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    SECTION_15_REF_TOKENS
        .iter()
        .any(|token| lower.contains(&token.to_ascii_lowercase()))
}

fn contains_scope_phrase(line_lower: &str, phrase: &str) -> bool {
    if phrase != "later" {
        return line_lower.contains(phrase);
    }

    contains_word(line_lower, "later")
        && (line_lower.contains("v1")
            || line_lower.contains("scope")
            || line_lower.contains("defer"))
}

fn contains_word(text: &str, needle: &str) -> bool {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .any(|token| token.eq_ignore_ascii_case(needle))
}

fn collect_markdown_files(workspace_root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Ok(entries) = host_fs::read_dir_paths(workspace_root) {
        for path in entries {
            if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("md") {
                out.push(path);
                continue;
            }

            if path.is_dir() && path.file_name().and_then(|name| name.to_str()) == Some("docs") {
                collect_files_with_extension(&path, "md", &mut out);
            }
        }
    }

    out.sort();
    out
}

fn collect_implementation_source_files(workspace_root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for rel in ["crates", "src", "tests"] {
        let path = workspace_root.join(rel);
        if path.is_dir() {
            collect_files_with_extension(&path, "rs", &mut out);
        }
    }
    out.sort();
    out
}

fn collect_files_with_extension(root: &Path, extension: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = host_fs::read_dir_paths(root) else {
        return;
    };

    let mut paths = entries;
    paths.sort();

    for path in paths {
        if path.is_dir() {
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if matches!(
                name,
                ".git"
                    | ".beads"
                    | "target"
                    | "target_lilaccave"
                    | "legacy_sqlite_code"
                    | "node_modules"
            ) {
                continue;
            }
            collect_files_with_extension(&path, extension, out);
            continue;
        }

        if path.extension().and_then(|ext| ext.to_str()) == Some(extension) {
            out.push(path);
        }
    }
}

fn relativize_path(workspace_root: &Path, path: &Path) -> String {
    path.strip_prefix(workspace_root).map_or_else(
        |_| path.display().to_string(),
        |rel| rel.display().to_string(),
    )
}

fn extract_bead_ids(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '.'))
        .filter_map(|token| {
            if token.len() < 4 {
                return None;
            }
            let lower = token.to_ascii_lowercase();
            if lower.starts_with("bd-") {
                Some(lower)
            } else {
                None
            }
        })
        .collect()
}

fn is_superseded_close_reason(reason_lower: &str) -> bool {
    ["duplicate", "superseded", "merged", "content merged"]
        .iter()
        .any(|needle| reason_lower.contains(needle))
}

fn find_section(lines: &[&str], accepted_headers: &[&str]) -> Option<SectionBounds> {
    let mut section_start = None;
    for (idx, line) in lines.iter().enumerate() {
        if !is_section_heading(line) {
            continue;
        }
        if heading_matches(line, accepted_headers) {
            section_start = Some(idx + 1);
            break;
        }
    }

    let start = section_start?;
    let mut end = lines.len();
    for (idx, line) in lines.iter().enumerate().skip(start) {
        if is_section_heading(line) {
            end = idx;
            break;
        }
    }

    Some(SectionBounds { start, end })
}

impl SectionBounds {
    fn slice<'a>(self, lines: &'a [&str]) -> &'a [&'a str] {
        &lines[self.start..self.end]
    }
}

fn is_section_heading(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }

    if is_markdown_heading(trimmed) {
        return true;
    }

    trimmed.ends_with(':') && trimmed.chars().any(char::is_alphabetic)
}

fn heading_matches(line: &str, accepted_headers: &[&str]) -> bool {
    let mut normalized = line.trim().to_ascii_lowercase();
    if normalized.starts_with('#') {
        normalized = normalized.trim_start_matches('#').trim().to_string();
    }
    normalized = normalized.trim_end_matches(':').trim().to_string();

    accepted_headers
        .iter()
        .any(|header| normalized.starts_with(&header.to_ascii_lowercase()))
}

fn contains_prefixed_identifier(lines: &[&str], prefixes: &[&str]) -> bool {
    lines.iter().any(|line| {
        line.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .filter(|token| !token.is_empty())
            .any(|token| {
                let lower = token.to_ascii_lowercase();
                prefixes.iter().any(|prefix| lower.starts_with(prefix))
            })
    })
}

fn has_log_level_marker(lines: &[&str]) -> bool {
    lines.iter().any(|line| {
        line.split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
            .filter(|token| !token.is_empty())
            .any(|token| {
                matches!(
                    token.to_ascii_uppercase().as_str(),
                    "DEBUG" | "INFO" | "WARN" | "ERROR"
                )
            })
    })
}

fn contains_bullet(lines: &[&str]) -> bool {
    lines.iter().any(|line| {
        let trimmed = line.trim();
        if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
            return true;
        }

        let digit_count = trimmed.chars().take_while(char::is_ascii_digit).count();
        if digit_count == 0 {
            return false;
        }

        let suffix = &trimmed[digit_count..];
        suffix.starts_with(". ")
    })
}

fn is_code_fence_delimiter(trimmed: &str) -> bool {
    trimmed.starts_with("```")
}

fn is_table_separator(trimmed: &str) -> bool {
    if !trimmed.starts_with('|') {
        return false;
    }
    let mut has_dash = false;
    for ch in trimmed.chars() {
        if ch == '-' {
            has_dash = true;
            continue;
        }
        if matches!(ch, '|' | ':' | ' ' | '\t') {
            continue;
        }
        return false;
    }
    has_dash
}

fn is_markdown_heading(trimmed: &str) -> bool {
    let hashes = trimmed.chars().take_while(|ch| *ch == '#').count();
    hashes > 0 && trimmed.chars().nth(hashes) == Some(' ')
}

fn strip_markdown_heading(trimmed: &str) -> &str {
    let hashes = trimmed.chars().take_while(|ch| *ch == '#').count();
    if hashes > 0 && trimmed.chars().nth(hashes) == Some(' ') {
        trimmed[hashes + 1..].trim()
    } else {
        trimmed
    }
}

fn truncate_line(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }

    let mut out = String::new();
    for ch in text.chars().take(max_chars.saturating_sub(1)) {
        out.push(ch);
    }
    out.push('…');
    out
}

fn keyword_hint(text: &str) -> String {
    let mut keywords = Vec::new();
    for token in text
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|token| token.len() >= 4)
    {
        let lower = token.to_ascii_lowercase();
        if keywords.contains(&lower) {
            continue;
        }
        keywords.push(lower);
        if keywords.len() == 3 {
            break;
        }
    }

    if keywords.is_empty() {
        "br search \"<keyword>\"".to_string()
    } else {
        format!("br search \"{}\"", keywords.join(" "))
    }
}

fn coverage_basis_points(covered: usize, total: usize) -> u32 {
    if total == 0 {
        return 10_000;
    }

    let covered_u128 = covered as u128;
    let total_u128 = total as u128;
    let bps = covered_u128.saturating_mul(10_000) / total_u128;
    u32::try_from(bps).unwrap_or(0)
}

fn internal_error(message: impl Into<String>) -> FrankenError {
    FrankenError::Internal(message.into())
}

impl Issue {
    fn full_text(&self) -> String {
        let mut text = self.description.clone();
        if !self.acceptance_criteria.is_empty() {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&self.acceptance_criteria);
        }
        for comment in &self.comments {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(comment);
        }
        text
    }
}
