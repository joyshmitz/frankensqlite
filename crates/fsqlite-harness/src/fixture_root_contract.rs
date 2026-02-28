//! Canonical fixture-root contract loader for Track C corpus gates.
//!
//! This module enforces a single source of truth for fixture roots and
//! cardinality floors, rooted in `corpus_manifest.toml`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Default canonical fixture-root manifest path relative to workspace root.
pub const DEFAULT_FIXTURE_ROOT_MANIFEST_PATH: &str = "corpus_manifest.toml";
/// Expected schema version for `[fixture_roots]`.
pub const FIXTURE_ROOT_SCHEMA_VERSION: &str = "1.0.0";

/// Loaded canonical fixture-root contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixtureRootContract {
    /// Absolute manifest path.
    pub manifest_path: PathBuf,
    /// SHA-256 of the raw manifest payload.
    pub manifest_sha256: String,
    /// Absolute fixtures directory.
    pub fixtures_dir: PathBuf,
    /// Absolute SLT directory.
    pub slt_dir: PathBuf,
    /// Minimum fixture JSON files required.
    pub min_fixture_json_files: usize,
    /// Minimum fixture entries required.
    pub min_fixture_entries: usize,
    /// Minimum fixture SQL statements required.
    pub min_fixture_sql_statements: usize,
    /// Minimum SLT files required.
    pub min_slt_files: usize,
    /// Minimum SLT entries required.
    pub min_slt_entries: usize,
    /// Minimum SLT SQL statements required.
    pub min_slt_sql_statements: usize,
    /// Required category families that must exist in `category_floors`.
    pub required_category_families: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CorpusManifestDocument {
    fixture_roots: Option<FixtureRootsSection>,
    #[serde(default)]
    category_floors: Vec<CategoryFloorEntry>,
}

#[derive(Debug, Deserialize)]
struct FixtureRootsSection {
    schema_version: String,
    fixtures_dir: String,
    slt_dir: String,
    min_fixture_json_files: usize,
    min_fixture_entries: usize,
    min_fixture_sql_statements: usize,
    min_slt_files: usize,
    min_slt_entries: usize,
    min_slt_sql_statements: usize,
    #[serde(default)]
    required_category_families: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CategoryFloorEntry {
    category: String,
    min_entries: usize,
}

/// Load and validate the canonical fixture-root contract.
///
/// # Errors
///
/// Returns `Err` if the manifest is missing, malformed, or violates contract
/// requirements.
pub fn load_fixture_root_contract(
    workspace_root: &Path,
    manifest_path: &Path,
) -> Result<FixtureRootContract, String> {
    let manifest_path = resolve_workspace_path(workspace_root, manifest_path);
    if !manifest_path.is_file() {
        return Err(format!(
            "fixture_root_manifest_missing path={}",
            manifest_path.display()
        ));
    }

    let raw = fs::read_to_string(&manifest_path).map_err(|error| {
        format!(
            "fixture_root_manifest_read_failed path={} error={error}",
            manifest_path.display()
        )
    })?;
    if raw.trim().is_empty() {
        return Err(format!(
            "fixture_root_manifest_empty path={}",
            manifest_path.display()
        ));
    }

    let manifest_sha256 = sha256_hex(raw.as_bytes());
    let doc = toml::from_str::<CorpusManifestDocument>(&raw).map_err(|error| {
        format!(
            "fixture_root_manifest_parse_failed path={} error={error}",
            manifest_path.display()
        )
    })?;

    let section = doc.fixture_roots.ok_or_else(|| {
        format!(
            "fixture_root_manifest_missing_section path={} section=fixture_roots",
            manifest_path.display()
        )
    })?;

    if section.schema_version != FIXTURE_ROOT_SCHEMA_VERSION {
        return Err(format!(
            "fixture_root_schema_version_mismatch expected={} observed={}",
            FIXTURE_ROOT_SCHEMA_VERSION, section.schema_version
        ));
    }

    let fixtures_dir = non_empty_string("fixture_roots.fixtures_dir", &section.fixtures_dir)?;
    let slt_dir = non_empty_string("fixture_roots.slt_dir", &section.slt_dir)?;
    require_positive(
        "fixture_roots.min_fixture_json_files",
        section.min_fixture_json_files,
    )?;
    require_positive(
        "fixture_roots.min_fixture_entries",
        section.min_fixture_entries,
    )?;
    require_positive(
        "fixture_roots.min_fixture_sql_statements",
        section.min_fixture_sql_statements,
    )?;
    require_positive("fixture_roots.min_slt_files", section.min_slt_files)?;
    require_positive("fixture_roots.min_slt_entries", section.min_slt_entries)?;
    require_positive(
        "fixture_roots.min_slt_sql_statements",
        section.min_slt_sql_statements,
    )?;

    let required_category_families =
        normalize_required_categories(section.required_category_families)?;
    validate_required_categories(&required_category_families, &doc.category_floors)?;

    Ok(FixtureRootContract {
        manifest_path,
        manifest_sha256,
        fixtures_dir: resolve_workspace_path(workspace_root, Path::new(&fixtures_dir)),
        slt_dir: resolve_workspace_path(workspace_root, Path::new(&slt_dir)),
        min_fixture_json_files: section.min_fixture_json_files,
        min_fixture_entries: section.min_fixture_entries,
        min_fixture_sql_statements: section.min_fixture_sql_statements,
        min_slt_files: section.min_slt_files,
        min_slt_entries: section.min_slt_entries,
        min_slt_sql_statements: section.min_slt_sql_statements,
        required_category_families,
    })
}

/// Enforce runtime fixture/slt settings against the canonical contract.
///
/// # Errors
///
/// Returns `Err` when any path/threshold differs from canonical values.
#[allow(clippy::too_many_arguments)]
pub fn enforce_fixture_contract_alignment(
    contract: &FixtureRootContract,
    fixtures_dir: &Path,
    slt_dir: &Path,
    min_fixture_json_files: usize,
    min_fixture_entries: usize,
    min_fixture_sql_statements: usize,
    min_slt_files: usize,
    min_slt_entries: usize,
    min_slt_sql_statements: usize,
) -> Result<(), String> {
    let mut mismatches = Vec::new();
    if !same_path(fixtures_dir, &contract.fixtures_dir) {
        mismatches.push(format!(
            "fixtures_dir mismatch expected={} observed={}",
            contract.fixtures_dir.display(),
            fixtures_dir.display()
        ));
    }
    if !same_path(slt_dir, &contract.slt_dir) {
        mismatches.push(format!(
            "slt_dir mismatch expected={} observed={}",
            contract.slt_dir.display(),
            slt_dir.display()
        ));
    }
    if min_fixture_json_files != contract.min_fixture_json_files {
        mismatches.push(format!(
            "min_fixture_json_files mismatch expected={} observed={}",
            contract.min_fixture_json_files, min_fixture_json_files
        ));
    }
    if min_fixture_entries != contract.min_fixture_entries {
        mismatches.push(format!(
            "min_fixture_entries mismatch expected={} observed={}",
            contract.min_fixture_entries, min_fixture_entries
        ));
    }
    if min_fixture_sql_statements != contract.min_fixture_sql_statements {
        mismatches.push(format!(
            "min_fixture_sql_statements mismatch expected={} observed={}",
            contract.min_fixture_sql_statements, min_fixture_sql_statements
        ));
    }
    if min_slt_files != contract.min_slt_files {
        mismatches.push(format!(
            "min_slt_files mismatch expected={} observed={}",
            contract.min_slt_files, min_slt_files
        ));
    }
    if min_slt_entries != contract.min_slt_entries {
        mismatches.push(format!(
            "min_slt_entries mismatch expected={} observed={}",
            contract.min_slt_entries, min_slt_entries
        ));
    }
    if min_slt_sql_statements != contract.min_slt_sql_statements {
        mismatches.push(format!(
            "min_slt_sql_statements mismatch expected={} observed={}",
            contract.min_slt_sql_statements, min_slt_sql_statements
        ));
    }

    if mismatches.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "fixture_root_contract_alignment_failed manifest={} {}",
            contract.manifest_path.display(),
            mismatches.join("; ")
        ))
    }
}

fn validate_required_categories(
    required_category_families: &[String],
    category_floors: &[CategoryFloorEntry],
) -> Result<(), String> {
    let mut categories = BTreeSet::new();
    for floor in category_floors {
        if floor.min_entries == 0 {
            return Err(format!(
                "fixture_root_manifest_invalid_category_floor category={} min_entries=0",
                floor.category
            ));
        }
        categories.insert(floor.category.trim().to_owned());
    }

    let mut missing = Vec::new();
    for required in required_category_families {
        if !categories.contains(required) {
            missing.push(required.clone());
        }
    }

    if missing.is_empty() {
        Ok(())
    } else {
        missing.sort();
        Err(format!(
            "fixture_root_manifest_missing_required_category_floors missing={}",
            missing.join(",")
        ))
    }
}

fn normalize_required_categories(values: Vec<String>) -> Result<Vec<String>, String> {
    let mut categories = values
        .into_iter()
        .map(|value| value.trim().to_owned())
        .collect::<Vec<_>>();
    categories.retain(|value| !value.is_empty());
    if categories.is_empty() {
        return Err(
            "fixture_root_manifest_required_category_families_must_not_be_empty".to_owned(),
        );
    }

    let mut unique = BTreeSet::new();
    let mut normalized = Vec::with_capacity(categories.len());
    for category in categories {
        if unique.insert(category.clone()) {
            normalized.push(category);
        }
    }
    Ok(normalized)
}

fn non_empty_string(field_name: &str, value: &str) -> Result<String, String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(format!("{field_name} must be non-empty"))
    } else {
        Ok(trimmed.to_owned())
    }
}

fn require_positive(field_name: &str, value: usize) -> Result<(), String> {
    if value == 0 {
        Err(format!("{field_name} must be > 0"))
    } else {
        Ok(())
    }
}

fn same_path(left: &Path, right: &Path) -> bool {
    match (left.canonicalize(), right.canonicalize()) {
        (Ok(left_path), Ok(right_path)) => left_path == right_path,
        _ => left == right,
    }
}

fn resolve_workspace_path(workspace_root: &Path, path: &Path) -> PathBuf {
    if path.is_relative() {
        workspace_root.join(path)
    } else {
        path.to_path_buf()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    format!("{digest:x}")
}
