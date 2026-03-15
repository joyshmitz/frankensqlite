//! Contract tests for db300_verification_matrix.toml (bd-db300.7.5.1).
//!
//! The goal is to keep Track I / Track J verification obligations concrete:
//! every code-changing bead must either map to explicit unit/property/fuzz
//! coverage or be explicitly excluded with a reason and gap-conversion rule.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use serde::Deserialize;

const BEAD_ID: &str = "bd-db300.7.5.1";
const CONTRACT_PATH: &str = "db300_verification_matrix.toml";

const INCLUDED_BEADS: [&str; 12] = [
    "bd-db300.10.1",
    "bd-db300.10.2",
    "bd-db300.10.3",
    "bd-db300.10.4",
    "bd-db300.10.5",
    "bd-db300.10.6",
    "bd-db300.10.7",
    "bd-db300.10.8",
    "bd-db300.10.9",
    "bd-db300.9.1",
    "bd-db300.9.2",
    "bd-db300.9.5",
];

const INCLUDED_BEADS_CONTD: [&str; 4] = [
    "bd-db300.9.6",
    "bd-db300.9.7",
    "bd-db300.9.8",
    "bd-db300.9.10",
];

const EXCLUDED_BEADS: [&str; 11] = [
    "bd-db300.9.3",
    "bd-db300.9.4",
    "bd-db300.9.9",
    "bd-db300.9.11",
    "bd-db300.9.11.1",
    "bd-db300.9.11.2",
    "bd-db300.9.11.3",
    "bd-db300.9.11.4",
    "bd-db300.9.11.5",
    "bd-db300.9.11.6",
    "bd-db300.9.12",
];

const EXCLUDED_BEADS_CONTD: [&str; 1] = ["bd-db300.10.10"];

#[derive(Debug, Deserialize)]
struct VerificationMatrixDocument {
    meta: MatrixMeta,
    #[serde(default, rename = "coverage_row")]
    coverage_rows: Vec<CoverageRow>,
    #[serde(default, rename = "excluded_bead")]
    excluded_beads: Vec<ExcludedBead>,
}

#[derive(Debug, Deserialize)]
struct MatrixMeta {
    schema_version: String,
    bead_id: String,
    track_id: String,
    generated_at: String,
    contract_owner: String,
}

#[derive(Debug, Deserialize)]
struct CoverageRow {
    bead_id: String,
    title: String,
    verification_owner: String,
    owner_crates: Vec<String>,
    unit_test_targets: Vec<String>,
    requires_property_coverage: bool,
    #[serde(default)]
    property_test_targets: Vec<String>,
    #[serde(default)]
    property_coverage_reason: String,
    requires_fuzz_coverage: bool,
    #[serde(default)]
    fuzz_test_targets: Vec<String>,
    #[serde(default)]
    fuzz_coverage_reason: String,
    gap_conversion_rule: String,
}

#[derive(Debug, Deserialize)]
struct ExcludedBead {
    bead_id: String,
    title: String,
    reason: String,
    gap_rule: String,
}

fn load_contract() -> VerificationMatrixDocument {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(CONTRACT_PATH);
    let content = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "failed to read {} at {}: {error}",
            CONTRACT_PATH,
            path.display()
        )
    });
    toml::from_str::<VerificationMatrixDocument>(&content).unwrap_or_else(|error| {
        panic!(
            "failed to parse {} at {}: {error}",
            CONTRACT_PATH,
            path.display()
        )
    })
}

fn expected_included_beads() -> BTreeSet<&'static str> {
    INCLUDED_BEADS
        .into_iter()
        .chain(INCLUDED_BEADS_CONTD)
        .collect::<BTreeSet<_>>()
}

fn expected_excluded_beads() -> BTreeSet<&'static str> {
    EXCLUDED_BEADS
        .into_iter()
        .chain(EXCLUDED_BEADS_CONTD)
        .collect::<BTreeSet<_>>()
}

#[test]
fn manifest_meta_is_pinned_to_g5_1_contract() {
    let document = load_contract();
    assert_eq!(document.meta.schema_version, "1.0.0");
    assert_eq!(document.meta.bead_id, BEAD_ID);
    assert_eq!(document.meta.track_id, "bd-db300.7.5");
    assert!(
        !document.meta.generated_at.trim().is_empty(),
        "generated_at must not be blank"
    );
    assert!(
        !document.meta.contract_owner.trim().is_empty(),
        "contract_owner must not be blank"
    );
}

#[test]
fn included_and_excluded_sets_cover_the_revised_track_i_and_j_surface() {
    let document = load_contract();

    let included = document
        .coverage_rows
        .iter()
        .map(|row| row.bead_id.as_str())
        .collect::<BTreeSet<_>>();
    let excluded = document
        .excluded_beads
        .iter()
        .map(|row| row.bead_id.as_str())
        .collect::<BTreeSet<_>>();

    assert_eq!(
        included,
        expected_included_beads(),
        "included code-changing bead set must stay exact and deterministic"
    );
    assert_eq!(
        excluded,
        expected_excluded_beads(),
        "excluded non-code-changing bead set must stay exact and deterministic"
    );
    assert!(
        included.is_disjoint(&excluded),
        "included and excluded bead sets must not overlap"
    );

    let combined = included.union(&excluded).copied().collect::<BTreeSet<_>>();
    let expected_all = expected_included_beads()
        .union(&expected_excluded_beads())
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        combined, expected_all,
        "every revised Track I / Track J bead must be covered exactly once"
    );
}

#[test]
fn each_coverage_row_has_concrete_owner_unit_surface_and_gap_rule() {
    let document = load_contract();

    for row in &document.coverage_rows {
        assert!(
            !row.title.trim().is_empty(),
            "blank title for {}",
            row.bead_id
        );
        assert!(
            !row.verification_owner.trim().is_empty(),
            "blank verification_owner for {}",
            row.bead_id
        );
        assert!(
            !row.owner_crates.is_empty(),
            "missing owner_crates for {}",
            row.bead_id
        );
        for owner in &row.owner_crates {
            assert!(
                !owner.trim().is_empty(),
                "blank owner crate in {}",
                row.bead_id
            );
        }
        assert!(
            !row.unit_test_targets.is_empty(),
            "missing unit_test_targets for {}",
            row.bead_id
        );
        for target in &row.unit_test_targets {
            assert!(
                target.contains("::tests::") || target.contains("/tests/"),
                "unit target must name a concrete test module/function surface: {} => {}",
                row.bead_id,
                target
            );
        }

        if row.requires_property_coverage {
            assert!(
                !row.property_test_targets.is_empty(),
                "property coverage required but no targets named for {}",
                row.bead_id
            );
        } else {
            assert!(
                !row.property_coverage_reason.trim().is_empty(),
                "property omission reason required for {}",
                row.bead_id
            );
        }

        if row.requires_fuzz_coverage {
            assert!(
                !row.fuzz_test_targets.is_empty(),
                "fuzz coverage required but no targets named for {}",
                row.bead_id
            );
        } else {
            assert!(
                !row.fuzz_coverage_reason.trim().is_empty(),
                "fuzz omission reason required for {}",
                row.bead_id
            );
        }

        assert!(
            !row.gap_conversion_rule.trim().is_empty(),
            "gap_conversion_rule required for {}",
            row.bead_id
        );
    }
}

#[test]
fn key_high_impact_rows_name_the_expected_test_families() {
    let document = load_contract();

    let j4 = document
        .coverage_rows
        .iter()
        .find(|row| row.bead_id == "bd-db300.10.4")
        .expect("missing J4 row");
    assert!(
        j4.unit_test_targets
            .iter()
            .any(|target| target.contains("statement_cache_reuses_prepare")),
        "J4 must name a concrete prepare-cache reuse test"
    );
    assert!(
        j4.unit_test_targets
            .iter()
            .any(|target| target.contains("invalidates_on_schema_change")),
        "J4 must name a concrete schema-invalidation test"
    );

    let i7 = document
        .coverage_rows
        .iter()
        .find(|row| row.bead_id == "bd-db300.9.7")
        .expect("missing I7 row");
    assert!(
        i7.requires_property_coverage,
        "I7 must require property coverage"
    );
    assert!(
        i7.property_test_targets
            .iter()
            .any(|target| target.contains("preserves_btree_invariants")),
        "I7 must name randomized B-tree invariant coverage"
    );

    let i10 = document
        .coverage_rows
        .iter()
        .find(|row| row.bead_id == "bd-db300.9.10")
        .expect("missing I10 row");
    assert!(
        i10.unit_test_targets
            .iter()
            .any(|target| target.contains("restarts_locally_on_version_mismatch")),
        "I10 must name the parent-version mismatch local-restart test"
    );
}

#[test]
fn excluded_rows_are_explicit_and_not_hand_wavy() {
    let document = load_contract();

    for row in &document.excluded_beads {
        assert!(
            !row.title.trim().is_empty(),
            "blank title for {}",
            row.bead_id
        );
        assert!(
            row.reason.trim().len() >= 32,
            "excluded bead {} must carry a non-trivial reason",
            row.bead_id
        );
        assert!(
            row.gap_rule.trim().len() >= 32,
            "excluded bead {} must carry a concrete gap rule",
            row.bead_id
        );
    }
}
