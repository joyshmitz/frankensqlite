//! Deterministic backend-wiring mode checks for bd-zjisk.1.
//!
//! This suite verifies startup/query behavior across runtime and certifying
//! modes without relying on nondeterministic workload generation.

use fsqlite::Connection;
use fsqlite_types::SqliteValue;
use tempfile::tempdir;

const BEAD_ID: &str = "bd-zjisk.1";
const SCENARIO_ID: &str = "BACKEND-WIRING-MODES";
const SEED: u64 = 3520;

fn setup_join_fixture(conn: &Connection) {
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT);")
        .expect("create items");
    conn.execute("CREATE TABLE tags (item_id INTEGER, tag TEXT);")
        .expect("create tags");
    conn.execute("INSERT INTO items VALUES (1, 'alpha');")
        .expect("insert item");
    conn.execute("INSERT INTO tags VALUES (1, 'fruit');")
        .expect("insert tag");
}

#[test]
fn certifying_mode_strict_rejects_fallback_query() {
    let temp = tempdir().expect("tempdir");
    let db_path = temp.path().join("zjisk1-certifying-strict.db");
    let db_path = db_path.to_string_lossy().to_string();
    let conn = Connection::open(&db_path).expect("open connection");

    setup_join_fixture(&conn);
    conn.execute("PRAGMA fsqlite.parity_cert=ON;")
        .expect("enable parity cert");
    conn.execute("PRAGMA fsqlite.parity_cert_strict=ON;")
        .expect("enable strict cert mode");

    let sql = "SELECT items.name, tags.tag \
               FROM items JOIN tags ON items.id = tags.item_id;";
    let err = conn
        .query(sql)
        .expect_err("strict certifying mode must reject interpreted fallback");
    assert!(
        err.to_string()
            .contains("in-memory fallback disabled in strict parity-cert mode"),
        "unexpected error in certifying strict mode: {err}"
    );
}

#[test]
fn runtime_mode_allows_same_query_path() {
    let temp = tempdir().expect("tempdir");
    let db_path = temp.path().join("zjisk1-runtime-mode.db");
    let db_path = db_path.to_string_lossy().to_string();
    let conn = Connection::open(&db_path).expect("open connection");

    setup_join_fixture(&conn);
    conn.execute("PRAGMA fsqlite.parity_cert=OFF;")
        .expect("disable certifying mode");
    conn.execute("PRAGMA fsqlite.parity_cert_strict=ON;")
        .expect("strict flag may stay on in runtime mode");

    let rows = conn
        .query(
            "SELECT items.name, tags.tag \
             FROM items JOIN tags ON items.id = tags.item_id;",
        )
        .expect("runtime mode should allow fallback query");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].values()[0], SqliteValue::Text("alpha".to_owned()));
    assert_eq!(rows[0].values()[1], SqliteValue::Text("fruit".to_owned()));
}

#[test]
fn mode_pragmas_report_expected_state() {
    let conn = Connection::open(":memory:").expect("open in-memory connection");
    let parity_rows = conn
        .query("PRAGMA fsqlite.parity_cert;")
        .expect("query parity_cert");
    assert_eq!(
        parity_rows[0].values()[0],
        SqliteValue::Integer(1),
        "parity_cert must default to ON"
    );

    let strict_rows = conn
        .query("PRAGMA fsqlite.parity_cert_strict;")
        .expect("query parity_cert_strict");
    assert_eq!(
        strict_rows[0].values()[0],
        SqliteValue::Integer(0),
        "parity_cert_strict defaults OFF for non-certifying runtime by default"
    );
}

#[test]
fn bead_metadata_constants_are_stable_for_replay() {
    assert_eq!(BEAD_ID, "bd-zjisk.1");
    assert_eq!(SCENARIO_ID, "BACKEND-WIRING-MODES");
    assert_eq!(SEED, 3520);
}
