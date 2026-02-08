#![cfg(unix)]

use std::path::Path;
use std::process::{Command, Output};

use fsqlite_types::LockLevel;
use fsqlite_types::cx::Cx;
use fsqlite_types::flags::VfsOpenFlags;
use fsqlite_vfs::shm::{SQLITE_SHM_SHARED, SQLITE_SHM_UNLOCK, wal_read_lock_slot};
use fsqlite_vfs::{UnixVfs, Vfs, VfsFile};
use tempfile::tempdir;

fn sqlite3_available() -> bool {
    Command::new("sqlite3").arg("--version").output().is_ok()
}

fn sqlite3_exec(db_path: &Path, sql: &str) -> Output {
    Command::new("sqlite3")
        .arg(db_path)
        .arg(sql)
        .output()
        .expect("sqlite3 command should execute")
}

fn setup_wal_db(db_path: &Path) {
    let output = sqlite3_exec(
        db_path,
        "PRAGMA journal_mode=WAL;\
         CREATE TABLE IF NOT EXISTS t(x INTEGER);\
         DELETE FROM t;\
         INSERT INTO t VALUES (1);",
    );
    assert!(
        output.status.success(),
        "sqlite3 setup failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_legacy_sqlite_reader_coexists() {
    if !sqlite3_available() {
        return;
    }

    let tmp = tempdir().expect("tempdir should be created");
    let db_path = tmp.path().join("interop_reader.db");
    setup_wal_db(&db_path);

    let cx = Cx::new();
    let vfs = UnixVfs::new();
    let flags = VfsOpenFlags::MAIN_DB | VfsOpenFlags::READWRITE;
    let (mut coordinator, _) = vfs
        .open(&cx, Some(&db_path), flags)
        .expect("frankensqlite unix vfs open should succeed");

    // Legacy sqlite3 interop is validated through classic DB lock grades.
    // WAL-slot protocol semantics are validated in `fsqlite-vfs` unit tests.
    coordinator
        .lock(&cx, LockLevel::Reserved)
        .expect("coordinator must hold reserved lock");

    let reader_output = sqlite3_exec(&db_path, "SELECT COUNT(*) FROM t;");
    assert!(
        reader_output.status.success(),
        "legacy sqlite3 reader should coexist while coordinator lock held: {}",
        String::from_utf8_lossy(&reader_output.stderr)
    );
    let stdout = String::from_utf8_lossy(&reader_output.stdout);
    assert!(
        stdout.contains('1'),
        "legacy reader should observe consistent data, got: {stdout}"
    );

    coordinator
        .unlock(&cx, LockLevel::None)
        .expect("coordinator must release lock");
}

#[test]
fn test_legacy_sqlite_writer_gets_busy() {
    if !sqlite3_available() {
        return;
    }

    let tmp = tempdir().expect("tempdir should be created");
    let db_path = tmp.path().join("interop_writer_busy.db");
    setup_wal_db(&db_path);

    let cx = Cx::new();
    let vfs = UnixVfs::new();
    let flags = VfsOpenFlags::MAIN_DB | VfsOpenFlags::READWRITE;
    let (mut coordinator, _) = vfs
        .open(&cx, Some(&db_path), flags)
        .expect("frankensqlite unix vfs open should succeed");

    // Use EXCLUSIVE to force sqlite3 writer `SQLITE_BUSY` from another process.
    coordinator
        .lock(&cx, LockLevel::Exclusive)
        .expect("coordinator must hold exclusive lock");

    let writer_output = sqlite3_exec(
        &db_path,
        "PRAGMA busy_timeout=100;\
         BEGIN IMMEDIATE;\
         INSERT INTO t VALUES (2);\
         COMMIT;",
    );
    assert!(
        !writer_output.status.success(),
        "legacy sqlite3 writer must fail with SQLITE_BUSY while coordinator lock held"
    );
    let stderr = String::from_utf8_lossy(&writer_output.stderr).to_lowercase();
    assert!(
        stderr.contains("locked") || stderr.contains("busy"),
        "expected SQLITE_BUSY/locked error, got stderr: {stderr}"
    );

    coordinator
        .unlock(&cx, LockLevel::None)
        .expect("coordinator must release lock");
}

#[test]
fn test_e2e_hybrid_shm_interop_with_c_sqlite() {
    if !sqlite3_available() {
        return;
    }

    let tmp = tempdir().expect("tempdir should be created");
    let db_path = tmp.path().join("interop_e2e.db");
    setup_wal_db(&db_path);

    let cx = Cx::new();
    let vfs = UnixVfs::new();
    let flags = VfsOpenFlags::MAIN_DB | VfsOpenFlags::READWRITE;
    let (mut coordinator, _) = vfs
        .open(&cx, Some(&db_path), flags)
        .expect("frankensqlite unix vfs open should succeed");
    let (mut reader2, _) = vfs
        .open(&cx, Some(&db_path), flags)
        .expect("second unix vfs open should succeed");

    coordinator
        .lock(&cx, LockLevel::Reserved)
        .expect("coordinator must hold reserved lock");

    let updated = coordinator
        .compat_reader_acquire_wal_read_lock(&cx, 0, 123)
        .expect("reader protocol update should succeed");
    assert!(updated, "first reader must update aReadMark");
    let joined = reader2
        .compat_reader_acquire_wal_read_lock(&cx, 0, 123)
        .expect("second reader protocol join should succeed");
    assert!(
        !joined,
        "second reader must join existing aReadMark with SHARED"
    );

    let reader_output = sqlite3_exec(&db_path, "SELECT COUNT(*) FROM t;");
    assert!(
        reader_output.status.success(),
        "legacy sqlite3 reader should succeed while coordinator is alive: {}",
        String::from_utf8_lossy(&reader_output.stderr)
    );

    coordinator
        .unlock(&cx, LockLevel::None)
        .expect("coordinator should release reserved lock before writer exclusion phase");
    // Escalate only for the writer-exclusion assertion.
    coordinator
        .lock(&cx, LockLevel::Exclusive)
        .expect("coordinator must hold exclusive lock for writer exclusion phase");

    let writer_output = sqlite3_exec(
        &db_path,
        "PRAGMA busy_timeout=100;\
         BEGIN IMMEDIATE;\
         INSERT INTO t VALUES (99);\
         COMMIT;",
    );
    assert!(
        !writer_output.status.success(),
        "legacy sqlite3 writer must observe SQLITE_BUSY while coordinator lock is held"
    );

    let read_slot0 = wal_read_lock_slot(0).expect("reader slot 0 should exist");
    reader2
        .shm_lock(&cx, read_slot0, 1, SQLITE_SHM_UNLOCK | SQLITE_SHM_SHARED)
        .expect("reader2 shared unlock should succeed");
    coordinator
        .shm_lock(&cx, read_slot0, 1, SQLITE_SHM_UNLOCK | SQLITE_SHM_SHARED)
        .expect("coordinator shared unlock should succeed");
    coordinator
        .unlock(&cx, LockLevel::None)
        .expect("coordinator must release lock");
}
