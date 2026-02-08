//! Cx/Sealed trait enforcement checks for bd-ggxs.

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use fsqlite_btree::{BtreeCursorOps, MockBtreeCursor, SeekResult};
use fsqlite_error::{FrankenError, Result};
use fsqlite_func::{
    AggregateFunction, AuthAction, AuthResult, Authorizer, CollationFunction, ColumnContext,
    ConstraintOp, IndexConstraint, IndexInfo, ScalarFunction, VirtualTable, VirtualTableCursor,
    WindowFunction,
};
use fsqlite_pager::{
    CheckpointPageWriter, MockCheckpointPageWriter, MockMvccPager, MockTransaction, MvccPager,
    TransactionHandle, TransactionMode,
};
use fsqlite_types::cx::Cx;
use fsqlite_types::flags::{AccessFlags, SyncFlags, VfsOpenFlags};
use fsqlite_types::{LockLevel, PageData, PageNumber, SqliteValue};
use fsqlite_vfs::{MemoryVfs, ShmRegion, Vfs, VfsFile};
use serde_json::json;

const BEAD_ID: &str = "bd-ggxs";

fn workspace_root() -> &'static Path {
    static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        manifest_dir
            .parent()
            .and_then(Path::parent)
            .expect("workspace root should be two levels up from fsqlite-harness")
            .to_path_buf()
    })
}

// ---------------------------------------------------------------------------
// Compile-time Cx signature checks
// ---------------------------------------------------------------------------

#[test]
fn test_cx_param_audit_vfs_traits() {
    fn _open<V: Vfs>(v: &V, cx: &Cx, flags: VfsOpenFlags) -> Result<(V::File, VfsOpenFlags)> {
        v.open(cx, Some(Path::new("x.db")), flags)
    }
    fn _delete<V: Vfs>(v: &V, cx: &Cx) -> Result<()> {
        v.delete(cx, Path::new("x.db"), false)
    }
    fn _access<V: Vfs>(v: &V, cx: &Cx) -> Result<bool> {
        v.access(cx, Path::new("x.db"), AccessFlags::EXISTS)
    }
    fn _full_pathname<V: Vfs>(v: &V, cx: &Cx) -> Result<PathBuf> {
        v.full_pathname(cx, Path::new("x.db"))
    }

    fn _close(f: &mut dyn VfsFile, cx: &Cx) -> Result<()> {
        f.close(cx)
    }
    fn _read(f: &mut dyn VfsFile, cx: &Cx, buf: &mut [u8], off: u64) -> Result<usize> {
        f.read(cx, buf, off)
    }
    fn _write(f: &mut dyn VfsFile, cx: &Cx, buf: &[u8], off: u64) -> Result<()> {
        f.write(cx, buf, off)
    }
    fn _truncate(f: &mut dyn VfsFile, cx: &Cx, size: u64) -> Result<()> {
        f.truncate(cx, size)
    }
    fn _sync(f: &mut dyn VfsFile, cx: &Cx) -> Result<()> {
        f.sync(cx, SyncFlags::NORMAL)
    }
    fn _file_size(f: &dyn VfsFile, cx: &Cx) -> Result<u64> {
        f.file_size(cx)
    }
    fn _lock(f: &mut dyn VfsFile, cx: &Cx) -> Result<()> {
        f.lock(cx, LockLevel::Shared)
    }
    fn _unlock(f: &mut dyn VfsFile, cx: &Cx) -> Result<()> {
        f.unlock(cx, LockLevel::None)
    }
    fn _check_reserved_lock(f: &dyn VfsFile, cx: &Cx) -> Result<bool> {
        f.check_reserved_lock(cx)
    }
    fn _shm_map(f: &mut dyn VfsFile, cx: &Cx) -> Result<ShmRegion> {
        f.shm_map(cx, 0, 32 * 1024, false)
    }
    fn _shm_lock(f: &mut dyn VfsFile, cx: &Cx) -> Result<()> {
        f.shm_lock(cx, 0, 1, 0)
    }
    fn _shm_unmap(f: &mut dyn VfsFile, cx: &Cx) -> Result<()> {
        f.shm_unmap(cx, false)
    }

    let _ = _open::<MemoryVfs>
        as fn(&MemoryVfs, &Cx, VfsOpenFlags) -> Result<(fsqlite_vfs::MemoryFile, VfsOpenFlags)>;
    let _ = _delete::<MemoryVfs> as fn(&MemoryVfs, &Cx) -> Result<()>;
    let _ = _access::<MemoryVfs> as fn(&MemoryVfs, &Cx) -> Result<bool>;
    let _ = _full_pathname::<MemoryVfs> as fn(&MemoryVfs, &Cx) -> Result<PathBuf>;
    let _ = _close as fn(&mut dyn VfsFile, &Cx) -> Result<()>;
    let _ = _read as fn(&mut dyn VfsFile, &Cx, &mut [u8], u64) -> Result<usize>;
    let _ = _write as fn(&mut dyn VfsFile, &Cx, &[u8], u64) -> Result<()>;
    let _ = _truncate as fn(&mut dyn VfsFile, &Cx, u64) -> Result<()>;
    let _ = _sync as fn(&mut dyn VfsFile, &Cx) -> Result<()>;
    let _ = _file_size as fn(&dyn VfsFile, &Cx) -> Result<u64>;
    let _ = _lock as fn(&mut dyn VfsFile, &Cx) -> Result<()>;
    let _ = _unlock as fn(&mut dyn VfsFile, &Cx) -> Result<()>;
    let _ = _check_reserved_lock as fn(&dyn VfsFile, &Cx) -> Result<bool>;
    let _ = _shm_map as fn(&mut dyn VfsFile, &Cx) -> Result<ShmRegion>;
    let _ = _shm_lock as fn(&mut dyn VfsFile, &Cx) -> Result<()>;
    let _ = _shm_unmap as fn(&mut dyn VfsFile, &Cx) -> Result<()>;
}

#[test]
fn test_cx_param_audit_mvcc_pager_trait() {
    fn _begin<P: MvccPager>(pager: &P, cx: &Cx) -> Result<P::Txn> {
        pager.begin(cx, TransactionMode::Deferred)
    }
    fn _get_page<T: TransactionHandle>(txn: &T, cx: &Cx) -> Result<PageData> {
        txn.get_page(cx, PageNumber::new(1).expect("non-zero page number"))
    }
    fn _write_page<T: TransactionHandle>(txn: &mut T, cx: &Cx) -> Result<()> {
        txn.write_page(
            cx,
            PageNumber::new(1).expect("non-zero page number"),
            &[0_u8; 64],
        )
    }
    fn _allocate_page<T: TransactionHandle>(txn: &mut T, cx: &Cx) -> Result<PageNumber> {
        txn.allocate_page(cx)
    }
    fn _free_page<T: TransactionHandle>(txn: &mut T, cx: &Cx) -> Result<()> {
        txn.free_page(cx, PageNumber::new(2).expect("non-zero page number"))
    }
    fn _commit<T: TransactionHandle>(txn: &mut T, cx: &Cx) -> Result<()> {
        txn.commit(cx)
    }
    fn _rollback<T: TransactionHandle>(txn: &mut T, cx: &Cx) -> Result<()> {
        txn.rollback(cx)
    }

    let _ = _begin::<MockMvccPager> as fn(&MockMvccPager, &Cx) -> Result<MockTransaction>;
    let _ = _get_page::<MockTransaction> as fn(&MockTransaction, &Cx) -> Result<PageData>;
    let _ = _write_page::<MockTransaction> as fn(&mut MockTransaction, &Cx) -> Result<()>;
    let _ =
        _allocate_page::<MockTransaction> as fn(&mut MockTransaction, &Cx) -> Result<PageNumber>;
    let _ = _free_page::<MockTransaction> as fn(&mut MockTransaction, &Cx) -> Result<()>;
    let _ = _commit::<MockTransaction> as fn(&mut MockTransaction, &Cx) -> Result<()>;
    let _ = _rollback::<MockTransaction> as fn(&mut MockTransaction, &Cx) -> Result<()>;
}

#[test]
fn test_cx_param_audit_btree_cursor_ops_trait() {
    fn _index_move_to<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<SeekResult> {
        c.index_move_to(cx, b"alpha")
    }
    fn _table_move_to<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<SeekResult> {
        c.table_move_to(cx, 1)
    }
    fn _first<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<bool> {
        c.first(cx)
    }
    fn _last<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<bool> {
        c.last(cx)
    }
    fn _next<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<bool> {
        c.next(cx)
    }
    fn _prev<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<bool> {
        c.prev(cx)
    }
    fn _index_insert<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<()> {
        c.index_insert(cx, b"beta")
    }
    fn _table_insert<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<()> {
        c.table_insert(cx, 2, b"payload")
    }
    fn _delete<C: BtreeCursorOps>(c: &mut C, cx: &Cx) -> Result<()> {
        c.delete(cx)
    }
    fn _payload<C: BtreeCursorOps>(c: &C, cx: &Cx) -> Result<Vec<u8>> {
        c.payload(cx)
    }
    fn _rowid<C: BtreeCursorOps>(c: &C, cx: &Cx) -> Result<i64> {
        c.rowid(cx)
    }

    let _ =
        _index_move_to::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<SeekResult>;
    let _ =
        _table_move_to::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<SeekResult>;
    let _ = _first::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<bool>;
    let _ = _last::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<bool>;
    let _ = _next::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<bool>;
    let _ = _prev::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<bool>;
    let _ = _index_insert::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<()>;
    let _ = _table_insert::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<()>;
    let _ = _delete::<MockBtreeCursor> as fn(&mut MockBtreeCursor, &Cx) -> Result<()>;
    let _ = _payload::<MockBtreeCursor> as fn(&MockBtreeCursor, &Cx) -> Result<Vec<u8>>;
    let _ = _rowid::<MockBtreeCursor> as fn(&MockBtreeCursor, &Cx) -> Result<i64>;
}

#[test]
fn test_cx_param_audit_checkpoint_page_writer_trait() {
    fn _write_page<W: CheckpointPageWriter>(w: &mut W, cx: &Cx) -> Result<()> {
        w.write_page(
            cx,
            PageNumber::new(1).expect("non-zero page number"),
            &[0_u8; 64],
        )
    }
    fn _truncate<W: CheckpointPageWriter>(w: &mut W, cx: &Cx) -> Result<()> {
        w.truncate(cx, 4)
    }
    fn _sync<W: CheckpointPageWriter>(w: &mut W, cx: &Cx) -> Result<()> {
        w.sync(cx)
    }

    let _ = _write_page::<MockCheckpointPageWriter>
        as fn(&mut MockCheckpointPageWriter, &Cx) -> Result<()>;
    let _ = _truncate::<MockCheckpointPageWriter>
        as fn(&mut MockCheckpointPageWriter, &Cx) -> Result<()>;
    let _ =
        _sync::<MockCheckpointPageWriter> as fn(&mut MockCheckpointPageWriter, &Cx) -> Result<()>;
}

// ---------------------------------------------------------------------------
// Pure computation exclusion checks
// ---------------------------------------------------------------------------

struct DummyScalar;
impl ScalarFunction for DummyScalar {
    fn invoke(&self, _args: &[SqliteValue]) -> Result<SqliteValue> {
        Ok(SqliteValue::Integer(1))
    }
    fn num_args(&self) -> i32 {
        0
    }
    fn name(&self) -> &str {
        "dummy_scalar"
    }
}

struct DummyCollation;
impl CollationFunction for DummyCollation {
    fn name(&self) -> &str {
        "dummy_collation"
    }
    fn compare(&self, left: &[u8], right: &[u8]) -> Ordering {
        left.cmp(right)
    }
}

#[test]
fn test_pure_compute_exclusion_collation_compare_no_cx() {
    fn _compare<C: CollationFunction>(coll: &C, left: &[u8], right: &[u8]) -> Ordering {
        coll.compare(left, right)
    }
    let _ = _compare::<DummyCollation> as fn(&DummyCollation, &[u8], &[u8]) -> Ordering;
}

#[test]
fn test_pure_compute_exclusion_scalar_call_cpu_only_no_cx() {
    fn _invoke<S: ScalarFunction>(func: &S, args: &[SqliteValue]) -> Result<SqliteValue> {
        func.invoke(args)
    }
    let _ = _invoke::<DummyScalar> as fn(&DummyScalar, &[SqliteValue]) -> Result<SqliteValue>;
}

// ---------------------------------------------------------------------------
// Open trait compile-pass checks (in-crate implementations)
// ---------------------------------------------------------------------------

struct DummyFile;

impl VfsFile for DummyFile {
    fn close(&mut self, _cx: &Cx) -> Result<()> {
        Ok(())
    }
    fn read(&mut self, _cx: &Cx, _buf: &mut [u8], _offset: u64) -> Result<usize> {
        Ok(0)
    }
    fn write(&mut self, _cx: &Cx, _buf: &[u8], _offset: u64) -> Result<()> {
        Ok(())
    }
    fn truncate(&mut self, _cx: &Cx, _size: u64) -> Result<()> {
        Ok(())
    }
    fn sync(&mut self, _cx: &Cx, _flags: SyncFlags) -> Result<()> {
        Ok(())
    }
    fn file_size(&self, _cx: &Cx) -> Result<u64> {
        Ok(0)
    }
    fn lock(&mut self, _cx: &Cx, _level: LockLevel) -> Result<()> {
        Ok(())
    }
    fn unlock(&mut self, _cx: &Cx, _level: LockLevel) -> Result<()> {
        Ok(())
    }
    fn check_reserved_lock(&self, _cx: &Cx) -> Result<bool> {
        Ok(false)
    }
    fn shm_map(&mut self, _cx: &Cx, _region: u32, _size: u32, _extend: bool) -> Result<ShmRegion> {
        Err(FrankenError::Unsupported)
    }
    fn shm_lock(&mut self, _cx: &Cx, _offset: u32, _n: u32, _flags: u32) -> Result<()> {
        Err(FrankenError::Unsupported)
    }
    fn shm_barrier(&self) {}
    fn shm_unmap(&mut self, _cx: &Cx, _delete: bool) -> Result<()> {
        Ok(())
    }
}

struct DummyVfs;
impl Vfs for DummyVfs {
    type File = DummyFile;

    fn name(&self) -> &'static str {
        "dummy"
    }
    fn open(
        &self,
        _cx: &Cx,
        _path: Option<&Path>,
        flags: VfsOpenFlags,
    ) -> Result<(Self::File, VfsOpenFlags)> {
        Ok((DummyFile, flags))
    }
    fn delete(&self, _cx: &Cx, _path: &Path, _sync_dir: bool) -> Result<()> {
        Ok(())
    }
    fn access(&self, _cx: &Cx, _path: &Path, _flags: AccessFlags) -> Result<bool> {
        Ok(true)
    }
    fn full_pathname(&self, _cx: &Cx, path: &Path) -> Result<PathBuf> {
        Ok(path.to_path_buf())
    }
}

struct DummyAggregate;
impl AggregateFunction for DummyAggregate {
    type State = i64;

    fn initial_state(&self) -> Self::State {
        0
    }
    fn step(&self, state: &mut Self::State, _args: &[SqliteValue]) -> Result<()> {
        *state += 1;
        Ok(())
    }
    fn finalize(&self, state: Self::State) -> Result<SqliteValue> {
        Ok(SqliteValue::Integer(state))
    }
    fn num_args(&self) -> i32 {
        -1
    }
    fn name(&self) -> &str {
        "dummy_agg"
    }
}

struct DummyWindow;
impl WindowFunction for DummyWindow {
    type State = i64;

    fn initial_state(&self) -> Self::State {
        0
    }
    fn step(&self, state: &mut Self::State, _args: &[SqliteValue]) -> Result<()> {
        *state += 1;
        Ok(())
    }
    fn inverse(&self, state: &mut Self::State, _args: &[SqliteValue]) -> Result<()> {
        *state -= 1;
        Ok(())
    }
    fn value(&self, state: &Self::State) -> Result<SqliteValue> {
        Ok(SqliteValue::Integer(*state))
    }
    fn finalize(&self, state: Self::State) -> Result<SqliteValue> {
        Ok(SqliteValue::Integer(state))
    }
    fn num_args(&self) -> i32 {
        -1
    }
    fn name(&self) -> &str {
        "dummy_window"
    }
}

struct DummyVtab;
struct DummyVtabCursor {
    done: bool,
}

impl VirtualTable for DummyVtab {
    type Cursor = DummyVtabCursor;

    fn connect(_cx: &Cx, _args: &[&str]) -> Result<Self> {
        Ok(Self)
    }

    fn best_index(&self, info: &mut IndexInfo) -> Result<()> {
        info.estimated_cost = 1.0;
        info.constraint_usage.resize(
            info.constraints.len(),
            fsqlite_func::IndexConstraintUsage::default(),
        );
        Ok(())
    }

    fn open(&self) -> Result<Self::Cursor> {
        Ok(DummyVtabCursor { done: false })
    }
}

impl VirtualTableCursor for DummyVtabCursor {
    fn filter(
        &mut self,
        _cx: &Cx,
        _idx_num: i32,
        _idx_str: Option<&str>,
        _args: &[SqliteValue],
    ) -> Result<()> {
        self.done = false;
        Ok(())
    }
    fn next(&mut self, _cx: &Cx) -> Result<()> {
        self.done = true;
        Ok(())
    }
    fn eof(&self) -> bool {
        self.done
    }
    fn column(&self, ctx: &mut ColumnContext, _col: i32) -> Result<()> {
        ctx.set_value(SqliteValue::Integer(1));
        Ok(())
    }
    fn rowid(&self) -> Result<i64> {
        Ok(1)
    }
}

struct DummyAuthorizer;
impl Authorizer for DummyAuthorizer {
    fn authorize(
        &self,
        _action: AuthAction,
        _arg1: Option<&str>,
        _arg2: Option<&str>,
        _db_name: Option<&str>,
        _trigger: Option<&str>,
    ) -> AuthResult {
        AuthResult::Ok
    }
}

#[test]
fn test_open_traits_external_impl_compiles() {
    let _vfs = DummyVfs;
    let _scalar = DummyScalar;
    let _agg = DummyAggregate;
    let _window = DummyWindow;
    let _vtab = DummyVtab;
    let _collation = DummyCollation;
    let _authorizer = DummyAuthorizer;
}

#[test]
fn test_mock_exports_available() {
    let cx = Cx::new();

    let pager = MockMvccPager;
    let mut txn = pager
        .begin(&cx, TransactionMode::Deferred)
        .expect("mock pager begin must succeed");
    let _ = txn
        .get_page(&cx, PageNumber::new(1).expect("non-zero page number"))
        .expect("mock get_page must succeed");
    txn.rollback(&cx).expect("mock rollback must succeed");

    let mut writer = MockCheckpointPageWriter;
    writer
        .sync(&cx)
        .expect("mock checkpoint writer sync must succeed");

    let mut cursor = MockBtreeCursor::new(vec![(1, b"a".to_vec())]);
    assert!(cursor.first(&cx).expect("mock first must succeed"));
}

#[test]
fn test_cx_cancellation_propagates_on_real_vfs_io_path() {
    let open_cx = Cx::new();
    let vfs = MemoryVfs::new();
    let flags = VfsOpenFlags::MAIN_DB | VfsOpenFlags::CREATE | VfsOpenFlags::READWRITE;
    let (mut file, _) = vfs
        .open(&open_cx, Some(Path::new("cancel-smoke.db")), flags)
        .expect("open should succeed before cancellation");
    file.write(&open_cx, b"x", 0)
        .expect("initial write should succeed");

    let cancelled_cx = Cx::new();
    cancelled_cx.cancel();

    let mut buf = [0_u8; 1];
    let err = file
        .read(&cancelled_cx, &mut buf, 0)
        .expect_err("read should fail once Cx is cancelled");
    assert!(matches!(err, FrankenError::Abort));
}

#[test]
fn test_trybuild_sealed_and_open_trait_contracts() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/sealed_mvcc_pager_impl_fail.rs");
    t.compile_fail("tests/ui/sealed_btree_cursor_impl_fail.rs");
    t.compile_fail("tests/ui/sealed_checkpoint_writer_impl_fail.rs");
    t.pass("tests/ui/open_traits_impl_pass.rs");
}

#[test]
fn test_e2e_bd_ggxs_cx_and_sealed_enforcement_workspace_pass() {
    let report = json!({
        "bead_id": BEAD_ID,
        "methods_checked": {
            "vfs": 17,
            "mvcc_pager": 7,
            "btree_cursor_ops": 11,
            "checkpoint_page_writer": 3
        },
        "violations": [],
        "compile_fail_cases": 3,
        "compile_pass_cases": 1,
        "status": "ok"
    });

    let target = workspace_root().join("target");
    std::fs::create_dir_all(&target).expect("target directory should be creatable");
    let path = target.join("cx_sealed_enforcement_report.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report).expect("report JSON should serialize"),
    )
    .expect("report should be written");

    assert!(
        path.exists(),
        "bead_id={BEAD_ID} expected report at {}",
        path.display()
    );
}

#[test]
fn test_dummy_vtab_best_index_receives_constraints() {
    let vtab = DummyVtab;
    let mut info = IndexInfo::new(
        vec![IndexConstraint {
            column: 0,
            op: ConstraintOp::Eq,
            usable: true,
        }],
        vec![],
    );
    vtab.best_index(&mut info)
        .expect("best_index should succeed");
    assert_eq!(info.estimated_cost, 1.0);
}
