//! Virtual table and cursor traits (§9.3).
//!
//! Virtual tables expose external data sources as SQL tables. They follow
//! the SQLite xCreate/xConnect/xBestIndex/xFilter/xNext protocol.
//!
//! These traits are **open** (user-implementable). Extension authors
//! implement them to create custom virtual table modules.
//!
//! # Cx on I/O Methods
//!
//! Methods that perform I/O accept `&Cx` for cancellation and deadline
//! propagation. Lightweight accessors (`eof`, `column`, `rowid`) do not
//! require `&Cx` since they operate on already-fetched row data.

use fsqlite_error::{FrankenError, Result};
use fsqlite_types::SqliteValue;
use fsqlite_types::cx::Cx;

// ---------------------------------------------------------------------------
// Query planner types
// ---------------------------------------------------------------------------

/// Comparison operator for an index constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstraintOp {
    Eq,
    Gt,
    Le,
    Lt,
    Ge,
    Match,
    Like,
    Glob,
    Regexp,
    Ne,
    IsNot,
    IsNotNull,
    IsNull,
    Is,
}

/// A single constraint from the WHERE clause that the planner is considering.
#[derive(Debug, Clone)]
pub struct IndexConstraint {
    /// Column index (0-based; `-1` for rowid).
    pub column: i32,
    /// The comparison operator.
    pub op: ConstraintOp,
    /// Whether the planner considers this constraint usable.
    pub usable: bool,
}

/// A single ORDER BY term from the query.
#[derive(Debug, Clone)]
pub struct IndexOrderBy {
    /// Column index (0-based).
    pub column: i32,
    /// `true` if descending, `false` if ascending.
    pub desc: bool,
}

/// Per-constraint usage information set by `best_index`.
#[derive(Debug, Clone, Default)]
pub struct IndexConstraintUsage {
    /// 1-based index into the `args` array passed to `filter`.
    /// 0 means this constraint is not consumed by the vtab.
    pub argv_index: i32,
    /// If `true`, the vtab guarantees this constraint is satisfied and
    /// the core need not double-check it.
    pub omit: bool,
}

/// Information exchanged between the query planner and virtual table
/// during index selection.
///
/// The planner fills `constraints` and `order_by`. The vtab fills
/// `constraint_usage`, `idx_num`, `idx_str`, `order_by_consumed`,
/// `estimated_cost`, and `estimated_rows`.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    /// WHERE clause constraints the planner is considering.
    pub constraints: Vec<IndexConstraint>,
    /// ORDER BY terms from the query.
    pub order_by: Vec<IndexOrderBy>,
    /// How each constraint maps to filter arguments (vtab fills this).
    pub constraint_usage: Vec<IndexConstraintUsage>,
    /// Integer identifier for the chosen index strategy.
    pub idx_num: i32,
    /// Optional string identifier for the chosen index strategy.
    pub idx_str: Option<String>,
    /// Whether the vtab guarantees the output is already sorted.
    pub order_by_consumed: bool,
    /// Estimated cost of the scan (lower is better).
    pub estimated_cost: f64,
    /// Estimated number of rows returned.
    pub estimated_rows: i64,
}

impl IndexInfo {
    /// Create a new `IndexInfo` with the given constraints and order-by terms.
    #[must_use]
    pub fn new(constraints: Vec<IndexConstraint>, order_by: Vec<IndexOrderBy>) -> Self {
        let usage_len = constraints.len();
        Self {
            constraints,
            order_by,
            constraint_usage: vec![IndexConstraintUsage::default(); usage_len],
            idx_num: 0,
            idx_str: None,
            order_by_consumed: false,
            estimated_cost: 1_000_000.0,
            estimated_rows: 1_000_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Column context
// ---------------------------------------------------------------------------

/// A context object passed to [`VirtualTableCursor::column`] for writing
/// the column value.
///
/// Analogous to C SQLite's `sqlite3_context*` used with `sqlite3_result_*`.
#[derive(Debug, Default)]
pub struct ColumnContext {
    value: Option<SqliteValue>,
}

impl ColumnContext {
    /// Create a new empty column context.
    #[must_use]
    pub fn new() -> Self {
        Self { value: None }
    }

    /// Set the value for this column.
    pub fn set_value(&mut self, val: SqliteValue) {
        self.value = Some(val);
    }

    /// Take the value out of this context, leaving `None`.
    pub fn take_value(&mut self) -> Option<SqliteValue> {
        self.value.take()
    }
}

// ---------------------------------------------------------------------------
// VirtualTable trait
// ---------------------------------------------------------------------------

/// A virtual table module.
///
/// Virtual tables expose external data sources as SQL tables. This trait
/// covers the full lifecycle: creation, connection, scanning, mutation,
/// and destruction.
///
/// This trait is **open** (user-implementable). The `Sized` bound on
/// constructor methods (`create`, `connect`) allows the trait to be used
/// as `dyn VirtualTable<Cursor = C>` for other methods.
///
/// # Default Implementations
///
/// Most methods have sensible defaults. At minimum, you must implement
/// `connect`, `best_index`, and `open`.
#[allow(clippy::missing_errors_doc)]
pub trait VirtualTable: Send + Sync {
    /// The cursor type for scanning this virtual table.
    type Cursor: VirtualTableCursor;

    /// Called for `CREATE VIRTUAL TABLE`.
    ///
    /// May create backing storage. Default delegates to `connect`
    /// (suitable for eponymous virtual tables).
    fn create(cx: &Cx, args: &[&str]) -> Result<Self>
    where
        Self: Sized,
    {
        Self::connect(cx, args)
    }

    /// Called for subsequent opens of an existing virtual table.
    fn connect(cx: &Cx, args: &[&str]) -> Result<Self>
    where
        Self: Sized;

    /// Inform the query planner about available indexes and their costs.
    fn best_index(&self, info: &mut IndexInfo) -> Result<()>;

    /// Open a new scan cursor.
    fn open(&self) -> Result<Self::Cursor>;

    /// Drop a virtual table instance (opposite of `connect`).
    fn disconnect(&mut self, _cx: &Cx) -> Result<()> {
        Ok(())
    }

    /// Called for `DROP VIRTUAL TABLE` — destroy backing storage.
    ///
    /// Default delegates to `disconnect`.
    fn destroy(&mut self, cx: &Cx) -> Result<()> {
        self.disconnect(cx)
    }

    /// INSERT/UPDATE/DELETE on the virtual table.
    ///
    /// - `args[0]`: old rowid (`None` for INSERT)
    /// - `args[1]`: new rowid
    /// - `args[2..]`: column values
    ///
    /// Returns the new rowid for INSERT, `None` for UPDATE/DELETE.
    ///
    /// Default returns [`FrankenError::ReadOnly`] (read-only virtual tables).
    fn update(&mut self, _cx: &Cx, _args: &[SqliteValue]) -> Result<Option<i64>> {
        Err(FrankenError::ReadOnly)
    }

    /// Begin a virtual table transaction.
    fn begin(&mut self, _cx: &Cx) -> Result<()> {
        Ok(())
    }

    /// Sync a virtual table transaction (phase 1 of 2PC).
    fn sync_txn(&mut self, _cx: &Cx) -> Result<()> {
        Ok(())
    }

    /// Commit a virtual table transaction.
    fn commit(&mut self, _cx: &Cx) -> Result<()> {
        Ok(())
    }

    /// Roll back a virtual table transaction.
    fn rollback(&mut self, _cx: &Cx) -> Result<()> {
        Ok(())
    }

    /// Rename the virtual table.
    ///
    /// Default returns [`FrankenError::Unsupported`].
    fn rename(&mut self, _cx: &Cx, _new_name: &str) -> Result<()> {
        Err(FrankenError::Unsupported)
    }

    /// Create a savepoint at level `n`.
    fn savepoint(&mut self, _cx: &Cx, _n: i32) -> Result<()> {
        Ok(())
    }

    /// Release savepoint at level `n`.
    fn release(&mut self, _cx: &Cx, _n: i32) -> Result<()> {
        Ok(())
    }

    /// Roll back to savepoint at level `n`.
    fn rollback_to(&mut self, _cx: &Cx, _n: i32) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// VirtualTableCursor trait
// ---------------------------------------------------------------------------

/// A cursor for scanning a virtual table.
///
/// Cursors are `Send` but **NOT** `Sync` — they are single-threaded
/// scan objects bound to a specific filter invocation.
///
/// # Lifecycle
///
/// 1. [`filter`](Self::filter) begins a scan with planner-chosen parameters.
/// 2. Iterate: check [`eof`](Self::eof), read [`column`](Self::column)/[`rowid`](Self::rowid), advance with [`next`](Self::next).
/// 3. The cursor is dropped when the scan is complete.
#[allow(clippy::missing_errors_doc)]
pub trait VirtualTableCursor: Send {
    /// Begin a scan with the filter parameters chosen by `best_index`.
    fn filter(
        &mut self,
        cx: &Cx,
        idx_num: i32,
        idx_str: Option<&str>,
        args: &[SqliteValue],
    ) -> Result<()>;

    /// Advance to the next row.
    fn next(&mut self, cx: &Cx) -> Result<()>;

    /// Whether the cursor has moved past the last row.
    fn eof(&self) -> bool;

    /// Write the value of column `col` into `ctx`.
    fn column(&self, ctx: &mut ColumnContext, col: i32) -> Result<()>;

    /// Return the rowid of the current row.
    fn rowid(&self) -> Result<i64>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::too_many_lines)]
mod tests {
    use super::*;

    // -- Mock: generate_series(start, stop) virtual table --

    struct GenerateSeries {
        destroyed: bool,
    }

    struct GenerateSeriesCursor {
        start: i64,
        stop: i64,
        current: i64,
    }

    impl VirtualTable for GenerateSeries {
        type Cursor = GenerateSeriesCursor;

        fn connect(_cx: &Cx, _args: &[&str]) -> Result<Self> {
            Ok(Self { destroyed: false })
        }

        fn best_index(&self, info: &mut IndexInfo) -> Result<()> {
            info.estimated_cost = 10.0;
            info.estimated_rows = 100;
            info.idx_num = 1;

            // Mark constraint 0 as consumed, mapped to filter arg 1.
            if !info.constraints.is_empty() && info.constraints[0].usable {
                info.constraint_usage[0].argv_index = 1;
                info.constraint_usage[0].omit = true;
            }
            Ok(())
        }

        fn open(&self) -> Result<GenerateSeriesCursor> {
            Ok(GenerateSeriesCursor {
                start: 0,
                stop: 0,
                current: 0,
            })
        }

        fn destroy(&mut self, _cx: &Cx) -> Result<()> {
            self.destroyed = true;
            Ok(())
        }
    }

    impl VirtualTableCursor for GenerateSeriesCursor {
        fn filter(
            &mut self,
            _cx: &Cx,
            _idx_num: i32,
            _idx_str: Option<&str>,
            args: &[SqliteValue],
        ) -> Result<()> {
            self.start = args.first().map_or(1, SqliteValue::to_integer);
            self.stop = args.get(1).map_or(10, SqliteValue::to_integer);
            self.current = self.start;
            Ok(())
        }

        fn next(&mut self, _cx: &Cx) -> Result<()> {
            self.current += 1;
            Ok(())
        }

        fn eof(&self) -> bool {
            self.current > self.stop
        }

        fn column(&self, ctx: &mut ColumnContext, _col: i32) -> Result<()> {
            ctx.set_value(SqliteValue::Integer(self.current));
            Ok(())
        }

        fn rowid(&self) -> Result<i64> {
            Ok(self.current)
        }
    }

    // -- Mock: read-only vtab for default update test --

    struct ReadOnlyVtab;

    struct ReadOnlyCursor;

    impl VirtualTable for ReadOnlyVtab {
        type Cursor = ReadOnlyCursor;

        fn connect(_cx: &Cx, _args: &[&str]) -> Result<Self> {
            Ok(Self)
        }

        fn best_index(&self, _info: &mut IndexInfo) -> Result<()> {
            Ok(())
        }

        fn open(&self) -> Result<ReadOnlyCursor> {
            Ok(ReadOnlyCursor)
        }
    }

    impl VirtualTableCursor for ReadOnlyCursor {
        fn filter(
            &mut self,
            _cx: &Cx,
            _idx_num: i32,
            _idx_str: Option<&str>,
            _args: &[SqliteValue],
        ) -> Result<()> {
            Ok(())
        }

        fn next(&mut self, _cx: &Cx) -> Result<()> {
            Ok(())
        }

        fn eof(&self) -> bool {
            true
        }

        fn column(&self, _ctx: &mut ColumnContext, _col: i32) -> Result<()> {
            Ok(())
        }

        fn rowid(&self) -> Result<i64> {
            Ok(0)
        }
    }

    // -- Mock: writable vtab for insert test --

    struct WritableVtab {
        rows: Vec<(i64, Vec<SqliteValue>)>,
        next_rowid: i64,
    }

    struct WritableCursor {
        rows: Vec<(i64, Vec<SqliteValue>)>,
        pos: usize,
    }

    impl VirtualTable for WritableVtab {
        type Cursor = WritableCursor;

        fn connect(_cx: &Cx, _args: &[&str]) -> Result<Self> {
            Ok(Self {
                rows: Vec::new(),
                next_rowid: 1,
            })
        }

        fn best_index(&self, _info: &mut IndexInfo) -> Result<()> {
            Ok(())
        }

        fn open(&self) -> Result<WritableCursor> {
            Ok(WritableCursor {
                rows: self.rows.clone(),
                pos: 0,
            })
        }

        fn update(&mut self, _cx: &Cx, args: &[SqliteValue]) -> Result<Option<i64>> {
            // args[0] = old rowid (Null for INSERT)
            if args[0].is_null() {
                // INSERT
                let rowid = self.next_rowid;
                self.next_rowid += 1;
                let cols: Vec<SqliteValue> = args[2..].to_vec();
                self.rows.push((rowid, cols));
                return Ok(Some(rowid));
            }
            Ok(None)
        }
    }

    impl VirtualTableCursor for WritableCursor {
        fn filter(
            &mut self,
            _cx: &Cx,
            _idx_num: i32,
            _idx_str: Option<&str>,
            _args: &[SqliteValue],
        ) -> Result<()> {
            self.pos = 0;
            Ok(())
        }

        fn next(&mut self, _cx: &Cx) -> Result<()> {
            self.pos += 1;
            Ok(())
        }

        fn eof(&self) -> bool {
            self.pos >= self.rows.len()
        }

        fn column(&self, ctx: &mut ColumnContext, col: i32) -> Result<()> {
            #[allow(clippy::cast_sign_loss)]
            let col_idx = col as usize;
            if let Some((_, ref cols)) = self.rows.get(self.pos) {
                if let Some(val) = cols.get(col_idx) {
                    ctx.set_value(val.clone());
                }
            }
            Ok(())
        }

        fn rowid(&self) -> Result<i64> {
            self.rows
                .get(self.pos)
                .map_or(Ok(0), |(rowid, _)| Ok(*rowid))
        }
    }

    // -- Tests --

    #[test]
    fn test_vtab_create_vs_connect() {
        let cx = Cx::new();

        // create delegates to connect by default.
        let vtab = GenerateSeries::create(&cx, &[]).unwrap();
        assert!(!vtab.destroyed);

        // connect works directly.
        let vtab2 = GenerateSeries::connect(&cx, &[]).unwrap();
        assert!(!vtab2.destroyed);
    }

    #[test]
    fn test_vtab_best_index_populates_info() {
        let cx = Cx::new();
        let vtab = GenerateSeries::connect(&cx, &[]).unwrap();

        let mut info = IndexInfo::new(
            vec![IndexConstraint {
                column: 0,
                op: ConstraintOp::Gt,
                usable: true,
            }],
            vec![],
        );

        vtab.best_index(&mut info).unwrap();

        assert_eq!(info.idx_num, 1);
        assert!((info.estimated_cost - 10.0).abs() < f64::EPSILON);
        assert_eq!(info.estimated_rows, 100);
        assert_eq!(info.constraint_usage[0].argv_index, 1);
        assert!(info.constraint_usage[0].omit);
    }

    #[test]
    fn test_vtab_cursor_filter_next_eof() {
        let cx = Cx::new();
        let vtab = GenerateSeries::connect(&cx, &[]).unwrap();
        let mut cursor = vtab.open().unwrap();

        cursor
            .filter(
                &cx,
                0,
                None,
                &[SqliteValue::Integer(1), SqliteValue::Integer(3)],
            )
            .unwrap();

        let mut values = Vec::new();
        while !cursor.eof() {
            let mut ctx = ColumnContext::new();
            cursor.column(&mut ctx, 0).unwrap();
            let rowid = cursor.rowid().unwrap();
            values.push((rowid, ctx.take_value().unwrap()));
            cursor.next(&cx).unwrap();
        }

        assert_eq!(values.len(), 3);
        assert_eq!(values[0], (1, SqliteValue::Integer(1)));
        assert_eq!(values[1], (2, SqliteValue::Integer(2)));
        assert_eq!(values[2], (3, SqliteValue::Integer(3)));
    }

    #[test]
    fn test_vtab_update_insert() {
        let cx = Cx::new();
        let mut vtab = WritableVtab::connect(&cx, &[]).unwrap();

        // INSERT: args[0] = Null (no old rowid), args[1] = new rowid (ignored),
        // args[2..] = column values
        let result = vtab
            .update(
                &cx,
                &[
                    SqliteValue::Null,
                    SqliteValue::Null,
                    SqliteValue::Text("hello".to_owned()),
                ],
            )
            .unwrap();

        assert_eq!(result, Some(1));
        assert_eq!(vtab.rows.len(), 1);
        assert_eq!(vtab.rows[0].0, 1);
    }

    #[test]
    fn test_vtab_update_readonly_default() {
        let cx = Cx::new();
        let mut vtab = ReadOnlyVtab::connect(&cx, &[]).unwrap();
        let err = vtab.update(&cx, &[SqliteValue::Null]).unwrap_err();
        assert!(matches!(err, FrankenError::ReadOnly));
    }

    #[test]
    fn test_vtab_destroy_vs_disconnect() {
        let cx = Cx::new();

        // Default: destroy delegates to disconnect (both no-ops for ReadOnlyVtab).
        let mut vtab = ReadOnlyVtab::connect(&cx, &[]).unwrap();
        vtab.disconnect(&cx).unwrap();
        vtab.destroy(&cx).unwrap();

        // Custom destroy sets a flag.
        let mut vtab = GenerateSeries::connect(&cx, &[]).unwrap();
        assert!(!vtab.destroyed);
        vtab.destroy(&cx).unwrap();
        assert!(vtab.destroyed);
    }

    #[test]
    fn test_vtab_cursor_send_but_not_sync() {
        fn assert_send<T: Send>() {}
        assert_send::<GenerateSeriesCursor>();

        // VirtualTableCursor is Send but NOT Sync.
        // We can't directly test "not Sync" at runtime, but we can verify
        // the trait bound: VirtualTableCursor: Send (not Send + Sync).
        // The type GenerateSeriesCursor IS actually Sync by coincidence
        // (all fields are i64), but the trait doesn't require it.
        // The key point: the trait signature says Send, not Send + Sync.
    }

    #[test]
    fn test_column_context_lifecycle() {
        let mut ctx = ColumnContext::new();
        assert!(ctx.take_value().is_none());

        ctx.set_value(SqliteValue::Integer(42));
        assert_eq!(ctx.take_value(), Some(SqliteValue::Integer(42)));

        // After take, it's None again.
        assert!(ctx.take_value().is_none());
    }

    #[test]
    fn test_index_info_new() {
        let info = IndexInfo::new(
            vec![
                IndexConstraint {
                    column: 0,
                    op: ConstraintOp::Eq,
                    usable: true,
                },
                IndexConstraint {
                    column: 1,
                    op: ConstraintOp::Gt,
                    usable: false,
                },
            ],
            vec![IndexOrderBy {
                column: 0,
                desc: false,
            }],
        );

        assert_eq!(info.constraints.len(), 2);
        assert_eq!(info.order_by.len(), 1);
        assert_eq!(info.constraint_usage.len(), 2);
        assert_eq!(info.idx_num, 0);
        assert!(info.idx_str.is_none());
        assert!(!info.order_by_consumed);
    }
}
