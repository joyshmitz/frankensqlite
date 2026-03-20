//! `execute_batch` support, analogous to `rusqlite::Connection::execute_batch`.

use fsqlite_error::FrankenError;

use crate::Connection;

/// Extension trait for executing multiple SQL statements in a batch.
pub trait BatchExt {
    /// Execute a string containing multiple SQL statements separated by
    /// semicolons. Each statement is executed in order; an error in any
    /// statement stops execution and returns that error.
    ///
    /// This is the fsqlite equivalent of `rusqlite::Connection::execute_batch`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use fsqlite::compat::BatchExt;
    ///
    /// conn.execute_batch("
    ///     PRAGMA journal_mode = WAL;
    ///     CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT);
    ///     CREATE INDEX IF NOT EXISTS idx_name ON users(name);
    /// ")?;
    /// ```
    fn execute_batch(&self, sql: &str) -> Result<(), FrankenError>;
}

impl BatchExt for Connection {
    fn execute_batch(&self, sql: &str) -> Result<(), FrankenError> {
        if batch_is_noop(sql) {
            return Ok(());
        }
        self.execute(sql).map(|_| ())
    }
}

fn batch_is_noop(sql: &str) -> bool {
    let bytes = sql.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            b';' | b' ' | b'\t' | b'\r' | b'\n' => {
                i += 1;
            }
            b'-' if i + 1 < bytes.len() && bytes[i + 1] == b'-' => {
                i += 2;
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'*' => {
                i += 2;
                while i + 1 < bytes.len() {
                    if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                        i += 2;
                        break;
                    }
                    i += 1;
                }
            }
            _ => return false,
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compat::RowExt;

    #[test]
    fn execute_batch_creates_tables() {
        let conn = Connection::open(":memory:").unwrap();
        conn.execute_batch(
            "
            CREATE TABLE a (id INTEGER PRIMARY KEY);
            CREATE TABLE b (id INTEGER PRIMARY KEY);
            INSERT INTO a (id) VALUES (1);
            INSERT INTO b (id) VALUES (2);
        ",
        )
        .unwrap();

        let a: i64 = conn
            .query_row("SELECT id FROM a")
            .map(|row| row.get_typed::<i64>(0).unwrap())
            .unwrap();
        assert_eq!(a, 1);

        let b: i64 = conn
            .query_row("SELECT id FROM b")
            .map(|row| row.get_typed::<i64>(0).unwrap())
            .unwrap();
        assert_eq!(b, 2);
    }

    #[test]
    fn execute_batch_empty_string() {
        let conn = Connection::open(":memory:").unwrap();
        conn.execute_batch("").unwrap();
        conn.execute_batch("   ;  ;  ").unwrap();
        conn.execute_batch("  -- nothing here\n/* still empty */ ; ")
            .unwrap();
    }

    #[test]
    fn execute_batch_semicolons_in_string_literals() {
        let conn = Connection::open(":memory:").unwrap();
        conn.execute_batch(
            "
            CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT);
            INSERT INTO t (val) VALUES ('hello;world');
            INSERT INTO t (val) VALUES ('a''b;c''d');
        ",
        )
        .unwrap();

        let rows = conn.query("SELECT val FROM t ORDER BY id").unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get_typed::<String>(0).unwrap(), "hello;world");
        assert_eq!(rows[1].get_typed::<String>(0).unwrap(), "a'b;c'd");
    }

    #[test]
    fn execute_batch_allows_triggers_with_internal_semicolons() {
        let conn = Connection::open(":memory:").unwrap();
        conn.execute_batch(
            "
            CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE item_audit (item_id INTEGER, seen_value TEXT);
            CREATE TRIGGER audit_items
            AFTER INSERT ON items
            BEGIN
                INSERT INTO item_audit (item_id, seen_value) VALUES (NEW.id, NEW.value);
            END;
            INSERT INTO items (id, value) VALUES (1, 'alpha');
        ",
        )
        .unwrap();

        let audit_rows = conn
            .query("SELECT item_id, seen_value FROM item_audit")
            .unwrap();
        assert_eq!(audit_rows.len(), 1);
        assert_eq!(audit_rows[0].get_typed::<i64>(0).unwrap(), 1);
        assert_eq!(audit_rows[0].get_typed::<String>(1).unwrap(), "alpha");
    }

    #[test]
    fn batch_is_noop_handles_whitespace_semicolons_and_comments() {
        assert!(batch_is_noop(""));
        assert!(batch_is_noop("  ;\n\t; "));
        assert!(batch_is_noop("-- comment only\n/* and block */ ;"));
        assert!(!batch_is_noop("SELECT 1"));
    }
}
