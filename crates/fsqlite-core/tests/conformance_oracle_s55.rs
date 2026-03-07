//! Session 55 conformance oracle tests — FK cascades, multi-way JOINs, CTE aggregates,
//! expression edges, complex UPDATEs, triggers, typeof, CAST, LIKE, BETWEEN, IN list edges.

use fsqlite_core::connection::Connection;
use fsqlite_types::value::SqliteValue;

/// Run queries against both FrankenSQLite and C SQLite, returning mismatches.
fn oracle_compare(
    fconn: &Connection,
    rconn: &rusqlite::Connection,
    queries: &[&str],
) -> Vec<String> {
    let mut mismatches = Vec::new();
    for query in queries {
        let frank_result = fconn.query(query);
        let csql_result: std::result::Result<Vec<Vec<String>>, String> = (|| {
            let mut stmt = rconn.prepare(query).map_err(|e| format!("prepare: {e}"))?;
            let col_count = stmt.column_count();
            let rows: Vec<Vec<String>> = stmt
                .query_map([], |row| {
                    let mut vals = Vec::new();
                    for i in 0..col_count {
                        let v: rusqlite::types::Value = row.get_unwrap(i);
                        let s = match v {
                            rusqlite::types::Value::Null => "NULL".to_owned(),
                            rusqlite::types::Value::Integer(n) => n.to_string(),
                            rusqlite::types::Value::Real(f) => format!("{f}"),
                            rusqlite::types::Value::Text(s) => format!("'{s}'"),
                            rusqlite::types::Value::Blob(b) => {
                                format!(
                                    "X'{}'",
                                    b.iter().map(|x| format!("{x:02X}")).collect::<String>()
                                )
                            }
                        };
                        vals.push(s);
                    }
                    Ok(vals)
                })
                .map_err(|e| format!("query: {e}"))?
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| format!("row: {e}"))?;
            Ok(rows)
        })();
        let csql_result = match csql_result {
            Ok(r) => r,
            Err(csql_err) => {
                if frank_result.is_ok() {
                    mismatches.push(format!(
                        "DIVERGE: {query}\n  frank: OK\n  csql:  ERROR({csql_err})"
                    ));
                }
                continue;
            }
        };

        match frank_result {
            Ok(rows) => {
                let frank_strs: Vec<Vec<String>> = rows
                    .iter()
                    .map(|row| {
                        row.values()
                            .iter()
                            .map(|v| match v {
                                SqliteValue::Null => "NULL".to_owned(),
                                SqliteValue::Integer(n) => n.to_string(),
                                SqliteValue::Float(f) => format!("{f}"),
                                SqliteValue::Text(s) => format!("'{s}'"),
                                SqliteValue::Blob(b) => {
                                    format!(
                                        "X'{}'",
                                        b.iter().map(|x| format!("{x:02X}")).collect::<String>()
                                    )
                                }
                            })
                            .collect()
                    })
                    .collect();

                if frank_strs != csql_result {
                    mismatches.push(format!(
                        "MISMATCH: {query}\n  frank: {frank_strs:?}\n  csql:  {csql_result:?}"
                    ));
                }
            }
            Err(e) => {
                mismatches.push(format!(
                    "FRANK_ERR: {query}\n  frank: {e}\n  csql:  {csql_result:?}"
                ));
            }
        }
    }
    mismatches
}

#[test]
fn test_conformance_fk_cascade_delete_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    // Set up FK enforcement first via pragma
    rconn.execute_batch("PRAGMA foreign_keys = ON").unwrap();
    fconn.execute("PRAGMA foreign_keys = ON").unwrap();

    for s in &[
        "CREATE TABLE fkp(id INTEGER PRIMARY KEY, name TEXT)",
        "CREATE TABLE fkc_cascade(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES fkp(id) ON DELETE CASCADE, label TEXT)",
        "CREATE TABLE fkc_setnull(id INTEGER PRIMARY KEY, pid INTEGER REFERENCES fkp(id) ON DELETE SET NULL, label TEXT)",
        "INSERT INTO fkp VALUES(1,'alpha'),(2,'beta'),(3,'gamma')",
        "INSERT INTO fkc_cascade VALUES(10,1,'c1'),(11,1,'c2'),(12,2,'c3'),(13,3,'c4')",
        "INSERT INTO fkc_setnull VALUES(20,1,'s1'),(21,2,'s2'),(22,3,'s3')",
        "DELETE FROM fkp WHERE id = 1",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM fkp ORDER BY id",
        "SELECT * FROM fkc_cascade ORDER BY id",
        "SELECT * FROM fkc_setnull ORDER BY id",
        "SELECT COUNT(*) FROM fkc_cascade",
        "SELECT COUNT(*) FROM fkc_setnull WHERE pid IS NULL",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} FK cascade/set null mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_multiway_left_join_null_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE mlj_a(id INTEGER PRIMARY KEY, x TEXT)",
        "CREATE TABLE mlj_b(id INTEGER PRIMARY KEY, aid INTEGER, y TEXT)",
        "CREATE TABLE mlj_c(id INTEGER PRIMARY KEY, bid INTEGER, z TEXT)",
        "INSERT INTO mlj_a VALUES(1,'a1'),(2,'a2'),(3,'a3')",
        "INSERT INTO mlj_b VALUES(10,1,'b1'),(11,1,'b2'),(12,3,'b3')",
        "INSERT INTO mlj_c VALUES(100,10,'c1'),(101,12,'c2')",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT a.x, b.y, c.z FROM mlj_a a LEFT JOIN mlj_b b ON b.aid = a.id LEFT JOIN mlj_c c ON c.bid = b.id ORDER BY a.id, b.id, c.id",
        "SELECT a.x, COUNT(b.id), COUNT(c.id) FROM mlj_a a LEFT JOIN mlj_b b ON b.aid = a.id LEFT JOIN mlj_c c ON c.bid = b.id GROUP BY a.x ORDER BY a.x",
        "SELECT a.x, COALESCE(b.y, 'no_b'), COALESCE(c.z, 'no_c') FROM mlj_a a LEFT JOIN mlj_b b ON b.aid = a.id LEFT JOIN mlj_c c ON c.bid = b.id ORDER BY a.id, b.id",
        "SELECT a.x FROM mlj_a a LEFT JOIN mlj_b b ON b.aid = a.id WHERE b.id IS NULL",
        "SELECT a.x, b.y FROM mlj_a a LEFT JOIN mlj_b b ON b.aid = a.id LEFT JOIN mlj_c c ON c.bid = b.id WHERE c.id IS NULL ORDER BY a.id, b.id",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} multi-way LEFT JOIN NULL mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_nested_cte_with_agg_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE nca_sales(id INTEGER PRIMARY KEY, region TEXT, product TEXT, amount REAL)",
        "INSERT INTO nca_sales VALUES(1,'N','Widget',100.0),(2,'N','Widget',150.0),(3,'N','Gadget',200.0),(4,'S','Widget',80.0),(5,'S','Gadget',120.0),(6,'S','Gadget',90.0)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "WITH regional AS (SELECT region, product, SUM(amount) AS total FROM nca_sales GROUP BY region, product) SELECT region, product, total FROM regional ORDER BY region, product",
        "WITH totals AS (SELECT product, SUM(amount) AS grand FROM nca_sales GROUP BY product) SELECT s.region, s.product, SUM(s.amount) AS reg_total, t.grand FROM nca_sales s JOIN totals t ON t.product = s.product GROUP BY s.region, s.product ORDER BY s.region, s.product",
        "WITH cnt AS (SELECT region, COUNT(*) AS n FROM nca_sales GROUP BY region) SELECT region, n FROM cnt WHERE n > 2 ORDER BY region",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} nested CTE with agg mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_union_all_order_limit_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE uao_t1(id INTEGER PRIMARY KEY, val TEXT, n INTEGER)",
        "CREATE TABLE uao_t2(id INTEGER PRIMARY KEY, val TEXT, n INTEGER)",
        "INSERT INTO uao_t1 VALUES(1,'a',10),(2,'b',20),(3,'c',30)",
        "INSERT INTO uao_t2 VALUES(4,'d',15),(5,'e',25),(6,'f',35)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT val, n FROM uao_t1 UNION ALL SELECT val, n FROM uao_t2 ORDER BY n",
        "SELECT val, n FROM uao_t1 UNION ALL SELECT val, n FROM uao_t2 ORDER BY n LIMIT 3",
        "SELECT val, n FROM uao_t1 UNION ALL SELECT val, n FROM uao_t2 ORDER BY n LIMIT 3 OFFSET 2",
        "SELECT val FROM uao_t1 UNION SELECT val FROM uao_t2 ORDER BY val",
        "SELECT val FROM uao_t1 EXCEPT SELECT val FROM uao_t2 ORDER BY val",
        "SELECT * FROM (SELECT val, n FROM uao_t1 UNION ALL SELECT val, n FROM uao_t2) ORDER BY n DESC LIMIT 2",
        "SELECT val, n FROM uao_t1 WHERE n > 15 UNION ALL SELECT val, n FROM uao_t2 WHERE n < 30 ORDER BY n",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} UNION ALL ORDER/LIMIT mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_expression_edge_nullif_iif_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT NULLIF(0, 0)",
        "SELECT NULLIF(1, 0)",
        "SELECT NULLIF('', '')",
        "SELECT NULLIF(NULL, NULL)",
        "SELECT NULLIF(NULL, 1)",
        "SELECT NULLIF(1, NULL)",
        "SELECT IIF(1, 'yes', 'no')",
        "SELECT IIF(0, 'yes', 'no')",
        "SELECT IIF(NULL, 'yes', 'no')",
        "SELECT IIF(1 > 2, 'gt', 'le')",
        "SELECT COALESCE(NULLIF(0, 0), 'fallback')",
        "SELECT COALESCE(NULLIF(1, 0), 'fallback')",
        "SELECT IIF(NULLIF(0, 0) IS NULL, 'was_zero', 'nonzero')",
        "SELECT NULLIF(CAST(1 AS TEXT), '1')",
        "SELECT NULLIF(1, CAST('1' AS INTEGER))",
        "SELECT IIF(typeof(1) = 'integer', 'int', 'other')",
        "SELECT IIF(typeof(1.0) = 'real', 'real', 'other')",
        "SELECT IIF(typeof('x') = 'text', 'text', 'other')",
        "SELECT IIF(typeof(NULL) = 'null', 'null', 'other')",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} NULLIF/IIF expression edge mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_order_by_case_expr_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE obc(id INTEGER PRIMARY KEY, status TEXT, priority INTEGER)",
        "INSERT INTO obc VALUES(1,'open',3),(2,'closed',1),(3,'open',2),(4,'pending',5),(5,'closed',4)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM obc ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, priority",
        "SELECT * FROM obc ORDER BY CASE WHEN priority > 3 THEN 0 ELSE 1 END, id",
        "SELECT status, COUNT(*) AS cnt FROM obc GROUP BY status ORDER BY CASE status WHEN 'open' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END",
        "SELECT *, CASE WHEN priority >= 3 THEN 'high' ELSE 'low' END AS tier FROM obc ORDER BY tier, priority DESC",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} ORDER BY CASE expression mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_complex_insert_defaults_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE cid(id INTEGER PRIMARY KEY, a TEXT NOT NULL, b INTEGER DEFAULT 42)",
        "INSERT INTO cid(a) VALUES('x'),('y'),('z')",
        "INSERT INTO cid(a, b) VALUES('w', 100)",
        "INSERT OR REPLACE INTO cid(id, a, b) VALUES(1, 'replaced', 99)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM cid ORDER BY id",
        "SELECT COUNT(*), SUM(b), AVG(b) FROM cid",
        "SELECT a, b FROM cid WHERE b = 42 ORDER BY a",
        "SELECT a FROM cid WHERE id = 1",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} complex INSERT defaults mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_subquery_in_select_correlated_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE ssg_dept(id INTEGER PRIMARY KEY, name TEXT)",
        "CREATE TABLE ssg_emp(id INTEGER PRIMARY KEY, dept_id INTEGER, salary REAL)",
        "INSERT INTO ssg_dept VALUES(1,'eng'),(2,'sales'),(3,'hr')",
        "INSERT INTO ssg_emp VALUES(1,1,100.0),(2,1,120.0),(3,2,90.0),(4,2,95.0),(5,3,80.0)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT d.name, (SELECT COUNT(*) FROM ssg_emp e WHERE e.dept_id = d.id) AS emp_count FROM ssg_dept d ORDER BY d.name",
        "SELECT d.name, (SELECT SUM(e.salary) FROM ssg_emp e WHERE e.dept_id = d.id) AS total_sal FROM ssg_dept d ORDER BY d.name",
        "SELECT d.name, (SELECT AVG(e.salary) FROM ssg_emp e WHERE e.dept_id = d.id) AS avg_sal FROM ssg_dept d ORDER BY d.name",
        "SELECT d.name FROM ssg_dept d WHERE (SELECT COUNT(*) FROM ssg_emp e WHERE e.dept_id = d.id) > 1 ORDER BY d.name",
        "SELECT d.name FROM ssg_dept d WHERE EXISTS (SELECT 1 FROM ssg_emp e WHERE e.dept_id = d.id AND e.salary > 100) ORDER BY d.name",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!(
            "{} subquery-in-SELECT correlated mismatches",
            mismatches.len()
        );
    }
}

#[test]
fn test_conformance_complex_update_with_subquery_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE cuj_items(id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT)",
        "INSERT INTO cuj_items VALUES(1,'a',10.0,'X'),(2,'b',20.0,'X'),(3,'c',30.0,'Y'),(4,'d',40.0,'Y'),(5,'e',50.0,'Z')",
        "UPDATE cuj_items SET price = price * 1.1 WHERE category = (SELECT category FROM cuj_items GROUP BY category ORDER BY SUM(price) DESC LIMIT 1)",
        "UPDATE cuj_items SET name = CASE WHEN price > 30 THEN UPPER(name) ELSE name END",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM cuj_items ORDER BY id",
        "SELECT category, SUM(price) FROM cuj_items GROUP BY category ORDER BY category",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!(
            "{} complex UPDATE with subquery mismatches",
            mismatches.len()
        );
    }
}

#[test]
fn test_conformance_trigger_insert_update_delete_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE tba_log(id INTEGER PRIMARY KEY, action TEXT, item_id INTEGER, ts TEXT DEFAULT 'now')",
        "CREATE TABLE tba_items(id INTEGER PRIMARY KEY, name TEXT, active INTEGER DEFAULT 1)",
        "CREATE TRIGGER tba_after_insert AFTER INSERT ON tba_items BEGIN INSERT INTO tba_log(action, item_id) VALUES('INSERT', NEW.id); END",
        "CREATE TRIGGER tba_after_delete AFTER DELETE ON tba_items BEGIN INSERT INTO tba_log(action, item_id) VALUES('DELETE', OLD.id); END",
        "CREATE TRIGGER tba_after_update AFTER UPDATE ON tba_items BEGIN INSERT INTO tba_log(action, item_id) VALUES('UPDATE', NEW.id); END",
        "INSERT INTO tba_items(id, name) VALUES(1, 'first')",
        "INSERT INTO tba_items(id, name) VALUES(2, 'second')",
        "INSERT INTO tba_items(id, name) VALUES(3, 'third')",
        "UPDATE tba_items SET active = 0 WHERE id = 2",
        "DELETE FROM tba_items WHERE id = 3",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM tba_items ORDER BY id",
        "SELECT action, item_id FROM tba_log ORDER BY id",
        "SELECT COUNT(*) FROM tba_log",
        "SELECT action, COUNT(*) FROM tba_log GROUP BY action ORDER BY action",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!(
            "{} trigger insert/update/delete mismatches",
            mismatches.len()
        );
    }
}

#[test]
fn test_conformance_coalesce_with_subquery_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE csq_t(id INTEGER PRIMARY KEY, val INTEGER)",
        "INSERT INTO csq_t VALUES(1,10),(2,NULL),(3,30)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT COALESCE(val, 0) FROM csq_t ORDER BY id",
        "SELECT COALESCE(val, (SELECT MAX(val) FROM csq_t)) FROM csq_t ORDER BY id",
        "SELECT id, COALESCE(val, -1) + 1 FROM csq_t ORDER BY id",
        "SELECT COALESCE(NULL, NULL, 'found')",
        "SELECT COALESCE(NULL, 42, 99)",
        "SELECT COALESCE(1, NULL, 99)",
        "SELECT id, COALESCE(val, id * 100) FROM csq_t ORDER BY id",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} COALESCE with subquery mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_group_concat_separator_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE gco(id INTEGER PRIMARY KEY, grp TEXT, val TEXT)",
        "INSERT INTO gco VALUES(1,'A','x'),(2,'A','y'),(3,'A','z'),(4,'B','m'),(5,'B','n')",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT grp, GROUP_CONCAT(val) FROM gco GROUP BY grp ORDER BY grp",
        "SELECT grp, GROUP_CONCAT(val, ';') FROM gco GROUP BY grp ORDER BY grp",
        "SELECT grp, GROUP_CONCAT(val, '') FROM gco GROUP BY grp ORDER BY grp",
        "SELECT GROUP_CONCAT(val) FROM gco",
        "SELECT GROUP_CONCAT(DISTINCT grp) FROM gco",
        "SELECT grp, GROUP_CONCAT(val || '!', ' ') FROM gco GROUP BY grp ORDER BY grp",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} GROUP_CONCAT separator mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_typeof_in_expressions_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE tie(id INTEGER PRIMARY KEY, a, b TEXT, c REAL, d BLOB)",
        "INSERT INTO tie VALUES(1, 42, 'hello', 3.14, X'DEADBEEF')",
        "INSERT INTO tie VALUES(2, NULL, NULL, NULL, NULL)",
        "INSERT INTO tie VALUES(3, 'text', '123', 0.0, X'')",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT typeof(a), typeof(b), typeof(c), typeof(d) FROM tie ORDER BY id",
        "SELECT typeof(a + 0), typeof(b || ''), typeof(c * 1) FROM tie WHERE id = 1",
        "SELECT typeof(CAST(42 AS TEXT)), typeof(CAST('42' AS INTEGER)), typeof(CAST('3.14' AS REAL))",
        "SELECT typeof(NULL), typeof(0), typeof(0.0), typeof(''), typeof(X'')",
        "SELECT typeof(1 + 1), typeof(1 + 1.0), typeof(1 || 'x')",
        "SELECT typeof(COALESCE(NULL, 1)), typeof(COALESCE(NULL, 'x'))",
        "SELECT typeof(IIF(1, 42, 'text')), typeof(IIF(0, 42, 'text'))",
        "SELECT typeof(MIN(1, 2)), typeof(MAX(1.0, 2))",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} typeof in expressions mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_cross_type_arithmetic_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT 1 + 1.0",
        "SELECT typeof(1 + 1.0)",
        "SELECT 1 + '2'",
        "SELECT typeof(1 + '2')",
        "SELECT '3' * '4'",
        "SELECT typeof('3' * '4')",
        "SELECT 1 + 'abc'",
        "SELECT typeof(1 + 'abc')",
        "SELECT 10 / 3",
        "SELECT typeof(10 / 3)",
        "SELECT 10.0 / 3",
        "SELECT typeof(10.0 / 3)",
        "SELECT 10 % 3",
        "SELECT -(-5)",
        "SELECT typeof(-(-5))",
        "SELECT 1 << 4",
        "SELECT 255 & 15",
        "SELECT 10 | 5",
        "SELECT ~0",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} cross-type arithmetic mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_recursive_cte_series_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "WITH RECURSIVE cnt(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM cnt WHERE x < 10) SELECT x FROM cnt",
        "WITH RECURSIVE cnt(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM cnt WHERE x < 5) SELECT SUM(x) FROM cnt",
        "WITH RECURSIVE p2(n, val) AS (VALUES(0, 1) UNION ALL SELECT n+1, val*2 FROM p2 WHERE n < 8) SELECT n, val FROM p2",
        "WITH RECURSIVE fact(n, f) AS (VALUES(1, 1) UNION ALL SELECT n+1, f*(n+1) FROM fact WHERE n < 10) SELECT n, f FROM fact",
        "WITH RECURSIVE s(n, acc) AS (VALUES(1, 'a') UNION ALL SELECT n+1, acc || CHAR(96+n+1) FROM s WHERE n < 5) SELECT acc FROM s ORDER BY n DESC LIMIT 1",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} recursive CTE series mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_having_without_group_by_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE hwg(id INTEGER PRIMARY KEY, val INTEGER)",
        "INSERT INTO hwg VALUES(1,10),(2,20),(3,30)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT COUNT(*) FROM hwg HAVING COUNT(*) > 2",
        "SELECT SUM(val) FROM hwg HAVING SUM(val) > 50",
        "SELECT COUNT(*) FROM hwg HAVING COUNT(*) > 10",
        "SELECT AVG(val) FROM hwg HAVING AVG(val) > 15",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} HAVING without GROUP BY mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_insert_or_ignore_replace_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE ioi(id INTEGER PRIMARY KEY, name TEXT UNIQUE, val INTEGER)",
        "INSERT INTO ioi VALUES(1,'alpha',10)",
        "INSERT INTO ioi VALUES(2,'beta',20)",
        "INSERT OR IGNORE INTO ioi VALUES(3,'alpha',30)",
        "INSERT OR IGNORE INTO ioi VALUES(4,'gamma',40)",
        "INSERT OR REPLACE INTO ioi VALUES(2,'beta',99)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM ioi ORDER BY id",
        "SELECT COUNT(*) FROM ioi",
        "SELECT name, val FROM ioi WHERE name = 'alpha'",
        "SELECT name, val FROM ioi WHERE name = 'beta'",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} INSERT OR IGNORE/REPLACE mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_multi_table_delete_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE mtd_orders(id INTEGER PRIMARY KEY, customer TEXT, total REAL)",
        "CREATE TABLE mtd_items(id INTEGER PRIMARY KEY, order_id INTEGER, product TEXT, qty INTEGER)",
        "INSERT INTO mtd_orders VALUES(1,'Alice',100.0),(2,'Bob',50.0),(3,'Carol',75.0)",
        "INSERT INTO mtd_items VALUES(10,1,'Widget',2),(11,1,'Gadget',1),(12,2,'Widget',5),(13,3,'Gizmo',3)",
        "DELETE FROM mtd_items WHERE order_id IN (SELECT id FROM mtd_orders WHERE total < 60)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM mtd_items ORDER BY id",
        "SELECT COUNT(*) FROM mtd_items",
        "SELECT o.customer, COUNT(i.id) FROM mtd_orders o LEFT JOIN mtd_items i ON i.order_id = o.id GROUP BY o.customer ORDER BY o.customer",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} multi-table DELETE mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_string_padding_trimming_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT TRIM('  hello  ')",
        "SELECT LTRIM('  hello  ')",
        "SELECT RTRIM('  hello  ')",
        "SELECT TRIM('xxhelloxx', 'x')",
        "SELECT LTRIM('xxhelloxx', 'x')",
        "SELECT RTRIM('xxhelloxx', 'x')",
        "SELECT LENGTH(TRIM('  abc  '))",
        "SELECT REPLACE('hello world', 'world', 'earth')",
        "SELECT REPLACE('aaa', 'a', 'bb')",
        "SELECT REPLACE('', 'a', 'b')",
        "SELECT REPLACE('hello', '', 'x')",
        "SELECT SUBSTR('hello', 1, 3)",
        "SELECT SUBSTR('hello', -3)",
        "SELECT SUBSTR('hello', 2)",
        "SELECT SUBSTR('hello', 0, 3)",
        "SELECT SUBSTR('hello', -10, 3)",
        "SELECT UPPER('hello'), LOWER('HELLO')",
        "SELECT UNICODE('A'), UNICODE('Z'), UNICODE('a')",
        "SELECT CHAR(65), CHAR(90), CHAR(97)",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} string padding/trimming mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_complex_view_join_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE cvq_products(id INTEGER PRIMARY KEY, name TEXT, price REAL, category TEXT)",
        "INSERT INTO cvq_products VALUES(1,'A',10.0,'elec'),(2,'B',20.0,'elec'),(3,'C',15.0,'books'),(4,'D',5.0,'books'),(5,'E',50.0,'elec')",
        "CREATE VIEW cvq_expensive AS SELECT * FROM cvq_products WHERE price > 12",
        "CREATE VIEW cvq_cat_stats AS SELECT category, COUNT(*) AS cnt, AVG(price) AS avg_price FROM cvq_products GROUP BY category",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM cvq_expensive ORDER BY id",
        "SELECT COUNT(*) FROM cvq_expensive",
        "SELECT category, SUM(price) FROM cvq_expensive GROUP BY category ORDER BY category",
        "SELECT * FROM cvq_cat_stats ORDER BY category",
        "SELECT e.name, s.avg_price FROM cvq_expensive e JOIN cvq_cat_stats s ON e.category = s.category ORDER BY e.name",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} complex view join mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_abs_zero_edge_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT ABS(0), ABS(-0), ABS(0.0), ABS(-0.0)",
        "SELECT ABS(-9223372036854775807)",
        "SELECT ABS(1), ABS(-1), ABS(1.5), ABS(-1.5)",
        "SELECT ABS(NULL)",
        "SELECT ABS('hello')",
        "SELECT ABS('-42')",
        "SELECT MIN(1, 2, 3), MAX(1, 2, 3)",
        "SELECT MIN(NULL, 1, 2), MAX(NULL, 1, 2)",
        "SELECT MIN('a', 'b', 'c'), MAX('a', 'b', 'c')",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} abs/zero edge mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_instr_hex_quote_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT INSTR('hello world', 'world')",
        "SELECT INSTR('hello world', 'xyz')",
        "SELECT INSTR('abcabc', 'bc')",
        "SELECT INSTR('', 'x')",
        "SELECT INSTR('hello', '')",
        "SELECT INSTR(NULL, 'x')",
        "SELECT INSTR('hello', NULL)",
        "SELECT HEX('hello')",
        "SELECT HEX(42)",
        "SELECT HEX(NULL)",
        "SELECT HEX(X'DEADBEEF')",
        "SELECT QUOTE('hello')",
        "SELECT QUOTE(42)",
        "SELECT QUOTE(3.14)",
        "SELECT QUOTE(NULL)",
        "SELECT QUOTE(X'ABCD')",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} instr/hex/quote mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_like_complex_patterns_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE lcp(id INTEGER PRIMARY KEY, val TEXT)",
        "INSERT INTO lcp VALUES(1,'hello'),(2,'HELLO'),(3,'Hello World'),(4,'%special%'),(5,'under_score'),(6,''),(7,NULL)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT id FROM lcp WHERE val LIKE 'hello' ORDER BY id",
        "SELECT id FROM lcp WHERE val LIKE 'HELLO' ORDER BY id",
        "SELECT id FROM lcp WHERE val LIKE '%llo%' ORDER BY id",
        "SELECT id FROM lcp WHERE val LIKE 'H_llo' ORDER BY id",
        "SELECT id FROM lcp WHERE val LIKE '' ORDER BY id",
        "SELECT id FROM lcp WHERE val LIKE '%' ORDER BY id",
        "SELECT id FROM lcp WHERE val NOT LIKE '%llo%' ORDER BY id",
        "SELECT id FROM lcp WHERE val LIKE '%!%%' ESCAPE '!' ORDER BY id",
        "SELECT id FROM lcp WHERE val LIKE '%!_%' ESCAPE '!' ORDER BY id",
        "SELECT 'abc' LIKE 'ABC'",
        "SELECT 'abc' LIKE 'a%'",
        "SELECT NULL LIKE 'x'",
        "SELECT 'x' LIKE NULL",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} LIKE complex pattern mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_cast_edge_cases_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT CAST(NULL AS INTEGER)",
        "SELECT CAST(NULL AS TEXT)",
        "SELECT CAST(NULL AS REAL)",
        "SELECT CAST(NULL AS BLOB)",
        "SELECT CAST('' AS INTEGER)",
        "SELECT CAST('' AS REAL)",
        "SELECT CAST('abc' AS INTEGER)",
        "SELECT CAST('abc' AS REAL)",
        "SELECT CAST('123abc' AS INTEGER)",
        "SELECT CAST('123.45abc' AS REAL)",
        "SELECT CAST(X'48454C4C4F' AS TEXT)",
        "SELECT CAST(9223372036854775807 AS REAL)",
        "SELECT typeof(CAST(9223372036854775807 AS REAL))",
        "SELECT CAST(1 AS TEXT), typeof(CAST(1 AS TEXT))",
        "SELECT CAST(3.14 AS INTEGER), typeof(CAST(3.14 AS INTEGER))",
        "SELECT CAST(3.99 AS INTEGER)",
        "SELECT CAST(-3.99 AS INTEGER)",
        "SELECT CAST(0 AS TEXT), CAST(0 AS REAL)",
        "SELECT CAST(1e20 AS INTEGER)",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} CAST edge case mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_zeroblob_length_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT LENGTH(ZEROBLOB(0))",
        "SELECT LENGTH(ZEROBLOB(1))",
        "SELECT LENGTH(ZEROBLOB(10))",
        "SELECT typeof(ZEROBLOB(5))",
        "SELECT HEX(ZEROBLOB(4))",
        "SELECT LENGTH(X'')",
        "SELECT typeof(X'')",
        "SELECT HEX(X'00FF')",
        "SELECT LENGTH(X'DEADBEEF')",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} zeroblob/length mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_between_null_type_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT 5 BETWEEN 1 AND 10",
        "SELECT 5 BETWEEN 10 AND 1",
        "SELECT NULL BETWEEN 1 AND 10",
        "SELECT 5 BETWEEN NULL AND 10",
        "SELECT 5 BETWEEN 1 AND NULL",
        "SELECT 'c' BETWEEN 'a' AND 'e'",
        "SELECT 'f' BETWEEN 'a' AND 'e'",
        "SELECT 5 NOT BETWEEN 1 AND 10",
        "SELECT 15 NOT BETWEEN 1 AND 10",
        "SELECT NULL NOT BETWEEN 1 AND 10",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} BETWEEN NULL/type mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_in_list_edge_cases_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT 1 IN (1, 2, 3)",
        "SELECT 4 IN (1, 2, 3)",
        "SELECT NULL IN (1, 2, 3)",
        "SELECT 1 IN (1, NULL, 3)",
        "SELECT 2 IN (1, NULL, 3)",
        "SELECT NULL IN (NULL)",
        "SELECT 1 NOT IN (1, 2, 3)",
        "SELECT 4 NOT IN (1, 2, 3)",
        "SELECT NULL NOT IN (1, 2, 3)",
        "SELECT 'a' IN ('a', 'b', 'c')",
        "SELECT 'x' IN ('a', 'b', 'c')",
        "SELECT 1 IN (SELECT 1 UNION SELECT 2 UNION SELECT 3)",
        "SELECT 5 IN (SELECT 1 UNION SELECT 2 UNION SELECT 3)",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} IN list edge case mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_transaction_savepoint_complex_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE tsc(id INTEGER PRIMARY KEY, val TEXT)",
        "INSERT INTO tsc VALUES(1, 'original')",
        "SAVEPOINT sp1",
        "INSERT INTO tsc VALUES(2, 'in_sp1')",
        "SAVEPOINT sp2",
        "INSERT INTO tsc VALUES(3, 'in_sp2')",
        "ROLLBACK TO sp2",
        "INSERT INTO tsc VALUES(4, 'after_rollback_sp2')",
        "RELEASE sp1",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = ["SELECT * FROM tsc ORDER BY id", "SELECT COUNT(*) FROM tsc"];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!(
            "{} transaction/savepoint complex mismatches",
            mismatches.len()
        );
    }
}

#[test]
fn test_conformance_multiple_default_values_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE mdv(id INTEGER PRIMARY KEY, a INTEGER DEFAULT 0, b TEXT DEFAULT 'hello', c REAL DEFAULT 3.14, d INTEGER DEFAULT (1 + 2))",
        "INSERT INTO mdv(id) VALUES(1)",
        "INSERT INTO mdv(id, a) VALUES(2, 99)",
        "INSERT INTO mdv(id, b) VALUES(3, 'world')",
        "INSERT INTO mdv VALUES(4, 10, 'custom', 2.71, 100)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM mdv ORDER BY id",
        "SELECT a, b, c, d FROM mdv WHERE id = 1",
        "SELECT COUNT(*), SUM(a), SUM(d) FROM mdv",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} multiple default values mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_multi_column_order_by_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE mco(id INTEGER PRIMARY KEY, a TEXT, b INTEGER, c REAL)",
        "INSERT INTO mco VALUES(1,'x',10,1.0),(2,'y',10,2.0),(3,'x',20,1.5),(4,'y',20,0.5),(5,'x',10,3.0),(6,'z',10,1.0)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM mco ORDER BY a, b, c",
        "SELECT * FROM mco ORDER BY a ASC, b DESC, c ASC",
        "SELECT * FROM mco ORDER BY b DESC, a ASC, id",
        "SELECT a, SUM(c) AS total FROM mco GROUP BY a ORDER BY total DESC, a",
        "SELECT a, b, COUNT(*) FROM mco GROUP BY a, b ORDER BY COUNT(*) DESC, a, b",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} multi-column ORDER BY mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_distinct_with_order_by_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE dwo(id INTEGER PRIMARY KEY, category TEXT, val INTEGER)",
        "INSERT INTO dwo VALUES(1,'A',10),(2,'B',20),(3,'A',30),(4,'C',10),(5,'B',20),(6,'A',10)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT DISTINCT category FROM dwo ORDER BY category",
        "SELECT DISTINCT val FROM dwo ORDER BY val",
        "SELECT DISTINCT category, val FROM dwo ORDER BY category, val",
        "SELECT DISTINCT category FROM dwo ORDER BY category DESC",
        "SELECT COUNT(DISTINCT category) FROM dwo",
        "SELECT COUNT(DISTINCT val) FROM dwo",
        "SELECT SUM(DISTINCT val) FROM dwo",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} DISTINCT with ORDER BY mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_create_table_as_select_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE ctas_src(id INTEGER PRIMARY KEY, name TEXT, val REAL)",
        "INSERT INTO ctas_src VALUES(1,'a',1.5),(2,'b',2.5),(3,'c',3.5)",
        "CREATE TABLE ctas_copy AS SELECT * FROM ctas_src WHERE val > 2",
        "CREATE TABLE ctas_agg AS SELECT name, val * 2 AS doubled FROM ctas_src",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM ctas_copy ORDER BY id",
        "SELECT COUNT(*) FROM ctas_copy",
        "SELECT * FROM ctas_agg ORDER BY name",
        "SELECT name, doubled FROM ctas_agg WHERE doubled > 4 ORDER BY name",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} CREATE TABLE AS SELECT mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_update_with_case_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE uwc(id INTEGER PRIMARY KEY, val INTEGER, label TEXT)",
        "INSERT INTO uwc VALUES(1,10,'low'),(2,20,'low'),(3,30,'low'),(4,40,'low'),(5,50,'low')",
        "UPDATE uwc SET label = CASE WHEN val > 30 THEN 'high' WHEN val > 15 THEN 'mid' ELSE 'low' END",
        "UPDATE uwc SET val = val + 100 WHERE val > 20",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM uwc ORDER BY id",
        "SELECT SUM(val) FROM uwc",
        "SELECT label, COUNT(*) FROM uwc GROUP BY label ORDER BY label",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} UPDATE with CASE mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_delete_with_subquery_where_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE dol(id INTEGER PRIMARY KEY, val INTEGER)",
        "INSERT INTO dol VALUES(1,10),(2,20),(3,30),(4,40),(5,50)",
        "DELETE FROM dol WHERE val > (SELECT AVG(val) FROM dol)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM dol ORDER BY id",
        "SELECT COUNT(*) FROM dol",
        "SELECT SUM(val) FROM dol",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} DELETE with subquery WHERE mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_rowid_edge_cases_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE roe(a TEXT, b INTEGER)",
        "INSERT INTO roe VALUES('x', 10)",
        "INSERT INTO roe VALUES('y', 20)",
        "INSERT INTO roe VALUES('z', 30)",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT rowid, a, b FROM roe ORDER BY rowid",
        "SELECT rowid, a FROM roe WHERE rowid = 2",
        "SELECT rowid, a FROM roe WHERE rowid BETWEEN 1 AND 2 ORDER BY rowid",
        "SELECT MAX(rowid) FROM roe",
        "SELECT COUNT(*) FROM roe WHERE rowid > 1",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} rowid edge case mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_insert_select_with_transform_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE ist_src(id INTEGER PRIMARY KEY, name TEXT, val INTEGER)",
        "INSERT INTO ist_src VALUES(1,'alpha',10),(2,'beta',20),(3,'gamma',30)",
        "CREATE TABLE ist_dest(id INTEGER PRIMARY KEY, label TEXT, doubled INTEGER)",
        "INSERT INTO ist_dest SELECT id, UPPER(name), val * 2 FROM ist_src WHERE val >= 20",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT * FROM ist_dest ORDER BY id",
        "SELECT COUNT(*) FROM ist_dest",
        "SELECT SUM(doubled) FROM ist_dest",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!(
            "{} INSERT SELECT with transform mismatches",
            mismatches.len()
        );
    }
}

#[test]
fn test_conformance_empty_table_aggregates_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &["CREATE TABLE eta(id INTEGER PRIMARY KEY, val INTEGER, name TEXT)"] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT COUNT(*) FROM eta",
        "SELECT COUNT(val) FROM eta",
        "SELECT SUM(val) FROM eta",
        "SELECT AVG(val) FROM eta",
        "SELECT MIN(val) FROM eta",
        "SELECT MAX(val) FROM eta",
        "SELECT TOTAL(val) FROM eta",
        "SELECT GROUP_CONCAT(name) FROM eta",
        "SELECT typeof(SUM(val)) FROM eta",
        "SELECT typeof(TOTAL(val)) FROM eta",
        "SELECT COALESCE(SUM(val), 0) FROM eta",
        "SELECT COUNT(*), SUM(val), AVG(val), MIN(val), MAX(val) FROM eta",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} empty table aggregate mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_mixed_type_sort_order_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    for s in &[
        "CREATE TABLE mts(id INTEGER PRIMARY KEY, val)",
        "INSERT INTO mts VALUES(1, NULL)",
        "INSERT INTO mts VALUES(2, 42)",
        "INSERT INTO mts VALUES(3, 3.14)",
        "INSERT INTO mts VALUES(4, 'text')",
        "INSERT INTO mts VALUES(5, X'ABCD')",
        "INSERT INTO mts VALUES(6, 0)",
        "INSERT INTO mts VALUES(7, '')",
    ] {
        fconn.execute(s).unwrap();
        rconn.execute_batch(s).unwrap();
    }

    let queries = [
        "SELECT id, val, typeof(val) FROM mts ORDER BY val",
        "SELECT id, val, typeof(val) FROM mts ORDER BY val DESC",
        "SELECT typeof(val), COUNT(*) FROM mts GROUP BY typeof(val) ORDER BY typeof(val)",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} mixed type sort order mismatches", mismatches.len());
    }
}

#[test]
fn test_conformance_compound_with_null_s55() {
    let fconn = Connection::open(":memory:").unwrap();
    let rconn = rusqlite::Connection::open_in_memory().unwrap();

    let queries = [
        "SELECT 1 UNION SELECT NULL ORDER BY 1",
        "SELECT NULL UNION SELECT 1 UNION SELECT NULL ORDER BY 1",
        "SELECT 1, 'a' UNION ALL SELECT NULL, 'b' UNION ALL SELECT 2, NULL ORDER BY 1",
        "SELECT 1 INTERSECT SELECT 1",
        "SELECT 1 EXCEPT SELECT 2",
        "SELECT 1 UNION SELECT 1 UNION SELECT 2",
        "SELECT * FROM (SELECT 1 AS x UNION ALL SELECT 2 UNION ALL SELECT 3) WHERE x > 1",
    ];

    let mismatches = oracle_compare(&fconn, &rconn, &queries);
    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("{m}\n");
        }
        panic!("{} compound with NULL mismatches", mismatches.len());
    }
}
