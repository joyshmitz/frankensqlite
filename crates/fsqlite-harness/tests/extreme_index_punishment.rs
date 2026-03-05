use fsqlite::Connection;
use fsqlite::compat::BatchExt;
use std::sync::{Arc, Barrier};
use std::thread;

#[test]
fn test_extreme_index_punishment() {
    let db_path = "file:memdb_punishment_4?mode=memory&cache=shared";
    
    let setup = Connection::open(db_path).unwrap();
    setup.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA busy_timeout=5000;
         CREATE TABLE items (id INTEGER PRIMARY KEY, category INTEGER, data TEXT);
         CREATE INDEX idx_category ON items(category);"
    ).unwrap();

    let n_threads = 20;
    let ops_per_thread = 200;
    let barrier = Arc::new(Barrier::new(n_threads));

    let mut handles = vec![];
    for tid in 0..n_threads {
        let bar = barrier.clone();
        let path = db_path.to_string();
        handles.push(thread::spawn(move || -> Result<(), String> {
            let conn = Connection::open(&path).map_err(|e| format!("{:?}", e))?;
            conn.execute_batch("PRAGMA busy_timeout=5000;").map_err(|e| format!("{:?}", e))?;
            bar.wait();
            
            for i in 0..ops_per_thread {
                let id = tid * 10000 + i;
                let category = 42; // ALL threads hammer the EXACT same index bucket to force maximum splits/merges
                let data = format!("thread_{}_op_{}", tid, i);
                
                loop {
                    match conn.execute("BEGIN CONCURRENT") {
                        Ok(_) => break,
                        Err(_) => thread::sleep(std::time::Duration::from_millis(1)),
                    }
                }
                
                conn.execute(
                    &format!("INSERT INTO items (id, category, data) VALUES ({}, {}, '{}')", id, category, data)
                ).map_err(|e| format!("{:?}", e))?;
                
                loop {
                    match conn.execute("COMMIT") {
                        Ok(_) => break,
                        Err(e) if e.to_string().contains("database is locked") || e.to_string().contains("busy") => {
                            conn.execute("ROLLBACK").map_err(|e| format!("{:?}", e))?;
                            loop {
                                match conn.execute("BEGIN CONCURRENT") {
                                    Ok(_) => break,
                                    Err(_) => thread::sleep(std::time::Duration::from_millis(1)),
                                }
                            }
                            conn.execute(
                                &format!("INSERT INTO items (id, category, data) VALUES ({}, {}, '{}')", id, category, data)
                            ).map_err(|e| format!("{:?}", e))?;
                        }
                        Err(e) => return Err(format!("unexpected error: {:?}", e)),
                    }
                }
            }
            Ok(())
        }));
    }

    for h in handles {
        let res = h.join().unwrap();
        assert!(res.is_ok(), "Thread failed: {:?}", res);
    }

    let rows = setup.query("SELECT COUNT(*) FROM items").unwrap();
    let total: i64 = match &rows[0].values()[0] {
        fsqlite_types::SqliteValue::Integer(i) => *i,
        _ => panic!("Expected integer"),
    };
    
    assert_eq!(total, (n_threads * ops_per_thread) as i64);
    
    // Also verify the index is intact
    let rows = setup.query("SELECT COUNT(*) FROM items INDEXED BY idx_category WHERE category = 42").unwrap();
    let index_total: i64 = match &rows[0].values()[0] {
        fsqlite_types::SqliteValue::Integer(i) => *i,
        _ => panic!("Expected integer"),
    };
    assert_eq!(index_total, (n_threads * ops_per_thread) as i64);
}