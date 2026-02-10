#[cfg(test)]
mod tests {
    use crate::engine::{MemDatabase, VdbeEngine, ExecOutcome};
    use crate::VdbeProgram;
    use fsqlite_types::opcode::{Opcode, P4};
    use fsqlite_types::value::SqliteValue;

    #[test]
    fn test_repro_delete_skips_next_row() {
        // Create table with 3 rows: 1, 2, 3.
        let mut db = MemDatabase::new();
        let root = db.create_table(1);
        db.insert_row(1, vec![SqliteValue::Integer(1)]);
        db.insert_row(2, vec![SqliteValue::Integer(2)]);
        db.insert_row(3, vec![SqliteValue::Integer(3)]);

        // Program:
        // 0. OpenRead cursor 0 on root
        // 1. Rewind 0 to End (label 6)
        // 2. Rowid 0 -> r1
        // 3. Eq r1, target(2) -> Delete
        // 4. Next 0 -> 2
        // 5. Halt
        // 6. Halt
        //
        // Delete step:
        // Delete rowid 2.
        // Then Next.
        // Expectation: Should visit row 3.
        // Bug: If Delete doesn't adjust cursor, Next skips row 3.

        let mut engine = VdbeEngine::new(5);
        engine.set_database(db);

        // Manually build program to iterate and delete 2.
        use fsqlite_types::opcode::VdbeOp;
        let ops = vec![
            VdbeOp { opcode: Opcode::OpenRead, p1: 0, p2: root, p3: 0, p4: P4::None, p5: 0 },
            VdbeOp { opcode: Opcode::Rewind, p1: 0, p2: 7, p3: 0, p4: P4::None, p5: 0 },
            // Loop start (addr 2)
            VdbeOp { opcode: Opcode::Rowid, p1: 0, p2: 1, p3: 0, p4: P4::None, p5: 0 },
            VdbeOp { opcode: Opcode::Integer, p1: 2, p2: 2, p3: 0, p4: P4::None, p5: 0 },
            VdbeOp { opcode: Opcode::Ne, p1: 2, p2: 6, p3: 1, p4: P4::None, p5: 0 }, // if r1 != 2, skip delete
            // Delete row 2
            VdbeOp { opcode: Opcode::Delete, p1: 0, p2: 0, p3: 0, p4: P4::None, p5: 0 },
            // Loop end (addr 6)
            VdbeOp { opcode: Opcode::Next, p1: 0, p2: 2, p3: 0, p4: P4::None, p5: 0 },
            VdbeOp { opcode: Opcode::Halt, p1: 0, p2: 0, p3: 0, p4: P4::None, p5: 0 },
        ];

        // We can't easily assert visited rows without ResultRow or tracing.
        // Instead, we check the DB content after.
        // But the bug is about *skipping* the next row during iteration.
        // If we skip row 3, we won't see it.
        // Let's modify the program to Count visited rows.
        // Or better, ResultRow.
        
        let ops_result = vec![
            VdbeOp { opcode: Opcode::OpenRead, p1: 0, p2: root, p3: 0, p4: P4::None, p5: 0 },
            VdbeOp { opcode: Opcode::Rewind, p1: 0, p2: 8, p3: 0, p4: P4::None, p5: 0 },
            // Loop start (addr 2)
            VdbeOp { opcode: Opcode::Rowid, p1: 0, p2: 1, p3: 0, p4: P4::None, p5: 0 },
            VdbeOp { opcode: Opcode::ResultRow, p1: 1, p2: 1, p3: 0, p4: P4::None, p5: 0 }, // Emit rowid
            VdbeOp { opcode: Opcode::Integer, p1: 2, p2: 2, p3: 0, p4: P4::None, p5: 0 },
            VdbeOp { opcode: Opcode::Ne, p1: 2, p2: 7, p3: 1, p4: P4::None, p5: 0 }, // if r1 != 2, skip delete
            // Delete row 2
            VdbeOp { opcode: Opcode::Delete, p1: 0, p2: 0, p3: 0, p4: P4::None, p5: 0 },
            // Loop end (addr 7)
            VdbeOp { opcode: Opcode::Next, p1: 0, p2: 2, p3: 0, p4: P4::None, p5: 0 },
            VdbeOp { opcode: Opcode::Halt, p1: 0, p2: 0, p3: 0, p4: P4::None, p5: 0 },
        ];
        
        // Use a wrapper or hack to run this since VdbeProgram fields are crate-private?
        // No, ops is pub(crate). But `VdbeProgram` struct fields are private.
        // I need to use `ProgramBuilder` or similar.
        // But `ProgramBuilder` is in `lib.rs`, `VdbeEngine` in `engine.rs`.
        // I can construct `VdbeProgram` if I can access fields? No.
        // I should use `ProgramBuilder` to build it.
    }
}
