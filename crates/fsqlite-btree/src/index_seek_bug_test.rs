#[cfg(test)]
mod tests {
    use crate::cursor::{BtCursor, BtreeCursorOps};
    use crate::cell::BtreePageType;
    use fsqlite_types::cx::Cx;
    use fsqlite_types::PageNumber;
    use fsqlite_pager::pager::SimplePager;
    use fsqlite_vfs::memory::MemoryVfs;

    #[test]
    fn test_index_seek_bug() {
        let vfs = MemoryVfs::new();
        let path = std::path::PathBuf::from("/test.db");
        let mut pager = SimplePager::open(vfs, &path, fsqlite_types::limits::PageSize::DEFAULT).unwrap();
        let cx = Cx::new();
        
        let root = pager.allocate_page(&cx).unwrap();
        let mut cursor = BtCursor::with_empty_index(root, 4096);
        cursor.bind(&mut pager);

        // Insert a middle value
        cursor.index_insert(&cx, b"M").unwrap();
        
        // Let's force a split? Actually, we can just insert A, Z, etc.
        for i in 0..100 {
            let key = format!("KEY_{:03}", i);
            cursor.index_insert(&cx, key.as_bytes()).unwrap();
        }
        
        // If the bug exists, inserting something that falls off a leaf but has a successor will panic.
        // Or we can just insert keys in order to force it.
        // Let's try inserting sequentially, then inserting one in the middle that falls at the end of a leaf
        
        // A better way is to do many inserts and see if it fails.
        let mut keys = Vec::new();
        for i in 0..250 {
            keys.push(format!("K{:03}", i));
        }
        for k in &keys {
            cursor.index_insert(&cx, k.as_bytes()).unwrap();
        }
    }
}
