use fsqlite_btree::{BtreeCursorOps, SeekResult};
use fsqlite_error::Result;
use fsqlite_types::cx::Cx;

struct ExternalCursor;

impl BtreeCursorOps for ExternalCursor {
    fn index_move_to(&mut self, _cx: &Cx, _key: &[u8]) -> Result<SeekResult> {
        unreachable!("compile-fail test should not execute")
    }

    fn table_move_to(&mut self, _cx: &Cx, _rowid: i64) -> Result<SeekResult> {
        unreachable!("compile-fail test should not execute")
    }

    fn first(&mut self, _cx: &Cx) -> Result<bool> {
        unreachable!("compile-fail test should not execute")
    }

    fn last(&mut self, _cx: &Cx) -> Result<bool> {
        unreachable!("compile-fail test should not execute")
    }

    fn next(&mut self, _cx: &Cx) -> Result<bool> {
        unreachable!("compile-fail test should not execute")
    }

    fn prev(&mut self, _cx: &Cx) -> Result<bool> {
        unreachable!("compile-fail test should not execute")
    }

    fn index_insert(&mut self, _cx: &Cx, _key: &[u8]) -> Result<()> {
        unreachable!("compile-fail test should not execute")
    }

    fn table_insert(&mut self, _cx: &Cx, _rowid: i64, _data: &[u8]) -> Result<()> {
        unreachable!("compile-fail test should not execute")
    }

    fn delete(&mut self, _cx: &Cx) -> Result<()> {
        unreachable!("compile-fail test should not execute")
    }

    fn payload(&self, _cx: &Cx) -> Result<Vec<u8>> {
        unreachable!("compile-fail test should not execute")
    }

    fn rowid(&self, _cx: &Cx) -> Result<i64> {
        unreachable!("compile-fail test should not execute")
    }

    fn eof(&self) -> bool {
        unreachable!("compile-fail test should not execute")
    }
}

fn main() {}
