use fsqlite_error::Result;
use fsqlite_pager::{MockTransaction, MvccPager, TransactionMode};
use fsqlite_types::cx::Cx;

struct ExternalPager;

impl MvccPager for ExternalPager {
    type Txn = MockTransaction;

    fn begin(&self, _cx: &Cx, _mode: TransactionMode) -> Result<Self::Txn> {
        unreachable!("compile-fail test should not execute")
    }
}

fn main() {}
