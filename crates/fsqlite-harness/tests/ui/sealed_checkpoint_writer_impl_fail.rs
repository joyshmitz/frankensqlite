use fsqlite_error::Result;
use fsqlite_pager::CheckpointPageWriter;
use fsqlite_types::PageNumber;
use fsqlite_types::cx::Cx;

struct ExternalCheckpointWriter;

impl CheckpointPageWriter for ExternalCheckpointWriter {
    fn write_page(&mut self, _cx: &Cx, _page_no: PageNumber, _data: &[u8]) -> Result<()> {
        unreachable!("compile-fail test should not execute")
    }

    fn truncate(&mut self, _cx: &Cx, _n_pages: u32) -> Result<()> {
        unreachable!("compile-fail test should not execute")
    }

    fn sync(&mut self, _cx: &Cx) -> Result<()> {
        unreachable!("compile-fail test should not execute")
    }
}

fn main() {}
