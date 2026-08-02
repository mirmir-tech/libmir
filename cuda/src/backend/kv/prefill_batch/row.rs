use runtime::kv::BlockTable;

#[derive(Debug)]
pub struct PrefillBatchRow {
    table: BlockTable,
    start: usize,
    tokens: usize,
}

impl PrefillBatchRow {
    pub(super) const fn new(table: BlockTable, start: usize, tokens: usize) -> Self {
        Self { table, start, tokens }
    }

    pub const fn table(&self) -> &BlockTable {
        &self.table
    }

    pub const fn start(&self) -> usize {
        self.start
    }

    pub const fn tokens(&self) -> usize {
        self.tokens
    }
}
