use runtime::kv::BlockTable;

use super::PagedPrefillBatch;
use crate::{Error, Result};

impl PagedPrefillBatch {
    pub(crate) fn prepare_decode(
        &mut self,
        tables: &[&BlockTable],
        starts: &[usize],
        query_tokens: &[usize],
    ) -> Result<()> {
        if query_tokens.iter().any(|count| *count != 1) {
            return Err(Error::InvalidPagedKv("paged decode batch requires one token per row"));
        }
        self.validate_batch(tables, starts, query_tokens)?;
        self.clear();
        let mut packed_context = 0;
        for (row, ((table, start), count)) in
            tables.iter().zip(starts).zip(query_tokens).enumerate()
        {
            self.rows.push(super::PrefillBatchRow::new((*table).clone(), *start, *count));
            self.prepare_row(row, table, *start, *count, row)?;
            self.query_starts.host[row + 1] = u32::try_from(row + 1)?;
            packed_context += table.token_len();
            self.context_starts.host[row + 1] = u32::try_from(packed_context)?;
            self.max_query_tokens = 1;
            self.max_context_tokens = self.max_context_tokens.max(table.token_len());
        }
        self.upload_decode(tables.len())?;
        self.active = tables.len();
        self.tokens = tables.len();
        Ok(())
    }

    fn upload_decode(&mut self, rows: usize) -> Result<()> {
        self.tables.upload(&self.stream)?;
        self.token_counts.upload(&self.stream)?;
        self.block_counts.upload(&self.stream)?;
        self.context_starts.upload(&self.stream)?;
        self.positions.upload(&self.stream)?;
        self.slot_mapping.upload(&self.stream)?;
        if self.decode_layout_rows != Some(rows) {
            self.query_starts.upload(&self.stream)?;
            self.request_indices.upload(&self.stream)?;
            self.decode_layout_rows = Some(rows);
        }
        Ok(())
    }
}
