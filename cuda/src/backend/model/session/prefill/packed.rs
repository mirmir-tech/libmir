use runtime::kv::BlockTable;

use super::CudaMoeModelSession;
use crate::{Error, Result};

const MAX_PACKED_BATCH_SHAPES: usize = 64;

impl CudaMoeModelSession {
    pub(crate) fn prepare_packed_prefill_batch(&mut self, maximum_rows: usize) -> Result<()> {
        for rows in batch_rows(maximum_rows) {
            let tokens = rows
                .checked_mul(self.prefill_tokens.capacity())
                .ok_or(Error::InvalidPagedKv("packed prefill capacity overflow"))?;
            let key = (rows, tokens);
            if !self.packed_batches.contains_key(&key) {
                let batch = self.new_packed_batch(rows, tokens)?;
                self.packed_batches.insert(key, batch);
            }
        }
        Ok(())
    }

    pub(crate) fn prefill_packed_chunk(
        &mut self,
        tokens: &[u32],
        tables: &[&BlockTable],
        starts: &[usize],
        query_tokens: &[usize],
    ) -> Result<()> {
        let total = query_tokens.iter().sum::<usize>();
        if tokens.len() != total
            || tables.len() != starts.len()
            || tables.len() != query_tokens.len()
        {
            return Err(Error::InvalidPagedKv("invalid packed model prefill geometry"));
        }
        self.ensure_prefill_capacity(total)?;
        let batch_key = self
            .packed_batches
            .iter()
            .filter(|(_, batch)| {
                batch.row_capacity() >= tables.len() && batch.token_capacity() >= total
            })
            .min_by_key(|(_, batch)| (batch.row_capacity(), batch.token_capacity()))
            .map_or((tables.len(), total), |(key, _)| *key);
        let mut batch = if let Some(batch) = self.packed_batches.remove(&batch_key) {
            batch
        } else {
            self.new_packed_batch(tables.len(), total)?
        };
        batch.prepare(tables, starts, query_tokens)?;
        self.prefill_tokens.upload(&self.stream, tokens)?;
        self.embedding.execute_batch(
            self.prefill_tokens.device(),
            0,
            total,
            &mut self.prefill_first,
        )?;
        let mut plans = self.packed_prefill.take(total).unwrap_or_default();
        let result = (|| {
            for (index, layer) in self.layers.iter_mut().enumerate() {
                let (input, output) = if index.is_multiple_of(2) {
                    (&self.prefill_first, &mut self.prefill_second)
                } else {
                    (&self.prefill_second, &mut self.prefill_first)
                };
                let signature = layer.prefill_signature();
                let plan_index = plans.iter().position(|plan| plan.supports(signature));
                let plan_index = if let Some(index) = plan_index {
                    index
                } else {
                    plans.push(layer.instantiate_shared_prefill(total)?);
                    plans.len() - 1
                };
                let prefill = plans[plan_index].borrow();
                if let Some(graph) = self.decode_graph.as_mut() {
                    graph.execute_prefill_batch(index, prefill, input, &batch, output)?;
                } else {
                    layer.execute_shared_prefill_batch(prefill, input, output, &batch)?;
                }
            }
            Ok(())
        })();
        if self.packed_batches.len() < MAX_PACKED_BATCH_SHAPES {
            self.packed_batches.insert(batch_key, batch);
        }
        self.packed_prefill.insert(total, plans);
        result
    }

    fn new_packed_batch(&self, rows: usize, tokens: usize) -> Result<crate::PagedPrefillBatch> {
        let attention = self
            .layers
            .first()
            .ok_or(Error::InvalidDecoderKernel("CUDA model session requires layers"))?
            .attention_config();
        self.backend.prepare_paged_prefill_batch(
            attention.cache,
            attention.max_sequence_blocks,
            rows,
            tokens,
        )
    }
}

fn batch_rows(maximum: usize) -> Vec<usize> {
    let mut sizes = std::iter::successors(Some(2_usize), |size| size.checked_mul(2))
        .take_while(|size| *size <= maximum)
        .collect::<Vec<_>>();
    sizes.extend([5, 10, maximum].into_iter().filter(|size| (2..=maximum).contains(size)));
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

#[cfg(test)]
mod tests {
    use super::batch_rows;

    #[test]
    fn prepares_canonical_metadata_buckets() {
        assert_eq!(batch_rows(10), [2, 4, 5, 8, 10]);
        assert_eq!(batch_rows(3), [2, 3]);
    }
}
