use mircuda::{DeviceBuffer, bf16};
use runtime::{
    backend::SamplingLogits,
    kv::{BlockTable, KvWritePlan},
};
use uuid::Uuid;

use super::CudaMoeModelSession;
use crate::{Error, Result};

impl CudaMoeModelSession {
    /// Prefills uncached prompt tokens from an explicit sequence offset.
    pub fn prefill_from(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        write_offset: usize,
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.prefill_from_for_sampling(
            session_id,
            tokens,
            write_offset,
            table,
            SamplingLogits::Full,
        )
    }

    pub(crate) fn prefill_from_for_sampling(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        write_offset: usize,
        table: &BlockTable,
        sampling: SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.validate_prefill(tokens, write_offset, table)?;
        let mut step_table = table.clone();
        let mut consumed = 0;
        let mut last_count = 0;
        while consumed < tokens.len() {
            let remaining = &tokens[consumed..];
            let count = self.prefill_chunk_len(remaining.len());
            let start = write_offset + consumed;
            step_table.set_token_len(start + count);
            self.prefill_chunk(session_id, &remaining[..count], start, &step_table)?;
            consumed += count;
            last_count = count;
        }
        self.finish_prefill(last_count, sampling)?;
        tracing::debug!(
            tokens = tokens.len(),
            write_offset,
            chunk_tokens = self.prefill_tokens.capacity(),
            "enqueued CUDA model prefill"
        );
        Ok(&self.logits)
    }

    pub(crate) fn prefill_chunk(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        write_offset: usize,
        table: &BlockTable,
    ) -> Result<()> {
        self.validate_prefill_chunk(tokens, write_offset, table)?;
        let count = tokens.len();
        self.prefill_tokens.upload(&self.stream, tokens)?;
        self.embedding.execute_batch(
            self.prefill_tokens.device(),
            0,
            count,
            &self.embedding_weight,
            &mut self.prefill_first,
        )?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            let plan = KvWritePlan::prefill(session_id, layer.layer(), table, write_offset, count)?;
            let (input, output) = if index.is_multiple_of(2) {
                (&self.prefill_first, &mut self.prefill_second)
            } else {
                (&self.prefill_second, &mut self.prefill_first)
            };
            layer.prepare_prefill(count)?;
            if let Some(graph) = self.decode_graph.as_mut() {
                graph.execute_prefill(
                    index,
                    layer.prefill_plan(count)?,
                    input,
                    &plan,
                    table,
                    write_offset,
                    output,
                )?;
            } else {
                layer.execute_prefill(input, output, &plan, table, write_offset, count)?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish_prefill(
        &mut self,
        tokens: usize,
        sampling: SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.select_prefill_output(tokens)?;
        self.project_logits(sampling)?;
        Ok(&self.logits)
    }

    pub(crate) fn prefill_chunk_len(&self, remaining: usize) -> usize {
        chunk_len(remaining, self.prefill_tokens.capacity())
    }

    fn validate_prefill(
        &self,
        tokens: &[u32],
        write_offset: usize,
        table: &BlockTable,
    ) -> Result<()> {
        let token_end = write_offset
            .checked_add(tokens.len())
            .ok_or(Error::InvalidPagedKv("CUDA prefill token range overflow"))?;
        let block_size = table
            .block_size()
            .ok_or(Error::InvalidPagedKv("CUDA prefill table has no block size"))?;
        if tokens.is_empty()
            || token_end != table.token_len()
            || token_end > table.capacity(block_size)
        {
            return Err(Error::InvalidPagedKv("CUDA prefill range differs from block table"));
        }
        for token in tokens {
            self.embedding.validate_token(*token)?;
        }
        Ok(())
    }

    fn validate_prefill_chunk(
        &self,
        tokens: &[u32],
        write_offset: usize,
        table: &BlockTable,
    ) -> Result<()> {
        let token_end = write_offset
            .checked_add(tokens.len())
            .ok_or(Error::InvalidPagedKv("CUDA prefill chunk range overflow"))?;
        if tokens.is_empty()
            || tokens.len() > self.prefill_tokens.capacity()
            || token_end != table.token_len()
        {
            return Err(Error::InvalidPagedKv("CUDA prefill chunk differs from block table"));
        }
        for token in tokens {
            self.embedding.validate_token(*token)?;
        }
        Ok(())
    }

    fn select_prefill_output(&mut self, tokens: usize) -> Result<()> {
        let (source, output) = if self.layers.len().is_multiple_of(2) {
            (&self.prefill_first, &mut self.first)
        } else {
            (&self.prefill_second, &mut self.second)
        };
        self.select_row.execute(&self.stream, source, output, tokens - 1, tokens)
    }
}

pub(super) fn chunk_len(remaining: usize, capacity: usize) -> usize {
    if remaining >= capacity {
        capacity
    } else {
        1 << (usize::BITS - 1 - remaining.leading_zeros())
    }
}

#[cfg(test)]
mod tests {
    use super::chunk_len;

    #[test]
    fn decomposes_tail_into_canonical_powers_of_two() {
        assert_eq!(chunk_len(257, 256), 256);
        assert_eq!(chunk_len(255, 256), 128);
        assert_eq!(chunk_len(127, 256), 64);
        assert_eq!(chunk_len(1, 256), 1);
    }
}
