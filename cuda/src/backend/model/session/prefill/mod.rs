use mircuda::{DeviceBuffer, bf16};
use runtime::{backend::SamplingLogits, kv::BlockTable};
use uuid::Uuid;

use super::CudaMoeModelSession;
use crate::{Error, Result};

pub(super) mod output;
mod packed;
pub(super) mod shapes;
mod shared;

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
            &SamplingLogits::Full,
        )
    }

    pub(crate) fn prefill_from_for_sampling(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        write_offset: usize,
        table: &BlockTable,
        sampling: &SamplingLogits,
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
            &mut self.prefill_first,
        )?;
        self.execute_shared_prefill(session_id, table, write_offset, count)
    }

    pub(crate) fn finish_prefill(
        &mut self,
        tokens: usize,
        sampling: &SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.select_prefill_output(tokens)?;
        self.project_logits(sampling)?;
        Ok(&self.logits)
    }

    pub(crate) fn finish_packed_prefill_row(
        &mut self,
        row: usize,
        tokens: usize,
        sampling: &SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.select_prefill_row(row, tokens)?;
        self.project_logits(sampling)?;
        Ok(&self.logits)
    }

    pub(crate) fn prefill_chunk_len(&self, remaining: usize) -> usize {
        chunk_len(remaining, self.prefill_tokens.capacity())
    }

    pub(super) fn ensure_prefill_capacity(&mut self, tokens: usize) -> Result<()> {
        if tokens <= self.prefill_tokens.capacity() {
            return Ok(());
        }
        let elements = tokens
            .checked_mul(self.hidden_size)
            .ok_or(Error::InvalidDecoderKernel("CUDA prefill size overflow"))?;
        self.prefill_tokens.ensure_capacity(&self.backend, tokens)?;
        self.prefill_first = self.backend.inner.pool.allocate(&self.stream, elements)?;
        self.prefill_second = self.backend.inner.pool.allocate(&self.stream, elements)?;
        Ok(())
    }

    pub(in crate::backend::model::session) fn validate_prefill(
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
        self.select_prefill_row(tokens - 1, tokens)
    }

    fn select_prefill_row(&mut self, row: usize, tokens: usize) -> Result<()> {
        let (source, output) = if self.layers.len().is_multiple_of(2) {
            (&self.prefill_first, &mut self.first)
        } else {
            (&self.prefill_second, &mut self.second)
        };
        self.select_row.execute(&self.stream, source, output, row, tokens)
    }
}

/// A chunk takes what the scheduler offers up to the token capacity. Packed
/// plans are keyed by the round's total, which the step budget fixes, so
/// rounding rows to powers of two only left budget unused in fair rounds
/// (ten rows of 103 became 640 of 1,024 tokens) and split a prompt's tail
/// into one chunk per generation step (15 tokens as 8, 4, 2, 1).
pub(super) const fn chunk_len(remaining: usize, capacity: usize) -> usize {
    if remaining >= capacity {
        capacity
    } else {
        remaining
    }
}

#[cfg(test)]
mod tests {
    use super::chunk_len;

    #[test]
    fn takes_the_offered_chunk_up_to_capacity() {
        assert_eq!(chunk_len(257, 256), 256);
        assert_eq!(chunk_len(255, 256), 255);
        assert_eq!(chunk_len(103, 256), 103);
        assert_eq!(chunk_len(1, 256), 1);
    }
}
