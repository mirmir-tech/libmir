use mircuda::{DeviceBuffer, bf16};

use super::{
    DecodeAttentionBf16, DecodeAttentionWeights, PrefillAttentionBf16, QkvProjectionBuffers,
};
use crate::{Error, PagedPrefillBatch, Result};

impl PrefillAttentionBf16 {
    pub fn execute_batch(
        &mut self,
        state: &mut DecodeAttentionBf16,
        input: &DeviceBuffer<bf16>,
        weights: DecodeAttentionWeights<'_>,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let mut state_config = state.config;
        let mut prefill_config = self.config;
        state_config.layer = 0;
        prefill_config.layer = 0;
        if state_config != prefill_config || batch.tokens() != self.tokens {
            return Err(Error::InvalidPagedKv(
                "prefill attention batch state or token count differs",
            ));
        }
        let separate = self.qkv.execute(
            input,
            &self.input_norm,
            weights.input_norm,
            weights.qkv,
            &mut QkvProjectionBuffers {
                normalized: &mut self.scratch.normalized,
                packed: &mut self.scratch.qkv,
                separate: &mut self.scratch.qkv_separate,
            },
        )?;
        let inputs = if separate {
            [
                &self.scratch.qkv_separate[0],
                &self.scratch.qkv_separate[1],
                &self.scratch.qkv_separate[2],
            ]
        } else {
            [&self.scratch.qkv, &self.scratch.qkv, &self.scratch.qkv]
        };
        self.qkv_postprocess_batch.execute(
            &self.stream,
            inputs,
            separate,
            weights.query_norm,
            weights.key_norm,
            batch.positions(),
            &mut self.scratch.query_rope,
            &mut self.scratch.key_rope,
            &mut self.scratch.value_norm,
        )?;
        state
            .cache
            .store_prefill_batch(batch, &self.scratch.key_rope, &self.scratch.value_norm)?;
        if self.config.sliding_window.is_none() && self.fmha.is_some() {
            return self.execute_paged_varlen_attention(state, weights, batch, output);
        }
        let mut packed = 0;
        for row in batch.rows() {
            self.scratch.ensure_row(
                &self.pool,
                &self.stream,
                row.tokens(),
                self.query_width,
                self.attention_width,
            )?;
            let row_scratch = self
                .scratch
                .rows
                .get_mut(&row.tokens())
                .ok_or(Error::InvalidPagedKv("missing packed prefill row scratch"))?;
            self.query_rows.execute(
                &self.stream,
                &self.scratch.query_rope,
                &mut row_scratch.query,
                packed,
                0,
                row.tokens(),
            )?;
            state.attention.execute_prefill(
                &row_scratch.query,
                &state.cache,
                row.table(),
                &mut row_scratch.attention,
                row.tokens(),
                row.start(),
                self.config.sliding_window,
                self.config.attention_scale,
            )?;
            self.output_rows.execute(
                &self.stream,
                &row_scratch.attention,
                &mut self.scratch.attention,
                0,
                packed,
                row.tokens(),
            )?;
            packed += row.tokens();
        }
        self.output_projection.execute(
            &self.stream,
            &self.scratch.attention,
            weights.output,
            output,
        )
    }
}
