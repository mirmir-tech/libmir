//! One shape of the prediction module: `tokens` rows of decoder output and
//! their following tokens in, the module's normalized output rows out.

use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};
use uuid::Uuid;

use super::PredictionTemplate;
use crate::{
    Bf16Linear, CudaAttentionHybridExecution, CudaBackend, CudaTensor, DecodeAttentionBf16, Error,
    PagedPrefillBatch, Result,
    backend::shared::routed::{CudaSharedRoutedModelTemplate, boundary::SharedRoutedEmbedding},
    kernels::{ConcatColumnsBf16, ShiftedRmsNorm},
};

#[derive(Debug)]
pub(in crate::backend::shared::routed) struct PredictionExecution {
    pub(in crate::backend::shared::routed) backend: CudaBackend,
    pub(in crate::backend::shared::routed) tokens: usize,
    weights: PredictionTemplate,
    /// The decoder's final norm over every row, not only the last.
    decoder_norm: ShiftedRmsNorm,
    norm: ShiftedRmsNorm,
    concat: ConcatColumnsBf16,
    fusion: Bf16Linear,
    layer: CudaAttentionHybridExecution,
    token_staging: PinnedBuffer<u32>,
    /// `t_{i+1}` for every row `i`.
    pub(in crate::backend::shared::routed) token_ids: DeviceBuffer<u32>,
    /// `h_i`: the decoder's normalized output rows.
    pub(in crate::backend::shared::routed) hidden: DeviceBuffer<bf16>,
    embedded: DeviceBuffer<bf16>,
    normalized_embedding: DeviceBuffer<bf16>,
    normalized_hidden: DeviceBuffer<bf16>,
    fused: DeviceBuffer<bf16>,
    projected: DeviceBuffer<bf16>,
    mixed: DeviceBuffer<bf16>,
    /// The module's normalized output: the shared head's input for draft
    /// logits and the hidden input of a further draft.
    pub(in crate::backend::shared::routed) drafted: DeviceBuffer<bf16>,
    pub(super) verification: Option<super::verify::Verification>,
}

impl PredictionExecution {
    pub(in crate::backend::shared::routed) fn new(
        template: &CudaSharedRoutedModelTemplate,
        weights: &PredictionTemplate,
        tokens: usize,
    ) -> Result<Self> {
        let backend = &template.backend;
        let hidden = template.decoder.hidden_size;
        let decoder_epsilon = template.decoder.rms_norm_eps.to_string().parse()?;
        let elements = tokens
            .checked_mul(hidden)
            .ok_or(Error::InvalidDecoderKernel("prediction activation size overflow"))?;
        let compiler = &backend.inner.compiler;
        let allocate = |elements| backend.inner.pool.allocate(&backend.inner.stream, elements);
        Ok(Self {
            backend: backend.clone(),
            tokens,
            weights: weights.clone(),
            decoder_norm: ShiftedRmsNorm::compile(
                compiler,
                tokens,
                hidden,
                decoder_epsilon,
                template.norm_shift,
            )?,
            norm: ShiftedRmsNorm::compile(compiler, tokens, hidden, weights.epsilon, 0.0)?,
            concat: ConcatColumnsBf16::compile(compiler, hidden)?,
            fusion: backend.prepare_bf16_linear(tokens, 2 * hidden, hidden)?,
            layer: weights.layer.prepare(tokens)?,
            token_staging: backend.inner.context.allocate_pinned(tokens)?,
            token_ids: backend.inner.pool.allocate(&backend.inner.stream, tokens)?,
            hidden: allocate(elements)?,
            embedded: allocate(elements)?,
            normalized_embedding: allocate(elements)?,
            normalized_hidden: allocate(elements)?,
            fused: allocate(2 * elements)?,
            projected: allocate(elements)?,
            mixed: allocate(elements)?,
            drafted: allocate(elements)?,
            verification: None,
        })
    }

    /// Normalizes every decoder output row into `hidden`.
    pub(in crate::backend::shared::routed) fn normalize_decoder(
        &mut self,
        decoder: &DeviceBuffer<bf16>,
        weight: &CudaTensor,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.decoder_norm
            .execute(stream, decoder, weight_bf16(weight)?, &mut self.hidden)
    }

    /// Stages the following token of every row from the host.
    pub(in crate::backend::shared::routed) fn upload_tokens(
        &mut self,
        tokens: &[u32],
    ) -> Result<()> {
        if tokens.len() != self.tokens {
            return Err(Error::InvalidDecoderKernel("prediction token count differs from plan"));
        }
        self.token_staging.copy_from_slice(tokens)?;
        self.backend
            .inner
            .stream
            .copy_to_device(&mut self.token_staging, &mut self.token_ids)?;
        Ok(())
    }

    /// Runs the module over `hidden` and `token_ids`, writing the module's K/V
    /// for positions `start_position..start_position + tokens`.
    pub(in crate::backend::shared::routed) fn execute(
        &mut self,
        embedding: &SharedRoutedEmbedding,
        state: &mut DecodeAttentionBf16,
        session_id: Uuid,
        table: &BlockTable,
        start_position: usize,
    ) -> Result<()> {
        self.fuse(embedding)?;
        let write = KvWritePlan::prefill(
            session_id,
            self.weights.index,
            table,
            start_position,
            self.tokens,
        )?;
        self.layer
            .execute(&self.projected, state, &write, table, start_position, &mut self.mixed)?;
        self.finish()
    }

    /// Runs the module over the logical rows of several sequences packed
    /// as `paging` lays them out; `state` provides the module's pages.
    pub(in crate::backend::shared::routed) fn execute_packed(
        &mut self,
        embedding: &SharedRoutedEmbedding,
        state: &mut DecodeAttentionBf16,
        paging: &PagedPrefillBatch,
    ) -> Result<()> {
        self.fuse(embedding)?;
        self.layer
            .execute_packed_prefill(&self.projected, state, paging, &mut self.mixed)?;
        self.finish()
    }

    /// `fusion([enorm(embed(t)) | hnorm(h)])` into `projected`.
    fn fuse(&mut self, embedding: &SharedRoutedEmbedding) -> Result<()> {
        let stream = &self.backend.inner.stream;
        embedding.execute_batch(&self.token_ids, self.tokens, &mut self.embedded)?;
        let weights = &self.weights;
        self.norm.execute(
            stream,
            &self.embedded,
            weight_bf16(&weights.embedding_norm)?,
            &mut self.normalized_embedding,
        )?;
        self.norm.execute(
            stream,
            &self.hidden,
            weight_bf16(&weights.hidden_norm)?,
            &mut self.normalized_hidden,
        )?;
        self.concat.execute(
            stream,
            &self.normalized_embedding,
            &self.normalized_hidden,
            &mut self.fused,
            self.tokens,
        )?;
        self.fusion.execute(&self.fused, &weights.fusion, &mut self.projected)
    }

    fn finish(&mut self) -> Result<()> {
        let stream = &self.backend.inner.stream;
        let weight = weight_bf16(&self.weights.final_norm)?;
        self.norm.execute(stream, &self.mixed, weight, &mut self.drafted)
    }

    pub(in crate::backend::shared::routed) const fn tokens(&self) -> usize {
        self.tokens
    }
}

fn weight_bf16(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
