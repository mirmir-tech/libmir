use std::collections::HashMap;

use mircuda::{DeviceBuffer, PinnedBuffer, Stream, bf16};
use prefill::output::PackedPrefillOutput;
use runtime::{backend::SamplingLogits, kv::BlockTable};
use uuid::Uuid;

use self::prefill::shapes::PrefillShapeCache;
use super::{
    graph::{CapturedModelDecode, execute_layers},
    layer::{SessionLayer, SharedLayerPrefill},
    prefill::PrefillTokenBuffer,
};
use crate::{
    CudaBackend, CudaTensor, DeviceSamplerBf16, Error, PagedPrefillBatch, Result, RmsNormBf16,
    backend::model::boundary::{ModelEmbedding, ModelOutputHead, ModelOutputHeadTemplate},
    kernels::SelectRowBf16,
};

mod prefill;
mod vision;
mod warmup;

/// Mutable device-resident decode state for one model session.
pub struct CudaMoeModelSession {
    backend: CudaBackend,
    hidden_size: usize,
    embedding: ModelEmbedding,
    final_norm: RmsNormBf16,
    output_projection: ModelOutputHead,
    output_template: ModelOutputHeadTemplate,
    final_norm_weight: CudaTensor,
    layers: Vec<SessionLayer>,
    packed_prefill: PrefillShapeCache<Vec<SharedLayerPrefill>>,
    packed_batches: HashMap<(usize, usize), PagedPrefillBatch>,
    packed_outputs: HashMap<usize, PackedPrefillOutput>,
    decode_graph: Option<CapturedModelDecode>,
    sampler: DeviceSamplerBf16,
    stream: Stream,
    input_token: DeviceBuffer<u32>,
    token_staging: PinnedBuffer<u32>,
    prefill_tokens: PrefillTokenBuffer,
    select_row: SelectRowBf16,
    first: DeviceBuffer<bf16>,
    second: DeviceBuffer<bf16>,
    prefill_first: DeviceBuffer<bf16>,
    prefill_second: DeviceBuffer<bf16>,
    logits: DeviceBuffer<bf16>,
    logit_softcap: Option<crate::kernels::LogitSoftcap>,
    logit_softcap_cap: Option<f32>,
}

impl CudaMoeModelSession {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        backend: CudaBackend,
        hidden_size: usize,
        embedding: ModelEmbedding,
        final_norm: RmsNormBf16,
        output_projection: ModelOutputHead,
        output_template: ModelOutputHeadTemplate,
        final_norm_weight: CudaTensor,
        layers: Vec<SessionLayer>,
        sampler: DeviceSamplerBf16,
        stream: Stream,
        input_token: DeviceBuffer<u32>,
        token_staging: PinnedBuffer<u32>,
        prefill_tokens: PrefillTokenBuffer,
        select_row: SelectRowBf16,
        first: DeviceBuffer<bf16>,
        second: DeviceBuffer<bf16>,
        prefill_first: DeviceBuffer<bf16>,
        prefill_second: DeviceBuffer<bf16>,
        logits: DeviceBuffer<bf16>,
        logit_softcap_cap: Option<f32>,
    ) -> Result<Self> {
        if layers.is_empty() {
            return Err(Error::InvalidDecoderKernel("CUDA model session requires layers"));
        }
        let logit_softcap = logit_softcap_cap
            .map(|cap| {
                crate::kernels::LogitSoftcap::compile(&backend.inner.compiler, logits.len(), cap)
            })
            .transpose()?;
        Ok(Self {
            backend,
            hidden_size,
            embedding,
            final_norm,
            output_projection,
            output_template,
            final_norm_weight,
            layers,
            packed_prefill: PrefillShapeCache::new(),
            packed_batches: HashMap::new(),
            packed_outputs: HashMap::new(),
            decode_graph: None,
            sampler,
            stream,
            input_token,
            token_staging,
            prefill_tokens,
            select_row,
            first,
            second,
            prefill_first,
            prefill_second,
            logits,
            logit_softcap,
            logit_softcap_cap,
        })
    }

    /// Enqueues one complete model token and returns device-resident logits.
    pub fn decode(
        &mut self,
        session_id: Uuid,
        token: u32,
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.decode_for_sampling(session_id, token, table, SamplingLogits::Full)
    }

    pub(crate) fn decode_for_sampling(
        &mut self,
        session_id: Uuid,
        token: u32,
        table: &BlockTable,
        sampling: SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.embedding.validate_token(token)?;
        self.token_staging.copy_from_slice(&[token])?;
        self.stream.copy_to_device(&mut self.token_staging, &mut self.input_token)?;
        self.embedding.execute(&self.input_token, 0, &mut self.first)?;
        self.forward(session_id, table, sampling)
    }

    /// Selects the next token while retaining it in device memory.
    pub fn sample(&mut self, policy: SamplingLogits) -> Result<&DeviceBuffer<u32>> {
        self.sampler.sample(&self.logits, policy)
    }

    /// Decodes the last device-sampled token without host synchronization.
    pub fn decode_sampled(
        &mut self,
        session_id: Uuid,
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.decode_sampled_for_sampling(session_id, table, SamplingLogits::Full)
    }

    pub(crate) fn decode_sampled_for_sampling(
        &mut self,
        session_id: Uuid,
        table: &BlockTable,
        sampling: SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        self.embedding.execute(self.sampler.selected(), 0, &mut self.first)?;
        self.forward(session_id, table, sampling)
    }

    fn forward(
        &mut self,
        session_id: Uuid,
        table: &BlockTable,
        sampling: SamplingLogits,
    ) -> Result<&DeviceBuffer<bf16>> {
        execute_layers(&mut self.decode_graph, &mut self.layers, &self.stream, session_id, table)?;
        self.project_logits(sampling)
    }

    fn project_logits(&mut self, sampling: SamplingLogits) -> Result<&DeviceBuffer<bf16>> {
        if self.layers.len().is_multiple_of(2) {
            self.final_norm
                .execute(&self.first, &self.final_norm_weight, &mut self.second)?;
            self.output_projection.execute(&self.second, &mut self.logits, sampling)?;
        } else {
            self.final_norm
                .execute(&self.second, &self.final_norm_weight, &mut self.first)?;
            self.output_projection.execute(&self.first, &mut self.logits, sampling)?;
        }
        if !matches!(sampling, SamplingLogits::None)
            && let Some(softcap) = &self.logit_softcap
        {
            softcap.execute(&self.stream, &mut self.logits)?;
        }
        Ok(&self.logits)
    }

    #[must_use]
    pub const fn logits(&self) -> &DeviceBuffer<bf16> {
        &self.logits
    }
}
