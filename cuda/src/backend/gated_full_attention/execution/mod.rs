use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvBackendStorage, KvWritePlan};

use super::{
    AffineGatedFullAttentionConfig, AffineGatedFullAttentionWeights,
    CudaAffineGatedFullAttentionState, batch::GatedFullAttentionBatch, checked,
    prefill::GatedFullAttentionPrefill, scratch::GatedAttentionScratch,
    validation::validate_execution,
};
use crate::{
    CudaBackend, DenseRole, Error, Result,
    backend::linear::{CheckpointProjection, CheckpointProjectionWeight},
    kernels::{
        AttentionTransform, BatchedSplitAttentionWorkspace, MropeSpec, ProjectionPackSplit,
        SigmoidElementwiseBf16,
    },
};

mod packed;
mod projection;

use packed::prepare_packed_projection;

#[derive(Debug)]
pub struct CudaAffineGatedFullAttentionExecution {
    pub(super) backend: CudaBackend,
    pub(super) config: AffineGatedFullAttentionConfig,
    pub(super) tokens: usize,
    query: Option<CheckpointProjection>,
    key: Option<CheckpointProjection>,
    value: Option<CheckpointProjection>,
    packed_qkv: Option<CheckpointProjection>,
    packed_split: Option<ProjectionPackSplit>,
    pub(super) output: CheckpointProjection,
    transform: AttentionTransform,
    pub(super) gate: SigmoidElementwiseBf16,
    weights: AffineGatedFullAttentionWeights,
    pub(super) scratch: GatedAttentionScratch,
    pub(super) batch: Option<GatedFullAttentionBatch>,
    pub(super) batch_workspace: Option<BatchedSplitAttentionWorkspace>,
    pub(super) prefill: Option<GatedFullAttentionPrefill>,
}

impl CudaAffineGatedFullAttentionExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedFullAttentionConfig,
        weights: &AffineGatedFullAttentionWeights,
        packed_qkv: Option<&CheckpointProjectionWeight>,
        tokens: usize,
    ) -> Result<Self> {
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("empty gated attention execution"));
        }
        let projection = |input, output, role, weight| {
            CheckpointProjection::new(backend, tokens, input, output, role, weight)
        };
        let query_width = config.query_width()?;
        let key_value_width = config.key_value_width()?;
        let projected_query = checked(query_width, 2)?;
        let (packed, packed_split) = prepare_packed_projection(
            backend, config, tokens, packed_qkv, projected_query, key_value_width,
        )?;
        Ok(Self {
            backend: backend.clone(),
            config,
            tokens,
            query: if packed.is_none() {
                Some(projection(
                    config.hidden_size,
                    projected_query,
                    DenseRole::AttentionQkv,
                    &weights.query,
                )?)
            } else {
                None
            },
            key: if packed.is_none() {
                Some(projection(
                    config.hidden_size,
                    key_value_width,
                    DenseRole::AttentionQkv,
                    &weights.key,
                )?)
            } else {
                None
            },
            value: if packed.is_none() {
                Some(projection(
                    config.hidden_size,
                    key_value_width,
                    DenseRole::AttentionQkv,
                    &weights.value,
                )?)
            } else {
                None
            },
            packed_qkv: packed,
            packed_split,
            output: projection(
                query_width,
                config.hidden_size,
                DenseRole::AttentionOutput,
                &weights.output,
            )?,
            transform: AttentionTransform::compile(
                &backend.inner.compiler,
                MropeSpec {
                    tokens,
                    heads: config.query_heads,
                    head_dim: config.head_dim,
                    rotary_dim: config.rotary_dim,
                    sections: config.rope_sections,
                    interleaved: config.rope_interleaved,
                    theta: config.rope_theta,
                },
                config.key_value_heads,
                config.rms_norm_epsilon,
                config.norm_weight_shift,
            )?,
            gate: SigmoidElementwiseBf16::compile(
                &backend.inner.compiler,
                checked(tokens, query_width)?,
            )?,
            weights: weights.clone(),
            scratch: GatedAttentionScratch::new(backend, config, tokens, packed_qkv.is_some())?,
            batch: None,
            batch_workspace: None,
            prefill: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        state: &mut CudaAffineGatedFullAttentionState,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        window: Option<usize>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_with_image_span(
            input, positions, state, write_plan, table, start_position, window, None, output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_image_span(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        state: &mut CudaAffineGatedFullAttentionState,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        window: Option<usize>,
        image_span: Option<(usize, usize)>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        validate_execution(
            self.config, self.tokens, input, positions, state, write_plan, table, start_position,
            output,
        )?;
        self.project_and_transform(input, positions)?;
        let written =
            state.cache.store(write_plan, &self.scratch.rotated_key, &self.scratch.value)?;
        if written != self.tokens {
            return Err(Error::InvalidPagedKv("gated attention KV write is incomplete"));
        }
        if self.tokens == 1 {
            state.attention.execute(
                &self.scratch.rotated_query,
                &state.cache,
                table,
                &mut self.scratch.attended,
                window,
                self.config.attention_scale,
            )?;
        } else {
            state.attention.execute_prefill_masked(
                &self.scratch.rotated_query,
                &state.cache,
                table,
                &mut self.scratch.attended,
                self.tokens,
                start_position,
                window,
                self.config.attention_scale,
                image_span,
            )?;
        }
        self.gate.execute(
            &self.backend.inner.stream,
            &self.scratch.attended,
            &self.scratch.gate,
            &mut self.scratch.gated,
        )?;
        self.output.execute(&self.scratch.gated, output)
    }
}
