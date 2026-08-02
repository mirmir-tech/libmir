use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvBackendStorage, KvWritePlan};

use super::{
    AffineGatedFullAttentionConfig, AffineGatedFullAttentionWeights,
    CudaAffineGatedFullAttentionState, checked, scratch::GatedAttentionScratch,
    validation::validate_execution,
};
use crate::{
    CudaBackend, CudaTensor, DenseRole, Error, Result,
    backend::linear::CheckpointProjection,
    kernels::{GatedAttentionSplit, Mrope, MropeSpec, ShiftedRmsNorm, SigmoidElementwiseBf16},
};

#[derive(Debug)]
pub struct CudaAffineGatedFullAttentionExecution {
    backend: CudaBackend,
    config: AffineGatedFullAttentionConfig,
    tokens: usize,
    query: CheckpointProjection,
    key: CheckpointProjection,
    value: CheckpointProjection,
    output: CheckpointProjection,
    split: GatedAttentionSplit,
    query_norm: ShiftedRmsNorm,
    key_norm: ShiftedRmsNorm,
    query_rope: Mrope,
    key_rope: Mrope,
    gate: SigmoidElementwiseBf16,
    weights: AffineGatedFullAttentionWeights,
    scratch: GatedAttentionScratch,
}

impl CudaAffineGatedFullAttentionExecution {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedFullAttentionConfig,
        weights: &AffineGatedFullAttentionWeights,
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
        let norm = |heads| {
            ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                checked(tokens, heads)?,
                config.head_dim,
                config.rms_norm_epsilon,
                config.norm_weight_shift,
            )
        };
        let rope = |heads| {
            Mrope::compile(
                &backend.inner.compiler,
                MropeSpec {
                    tokens,
                    heads,
                    head_dim: config.head_dim,
                    rotary_dim: config.rotary_dim,
                    sections: config.rope_sections,
                    interleaved: config.rope_interleaved,
                    theta: config.rope_theta,
                },
            )
        };
        Ok(Self {
            backend: backend.clone(),
            config,
            tokens,
            query: projection(
                config.hidden_size,
                checked(query_width, 2)?,
                DenseRole::AttentionQkv,
                &weights.query,
            )?,
            key: projection(
                config.hidden_size,
                key_value_width,
                DenseRole::AttentionQkv,
                &weights.key,
            )?,
            value: projection(
                config.hidden_size,
                key_value_width,
                DenseRole::AttentionQkv,
                &weights.value,
            )?,
            output: projection(
                query_width,
                config.hidden_size,
                DenseRole::AttentionOutput,
                &weights.output,
            )?,
            split: GatedAttentionSplit::compile(
                &backend.inner.compiler,
                tokens,
                config.query_heads,
                config.head_dim,
            )?,
            query_norm: norm(config.query_heads)?,
            key_norm: norm(config.key_value_heads)?,
            query_rope: rope(config.query_heads)?,
            key_rope: rope(config.key_value_heads)?,
            gate: SigmoidElementwiseBf16::compile(
                &backend.inner.compiler,
                checked(tokens, query_width)?,
            )?,
            weights: weights.clone(),
            scratch: GatedAttentionScratch::new(backend, config, tokens)?,
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

    fn project_and_transform(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
    ) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.query.execute(input, &mut self.scratch.query_projected)?;
        self.split.execute(
            stream,
            &self.scratch.query_projected,
            &mut self.scratch.query,
            &mut self.scratch.gate,
        )?;
        self.key.execute(input, &mut self.scratch.key)?;
        self.value.execute(input, &mut self.scratch.value)?;
        self.query_norm.execute(
            stream,
            &self.scratch.query,
            bf16_tensor(&self.weights.query_norm)?,
            &mut self.scratch.normalized_query,
        )?;
        self.key_norm.execute(
            stream,
            &self.scratch.key,
            bf16_tensor(&self.weights.key_norm)?,
            &mut self.scratch.normalized_key,
        )?;
        self.query_rope.execute(
            stream,
            &self.scratch.normalized_query,
            positions,
            &mut self.scratch.rotated_query,
        )?;
        self.key_rope.execute(
            stream,
            &self.scratch.normalized_key,
            positions,
            &mut self.scratch.rotated_key,
        )
    }
}

fn bf16_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
