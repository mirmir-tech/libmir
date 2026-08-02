use mircuda::{DeviceBuffer, FmhaBf16Plan, MemoryPool, Stream, bf16};
use runtime::kv::{BlockTable, KvBackendStorage, KvWritePlan};

use super::{
    AttentionOutputProjection, AttentionQkvProjection, DecodeAttentionBf16, DecodeAttentionConfig,
    DecodeAttentionOutputWeight, DecodeAttentionWeights, ProjectionFormat, QkvProjectionBuffers,
    validate,
};
use crate::{
    CudaBackend, DensePlanRequest, DenseRole, Error, ExecutionPhase, NvFp4Bf16Linear, Result,
    RmsNormBf16,
    kernels::{BatchedQkvPostprocess, CopyRowsBf16, QkvPostprocess, QkvPostprocessSpec},
};
mod batch;
mod image;
mod paged;
mod scratch;
mod varlen;
pub(in crate::backend) use image::ImageAttentionSpan;
use scratch::PrefillAttentionScratch;
use varlen::prepare_varlen_fmha;
#[derive(Debug)]
pub struct PrefillAttentionBf16 {
    input_norm: RmsNormBf16,
    qkv: AttentionQkvProjection,
    qkv_postprocess: QkvPostprocess,
    qkv_postprocess_batch: BatchedQkvPostprocess,
    fmha: Option<FmhaBf16Plan>,
    query_rows: CopyRowsBf16,
    output_rows: CopyRowsBf16,
    output_projection: AttentionOutputProjection,
    scratch: PrefillAttentionScratch,
    stream: Stream,
    pool: MemoryPool,
    config: DecodeAttentionConfig,
    query_width: usize,
    attention_width: usize,
    tokens: usize,
}

impl CudaBackend {
    pub fn prepare_prefill_attention_bf16(
        &self,
        config: DecodeAttentionConfig,
        tokens: usize,
    ) -> Result<PrefillAttentionBf16> {
        PrefillAttentionBf16::new(self, config, tokens, None)
    }
}
impl PrefillAttentionBf16 {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        config: DecodeAttentionConfig,
        tokens: usize,
        weights: Option<DecodeAttentionWeights<'_>>,
    ) -> Result<Self> {
        validate(config)?;
        if tokens == 0 {
            return Err(Error::InvalidPagedKv("prefill attention chunk is empty"));
        }
        let hidden = config.hidden_size;
        let query_width = config.query_heads * config.cache.key_head_dim;
        let attention_width = config.query_heads * config.cache.value_head_dim;
        let epsilon = config.rms_norm_epsilon;
        let qkv_spec = QkvPostprocessSpec {
            tokens,
            query_heads: config.query_heads,
            kv_heads: config.cache.kv_heads,
            head_dim: config.cache.key_head_dim,
            value_head_dim: config.cache.value_head_dim,
            rotary_dim: config.rotary_dim,
            pairing_dim: config.rope_pairing_dim,
            theta: config.rope_theta,
            epsilon,
            normalization: config.qkv_normalization,
        };
        let fmha = prepare_varlen_fmha(backend, config)?;
        let output_request = DensePlanRequest {
            phase: ExecutionPhase::Prefill,
            role: DenseRole::AttentionOutput,
            tokens,
            input_features: attention_width,
            output_features: hidden,
        };
        let output_projection = match config.projection_format {
            ProjectionFormat::Affine
            | ProjectionFormat::DirectFp8
            | ProjectionFormat::MxFp4
            | ProjectionFormat::MxFp8 => AttentionOutputProjection::new(
                backend,
                config,
                tokens,
                weights.map(|weights| weights.output),
            )?,
            ProjectionFormat::Bf16 => {
                AttentionOutputProjection::Bf16(backend.prepare_bf16_projection(output_request)?)
            },
            ProjectionFormat::PackedInteger => {
                let Some(DecodeAttentionOutputWeight::PackedInteger(weight)) =
                    weights.map(|weights| weights.output)
                else {
                    return Err(Error::InvalidExecutionPlan(
                        "packed integer prefill requires a packed output",
                    ));
                };
                AttentionOutputProjection::PackedInteger(crate::PackedIntegerBf16Linear::new(
                    backend, tokens, attention_width, hidden, weight,
                )?)
            },
            ProjectionFormat::NvFp4 => {
                let Some(DecodeAttentionOutputWeight::NvFp4(weight)) =
                    weights.map(|weights| weights.output)
                else {
                    return Err(Error::InvalidExecutionPlan("NVFP4 prefill requires NVFP4 output"));
                };
                AttentionOutputProjection::NvFp4(NvFp4Bf16Linear::from_weight(
                    backend,
                    tokens,
                    weight.clone(),
                )?)
            },
        };
        Ok(Self {
            input_norm: RmsNormBf16::new(backend, tokens, hidden, epsilon)?,
            qkv: AttentionQkvProjection::new(
                backend,
                config,
                tokens,
                weights.map(|value| value.qkv),
            )?,
            qkv_postprocess: QkvPostprocess::compile(&backend.inner.compiler, qkv_spec)?,
            qkv_postprocess_batch: BatchedQkvPostprocess::compile(
                &backend.inner.compiler,
                qkv_spec,
            )?,
            fmha,
            query_rows: CopyRowsBf16::compile(&backend.inner.compiler, query_width)?,
            output_rows: CopyRowsBf16::compile(&backend.inner.compiler, attention_width)?,
            output_projection,
            scratch: PrefillAttentionScratch::new(backend, config, tokens)?,
            stream: backend.inner.stream.clone(),
            pool: backend.inner.pool.clone(),
            config,
            query_width,
            attention_width,
            tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        state: &mut DecodeAttentionBf16,
        input: &DeviceBuffer<bf16>,
        weights: DecodeAttentionWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_masked(state, input, weights, write_plan, table, start_position, output, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn execute_masked(
        &mut self,
        state: &mut DecodeAttentionBf16,
        input: &DeviceBuffer<bf16>,
        weights: DecodeAttentionWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
        image: Option<ImageAttentionSpan>,
    ) -> Result<()> {
        let end = start_position
            .checked_add(self.tokens)
            .ok_or(Error::InvalidPagedKv("prefill attention range overflow"))?;
        let mut state_config = state.config;
        let mut prefill_config = self.config;
        state_config.layer = 0;
        prefill_config.layer = 0;
        if state_config != prefill_config
            || write_plan.token_count() != self.tokens
            || write_plan.written_tokens() != self.tokens
            || table.token_len() != end
        {
            return Err(Error::InvalidPagedKv("prefill attention state or range mismatch"));
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
        if separate {
            self.qkv_postprocess.execute_separate(
                &self.stream,
                [
                    &self.scratch.qkv_separate[0],
                    &self.scratch.qkv_separate[1],
                    &self.scratch.qkv_separate[2],
                ],
                weights.query_norm,
                weights.key_norm,
                &mut self.scratch.query_rope,
                &mut self.scratch.key_rope,
                &mut self.scratch.value_norm,
                start_position,
            )?;
        } else {
            self.qkv_postprocess.execute(
                &self.stream,
                &self.scratch.qkv,
                weights.query_norm,
                weights.key_norm,
                &mut self.scratch.query_rope,
                &mut self.scratch.key_rope,
                &mut self.scratch.value_norm,
                start_position,
            )?;
        }
        state
            .cache
            .store(write_plan, &self.scratch.key_rope, &self.scratch.value_norm)?;
        let image = self.config.sliding_window.and(image);
        state.attention.execute_prefill_masked(
            &self.scratch.query_rope,
            &state.cache,
            table,
            &mut self.scratch.attention,
            self.tokens,
            start_position,
            self.config.sliding_window,
            self.config.attention_scale,
            image.map(|span| (span.start, span.end)),
        )?;
        self.output_projection.execute(
            &self.stream,
            &self.scratch.attention,
            weights.output,
            output,
        )
    }
}
