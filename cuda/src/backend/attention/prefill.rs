use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::kv::{BlockTable, KvBackendStorage, KvWritePlan};

use super::{
    AttentionOutputProjection, AttentionQkvProjection, DecodeAttentionBf16, DecodeAttentionConfig,
    DecodeAttentionOutputWeight, DecodeAttentionWeights, ProjectionFormat, QkvProjectionBuffers,
    validate,
};
use crate::{
    CudaBackend, DensePlanRequest, DenseRole, Error, ExecutionPhase, NvFp4Bf16Linear, Result,
    RmsNormBf16,
    kernels::{QkvPostprocess, QkvPostprocessSpec},
};
/// Fixed-chunk BF16 attention prefill sharing paged state with decode.
#[derive(Debug)]
pub struct PrefillAttentionBf16 {
    input_norm: RmsNormBf16,
    qkv: AttentionQkvProjection,
    qkv_postprocess: QkvPostprocess,
    output_projection: AttentionOutputProjection,
    scratch: PrefillAttentionScratch,
    stream: Stream,
    config: DecodeAttentionConfig,
    tokens: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::backend) struct ImageAttentionSpan {
    pub start: usize,
    pub end: usize,
}
#[derive(Debug)]
struct PrefillAttentionScratch {
    normalized: DeviceBuffer<bf16>,
    qkv: DeviceBuffer<bf16>,
    qkv_separate: [DeviceBuffer<bf16>; 3],
    value_norm: DeviceBuffer<bf16>,
    query_rope: DeviceBuffer<bf16>,
    key_rope: DeviceBuffer<bf16>,
    attention: DeviceBuffer<bf16>,
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
        let attention_width = config.query_heads * config.cache.value_head_dim;
        let epsilon = config.rms_norm_epsilon;
        let qkv_postprocess = QkvPostprocessSpec {
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
        let output_request = DensePlanRequest {
            phase: ExecutionPhase::Prefill,
            role: DenseRole::AttentionOutput,
            tokens,
            input_features: attention_width,
            output_features: hidden,
        };
        let output_projection = match config.projection_format {
            ProjectionFormat::Bf16 => {
                AttentionOutputProjection::Bf16(backend.prepare_bf16_projection(output_request)?)
            },
            ProjectionFormat::Int8 => {
                let Some(DecodeAttentionOutputWeight::Int8(weight)) =
                    weights.map(|weights| weights.output)
                else {
                    return Err(Error::InvalidExecutionPlan("INT8 prefill requires INT8 output"));
                };
                AttentionOutputProjection::Int8(crate::CompressedInt8Bf16Linear::new(
                    backend,
                    tokens,
                    attention_width,
                    hidden,
                    weight.clone(),
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
            qkv_postprocess: QkvPostprocess::compile(&backend.inner.compiler, qkv_postprocess)?,
            output_projection,
            scratch: PrefillAttentionScratch::new(backend, config, tokens)?,
            stream: backend.inner.stream.clone(),
            config,
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
        if state.config != self.config
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
impl PrefillAttentionScratch {
    fn new(backend: &CudaBackend, config: DecodeAttentionConfig, tokens: usize) -> Result<Self> {
        let allocate = |width| -> Result<DeviceBuffer<bf16>> {
            let elements = tokens
                .checked_mul(width)
                .ok_or(Error::InvalidPagedKv("prefill attention scratch overflow"))?;
            Ok(backend.inner.pool.allocate::<bf16>(&backend.inner.stream, elements)?)
        };
        let query = config.query_heads * config.cache.key_head_dim;
        let key = config.cache.kv_heads * config.cache.key_head_dim;
        let value = config.cache.kv_heads * config.cache.value_head_dim;
        Ok(Self {
            normalized: allocate(config.hidden_size)?,
            qkv: allocate(query + key + value)?,
            qkv_separate: [allocate(query)?, allocate(key)?, allocate(value)?],
            value_norm: allocate(value)?,
            query_rope: allocate(query)?,
            key_rope: allocate(key)?,
            attention: allocate(config.query_heads * config.cache.value_head_dim)?,
        })
    }
}
