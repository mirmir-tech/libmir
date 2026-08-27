use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{AffineGatedFullAttentionMoeLayerConfig, scratch::FullAttentionLayerScratch};
use crate::{
    CudaAffineGatedFullAttention, CudaAffineGatedFullAttentionExecution,
    CudaAffineGatedFullAttentionState, CudaAffineSharedExpertMoe,
    CudaAffineSharedExpertMoeExecution, CudaBackend, CudaTensor, Error, ExecutionPhase,
    PagedPrefillBatch, Result,
    kernels::{ElementwiseBf16, ShiftedRmsNorm},
};

#[derive(Debug)]
pub struct CudaAffineGatedFullAttentionMoeExecution {
    pub(super) backend: CudaBackend,
    pub(super) attention: CudaAffineGatedFullAttentionExecution,
    pub(super) moe: CudaAffineSharedExpertMoeExecution,
    pub(super) input_norm: ShiftedRmsNorm,
    pub(super) post_attention_norm: ShiftedRmsNorm,
    pub(super) residual: ElementwiseBf16,
    pub(super) input_norm_weight: CudaTensor,
    pub(super) post_attention_norm_weight: CudaTensor,
    pub(super) scratch: FullAttentionLayerScratch,
}

impl CudaAffineGatedFullAttentionMoeExecution {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedFullAttentionMoeLayerConfig,
        attention: &CudaAffineGatedFullAttention,
        moe: &CudaAffineSharedExpertMoe,
        input_norm_weight: &CudaTensor,
        post_attention_norm_weight: &CudaTensor,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<Self> {
        let hidden = config.attention.hidden_size;
        let norm = || {
            ShiftedRmsNorm::compile(
                &backend.inner.compiler,
                tokens,
                hidden,
                config.rms_norm_epsilon,
                config.norm_weight_shift,
            )
        };
        let elements = tokens
            .checked_mul(hidden)
            .ok_or(Error::InvalidDecoderKernel("gated full-attention residual shape overflow"))?;
        Ok(Self {
            backend: backend.clone(),
            attention: attention.prepare(tokens)?,
            moe: moe.prepare_phase(tokens, phase)?,
            input_norm: norm()?,
            post_attention_norm: norm()?,
            residual: ElementwiseBf16::compile(&backend.inner.compiler, elements)?,
            input_norm_weight: input_norm_weight.clone(),
            post_attention_norm_weight: post_attention_norm_weight.clone(),
            scratch: FullAttentionLayerScratch::new(backend, tokens, hidden)?,
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

    pub(crate) fn execute_packed_prefill(
        &mut self,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        states: &mut [&mut CudaAffineGatedFullAttentionState],
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(input, output)?;
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16_tensor(&self.input_norm_weight)?,
            &mut self.scratch.normalized,
        )?;
        self.attention.execute_packed_prefill(
            &self.scratch.normalized,
            positions,
            states,
            batch,
            &mut self.scratch.attention,
        )?;
        self.moe.execute_residual_norm(
            &self.post_attention_norm,
            input,
            &self.scratch.attention,
            bf16_tensor(&self.post_attention_norm_weight)?,
            &mut self.scratch.residual,
            &mut self.scratch.normalized,
            &mut self.scratch.moe,
            output,
            &self.residual,
        )?;
        Ok(())
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
        self.validate(input, output)?;
        let stream = &self.backend.inner.stream;
        self.input_norm.execute(
            stream,
            input,
            bf16_tensor(&self.input_norm_weight)?,
            &mut self.scratch.normalized,
        )?;
        self.attention.execute_with_image_span(
            &self.scratch.normalized,
            positions,
            state,
            write_plan,
            table,
            start_position,
            window,
            image_span,
            &mut self.scratch.attention,
        )?;
        self.moe.execute_residual_norm(
            &self.post_attention_norm,
            input,
            &self.scratch.attention,
            bf16_tensor(&self.post_attention_norm_weight)?,
            &mut self.scratch.residual,
            &mut self.scratch.normalized,
            &mut self.scratch.moe,
            output,
            &self.residual,
        )?;
        Ok(())
    }

    pub(super) fn validate(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        let expected = self.scratch.residual.len();
        if input.len() != expected || output.len() != expected {
            return Err(Error::InvalidDecoderKernel(
                "gated full-attention MoE layer buffer mismatch",
            ));
        }
        Ok(())
    }
}

fn bf16_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
