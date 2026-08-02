use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{
    DecodeDenseSwiGlu, DenseScratch, DenseSwiGluConfig, DenseSwiGluWeights, DownProjection,
    GateUpBuffers, GateUpProjection, validate,
};
use crate::{
    CudaBackend, Error, PagedPrefillBatch, PrefillAttentionBf16, Result, RmsNormBf16,
    kernels::{ElementwiseBf16, PackedGatedBf16},
};

pub struct PrefillDenseSwiGlu {
    attention: PrefillAttentionBf16,
    post_attention_norm: RmsNormBf16,
    gate_up: GateUpProjection,
    down: DownProjection,
    activation: PackedGatedBf16,
    hidden_ops: ElementwiseBf16,
    scratch: DenseScratch,
    stream: Stream,
    config: DenseSwiGluConfig,
    tokens: usize,
}

impl PrefillDenseSwiGlu {
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        tokens: usize,
        weights: DenseSwiGluWeights<'_>,
    ) -> Result<Self> {
        validate(config)?;
        if tokens == 0 {
            return Err(Error::InvalidDecoderKernel("dense prefill chunk is empty"));
        }
        let hidden = config.attention.hidden_size;
        let intermediate = config.intermediate_size;
        Ok(Self {
            attention: PrefillAttentionBf16::new(
                backend,
                config.attention,
                tokens,
                Some(weights.attention),
            )?,
            post_attention_norm: RmsNormBf16::new(
                backend,
                tokens,
                hidden,
                config.attention.rms_norm_epsilon,
            )?,
            gate_up: GateUpProjection::new(backend, config, tokens, Some(weights.gate_up))?,
            down: DownProjection::new(backend, config, tokens, Some(weights.down))?,
            activation: PackedGatedBf16::compile(&backend.inner.compiler, tokens, intermediate)?,
            hidden_ops: ElementwiseBf16::compile(
                &backend.inner.compiler,
                super::scratch::elements(tokens, hidden)?,
            )?,
            scratch: DenseScratch::new(backend, config, tokens)?,
            stream: backend.inner.stream.clone(),
            config,
            tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &mut self,
        state: &mut DecodeDenseSwiGlu,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        start_position: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.attention.execute(
            &mut state.attention,
            input,
            weights.attention,
            write_plan,
            table,
            start_position,
            &mut self.scratch.attention,
        )?;
        self.execute_tail(input, weights, output)
    }

    pub fn execute_batch(
        &mut self,
        state: &mut DecodeDenseSwiGlu,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        batch: &PagedPrefillBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.attention.execute_batch(
            &mut state.attention,
            input,
            weights.attention,
            batch,
            &mut self.scratch.attention,
        )?;
        self.execute_tail(input, weights, output)
    }

    fn execute_tail(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.hidden_ops.add(
            &self.stream,
            input,
            &self.scratch.attention,
            &mut self.scratch.residual,
        )?;
        let separate = self.gate_up.execute(
            &self.stream,
            &self.scratch.residual,
            &self.post_attention_norm,
            weights.post_attention_norm,
            weights.gate_up,
            &mut GateUpBuffers {
                normalized: &mut self.scratch.normalized,
                packed: &mut self.scratch.gate_up,
                separate: &mut self.scratch.gate_up_separate,
            },
        )?;
        let fused_down = separate
            && self.down.execute_gated(
                &self.scratch.gate_up_separate[0],
                &self.scratch.gate_up_separate[1],
                self.config.activation,
                weights.down,
                &mut self.scratch.mlp,
            )?;
        if separate && !fused_down {
            self.activation.execute_separate(
                &self.stream,
                &self.scratch.gate_up_separate[0],
                &self.scratch.gate_up_separate[1],
                &mut self.scratch.activated,
                self.config.activation.into(),
            )?;
        } else if !fused_down {
            self.activation.execute(
                &self.stream,
                &self.scratch.gate_up,
                &mut self.scratch.activated,
                self.config.activation.into(),
            )?;
        }
        if !fused_down {
            self.down.execute(
                &self.stream,
                &self.scratch.activated,
                weights.down,
                &mut self.scratch.mlp,
            )?;
        }
        self.hidden_ops
            .add(&self.stream, &self.scratch.residual, &self.scratch.mlp, output)
    }

    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }
}
