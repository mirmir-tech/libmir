use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{
    DenseScratch, DenseSwiGluConfig, DenseSwiGluWeights, DownProjection, GateUpBuffers,
    GateUpProjection,
};
use crate::{
    CudaBackend, DecodeAttentionBf16, Error, Result, RmsNormBf16,
    kernels::{ElementwiseBf16, PackedGatedBf16},
};

pub struct DecodeDenseSwiGlu {
    pub(in crate::backend) attention: DecodeAttentionBf16,
    pub(super) post_attention_norm: RmsNormBf16,
    pub(super) gate_up: GateUpProjection,
    pub(super) down: DownProjection,
    pub(super) activation: PackedGatedBf16,
    pub(super) hidden_ops: ElementwiseBf16,
    pub(super) scratch: DenseScratch,
    pub(super) stream: Stream,
    pub(super) config: DenseSwiGluConfig,
}

impl DecodeDenseSwiGlu {
    pub(in crate::backend) fn new_with_cache(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        cache: crate::PagedKvCache,
        weights: DenseSwiGluWeights<'_>,
    ) -> Result<Self> {
        validate(config)?;
        let hidden = config.attention.hidden_size;
        let intermediate = config.intermediate_size;
        Ok(Self {
            attention: DecodeAttentionBf16::new_with_cache_and_weights(
                backend,
                config.attention,
                cache,
                Some(weights.attention),
            )?,
            post_attention_norm: RmsNormBf16::new(
                backend,
                1,
                hidden,
                config.attention.rms_norm_epsilon,
            )?,
            gate_up: GateUpProjection::new(backend, config, 1, Some(weights.gate_up))?,
            down: DownProjection::new(backend, config, 1, Some(weights.down))?,
            activation: PackedGatedBf16::compile(&backend.inner.compiler, 1, intermediate)?,
            hidden_ops: ElementwiseBf16::compile(&backend.inner.compiler, hidden)?,
            scratch: DenseScratch::new(backend, config, 1)?,
            stream: backend.inner.stream.clone(),
            config,
        })
    }

    pub fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        write_plan: &KvWritePlan,
        table: &BlockTable,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.attention.execute(
            input,
            weights.attention,
            write_plan,
            table,
            &mut self.scratch.attention,
        )?;
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
}

pub(super) fn validate(config: DenseSwiGluConfig) -> Result<()> {
    if config.intermediate_size == 0 {
        Err(Error::InvalidDecoderKernel("dense SwiGLU intermediate size is zero"))
    } else {
        Ok(())
    }
}
