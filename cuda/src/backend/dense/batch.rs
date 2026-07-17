use mircuda::{DeviceBuffer, Stream, bf16};

use super::{
    DenseScratch, DenseSwiGluConfig, DenseSwiGluLayerTemplate, DenseSwiGluWeights, DownProjection,
    GateUpBuffers, GateUpProjection, validate,
};
use crate::{
    BatchedDecodeAttentionBf16, CudaBackend, Error, PagedDecodeBatch, PagedKvCache, Result,
    RmsNormBf16,
    kernels::{ElementwiseBf16, PackedGatedBf16},
};

/// Fixed-shape dense `SwiGLU` layer for independent decode rows.
pub struct BatchedDecodeDenseSwiGlu {
    attention: BatchedDecodeAttentionBf16,
    post_attention_norm: RmsNormBf16,
    gate_up: GateUpProjection,
    down: DownProjection,
    activation: PackedGatedBf16,
    hidden_ops: ElementwiseBf16,
    scratch: DenseScratch,
    stream: Stream,
    config: DenseSwiGluConfig,
    rows: usize,
}

/// Batched dense layer retaining immutable checkpoint weights.
pub struct BatchedDecodeDenseLayer {
    block: BatchedDecodeDenseSwiGlu,
    template: DenseSwiGluLayerTemplate,
}

impl BatchedDecodeDenseSwiGlu {
    fn new_with_cache(
        backend: &CudaBackend,
        config: DenseSwiGluConfig,
        rows: usize,
        cache: PagedKvCache,
        weights: DenseSwiGluWeights<'_>,
    ) -> Result<Self> {
        validate(config)?;
        if rows == 0 {
            return Err(Error::InvalidDecoderKernel("batched dense decode is empty"));
        }
        let hidden = config.attention.hidden_size;
        let elements = rows
            .checked_mul(hidden)
            .ok_or(Error::InvalidDecoderKernel("batched dense hidden size overflow"))?;
        Ok(Self {
            attention: BatchedDecodeAttentionBf16::new_with_cache_and_weights(
                backend,
                config.attention,
                rows,
                cache,
                Some(weights.attention),
            )?,
            post_attention_norm: RmsNormBf16::new(
                backend,
                rows,
                hidden,
                config.attention.rms_norm_epsilon,
            )?,
            gate_up: GateUpProjection::new(backend, config, rows, Some(weights.gate_up))?,
            down: DownProjection::new(backend, config, rows, Some(weights.down))?,
            activation: PackedGatedBf16::compile(
                &backend.inner.compiler,
                rows,
                config.intermediate_size,
            )?,
            hidden_ops: ElementwiseBf16::compile(&backend.inner.compiler, elements)?,
            scratch: DenseScratch::new(backend, config, rows)?,
            stream: backend.inner.stream.clone(),
            config,
            rows,
        })
    }

    fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: DenseSwiGluWeights<'_>,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if batch.active() != self.rows {
            return Err(Error::InvalidDecoderKernel("dense decode batch differs from plan"));
        }
        self.attention
            .execute(input, weights.attention, batch, &mut self.scratch.attention)?;
        self.hidden_ops.add(
            &self.stream,
            input,
            &self.scratch.attention,
            &mut self.scratch.residual,
        )?;
        let separate = self.gate_up.execute(
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
            self.down
                .execute(&self.scratch.activated, weights.down, &mut self.scratch.mlp)?;
        }
        self.hidden_ops
            .add(&self.stream, &self.scratch.residual, &self.scratch.mlp, output)
    }
}

impl BatchedDecodeDenseLayer {
    pub(super) fn new(
        backend: &CudaBackend,
        template: DenseSwiGluLayerTemplate,
        rows: usize,
        cache: PagedKvCache,
    ) -> Result<Self> {
        let block = BatchedDecodeDenseSwiGlu::new_with_cache(
            backend,
            template.config(),
            rows,
            cache,
            template.weights(),
        )?;
        Ok(Self { block, template })
    }

    pub(in crate::backend) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        batch: &PagedDecodeBatch,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.block.execute(input, self.template.weights(), batch, output)
    }
}
