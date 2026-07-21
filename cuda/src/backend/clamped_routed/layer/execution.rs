use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvBackendStorage, KvWritePlan};

use super::{ClampedRoutedLayerExecution, ClampedRoutedLayerTemplate};
use crate::{
    CudaTensor, Error, PagedKvCache, Result,
    backend::clamped_routed::weights::ClampedRoutedExpertWeights,
};

impl ClampedRoutedLayerExecution {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::clamped_routed) fn execute(
        &mut self,
        template: &ClampedRoutedLayerTemplate,
        input: &DeviceBuffer<bf16>,
        cache: &mut PagedKvCache,
        write: &KvWritePlan,
        table: &BlockTable,
        table_device: &DeviceBuffer<u32>,
        start: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let weights = template.weights();
        self.input_norm
            .execute(input, &weights.input_norm, &mut self.scratch.normalized)?;
        self.qkv.execute(&self.kernels, &self.stream, &mut self.scratch, start)?;
        cache.store(write, &self.scratch.key, &self.scratch.value)?;
        self.attend(cache, table, table_device, bf16s(&weights.sinks)?, start)?;
        self.output.execute(&self.scratch.attended, &mut self.scratch.projected)?;
        self.kernels.add_bias(
            &self.stream,
            &self.scratch.projected,
            bf16s(&weights.output_bias)?,
            &mut self.scratch.biased,
            self.config.hidden,
        )?;
        self.add
            .add(&self.stream, input, &self.scratch.biased, &mut self.scratch.residual)?;
        self.post_norm.execute(
            &self.scratch.residual,
            &weights.post_norm,
            &mut self.scratch.normalized,
        )?;
        self.router.execute(&self.scratch.normalized, &mut self.scratch.router)?;
        self.kernels.add_bias(
            &self.stream,
            &self.scratch.router,
            bf16s(&weights.router_bias)?,
            &mut self.scratch.router_biased,
            self.config.experts,
        )?;
        self.top_k.execute(
            &self.stream,
            &self.scratch.router_biased,
            &mut self.scratch.selected,
            &mut self.scratch.routing,
        )?;
        self.experts(&weights.experts)?;
        self.add.add(&self.stream, &self.scratch.residual, &self.scratch.moe, output)
    }

    fn experts(&mut self, weights: &ClampedRoutedExpertWeights) -> Result<()> {
        match weights {
            ClampedRoutedExpertWeights::Native(weights) => {
                self.kernels.gate_up_native(
                    &self.stream,
                    &self.scratch.normalized,
                    u8s(&weights.gate_up_blocks)?,
                    u8s(&weights.gate_up_scales)?,
                    bf16s(&weights.gate_up_bias)?,
                    &self.scratch.selected,
                    &mut self.scratch.activated,
                )?;
                self.kernels.down_native(
                    &self.stream,
                    &self.scratch.activated,
                    u8s(&weights.down_blocks)?,
                    u8s(&weights.down_scales)?,
                    bf16s(&weights.down_bias)?,
                    &self.scratch.selected,
                    &self.scratch.routing,
                    &mut self.scratch.moe,
                )
            },
            ClampedRoutedExpertWeights::Mlx(weights) => {
                self.kernels.gate_up_mlx(
                    &self.stream,
                    &self.scratch.normalized,
                    u32s(&weights.gate_blocks)?,
                    u8s(&weights.gate_scales)?,
                    bf16s(&weights.gate_bias)?,
                    u32s(&weights.up_blocks)?,
                    u8s(&weights.up_scales)?,
                    bf16s(&weights.up_bias)?,
                    &self.scratch.selected,
                    &mut self.scratch.activated,
                )?;
                self.kernels.down_mlx(
                    &self.stream,
                    &self.scratch.activated,
                    u32s(&weights.down_blocks)?,
                    u8s(&weights.down_scales)?,
                    bf16s(&weights.down_bias)?,
                    &self.scratch.selected,
                    &self.scratch.routing,
                    &mut self.scratch.moe,
                )
            },
        }
    }

    fn attend(
        &mut self,
        cache: &PagedKvCache,
        table: &BlockTable,
        table_device: &DeviceBuffer<u32>,
        sinks: &DeviceBuffer<bf16>,
        start: usize,
    ) -> Result<()> {
        let blocks = table.blocks().len();
        if self.tokens == 1 {
            self.attention.execute(
                &self.stream,
                &self.scratch.query,
                cache.key_pages(),
                cache.value_pages(),
                table_device,
                sinks,
                &mut self.scratch.attended,
                table.token_len(),
                blocks,
                self.window,
                self.config.scale,
            )
        } else {
            self.attention.execute_prefill(
                &self.stream,
                &self.scratch.query,
                cache.key_pages(),
                cache.value_pages(),
                table_device,
                sinks,
                &mut self.scratch.attended,
                self.tokens,
                start,
                blocks,
                self.window,
                self.config.scale,
            )
        }
    }
}

fn bf16s(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}

fn u8s(tensor: &CudaTensor) -> Result<&DeviceBuffer<u8>> {
    tensor.as_u8().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "U8",
    })
}

fn u32s(tensor: &CudaTensor) -> Result<&DeviceBuffer<u32>> {
    tensor.as_u32().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "U32",
    })
}
