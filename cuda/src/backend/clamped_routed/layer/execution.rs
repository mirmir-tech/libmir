use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockTable, KvWritePlan};

use super::{ClampedRoutedLayerExecution, ClampedRoutedLayerTemplate};
use crate::{
    CudaTensor, Error, PagedKvCache, PagedPrefillBatch, Result,
    backend::{
        WindowedPrefillStaging,
        clamped_routed::{scratch::ClampedRoutedScratch, weights::ClampedRoutedExpertWeights},
        linear::SelectedDenseMoeBf16,
    },
    kernels::{ClampedRoutedBatchSplitDecode, ClampedRoutedSplitDecode},
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
        ring_table_device: &DeviceBuffer<u32>,
        ring_slot: usize,
        start: usize,
        cached_until: usize,
        scratch: &mut ClampedRoutedScratch,
        dense_experts: &mut Option<SelectedDenseMoeBf16>,
        split_decode: &mut Option<ClampedRoutedSplitDecode>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let weights = template.weights();
        self.input_norm.execute(input, &weights.input_norm, &mut scratch.normalized)?;
        self.qkv.execute(&self.kernels, &self.stream, scratch)?;
        if !cache.is_windowed() {
            let mut missing = write.clone();
            missing.skip_prefix(cached_until.saturating_sub(start));
            cache.store_for_session(&missing, ring_slot, &scratch.key, &scratch.value)?;
        }
        let table_device = if cache.is_windowed() {
            ring_table_device
        } else {
            table_device
        };
        self.attend(
            cache,
            &scratch.key,
            &scratch.value,
            table,
            table_device,
            bf16s(&weights.sinks)?,
            start,
            &scratch.query,
            &mut scratch.attended,
            split_decode,
        )?;
        if cache.is_windowed() {
            cache.store_for_session(write, ring_slot, &scratch.key, &scratch.value)?;
        }
        self.execute_tail(input, weights, scratch, dense_experts, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend::clamped_routed) fn execute_batch(
        &mut self,
        template: &ClampedRoutedLayerTemplate,
        input: &DeviceBuffer<bf16>,
        cache: &mut PagedKvCache,
        batch: &PagedPrefillBatch,
        windowed_prefill: Option<&mut WindowedPrefillStaging>,
        scratch: &mut ClampedRoutedScratch,
        dense_experts: &mut Option<SelectedDenseMoeBf16>,
        batch_split_decode: &mut Option<ClampedRoutedBatchSplitDecode>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let weights = template.weights();
        self.input_norm.execute(input, &weights.input_norm, &mut scratch.normalized)?;
        self.qkv.execute(&self.kernels, &self.stream, scratch)?;
        if !cache.is_windowed() {
            cache.store_prefill_batch(batch, &scratch.key, &scratch.value)?;
        }
        let windowed_fmha = if cache.is_windowed()
            && batch.max_query_tokens() >= super::super::WINDOWED_FMHA_MIN_QUERY_TOKENS
        {
            let staged = windowed_prefill
                .ok_or(Error::InvalidExecutionPlan("windowed prefill staging was not prepared"))?;
            staged.stage(
                batch,
                &scratch.key,
                &scratch.value,
                cache.key_pages(),
                cache.value_pages(),
                self.window.ok_or(Error::InvalidExecutionPlan(
                    "windowed cache layer has no attention window",
                ))?,
            )?;
            self.attention.execute_windowed_fmha(
                &self.stream,
                &scratch.query,
                batch,
                staged,
                bf16s(&weights.sinks)?,
                &mut scratch.normalized,
                &mut scratch.attended,
                self.config.scale,
            )?
        } else {
            false
        };
        let split = (!cache.is_windowed() && !windowed_fmha)
            .then_some(batch_split_decode.as_mut())
            .flatten()
            .map(|split| {
                split.execute(
                    &self.stream,
                    &scratch.query,
                    cache.key_pages(),
                    cache.value_pages(),
                    batch,
                    bf16s(&weights.sinks)?,
                    &mut scratch.attended,
                    self.window,
                    self.config.scale,
                )
            })
            .transpose()?
            .unwrap_or(false);
        if !split && !windowed_fmha {
            let tables = if cache.is_windowed() {
                batch.ring_tables()
            } else {
                batch.tables()
            };
            self.attention.execute_prefill_batch(
                &self.stream,
                &scratch.query,
                &scratch.key,
                &scratch.value,
                cache.key_pages(),
                cache.value_pages(),
                batch,
                tables,
                bf16s(&weights.sinks)?,
                &mut scratch.normalized,
                &mut scratch.attended,
                self.window,
                self.config.scale,
            )?;
        }
        if cache.is_windowed() {
            cache.store_prefill_batch(batch, &scratch.key, &scratch.value)?;
        }
        self.execute_tail(input, weights, scratch, dense_experts, output)
    }

    fn execute_tail(
        &mut self,
        input: &DeviceBuffer<bf16>,
        weights: &super::super::weights::ClampedRoutedLayerWeights,
        scratch: &mut ClampedRoutedScratch,
        dense_experts: &mut Option<SelectedDenseMoeBf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.output.execute(&scratch.attended, &mut scratch.projected)?;
        self.kernels.add_bias(
            &self.stream,
            &scratch.projected,
            bf16s(&weights.output_bias)?,
            &mut scratch.biased,
            self.config.hidden,
        )?;
        self.add.add(&self.stream, input, &scratch.biased, &mut scratch.residual)?;
        self.post_norm
            .execute(&scratch.residual, &weights.post_norm, &mut scratch.normalized)?;
        self.router.execute(&scratch.normalized, &mut scratch.router)?;
        self.kernels.add_bias(
            &self.stream,
            &scratch.router,
            bf16s(&weights.router_bias)?,
            &mut scratch.router_biased,
            self.config.experts,
        )?;
        self.top_k.execute(
            &self.stream,
            &scratch.router_biased,
            &mut scratch.selected,
            &mut scratch.routing,
        )?;
        self.experts(&weights.experts, scratch, dense_experts)?;
        self.add.add(&self.stream, &scratch.residual, &scratch.moe, output)
    }

    fn experts(
        &mut self,
        weights: &ClampedRoutedExpertWeights,
        scratch: &mut ClampedRoutedScratch,
        dense_experts: &mut Option<SelectedDenseMoeBf16>,
    ) -> Result<()> {
        match weights {
            ClampedRoutedExpertWeights::Native(_) | ClampedRoutedExpertWeights::Mlx(_) => {
                let partial = scratch
                    .route_partial
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("clamped MXFP4 partial was not prepared"))?;
                self.experts
                    .as_mut()
                    .ok_or(Error::InvalidExecutionPlan("clamped MXFP4 execution was not prepared"))?
                    .execute(
                        weights,
                        &scratch.normalized,
                        &scratch.selected,
                        &scratch.routing,
                        &mut scratch.activated,
                        partial,
                        &mut scratch.moe,
                    )
            },
            ClampedRoutedExpertWeights::Dense(weights) => dense_experts
                .as_mut()
                .ok_or(Error::InvalidExecutionPlan(
                    "dense clamped-routed execution was not prepared",
                ))?
                .execute(
                    &scratch.normalized,
                    &scratch.selected,
                    &scratch.routing,
                    weights,
                    &mut scratch.activated,
                    &mut scratch.moe,
                ),
        }
    }
}

fn bf16s(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
