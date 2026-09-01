use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use super::{
    PagedArenaPool, Result, attention_tuning,
    compiled::CompiledGraphs,
    gate_up_tuning::MetalTuner,
    kernels::{
        Kernels, MxFp4Shape, PageWriteOptions, PreparedPageWrite, PreparedQuantizedPageWrite,
        QuantizedPageWriteOptions,
    },
};

mod construction;
mod tuning;

#[derive(Debug)]
pub struct Stream {
    native: mirtal::Stream,
    compiled: CompiledGraphs,
    kernels: Kernels,
    config: Arc<crate::MetalConfig>,
    pub(super) tuner: Mutex<MetalTuner>,
    graph_dumped: AtomicBool,
    paged_arenas: Arc<PagedArenaPool>,
}

impl Stream {
    pub fn synchronize(&self) -> Result<()> {
        Ok(self.native.synchronize()?)
    }

    pub(crate) fn eval_many(&self, arrays: &[&super::Array]) -> Result<()> {
        let arrays = arrays.iter().map(|array| array.native()).collect::<Vec<_>>();
        Ok(self.native.eval_many(&arrays)?)
    }

    pub(crate) fn eval_many_with_paged_arenas(&self, arrays: &[&super::Array]) -> Result<()> {
        self.paged_arenas.eval_with_graph_roots(arrays, self)
    }

    pub(super) const fn native(&self) -> &mirtal::Stream {
        &self.native
    }

    pub(super) const fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub(crate) fn config(&self) -> &crate::MetalConfig {
        &self.config
    }

    pub(crate) const fn paged_arenas(&self) -> &Arc<PagedArenaPool> {
        &self.paged_arenas
    }

    pub(crate) fn detach_paged_arena_graphs(&self) -> Result<()> {
        self.paged_arenas.detach_evaluated_graphs(self)
    }

    pub(crate) fn finish_startup_tuning(&self) {
        self.tuner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .finish_startup();
    }

    pub(crate) fn take_graph_dump_path(&self) -> Option<&std::path::Path> {
        self.config
            .diagnostics
            .graph_dump
            .as_deref()
            .filter(|_| !self.graph_dumped.swap(true, Ordering::Relaxed))
    }

    pub(super) fn gelu_approx_mul(
        &self,
        gate: &mirtal::Array,
        input: &mirtal::Array,
    ) -> Result<mirtal::Array> {
        Ok(self.compiled.geglu(gate, input, &self.native)?)
    }

    pub(super) fn silu_mul(
        &self,
        gate: &mirtal::Array,
        input: &mirtal::Array,
    ) -> Result<mirtal::Array> {
        Ok(self.compiled.swiglu(gate, input, &self.native)?)
    }

    pub(super) fn precise_silu_mul(
        &self,
        reference: &mirtal::Array,
        gate: &mirtal::Array,
        input: &mirtal::Array,
    ) -> Result<mirtal::Array> {
        Ok(self.compiled.precise_swiglu(reference, gate, input, &self.native)?)
    }

    pub(super) fn logit_softcap(&self, input: &mirtal::Array, cap: f32) -> Result<mirtal::Array> {
        let cap = mirtal::Array::from_slice(&[cap], [])?;
        Ok(self.compiled.logit_softcap(input, &cap, &self.native)?)
    }

    pub(super) fn affine_router(&self, inputs: [&mirtal::Array; 6]) -> Result<[mirtal::Array; 2]> {
        Ok(self.compiled.router(inputs, &self.native)?)
    }

    pub(super) fn mxfp4_gate_up(
        &self,
        inputs: [&super::Array; 6],
        shape: MxFp4Shape,
    ) -> Result<super::Array> {
        self.kernels.mxfp4_gate_up(inputs, shape, self)
    }

    pub(super) fn mxfp4_down(
        &self,
        inputs: [&super::Array; 6],
        shape: MxFp4Shape,
    ) -> Result<super::Array> {
        self.kernels.mxfp4_down(inputs, shape, self)
    }

    pub(super) fn mxfp4_split_gate_up(
        &self,
        inputs: [&super::Array; 9],
        shape: MxFp4Shape,
    ) -> Result<super::Array> {
        self.kernels.mxfp4_split_gate_up(inputs, shape, self)
    }

    pub(super) fn mxfp4_u32_down(
        &self,
        inputs: [&super::Array; 6],
        shape: MxFp4Shape,
    ) -> Result<super::Array> {
        self.kernels.mxfp4_u32_down(inputs, shape, self)
    }

    pub(super) fn paged_attention(
        &self,
        inputs: [&mirtal::Array; 5],
        scratch: &super::attention::PagedAttentionScratch,
        page_size: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<mirtal::Array> {
        attention_tuning::forward(
            &self.kernels, inputs, scratch, page_size, context_tokens, scale, self,
        )
    }

    pub(super) fn batched_paged_attention(
        &self,
        inputs: [&mirtal::Array; 28],
        page_size: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<mirtal::Array> {
        self.kernels
            .batched_paged_attention(&self.native, inputs, page_size, context_tokens, scale)
    }

    pub(super) fn expert_restore_reduce(
        &self,
        inputs: [&mirtal::Array; 3],
    ) -> Result<mirtal::Array> {
        self.kernels.expert_restore_reduce(&self.native, inputs)
    }

    pub(super) fn expert_group(
        &self,
        indices: &mirtal::Array,
        experts: usize,
    ) -> Result<[mirtal::Array; 2]> {
        self.kernels.expert_group(&self.native, indices, experts)
    }

    pub(super) fn quantized_paged_attention(
        &self,
        inputs: [&mirtal::Array; 7],
        page_size: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<mirtal::Array> {
        self.kernels
            .quantized_paged_attention(&self.native, inputs, page_size, context_tokens, scale)
    }

    pub(super) fn gated_delta_gates(
        &self,
        inputs: [&mirtal::Array; 4],
    ) -> Result<[mirtal::Array; 2]> {
        self.kernels.gated_delta_gates(&self.native, inputs)
    }

    pub(super) fn gated_delta_recurrence(
        &self,
        inputs: [&mirtal::Array; 6],
    ) -> Result<[mirtal::Array; 2]> {
        self.kernels.gated_delta_recurrence(&self.native, inputs)
    }

    pub(super) fn gated_delta_decode(
        &self,
        inputs: [&mirtal::Array; 8],
        normalize: bool,
    ) -> Result<[mirtal::Array; 2]> {
        self.kernels.gated_delta_decode(&self.native, inputs, normalize)
    }

    pub(super) fn page_write(
        &self,
        inputs: [&mirtal::Array; 5],
        options: PageWriteOptions,
        prepared: &mut PreparedPageWrite,
    ) -> Result<[mirtal::Array; 2]> {
        self.kernels.page_write(&self.native, inputs, options, prepared)
    }

    pub(super) fn quantized_page_write(
        &self,
        inputs: [&mirtal::Array; 7],
        options: QuantizedPageWriteOptions,
        prepared: &mut PreparedQuantizedPageWrite,
    ) -> Result<[mirtal::Array; 4]> {
        self.kernels.quantized_page_write(&self.native, inputs, options, prepared)
    }
}
