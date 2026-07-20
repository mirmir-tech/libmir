use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::{
    Result,
    compiled::CompiledGraphs,
    kernels::{
        Kernels, PageWriteOptions, PreparedPageWrite, PreparedQuantizedPageWrite,
        QuantizedPageWriteOptions,
    },
};

#[derive(Debug)]
pub struct Stream {
    native: mirtal::Stream,
    compiled: CompiledGraphs,
    kernels: Kernels,
    config: Arc<crate::MetalConfig>,
    graph_dumped: AtomicBool,
}

impl Stream {
    pub fn new_gpu() -> Result<Self> {
        Self::new(mirtal::Device::gpu(0), Arc::default())
    }

    pub(crate) fn new_gpu_with_config(config: Arc<crate::MetalConfig>) -> Result<Self> {
        Self::new(mirtal::Device::gpu(0), config)
    }

    pub fn new_cpu() -> Result<Self> {
        Self::new(mirtal::Device::cpu(0), Arc::default())
    }

    fn new(device: mirtal::Device, config: Arc<crate::MetalConfig>) -> Result<Self> {
        let native = device.new_stream()?;
        let compiled = CompiledGraphs::new(&native)?;
        let kernels = Kernels::new()?;
        Ok(Self {
            native,
            compiled,
            kernels,
            config,
            graph_dumped: AtomicBool::new(false),
        })
    }

    pub fn synchronize(&self) -> Result<()> {
        Ok(self.native.synchronize()?)
    }

    pub(super) const fn native(&self) -> &mirtal::Stream {
        &self.native
    }

    pub(crate) fn config(&self) -> &crate::MetalConfig {
        &self.config
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

    pub(super) fn paged_attention(
        &self,
        inputs: [&mirtal::Array; 5],
        scratch: &super::attention::PagedAttentionScratch,
        page_size: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<mirtal::Array> {
        self.kernels
            .paged_attention(&self.native, inputs, scratch, page_size, context_tokens, scale)
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
