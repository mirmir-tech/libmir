mod attention;
mod page_write;

pub use page_write::{PreparedQuantizedPageWrite, QuantizedPageWriteOptions};

use super::Result;

mirtal::metal_library! {
    fn page_write_library {
        name: "mirmir_quantized_page_write",
        source: file "kernels/quantized_kv/page_write.metal",
    }
}

mirtal::metal_kernel! {
    fn attention_kernel {
        name: "mirmir_quantized_paged_sdpa",
        templates: [
            T: dtype = bf16, QUERY_HEADS: int = 32, KV_HEADS: int = 16,
            QUERY_TOKENS: int = 1, PAGE_CAPACITY: int = 128,
            HEAD_DIM: int = 128, PACKED_DIM: int = 32,
            QK_PER_THREAD: int = 4, V_PER_THREAD: int = 4, PAGE_SIZE: int = 16,
        ],
        inputs: [
            queries: T, key_pages: u32, value_pages: u32,
            key_scales: f32, value_scales: f32, page_table: u32,
            page_dependency: u32, attention_scale: scalar<f32>,
        ],
        outputs: [output: T],
        source: file "kernels/quantized_kv/attention.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct QuantizedKvKernels {
    attention: mirtal::MetalKernel<8, 1>,
    page_write: mirtal::MetalLibrary,
}

impl QuantizedKvKernels {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            attention: attention_kernel()?,
            page_write: page_write_library()?,
        })
    }
}

impl super::Kernels {
    pub(crate) fn quantized_page_write(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 7],
        options: QuantizedPageWriteOptions,
        prepared: &mut PreparedQuantizedPageWrite,
    ) -> Result<[mirtal::Array; 4]> {
        self.quantized_kv.page_write(stream, inputs, options, prepared)
    }

    pub(crate) fn quantized_paged_attention(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 7],
        page_size: usize,
        context_tokens: usize,
        scale: f32,
    ) -> Result<mirtal::Array> {
        self.quantized_kv.attention(stream, inputs, page_size, context_tokens, scale)
    }
}
