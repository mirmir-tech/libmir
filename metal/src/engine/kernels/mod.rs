mod awq;
mod bitsandbytes;
mod direct_fp8;
mod direct_fp8_embedding;
mod expert_group;
mod expert_reduce;
pub(super) mod gated_delta;
mod gptq;
mod mxfp4;
mod mxfp4_embedding;
mod mxfp4_gathered_linear;
mod mxfp4_linear;
mod nvfp4_convert;
mod nvfp4_gathered_linear;
mod page_write;
mod paged_attention;
pub(super) use paged_attention::{PagedExecution, partial_blocks, two_pass_supported};
mod quantized_kv;
mod template;

pub use direct_fp8::DirectFp8Spec;
pub(super) use direct_fp8_embedding::DirectFp8EmbeddingSpec;
pub(super) use mxfp4::MxFp4Shape;
pub(super) use page_write::{PageWriteOptions, PreparedPageWrite};
pub(super) use quantized_kv::{PreparedQuantizedPageWrite, QuantizedPageWriteOptions};
pub(super) use template::template;

use super::Result;
pub(super) const BATCHED_PAGED_ROWS: usize = 12;

mirtal::metal_kernel! {
    fn mxfp4_gate_up {
        name: "mirmir_mxfp4_gate_up",
        templates: [
            T: dtype = bf16, HIDDEN: int = 2880, INTERMEDIATE: int = 2880,
            TOP_K: int = 4,
        ],
        inputs: [
            input: T, blocks: u8, scales: u8, bias: T, indices: u32,
            limit: scalar<f32>,
        ],
        outputs: [output: T],
        source: file "kernels/mxfp4_gate_up.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn mxfp4_down {
        name: "mirmir_mxfp4_down",
        templates: [
            T: dtype = bf16, HIDDEN: int = 2880, INTERMEDIATE: int = 2880,
            TOP_K: int = 4,
        ],
        inputs: [input: T, blocks: u8, scales: u8, bias: T, indices: u32, routing: T],
        outputs: [output: T],
        source: file "kernels/mxfp4_down.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn mxfp4_split_gate_up {
        name: "mirmir_mxfp4_split_gate_up",
        templates: [
            T: dtype = bf16, HIDDEN: int = 2880, INTERMEDIATE: int = 2880,
            TOP_K: int = 4,
        ],
        inputs: [
            input: T, gate_blocks: u32, gate_scales: u8, gate_bias: T,
            up_blocks: u32, up_scales: u8, up_bias: T, indices: u32,
            limit: scalar<f32>,
        ],
        outputs: [output: T],
        source: file "kernels/mxfp4_split_gate_up.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn mxfp4_u32_down {
        name: "mirmir_mxfp4_u32_down",
        templates: [
            T: dtype = bf16, HIDDEN: int = 2880, INTERMEDIATE: int = 2880,
            TOP_K: int = 4,
        ],
        inputs: [
            input: T, blocks: u32, scales: u8, bias: T, indices: u32,
            routing: T,
        ],
        outputs: [output: T],
        source: file "kernels/mxfp4_u32_down.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_library! {
    fn paged_kv_library {
        name: "mirmir_paged_kv",
        source: file "kernels/paged_kv.metal",
    }
}

mirtal::metal_kernel! {
    fn gated_delta_recurrence {
        name: "mirmir_gated_delta",
        templates: [
            InT: dtype = bf16, StT: dtype = f32, DK: int = 128, DV: int = 128,
            HK: int = 8, HV: int = 16, STEPS: int = 4,
        ],
        inputs: [query: InT, key: InT, value: InT, decay: f32, update: f32, state: StT],
        outputs: [output: InT, next_state: StT],
        source: file "kernels/gated_delta_recurrence.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn gated_delta_decode {
        name: "mirmir_gated_delta_decode_headwide_gates",
        templates: [
            InT: dtype = bf16, StT: dtype = f32, DK: int = 128, DV: int = 128,
            HK: int = 8, HV: int = 16, NORMALIZE: bool = true,
        ],
        inputs: [
            query: InT, key: InT, value: InT, alpha: float, beta: float,
            a_log: float, dt_bias: float, state: StT,
        ],
        outputs: [output: InT, next_state: StT],
        source: file "kernels/gated_delta_decode.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    fn paged_attention {
        name: "mirmir_paged_sdpa",
        templates: [
            T: dtype = bf16, QUERY_HEADS: int = 32, KV_HEADS: int = 16,
            PAGE_CAPACITY: int = 128, HEAD_DIM: int = 128,
            QK_PER_THREAD: int = 4, V_PER_THREAD: int = 4, PAGE_SIZE: int = 64,
        ],
        inputs: [
            queries: T, key_pages: T, value_pages: T, page_table: u32,
            page_dependency: u32, attention_scale: scalar<f32>,
        ],
        outputs: [output: T],
        source: file "kernels/paged_attention.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_library! {
    fn paged_attention_partial_library {
        name: "mirmir_paged_sdpa_partial",
        source: file "kernels/paged_attention/partial.metal",
    }
}

mirtal::metal_kernel! {
    fn paged_attention_reduce {
        name: "mirmir_paged_sdpa_reduce",
        templates: [
            T: dtype = bf16, HEAD_DIM: int = 128, BLOCKS: int = 64,
            REDUCTION_GROUPS: int = 32,
        ],
        inputs: [partials: T, sums: f32, maximums: f32],
        outputs: [output: T],
        source: file "kernels/paged_attention/reduce.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct Kernels {
    awq_repack: awq::AwqRepackKernel,
    bitsandbytes_4bit: bitsandbytes::BitsAndBytes4BitKernel,
    direct_fp8: direct_fp8::DirectFp8Kernel,
    direct_fp8_embedding: direct_fp8_embedding::DirectFp8EmbeddingKernel,
    gated_delta_gates: mirtal::MetalKernel<4, 2>,
    expert_group: expert_group::ExpertGroupKernel,
    expert_reduce: expert_reduce::ExpertReduceKernel,
    gated_delta_recurrence: mirtal::MetalKernel<6, 2>,
    gated_delta_decode: mirtal::MetalKernel<8, 2>,
    gptq: gptq::GptqKernels,
    paged_attention: mirtal::MetalKernel<6, 1>,
    paged_attention_batched: mirtal::MetalKernel<29, 1>,
    paged_attention_partial: mirtal::MetalLibrary,
    paged_attention_reduce: mirtal::MetalKernel<3, 1>,
    paged_kv: mirtal::MetalLibrary,
    mxfp4_gate_up: mirtal::MetalKernel<6, 1>,
    mxfp4_embedding: mxfp4_embedding::MxFp4EmbeddingKernel,
    mxfp4_gathered_linear: mxfp4_gathered_linear::MxFp4GatheredLinearKernel,
    mxfp4_linear: mxfp4_linear::MxFp4LinearKernel,
    mxfp4_down: mirtal::MetalKernel<6, 1>,
    mxfp4_split_gate_up: mirtal::MetalKernel<9, 1>,
    mxfp4_u32_down: mirtal::MetalKernel<6, 1>,
    nvfp4_convert: nvfp4_convert::NvFp4ConvertKernel,
    nvfp4_gathered_linear: nvfp4_gathered_linear::NvFp4GatheredLinearKernel,
    quantized_kv: quantized_kv::QuantizedKvKernels,
}

impl Kernels {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            awq_repack: awq::AwqRepackKernel::new()?,
            bitsandbytes_4bit: bitsandbytes::BitsAndBytes4BitKernel::new()?,
            direct_fp8: direct_fp8::DirectFp8Kernel::new()?,
            direct_fp8_embedding: direct_fp8_embedding::DirectFp8EmbeddingKernel::new()?,
            gated_delta_gates: gated_delta::new_gated_delta_gates_kernel()?,
            expert_group: expert_group::ExpertGroupKernel::new()?,
            expert_reduce: expert_reduce::ExpertReduceKernel::new()?,
            gated_delta_recurrence: gated_delta_recurrence()?,
            gated_delta_decode: gated_delta_decode()?,
            gptq: gptq::GptqKernels::new()?,
            paged_attention: paged_attention()?,
            paged_attention_batched: paged_attention::batched::new()?,
            paged_attention_partial: paged_attention_partial_library()?,
            paged_attention_reduce: paged_attention_reduce()?,
            paged_kv: paged_kv_library()?,
            mxfp4_gate_up: mxfp4_gate_up()?,
            mxfp4_embedding: mxfp4_embedding::MxFp4EmbeddingKernel::new()?,
            mxfp4_gathered_linear: mxfp4_gathered_linear::MxFp4GatheredLinearKernel::new()?,
            mxfp4_linear: mxfp4_linear::MxFp4LinearKernel::new()?,
            mxfp4_down: mxfp4_down()?,
            mxfp4_split_gate_up: mxfp4_split_gate_up()?,
            mxfp4_u32_down: mxfp4_u32_down()?,
            nvfp4_convert: nvfp4_convert::NvFp4ConvertKernel::new()?,
            nvfp4_gathered_linear: nvfp4_gathered_linear::NvFp4GatheredLinearKernel::new()?,
            quantized_kv: quantized_kv::QuantizedKvKernels::new()?,
        })
    }
}
