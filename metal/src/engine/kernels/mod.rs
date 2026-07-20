mod gated_delta;
mod page_write;
mod paged_attention;
mod quantized_kv;

pub(super) use page_write::{PageWriteOptions, PreparedPageWrite};
pub(super) use quantized_kv::{PreparedQuantizedPageWrite, QuantizedPageWriteOptions};

use super::Result;

mirtal::metal_kernel! {
    fn gated_delta_gates {
        name: "mirmir_gated_delta_gates",
        templates: [HV: int = 16],
        inputs: [alpha: float, beta: float, a_log: float, dt_bias: float],
        outputs: [decay: f32, update: f32],
        source: file "kernels/gated_delta_gates.metal",
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
        templates: [T: dtype = bf16, HEAD_DIM: int = 128, BLOCKS: int = 64],
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
    gated_delta_gates: mirtal::MetalKernel<4, 2>,
    gated_delta_recurrence: mirtal::MetalKernel<6, 2>,
    gated_delta_decode: mirtal::MetalKernel<8, 2>,
    paged_attention: mirtal::MetalKernel<6, 1>,
    paged_attention_partial: mirtal::MetalLibrary,
    paged_attention_reduce: mirtal::MetalKernel<3, 1>,
    paged_kv: mirtal::MetalLibrary,
    quantized_kv: quantized_kv::QuantizedKvKernels,
}

impl Kernels {
    pub(super) fn new() -> Result<Self> {
        Ok(Self {
            gated_delta_gates: gated_delta_gates()?,
            gated_delta_recurrence: gated_delta_recurrence()?,
            gated_delta_decode: gated_delta_decode()?,
            paged_attention: paged_attention()?,
            paged_attention_partial: paged_attention_partial_library()?,
            paged_attention_reduce: paged_attention_reduce()?,
            paged_kv: paged_kv_library()?,
            quantized_kv: quantized_kv::QuantizedKvKernels::new()?,
        })
    }
}

fn template(name: &'static str, value: usize) -> Result<mirtal::TemplateArg> {
    Ok(mirtal::TemplateArg::int(name, i32::try_from(value)?))
}

pub(super) fn new_gated_delta_decode_kernel() -> Result<mirtal::MetalKernel<8, 2>> {
    Ok(gated_delta_decode()?)
}
