use super::{
    Kernels, Result, awq, bitsandbytes, direct_fp8, direct_fp8_embedding, expert_group,
    expert_reduce, gated_delta, gated_delta_decode, gated_delta_recurrence, gptq, mxfp4_down,
    mxfp4_embedding, mxfp4_gate_up, mxfp4_gathered_linear, mxfp4_linear, mxfp4_split_gate_up,
    mxfp4_u32_down, nvfp4_convert, nvfp4_gathered_linear, paged_attention,
    paged_attention_partial_library, paged_attention_reduce, paged_kv_library, quantized_kv,
};

impl Kernels {
    pub(crate) fn new() -> Result<Self> {
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
