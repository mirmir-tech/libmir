use mircuda::{
    Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::{QkvPostprocessSpec, validate};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(BatchedQkvPostprocessKernel = "libmir_cuda_qkv_postprocess_batch_bf16"(
    query_input: &DeviceBuffer<bf16>, key_input: &DeviceBuffer<bf16>,
    value_input: &DeviceBuffer<bf16>, query_weight: &DeviceBuffer<bf16>,
    key_weight: &DeviceBuffer<bf16>, token_counts: &DeviceBuffer<u32>,
    query_output: &mut DeviceBuffer<bf16>, key_output: &mut DeviceBuffer<bf16>,
    value_output: &mut DeviceBuffer<bf16>, tokens: u32, query_heads: u32,
    kv_heads: u32, head_dim: u32, value_head_dim: u32, rotary_dim: u32,
    pairing_dim: u32, theta: f32, epsilon: f32, separate_inputs: u32, normalize_query: u32,
    normalize_key: u32, normalize_value: u32,
));

#[derive(Clone, Debug)]
pub struct BatchedQkvPostprocess {
    kernel: TypedKernel<BatchedQkvPostprocessKernel>,
    spec: QkvPostprocessSpec,
}

impl BatchedQkvPostprocess {
    pub(crate) fn compile(compiler: &Compiler, spec: QkvPostprocessSpec) -> Result<Self> {
        validate(spec)?;
        let source = cuda_kernel_file!("../../../kernels/qkv_postprocess_bf16.cu");
        let module = compiler.compile(source, &mircuda::CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute(
        &self,
        stream: &Stream,
        inputs: [&DeviceBuffer<bf16>; 3],
        separate: bool,
        query_weight: &DeviceBuffer<bf16>,
        key_weight: &DeviceBuffer<bf16>,
        token_counts: &DeviceBuffer<u32>,
        query_output: &mut DeviceBuffer<bf16>,
        key_output: &mut DeviceBuffer<bf16>,
        value_output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let query = product(self.spec.query_heads, self.spec.head_dim)?;
        let key = product(self.spec.kv_heads, self.spec.head_dim)?;
        let value = product(self.spec.kv_heads, self.spec.value_head_dim)?;
        let packed_width = query
            .checked_add(key)
            .and_then(|width| width.checked_add(value))
            .ok_or(Error::InvalidDecoderKernel("batched QKV packed width overflow"))?;
        if separate {
            for (label, width, input) in
                [("Q", query, inputs[0]), ("K", key, inputs[1]), ("V", value, inputs[2])]
            {
                require(label, product(self.spec.tokens, width)?, input.len())?;
            }
        } else {
            require(
                "batched QKV input",
                product(self.spec.tokens, packed_width)?,
                inputs[0].len(),
            )?;
        }
        require("batched Q positions", self.spec.tokens, token_counts.len())?;
        require("batched Q norm", self.spec.head_dim, query_weight.len())?;
        require("batched K norm", self.spec.head_dim, key_weight.len())?;
        require("batched Q output", product(self.spec.tokens, query)?, query_output.len())?;
        require("batched K output", product(self.spec.tokens, key)?, key_output.len())?;
        require("batched V output", product(self.spec.tokens, value)?, value_output.len())?;
        let heads = self
            .spec
            .query_heads
            .checked_add(self.spec.kv_heads.saturating_mul(2))
            .and_then(|heads| heads.checked_mul(self.spec.tokens))
            .ok_or(Error::InvalidDecoderKernel("batched QKV launch overflow"))?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(heads)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                inputs[0],
                inputs[1],
                inputs[2],
                query_weight,
                key_weight,
                token_counts,
                query_output,
                key_output,
                value_output,
                narrow(self.spec.tokens)?,
                narrow(self.spec.query_heads)?,
                narrow(self.spec.kv_heads)?,
                narrow(self.spec.head_dim)?,
                narrow(self.spec.value_head_dim)?,
                narrow(self.spec.rotary_dim)?,
                narrow(self.spec.pairing_dim)?,
                self.spec.theta,
                self.spec.epsilon,
                u32::from(separate),
                u32::from(self.spec.normalization.query),
                u32::from(self.spec.normalization.key),
                u32::from(self.spec.normalization.value),
            ),
        )?)
    }
}
