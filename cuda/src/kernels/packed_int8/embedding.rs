use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(PackedInt8EmbeddingKernel = "libmir_cuda_packed_int8_embedding_bf16"(
    selected: &DeviceBuffer<u32>, weight: &DeviceBuffer<i32>, scales: &DeviceBuffer<bf16>,
    output: &mut DeviceBuffer<bf16>, selected_start: u32, tokens: u32, vocab: u32, hidden: u32,
    output_scale: f32, bits: u32, group_size: u32,
));

#[derive(Clone, Copy, Debug)]
pub struct PackedInt8EmbeddingSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub output_scale: f32,
    bits: usize,
    group_size: usize,
}

impl PackedInt8EmbeddingSpec {
    pub const fn new(vocab: usize, hidden: usize, output_scale: f32) -> Result<Self> {
        Self::new_packed(vocab, hidden, output_scale, 8, hidden)
    }

    pub const fn new_packed(
        vocab: usize,
        hidden: usize,
        output_scale: f32,
        bits: usize,
        group_size: usize,
    ) -> Result<Self> {
        if vocab == 0
            || hidden == 0
            || !matches!(bits, 4 | 8)
            || group_size == 0
            || !hidden.is_multiple_of(group_size)
            || !(hidden * bits).is_multiple_of(32)
        {
            return Err(Error::InvalidQuantizedGemv(
                "packed integer embedding has invalid dimensions, bits, or group size",
            ));
        }
        if !output_scale.is_finite() {
            return Err(Error::InvalidQuantizedGemv(
                "packed integer embedding scale must be finite",
            ));
        }
        Ok(Self {
            vocab,
            hidden,
            output_scale,
            bits,
            group_size,
        })
    }
}

pub struct PackedInt8EmbeddingLaunch<'a> {
    pub selected: &'a DeviceBuffer<u32>,
    pub selected_start: usize,
    pub tokens: usize,
    pub weight: &'a DeviceBuffer<i32>,
    pub scales: &'a DeviceBuffer<bf16>,
    pub output: &'a mut DeviceBuffer<bf16>,
}

#[derive(Clone, Debug)]
pub struct PackedInt8Embedding {
    kernel: TypedKernel<PackedInt8EmbeddingKernel>,
    spec: PackedInt8EmbeddingSpec,
}

impl PackedInt8Embedding {
    pub fn compile(compiler: &Compiler, spec: PackedInt8EmbeddingSpec) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/packed_int8_bf16.cu");
        let module =
            compiler.compile(source, &CompileOptions { fast_math: true, ..Default::default() })?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        launch: &mut PackedInt8EmbeddingLaunch<'_>,
    ) -> Result<()> {
        let selected_end = launch
            .selected_start
            .checked_add(launch.tokens)
            .ok_or(Error::InvalidQuantizedGemv("packed integer selected range overflow"))?;
        require("packed integer selected tokens", selected_end, launch.selected.len())?;
        require(
            "packed integer embedding weight",
            product(self.spec.vocab, self.spec.hidden * self.spec.bits / 32)?,
            launch.weight.len(),
        )?;
        require(
            "packed integer embedding scales",
            product(self.spec.vocab, self.spec.hidden / self.spec.group_size)?,
            launch.scales.len(),
        )?;
        require(
            "packed integer embedding output",
            product(launch.tokens, self.spec.hidden)?,
            launch.output.len(),
        )?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.hidden.div_ceil(256))?, narrow(launch.tokens)?, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                launch.selected,
                launch.weight,
                launch.scales,
                &mut *launch.output,
                narrow(launch.selected_start)?,
                narrow(launch.tokens)?,
                narrow(self.spec.vocab)?,
                narrow(self.spec.hidden)?,
                self.spec.output_scale,
                narrow(self.spec.bits)?,
                narrow(self.spec.group_size)?,
            ),
        )?)
    }
}
