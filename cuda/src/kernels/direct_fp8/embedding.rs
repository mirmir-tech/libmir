use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::{DirectFp8Format, DirectFp8Scale, scale_geometry};
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(F32ScaleKernel = "libmir_cuda_direct_fp8_embedding_f32_scale"(
    weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<f32>, selected: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, selected_start: u32, tokens: u32, vocab: u32, hidden: u32,
    scale_rows: u32, scale_columns: u32, scale_row_size: u32, scale_group_size: u32,
    inverse_scale: u32, output_scale: f32, e5m2: u32,
));
cuda_export!(Bf16ScaleKernel = "libmir_cuda_direct_fp8_embedding_bf16_scale"(
    weight: &DeviceBuffer<u8>, scales: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    output: &mut DeviceBuffer<bf16>, selected_start: u32, tokens: u32, vocab: u32, hidden: u32,
    scale_rows: u32, scale_columns: u32, scale_row_size: u32, scale_group_size: u32,
    inverse_scale: u32, output_scale: f32, e5m2: u32,
));

#[derive(Clone, Copy, Debug)]
pub struct DirectFp8EmbeddingSpec {
    pub format: DirectFp8Format,
    pub vocab: usize,
    pub hidden: usize,
    pub scale: DirectFp8Scale,
    pub inverse_scale: bool,
    pub output_scale: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct DirectFp8EmbeddingBatch<'a> {
    pub selected: &'a DeviceBuffer<u32>,
    pub selected_start: usize,
    pub tokens: usize,
}

#[derive(Clone, Debug)]
pub struct DirectFp8Embedding {
    f32_scale: TypedKernel<F32ScaleKernel>,
    bf16_scale: TypedKernel<Bf16ScaleKernel>,
    spec: DirectFp8EmbeddingSpec,
}

impl DirectFp8Embedding {
    pub fn compile(compiler: &Compiler, spec: DirectFp8EmbeddingSpec) -> Result<Self> {
        if spec.vocab == 0
            || spec.hidden == 0
            || !spec.hidden.is_multiple_of(4)
            || !spec.output_scale.is_finite()
            || spec.output_scale <= 0.0
        {
            return Err(Error::InvalidDecoderKernel("invalid direct FP8 embedding geometry"));
        }
        let _ = scale_geometry(spec.scale, spec.vocab, spec.hidden)?;
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/direct_fp8_embedding.cu"),
            &CompileOptions {
                fast_math: false,
                ..CompileOptions::default()
            },
        )?;
        Ok(Self {
            f32_scale: module.kernel()?,
            bf16_scale: module.kernel()?,
            spec,
        })
    }

    pub fn execute_f32_scales(
        &self,
        stream: &Stream,
        weight: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<f32>,
        batch: DirectFp8EmbeddingBatch<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(weight, scales.len(), batch, output)?;
        let (scale_rows, scale_columns, scale_row_size, scale_group_size) = self.geometry()?;
        Ok(self.f32_scale.launch(
            stream,
            self.launch(batch.tokens)?,
            (
                weight,
                scales,
                batch.selected,
                output,
                narrow(batch.selected_start)?,
                narrow(batch.tokens)?,
                narrow(self.spec.vocab)?,
                narrow(self.spec.hidden)?,
                narrow(scale_rows)?,
                narrow(scale_columns)?,
                narrow(scale_row_size)?,
                narrow(scale_group_size)?,
                u32::from(self.spec.inverse_scale),
                self.spec.output_scale,
                u32::from(self.spec.format == DirectFp8Format::E5M2),
            ),
        )?)
    }

    pub fn execute_bf16_scales(
        &self,
        stream: &Stream,
        weight: &DeviceBuffer<u8>,
        scales: &DeviceBuffer<bf16>,
        batch: DirectFp8EmbeddingBatch<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.validate(weight, scales.len(), batch, output)?;
        let (scale_rows, scale_columns, scale_row_size, scale_group_size) = self.geometry()?;
        Ok(self.bf16_scale.launch(
            stream,
            self.launch(batch.tokens)?,
            (
                weight,
                scales,
                batch.selected,
                output,
                narrow(batch.selected_start)?,
                narrow(batch.tokens)?,
                narrow(self.spec.vocab)?,
                narrow(self.spec.hidden)?,
                narrow(scale_rows)?,
                narrow(scale_columns)?,
                narrow(scale_row_size)?,
                narrow(scale_group_size)?,
                u32::from(self.spec.inverse_scale),
                self.spec.output_scale,
                u32::from(self.spec.format == DirectFp8Format::E5M2),
            ),
        )?)
    }

    fn validate<T: mircuda::DeviceElement>(
        &self,
        weight: &DeviceBuffer<u8>,
        scales: usize,
        batch: DirectFp8EmbeddingBatch<'_>,
        output: &DeviceBuffer<T>,
    ) -> Result<()> {
        if batch.tokens == 0 {
            return Err(Error::InvalidDecoderKernel("direct FP8 embedding batch is empty"));
        }
        require(
            "direct FP8 embedding weight",
            product(self.spec.vocab, self.spec.hidden)?,
            weight.len(),
        )?;
        let (scale_rows, scale_columns, _, _) = self.geometry()?;
        require("direct FP8 embedding scales", product(scale_rows, scale_columns)?, scales)?;
        let required = batch
            .selected_start
            .checked_add(batch.tokens)
            .ok_or(Error::InvalidDecoderKernel("direct FP8 embedding token range overflow"))?;
        require("direct FP8 embedding tokens", required, batch.selected.len())?;
        require(
            "direct FP8 embedding output",
            product(batch.tokens, self.spec.hidden)?,
            output.len(),
        )
    }

    fn geometry(&self) -> Result<(usize, usize, usize, usize)> {
        scale_geometry(self.spec.scale, self.spec.vocab, self.spec.hidden)
    }

    fn launch(&self, tokens: usize) -> Result<LaunchConfig> {
        let elements = product(tokens, self.spec.hidden)?;
        Ok(LaunchConfig {
            grid: (narrow(elements.div_ceil(256))?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        })
    }
}
