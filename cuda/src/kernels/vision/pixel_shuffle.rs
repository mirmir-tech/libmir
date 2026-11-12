//! Kernels of the pixel-shuffle encoder: the prefixed embedding with
//! positions added, the fold of patch blocks into tokens, and the
//! projector's RMS norm and squared rectifier.

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    EmbedKernel = "libmir_cuda_pixel_shuffle_embed_bf16"(
        patches: &DeviceBuffer<bf16>, table: &DeviceBuffer<bf16>, prefix: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>, rows: u32, cols: u32, grid: u32, prefix_tokens: u32,
        hidden: u32,
    )
);
cuda_export!(
    FoldKernel = "libmir_cuda_pixel_shuffle_fold_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, rows: u32, cols: u32,
        factor: u32, prefix_tokens: u32, hidden: u32,
    )
);
cuda_export!(
    RmsNormKernel = "libmir_cuda_pixel_shuffle_rms_norm_bf16"(
        input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        rows: u32, columns: u32, epsilon: f32,
    )
);
cuda_export!(
    ReluSquareKernel = "libmir_cuda_pixel_shuffle_relu_square_bf16"(
        input: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>, elements: u32,
    )
);

/// The patch grid and widths one image's kernels run over.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelShuffleGeometry {
    pub rows: usize,
    pub cols: usize,
    pub prefix_tokens: usize,
    pub hidden: usize,
    pub factor: usize,
}

impl PixelShuffleGeometry {
    pub fn tokens(&self) -> Result<usize> {
        product(self.rows, self.cols)?
            .checked_add(self.prefix_tokens)
            .ok_or(Error::InvalidVisionKernel("pixel-shuffle token count overflow"))
    }

    pub fn folded_tokens(&self) -> Result<usize> {
        product(self.rows / self.factor, self.cols / self.factor)
    }

    pub fn folded_width(&self) -> Result<usize> {
        product(product(self.factor, self.factor)?, self.hidden)
    }

    fn validate(&self) -> Result<()> {
        if self.rows == 0
            || self.cols == 0
            || self.hidden == 0
            || self.factor == 0
            || !self.rows.is_multiple_of(self.factor)
            || !self.cols.is_multiple_of(self.factor)
        {
            return Err(Error::InvalidVisionKernel("invalid pixel-shuffle geometry"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PixelShuffleKernels {
    embed: TypedKernel<EmbedKernel>,
    fold: TypedKernel<FoldKernel>,
    rms_norm: TypedKernel<RmsNormKernel>,
    relu_square: TypedKernel<ReluSquareKernel>,
}

impl PixelShuffleKernels {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/vision_pixel_shuffle_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            embed: module.kernel()?,
            fold: module.kernel()?,
            rms_norm: module.kernel()?,
            relu_square: module.kernel()?,
        })
    }

    /// `patches` are the projected patches, `table` the square position
    /// table of `grid`² rows, `prefix` the prefix tokens.
    pub fn embed(
        &self,
        stream: &Stream,
        geometry: PixelShuffleGeometry,
        (patches, table, prefix): (&DeviceBuffer<bf16>, &DeviceBuffer<bf16>, &DeviceBuffer<bf16>),
        grid: usize,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        geometry.validate()?;
        let hidden = geometry.hidden;
        require(
            "pixel-shuffle patches",
            product(product(geometry.rows, geometry.cols)?, hidden)?,
            patches.len(),
        )?;
        require(
            "pixel-shuffle position table",
            product(product(grid, grid)?, hidden)?,
            table.len(),
        )?;
        require("pixel-shuffle prefix", product(geometry.prefix_tokens, hidden)?, prefix.len())?;
        let elements = product(geometry.tokens()?, hidden)?;
        require("pixel-shuffle embedding", elements, output.len())?;
        if grid == 0 {
            return Err(Error::InvalidVisionKernel("empty pixel-shuffle position table"));
        }
        Ok(self.embed.launch(
            stream,
            launch(elements)?,
            (
                patches,
                table,
                prefix,
                output,
                narrow(geometry.rows)?,
                narrow(geometry.cols)?,
                narrow(grid)?,
                narrow(geometry.prefix_tokens)?,
                narrow(hidden)?,
            ),
        )?)
    }

    pub fn fold(
        &self,
        stream: &Stream,
        geometry: PixelShuffleGeometry,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        geometry.validate()?;
        require(
            "pixel-shuffle encoder output",
            product(geometry.tokens()?, geometry.hidden)?,
            input.len(),
        )?;
        let elements = product(geometry.folded_tokens()?, geometry.folded_width()?)?;
        require("pixel-shuffle folded tokens", elements, output.len())?;
        Ok(self.fold.launch(
            stream,
            launch(elements)?,
            (
                input,
                output,
                narrow(geometry.rows)?,
                narrow(geometry.cols)?,
                narrow(geometry.factor)?,
                narrow(geometry.prefix_tokens)?,
                narrow(geometry.hidden)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        rows: usize,
        columns: usize,
        epsilon: f32,
    ) -> Result<()> {
        require("pixel-shuffle norm input", product(rows, columns)?, input.len())?;
        require("pixel-shuffle norm weight", columns, weight.len())?;
        require("pixel-shuffle norm output", product(rows, columns)?, output.len())?;
        if rows == 0 || columns == 0 || !epsilon.is_finite() || epsilon < 0.0 {
            return Err(Error::InvalidVisionKernel("invalid pixel-shuffle norm geometry"));
        }
        Ok(self.rms_norm.launch(
            stream,
            LaunchConfig {
                grid: (narrow(rows)?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, weight, output, narrow(rows)?, narrow(columns)?, epsilon),
        )?)
    }

    pub fn relu_square(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        elements: usize,
    ) -> Result<()> {
        require("pixel-shuffle activation input", elements, input.len())?;
        require("pixel-shuffle activation output", elements, output.len())?;
        Ok(self.relu_square.launch(
            stream,
            launch(elements)?,
            (input, output, narrow(elements)?),
        )?)
    }
}

fn launch(elements: usize) -> Result<LaunchConfig> {
    let threads = 256_usize;
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(threads))?, 1, 1),
        block: (narrow(threads)?, 1, 1),
        shared_memory_bytes: 0,
    })
}
