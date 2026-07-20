use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    InterpolateKernel = "libmir_cuda_spatial_merge_interpolate_bf16"(
        table: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
        grid_height: u32, grid_width: u32, source_side: u32, merge: u32, hidden: u32,
    )
);
cuda_export!(
    QkvSplitKernel = "libmir_cuda_spatial_merge_qkv_split_bf16"(
        input: &DeviceBuffer<bf16>, query: &mut DeviceBuffer<bf16>,
        key: &mut DeviceBuffer<bf16>, value: &mut DeviceBuffer<bf16>,
        tokens: u32, hidden: u32,
    )
);
cuda_export!(
    RopeKernel = "libmir_cuda_spatial_merge_rope_bf16"(
        input: &DeviceBuffer<bf16>, positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>, tokens: u32, heads: u32, head_dim: u32,
    )
);

#[derive(Clone, Debug)]
pub struct SpatialMergeKernels {
    interpolate: TypedKernel<InterpolateKernel>,
    split: TypedKernel<QkvSplitKernel>,
    rope: TypedKernel<RopeKernel>,
}

impl SpatialMergeKernels {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/vision_spatial_merge_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self {
            interpolate: module.kernel()?,
            split: module.kernel()?,
            rope: module.kernel()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn interpolate(
        &self,
        stream: &Stream,
        table: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        grid_height: usize,
        grid_width: usize,
        source_side: usize,
        merge: usize,
        hidden: usize,
    ) -> Result<()> {
        let tokens = product(grid_height, grid_width)?;
        require(
            "spatial position table",
            product(product(source_side, source_side)?, hidden)?,
            table.len(),
        )?;
        require("spatial interpolated positions", product(tokens, hidden)?, output.len())?;
        if source_side == 0
            || merge == 0
            || !grid_height.is_multiple_of(merge)
            || !grid_width.is_multiple_of(merge)
        {
            return Err(Error::InvalidVisionKernel("invalid spatial interpolation geometry"));
        }
        Ok(self.interpolate.launch(
            stream,
            launch(product(tokens, hidden)?)?,
            (
                table,
                output,
                narrow(grid_height)?,
                narrow(grid_width)?,
                narrow(source_side)?,
                narrow(merge)?,
                narrow(hidden)?,
            ),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split_qkv(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        query: &mut DeviceBuffer<bf16>,
        key: &mut DeviceBuffer<bf16>,
        value: &mut DeviceBuffer<bf16>,
        tokens: usize,
        hidden: usize,
    ) -> Result<()> {
        let elements = product(tokens, hidden)?;
        require("spatial qkv", product(elements, 3)?, input.len())?;
        require("spatial query", elements, query.len())?;
        require("spatial key", elements, key.len())?;
        require("spatial value", elements, value.len())?;
        Ok(self.split.launch(
            stream,
            launch(elements)?,
            (input, query, key, value, narrow(tokens)?, narrow(hidden)?),
        )?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        positions: &DeviceBuffer<u32>,
        output: &mut DeviceBuffer<bf16>,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        if !head_dim.is_multiple_of(8) || head_dim > 256 {
            return Err(Error::InvalidVisionKernel("invalid spatial-merge RoPE geometry"));
        }
        let elements = product(product(tokens, heads)?, head_dim)?;
        require("spatial RoPE input", elements, input.len())?;
        require("spatial RoPE positions", product(tokens, 2)?, positions.len())?;
        require("spatial RoPE output", elements, output.len())?;
        Ok(self.rope.launch(
            stream,
            launch(elements)?,
            (input, positions, output, narrow(tokens)?, narrow(heads)?, narrow(head_dim)?),
        )?)
    }
}

fn launch(elements: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(elements.div_ceil(256))?, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    })
}
