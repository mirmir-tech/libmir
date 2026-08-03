mod quantize;

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

cuda_export!(PrepareBucketsKernel = "libmir_cuda_nvfp4_prepare_buckets"(
    selected: &DeviceBuffer<u32>, counts: &mut DeviceBuffer<u32>,
    offsets: &mut DeviceBuffer<u32>, scale_offsets: &mut DeviceBuffer<u32>,
    order: &mut DeviceBuffer<u32>,
    positions: &mut DeviceBuffer<u32>, indices: &mut DeviceBuffer<u32>,
    assignments: u32, experts: u32,
));
cuda_export!(QuantizeBucketsKernel = "libmir_cuda_nvfp4_quantize_buckets_bf16"(
    input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    order: &DeviceBuffer<u32>, offsets: &DeviceBuffer<u32>,
    scale_offsets: &DeviceBuffer<u32>,
    global_scales: &DeviceBuffer<f32>, packed: &mut DeviceBuffer<u8>,
    scales: &mut DeviceBuffer<u8>, assignments: u32, selected_count: u32,
    input_rows: u32, columns: u32, ranked: u32,
));
cuda_export!(QuantizeBucketPairKernel = "libmir_cuda_nvfp4_quantize_bucket_pair_bf16"(
    input: &DeviceBuffer<bf16>, selected: &DeviceBuffer<u32>,
    order: &DeviceBuffer<u32>, offsets: &DeviceBuffer<u32>,
    scale_offsets: &DeviceBuffer<u32>,
    left_globals: &DeviceBuffer<f32>, right_globals: &DeviceBuffer<f32>,
    left_packed: &mut DeviceBuffer<u8>, right_packed: &mut DeviceBuffer<u8>,
    left_scales: &mut DeviceBuffer<u8>, right_scales: &mut DeviceBuffer<u8>,
    assignments: u32, selected_count: u32, input_rows: u32,
    columns: u32,
));

#[derive(Clone, Debug)]
pub struct NvFp4BucketPreparation {
    prepare: TypedKernel<PrepareBucketsKernel>,
    quantize: TypedKernel<QuantizeBucketsKernel>,
    quantize_pair: TypedKernel<QuantizeBucketPairKernel>,
}

impl NvFp4BucketPreparation {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let source = cuda_kernel_file!("../../../../kernels/nvfp4_buckets.cu");
        let options = CompileOptions { fast_math: false, ..Default::default() };
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            prepare: module.kernel()?,
            quantize: module.kernel()?,
            quantize_pair: module.kernel()?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &self,
        stream: &Stream,
        selected: &DeviceBuffer<u32>,
        counts: &mut DeviceBuffer<u32>,
        offsets: &mut DeviceBuffer<u32>,
        scale_offsets: &mut DeviceBuffer<u32>,
        order: &mut DeviceBuffer<u32>,
        positions: &mut DeviceBuffer<u32>,
        indices: &mut DeviceBuffer<u32>,
        geometry: BucketGeometry,
    ) -> Result<()> {
        geometry.validate(selected, counts, offsets, scale_offsets, order, positions, indices)?;
        Ok(self.prepare.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: narrow(product(geometry.experts, 2 * size_of::<u32>())?)?,
            },
            (
                selected,
                counts,
                offsets,
                scale_offsets,
                order,
                positions,
                indices,
                narrow(geometry.assignments)?,
                narrow(geometry.experts)?,
            ),
        )?)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BucketGeometry {
    pub assignments: usize,
    pub experts: usize,
}

impl BucketGeometry {
    #[allow(clippy::too_many_arguments)]
    fn validate(
        self,
        selected: &DeviceBuffer<u32>,
        counts: &DeviceBuffer<u32>,
        offsets: &DeviceBuffer<u32>,
        scale_offsets: &DeviceBuffer<u32>,
        order: &DeviceBuffer<u32>,
        positions: &DeviceBuffer<u32>,
        indices: &DeviceBuffer<u32>,
    ) -> Result<()> {
        if self.assignments == 0 || self.experts == 0 {
            return Err(Error::InvalidNvFp4("invalid bucket geometry"));
        }
        require("bucket selections", self.assignments, selected.len())?;
        require("bucket counts", self.experts, counts.len())?;
        require("bucket offsets", self.experts, offsets.len())?;
        require("bucket scale offsets", self.experts, scale_offsets.len())?;
        require("bucket order", self.assignments, order.len())?;
        require("bucket positions", self.assignments, positions.len())?;
        require("bucket indices", self.experts, indices.len())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BucketQuantize {
    pub assignments: usize,
    pub experts: usize,
    pub selected: usize,
    pub input_rows: usize,
    pub columns: usize,
    pub ranked: bool,
}
