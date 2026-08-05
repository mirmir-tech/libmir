use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, MemoryPool, Stream, TypedKernel,
    cuda_export, cuda_kernel_file,
};

use crate::{Result, kernels::geometry::narrow};

cuda_export!(CacheEvictionKernel = "libmir_cuda_tuning_cache_evict"(
    values: &mut DeviceBuffer<u8>, elements: u32,
));

const EVICTION_BYTES: usize = 64 * 1024 * 1024;
const VECTOR_BYTES: usize = 16;

#[derive(Debug)]
pub struct CacheEviction {
    kernel: TypedKernel<CacheEvictionKernel>,
    values: DeviceBuffer<u8>,
}

impl CacheEviction {
    pub(crate) fn compile(compiler: &Compiler, pool: &MemoryPool, stream: &Stream) -> Result<Self> {
        let module = compiler.compile(
            cuda_kernel_file!("../../../kernels/tuning_cache.cu"),
            &CompileOptions {
                fast_math: false,
                ..CompileOptions::default()
            },
        )?;
        Ok(Self {
            kernel: module.kernel()?,
            values: pool.allocate_zeroed(stream, EVICTION_BYTES)?,
        })
    }

    pub(crate) fn execute(&self, stream: &Stream) -> Result<()> {
        let elements = EVICTION_BYTES / VECTOR_BYTES;
        let threads = 256_usize;
        let mut values = self.values.clone();
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(threads))?, 1, 1),
                block: (narrow(threads)?, 1, 1),
                shared_memory_bytes: 0,
            },
            (&mut values, narrow(elements)?),
        )?)
    }
}
