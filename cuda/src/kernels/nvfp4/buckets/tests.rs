use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};

use super::*;
use crate::{
    GatedActivation as BackendGatedActivation,
    kernels::{ElementwiseBf16, GatedActivation as KernelGatedActivation, scale_elements},
};

struct Runtime {
    context: Context,
    compiler: Compiler,
    pool: MemoryPool,
    stream: Stream,
}

impl Runtime {
    fn new() -> Result<Self> {
        let driver = Driver::initialize()?;
        let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
        let context = driver.create_context(device)?;
        let stream = context.create_stream()?;
        let pool = context.default_memory_pool()?;
        let compiler = Compiler::with_include_paths(
            context.clone(),
            [PathBuf::from("/usr/local/cuda/include")],
        )?;
        Ok(Self { context, compiler, pool, stream })
    }

    fn copy<T: DeviceElement + Copy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = self.context.allocate_pinned(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = self.pool.allocate(&self.stream, values.len())?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    }

    fn read<T: DeviceElement + Copy>(&self, device: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned(device.len())?;
        self.stream.copy_to_host(device, &mut host)?;
        Ok(host.to_vec()?)
    }
}

#[test]
fn gated_quantization_matches_split_bucket_path() -> Result<()> {
    const TOKENS: usize = 3;
    const TOP_K: usize = 2;
    const EXPERTS: usize = 3;
    const COLUMNS: usize = 64;
    let runtime = Runtime::new()?;
    let assignments = TOKENS * TOP_K;
    let values = |modulus: u16, shift: f32| {
        (0..assignments * COLUMNS)
            .map(|index| {
                Ok(bf16::from_f32(f32::from(u16::try_from(index)? % modulus) / 16.0 - shift))
            })
            .collect::<Result<Vec<_>>>()
    };
    let gate = runtime.copy(&values(37, 1.0)?)?;
    let up = runtime.copy(&values(29, 0.75)?)?;
    let selected = runtime.copy(&[0_u32, 1, 2, 0, 1, 2])?;
    let globals = runtime.copy(&[0.25_f32, 0.5, 1.0])?;
    let mut counts = runtime.pool.allocate_zeroed(&runtime.stream, EXPERTS)?;
    let mut offsets = runtime.pool.allocate_zeroed(&runtime.stream, EXPERTS)?;
    let mut scale_offsets = runtime.pool.allocate_zeroed(&runtime.stream, EXPERTS)?;
    let mut order = runtime.pool.allocate_zeroed(&runtime.stream, assignments)?;
    let mut positions = runtime.pool.allocate_zeroed(&runtime.stream, assignments)?;
    let mut indices = runtime.pool.allocate_zeroed(&runtime.stream, EXPERTS)?;
    let bucket = NvFp4BucketPreparation::compile(&runtime.compiler)?;
    bucket.prepare(
        &runtime.stream,
        &selected,
        &mut counts,
        &mut offsets,
        &mut scale_offsets,
        &mut order,
        &mut positions,
        &mut indices,
        BucketGeometry { assignments, experts: EXPERTS },
    )?;
    let geometry = BucketQuantize {
        assignments,
        experts: EXPERTS,
        selected: TOP_K,
        input_rows: assignments,
        columns: COLUMNS,
        ranked: true,
    };
    let scale_count = scale_elements(EXPERTS * 128, COLUMNS)?;
    let mut intermediate = runtime.pool.allocate(&runtime.stream, assignments * COLUMNS)?;
    ElementwiseBf16::compile(&runtime.compiler, assignments * COLUMNS)?.gated(
        &runtime.stream,
        &gate,
        &up,
        &mut intermediate,
        KernelGatedActivation::Silu,
    )?;
    let mut expected = runtime.pool.allocate_zeroed(&runtime.stream, assignments * COLUMNS / 2)?;
    let mut expected_scales = runtime.pool.allocate_zeroed(&runtime.stream, scale_count)?;
    bucket.quantize(
        &runtime.stream,
        &intermediate,
        &selected,
        &order,
        &offsets,
        &scale_offsets,
        &globals,
        &mut expected,
        &mut expected_scales,
        geometry,
    )?;
    let mut actual = runtime.pool.allocate_zeroed(&runtime.stream, assignments * COLUMNS / 2)?;
    let mut actual_scales = runtime.pool.allocate_zeroed(&runtime.stream, scale_count)?;
    bucket.quantize_gated(
        &runtime.stream,
        &gate,
        &up,
        &selected,
        &order,
        &offsets,
        &scale_offsets,
        &globals,
        &mut actual,
        &mut actual_scales,
        geometry,
        BackendGatedActivation::Silu,
    )?;
    assert_eq!(runtime.read(&actual)?, runtime.read(&expected)?);
    assert_eq!(runtime.read(&actual_scales)?, runtime.read(&expected_scales)?);
    Ok(())
}
