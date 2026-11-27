use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};

use crate::Result;

pub(super) struct Runtime {
    pub context: Context,
    pub compiler: Compiler,
    pub pool: MemoryPool,
    pub stream: Stream,
}

impl Runtime {
    pub fn new() -> Result<Self> {
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

    pub fn copy<T: DeviceElement + Copy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = self.pool.allocate::<T>(&self.stream, values.len())?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        self.stream.synchronize()?;
        Ok(device)
    }

    pub fn copy_f32(&self, values: &[f32]) -> Result<DeviceBuffer<bf16>> {
        self.copy(&values.iter().copied().map(bf16::from_f32).collect::<Vec<_>>())
    }

    pub fn read<T: DeviceElement + Copy>(&self, device: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned::<T>(device.len())?;
        self.stream.copy_to_host(device, &mut host)?;
        self.stream.synchronize()?;
        Ok(host.to_vec()?)
    }
}

impl Runtime {
    /// Mean microseconds of `operation` over `cycles` warm launches.
    pub fn measure(&self, cycles: u16, mut operation: impl FnMut() -> Result<()>) -> Result<f32> {
        for _ in 0..8 {
            operation()?;
        }
        self.stream.synchronize()?;
        let started = self.context.create_event(true)?;
        let completed = self.context.create_event(true)?;
        started.record(&self.stream)?;
        for _ in 0..cycles {
            operation()?;
        }
        completed.record(&self.stream)?;
        completed.synchronize()?;
        Ok(started.elapsed_ms(&completed)? * 1_000.0 / f32::from(cycles))
    }
}

/// Deterministic uniform values in `[-scale, scale)`, rounded through BF16 so
/// the reference sees exactly what the device reads.
pub(super) struct Values(u64);

impl Values {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn take(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = f32::from(u16::try_from(self.0 >> 48).unwrap_or(0)) / 32_768.0 - 1.0;
                bf16::from_f32(unit * scale).to_f32()
            })
            .collect()
    }
}
