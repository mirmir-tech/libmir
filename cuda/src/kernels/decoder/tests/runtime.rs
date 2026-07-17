use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream};

use super::Result;

pub(super) struct Runtime {
    context: Context,
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

    pub fn read<T: DeviceElement + Copy>(&self, device: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned::<T>(device.len())?;
        self.stream.copy_to_host(device, &mut host)?;
        Ok(host.to_vec()?)
    }

    pub fn measure(&self, cycles: usize, mut operation: impl FnMut() -> Result<()>) -> Result<f32> {
        for _ in 0..16 {
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
        Ok(started.elapsed_ms(&completed)? * 1_000.0 / f32::from(u16::try_from(cycles)?))
    }
}
