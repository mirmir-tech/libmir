use mircuda::{DeviceBuffer, DeviceElement};

use super::{CudaBackend, Result};

pub(in crate::backend::attention) fn read<T: DeviceElement>(
    backend: &CudaBackend,
    source: &DeviceBuffer<T>,
) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
