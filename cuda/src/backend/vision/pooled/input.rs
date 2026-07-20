use mircuda::{DeviceBuffer, DeviceElement, PinnedBuffer};
use models::{layout::PooledVisionConfig, vision::PooledPreprocessedImage};

use super::super::super::CudaBackend;
use crate::{Error, Result};

pub(super) struct PooledInput {
    pub patch_staging: PinnedBuffer<f32>,
    pub patches: DeviceBuffer<f32>,
    pub position_staging: PinnedBuffer<u32>,
    pub positions: DeviceBuffer<u32>,
    pub tokens: usize,
    pub patch_width: usize,
    pub pooled_tokens: usize,
}

impl PooledInput {
    pub(super) fn prepare(
        backend: &CudaBackend,
        config: &PooledVisionConfig,
        image: &PooledPreprocessedImage,
    ) -> Result<Self> {
        let tokens = image
            .grid_height
            .checked_mul(image.grid_width)
            .ok_or(Error::InvalidVisionKernel("pooled token count overflow"))?;
        let patch_width = config
            .patch_size
            .checked_mul(config.patch_size)
            .and_then(|area| area.checked_mul(3))
            .ok_or(Error::InvalidVisionKernel("pooled patch width overflow"))?;
        validate(image, tokens, patch_width, config.pooling_kernel_size)?;
        let (patch_staging, patches) = upload(backend, &image.patches)?;
        let (position_staging, positions) = upload(backend, &image.position_ids)?;
        Ok(Self {
            patch_staging,
            patches,
            position_staging,
            positions,
            tokens,
            patch_width,
            pooled_tokens: image.soft_tokens,
        })
    }
}

pub(super) fn validate(
    image: &PooledPreprocessedImage,
    tokens: usize,
    patch_width: usize,
    kernel: usize,
) -> Result<()> {
    if kernel == 0 {
        return Err(Error::InvalidVisionKernel("pooled kernel must be non-zero"));
    }
    let expected = tokens
        .checked_mul(patch_width)
        .ok_or(Error::InvalidVisionKernel("pooled patch payload overflow"))?;
    let positions = tokens
        .checked_mul(2)
        .ok_or(Error::InvalidVisionKernel("pooled position payload overflow"))?;
    let pooled = (image.grid_height / kernel) * (image.grid_width / kernel);
    if image.patches.len() != expected
        || image.position_ids.len() != positions
        || image.soft_tokens != pooled
        || !image.grid_height.is_multiple_of(kernel)
        || !image.grid_width.is_multiple_of(kernel)
    {
        Err(Error::InvalidVisionKernel("inconsistent pooled preprocessed image"))
    } else {
        Ok(())
    }
}

fn upload<T: DeviceElement>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<(PinnedBuffer<T>, DeviceBuffer<T>)> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok((host, device))
}
