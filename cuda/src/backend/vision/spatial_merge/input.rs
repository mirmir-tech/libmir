use mircuda::{DeviceBuffer, PinnedBuffer};
use models::{layout::SpatialMergeVisionConfig, vision::SpatialMergePreprocessedImage};

use super::super::super::CudaBackend;
use crate::{CudaTensor, CudaTensorSet, Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PatchProjectionLayout {
    ChannelFirst,
    ChannelLast,
}

pub(super) struct SpatialInput {
    pub patch_staging: PinnedBuffer<f32>,
    pub patches: DeviceBuffer<f32>,
    pub position_staging: PinnedBuffer<u32>,
    pub positions: DeviceBuffer<u32>,
    pub tokens: usize,
    pub patch_width: usize,
    pub soft_tokens: usize,
    pub grid_height: usize,
    pub grid_width: usize,
}

impl SpatialInput {
    pub(super) fn prepare(
        backend: &CudaBackend,
        config: &SpatialMergeVisionConfig,
        image: &SpatialMergePreprocessedImage,
    ) -> Result<Self> {
        let tokens = checked(image.grid_height, image.grid_width)?;
        let patch_width = checked(
            checked(config.in_channels, config.temporal_patch_size)?,
            checked(config.patch_size, config.patch_size)?,
        )?;
        validate(image, config.spatial_merge_size, tokens, patch_width)?;
        let (patch_staging, patches) = upload(backend, &image.patches)?;
        let (position_staging, positions) =
            upload(backend, &position_ids(image, config.spatial_merge_size)?)?;
        Ok(Self {
            patch_staging,
            patches,
            position_staging,
            positions,
            tokens,
            patch_width,
            soft_tokens: image.soft_tokens,
            grid_height: image.grid_height,
            grid_width: image.grid_width,
        })
    }
}

pub(super) fn vision_prefix(tensors: &CudaTensorSet) -> Result<String> {
    ["model.visual", "vision_tower"]
        .into_iter()
        .find(|prefix| tensors.get(&format!("{prefix}.patch_embed.proj.weight")).is_some())
        .map(str::to_owned)
        .ok_or_else(|| Error::MissingTensor("spatial-merge patch projection".into()))
}

pub(super) fn patch_projection_layout(
    weight: &CudaTensor,
    config: &SpatialMergeVisionConfig,
) -> Result<PatchProjectionLayout> {
    let output = config.hidden_size;
    let channels = config.in_channels;
    let temporal = config.temporal_patch_size;
    let patch = config.patch_size;
    let flattened = checked(checked(channels, temporal)?, checked(patch, patch)?)?;
    match weight.shape() {
        [actual_output, width] if *actual_output == output && *width == flattened => {
            Ok(PatchProjectionLayout::ChannelFirst)
        },
        [actual_output, actual_channels, actual_temporal, height, width]
            if *actual_output == output
                && *actual_channels == channels
                && *actual_temporal == temporal
                && *height == patch
                && *width == patch =>
        {
            Ok(PatchProjectionLayout::ChannelFirst)
        },
        [actual_output, actual_temporal, height, width, actual_channels]
            if *actual_output == output
                && *actual_temporal == temporal
                && *height == patch
                && *width == patch
                && *actual_channels == channels =>
        {
            Ok(PatchProjectionLayout::ChannelLast)
        },
        _ => Err(Error::UnsupportedVisionContract(format!(
            "spatial patch projection shape {:?} does not match the parsed image geometry",
            weight.shape()
        ))),
    }
}

pub(super) fn position_side(count: usize) -> Result<usize> {
    let mut root = 0_usize;
    while (root + 1).checked_mul(root + 1).is_some_and(|square| square <= count) {
        root += 1;
    }
    if checked(root, root)? == count {
        Ok(root)
    } else {
        Err(Error::UnsupportedVisionContract(
            "spatial-merge position embedding count must be square".into(),
        ))
    }
}

pub(super) fn checked(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidVisionKernel("spatial-merge geometry overflow"))
}

pub(super) fn validate(
    image: &SpatialMergePreprocessedImage,
    merge: usize,
    tokens: usize,
    patch_width: usize,
) -> Result<()> {
    let expected_soft = checked(image.grid_height / merge, image.grid_width / merge)?;
    if image.grid_t != 1
        || !image.grid_height.is_multiple_of(merge)
        || !image.grid_width.is_multiple_of(merge)
        || image.patches.len() != checked(tokens, patch_width)?
        || image.soft_tokens != expected_soft
    {
        return Err(Error::InvalidVisionKernel("inconsistent spatial-merge preprocessed image"));
    }
    Ok(())
}

fn position_ids(image: &SpatialMergePreprocessedImage, merge: usize) -> Result<Vec<u32>> {
    let mut values = Vec::with_capacity(checked(checked(image.grid_height, image.grid_width)?, 2)?);
    for block_y in 0..image.grid_height / merge {
        for block_x in 0..image.grid_width / merge {
            for merge_y in 0..merge {
                for merge_x in 0..merge {
                    values.push(u32::try_from(block_y * merge + merge_y)?);
                    values.push(u32::try_from(block_x * merge + merge_x)?);
                }
            }
        }
    }
    Ok(values)
}

fn upload<T: mircuda::DeviceElement>(
    backend: &CudaBackend,
    values: &[T],
) -> Result<(PinnedBuffer<T>, DeviceBuffer<T>)> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok((host, device))
}
