use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use models::{layout::PixelShuffleVisionConfig, vision::PixelShufflePreprocessedImage};

use super::{
    super::{
        super::CudaBackend,
        linear::{VisionLinear, required},
    },
    layer::PixelShuffleLayer,
    projector::PixelShuffleProjector,
    scratch::{PixelShuffleScratch, bf16, elements},
};
use crate::{
    CudaTensor, CudaTensorSet, Error, Result,
    kernels::{
        PixelShuffleGeometry, PixelShuffleKernels, SpatialMergeKernels, VisionElementwise,
        VisionElementwiseSpec,
    },
};

/// The encoder for one patch grid: its linears and attention are sized to
/// the grid's token count, so a new grid needs a new runner.
#[derive(Debug)]
pub(super) struct PixelShuffleRunner {
    backend: CudaBackend,
    config: PixelShuffleVisionConfig,
    geometry: PixelShuffleGeometry,
    patch_width: usize,
    patch_staging: PinnedBuffer<f32>,
    patch_values: DeviceBuffer<f32>,
    conversion: VisionElementwise,
    patch_linear: VisionLinear,
    position_table: CudaTensor,
    prefix: CudaTensor,
    kernels: PixelShuffleKernels,
    layers: Vec<PixelShuffleLayer>,
    projector: PixelShuffleProjector,
    scratch: PixelShuffleScratch,
}

impl PixelShuffleRunner {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &PixelShuffleVisionConfig,
        tensors: &CudaTensorSet,
        image: &PixelShufflePreprocessedImage,
    ) -> Result<Self> {
        let geometry = geometry(config, image);
        let patch_width =
            elements(config.num_channels, elements(config.patch_size, config.patch_size)?)?;
        validate(image, geometry, patch_width)?;
        let patches = elements(geometry.rows, geometry.cols)?;
        let tokens = geometry.tokens()?;
        let encoder = models::weights::pixel_shuffle_encoder_tensor;
        let compiler = &backend.inner.compiler;
        let split = SpatialMergeKernels::compile(compiler)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                PixelShuffleLayer::new(backend, config, tensors, index, tokens, split.clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let mut patch_staging = backend.inner.context.allocate_pinned(image.patches.len())?;
        patch_staging.copy_from_slice(&image.patches)?;
        let mut patch_values =
            backend.inner.pool.allocate(&backend.inner.stream, image.patches.len())?;
        backend.inner.stream.copy_to_device(&mut patch_staging, &mut patch_values)?;
        Ok(Self {
            backend: backend.clone(),
            config: config.clone(),
            geometry,
            patch_width,
            patch_staging,
            patch_values,
            conversion: VisionElementwise::compile(
                compiler,
                VisionElementwiseSpec {
                    rows: patches,
                    columns: patch_width,
                    epsilon: 0.0,
                },
            )?,
            patch_linear: VisionLinear::new(
                backend,
                tensors,
                &encoder("patch_generator.embedder"),
                patches,
                patch_width,
                config.hidden_size,
                false,
            )?,
            position_table: required(tensors, &encoder("patch_generator.pos_embed"))?,
            prefix: required(tensors, &encoder("patch_generator.cls_token.token"))?,
            kernels: PixelShuffleKernels::compile(compiler)?,
            layers,
            projector: PixelShuffleProjector::new(backend, config, tensors, geometry)?,
            scratch: PixelShuffleScratch::new(backend, config, geometry, patch_width)?,
        })
    }

    pub(super) fn update_input(&mut self, image: &PixelShufflePreprocessedImage) -> Result<()> {
        if geometry(&self.config, image) != self.geometry {
            return Err(Error::InvalidVisionKernel(
                "pixel-shuffle runner geometry differs from the image",
            ));
        }
        validate(image, self.geometry, self.patch_width)?;
        self.patch_staging.copy_from_slice(&image.patches)?;
        self.backend
            .inner
            .stream
            .copy_to_device(&mut self.patch_staging, &mut self.patch_values)?;
        Ok(())
    }

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn execute(&mut self) -> Result<()> {
        self.execute_input()?;
        for index in 0..self.layers.len() {
            self.execute_layer(index)?;
        }
        self.execute_projector()
    }

    pub(super) fn execute_input(&mut self) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.conversion
            .convert(stream, &self.patch_values, &mut self.scratch.patches, 1.0, 0.0)?;
        self.patch_linear.execute(&self.scratch.patches, &mut self.scratch.projected)?;
        self.kernels.embed(
            stream,
            self.geometry,
            (&self.scratch.projected, bf16(&self.position_table)?, bf16(&self.prefix)?),
            self.config.position_grid,
            &mut self.scratch.hidden_a,
        )
    }

    pub(super) fn execute_layer(&mut self, index: usize) -> Result<()> {
        self.layers
            .get_mut(index)
            .ok_or(Error::InvalidVisionKernel("pixel-shuffle layer index is out of range"))?
            .execute(&mut self.scratch)
    }

    pub(super) fn execute_projector(&mut self) -> Result<()> {
        self.projector.execute(
            &self.backend.inner.stream,
            &self.kernels,
            self.geometry,
            &mut self.scratch,
        )
    }

    pub(super) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(super) fn output(&self) -> Result<(DeviceBuffer<bf16>, usize, usize)> {
        Ok((
            self.scratch.output.clone(),
            self.geometry.folded_tokens()?,
            self.config.output_hidden_size,
        ))
    }
}

pub(super) fn geometry(
    config: &PixelShuffleVisionConfig,
    image: &PixelShufflePreprocessedImage,
) -> PixelShuffleGeometry {
    PixelShuffleGeometry {
        rows: image.grid_height,
        cols: image.grid_width,
        prefix_tokens: config.prefix_tokens,
        hidden: config.hidden_size,
        factor: config.shuffle_factor,
    }
}

fn validate(
    image: &PixelShufflePreprocessedImage,
    geometry: PixelShuffleGeometry,
    patch_width: usize,
) -> Result<()> {
    if image.patches.len() != elements(elements(geometry.rows, geometry.cols)?, patch_width)?
        || image.soft_tokens != geometry.folded_tokens()?
    {
        return Err(Error::InvalidVisionKernel("inconsistent pixel-shuffle preprocessed image"));
    }
    Ok(())
}
