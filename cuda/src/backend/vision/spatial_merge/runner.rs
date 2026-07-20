use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use models::{layout::SpatialMergeVisionConfig, vision::SpatialMergePreprocessedImage};

use super::{
    input::{
        PatchProjectionLayout, SpatialInput, checked, patch_projection_layout, position_side,
        validate, vision_prefix,
    },
    layer::SpatialMergeLayer,
    primitives::{bf16, elementwise},
    scratch::SpatialMergeScratch,
};
use crate::{
    CudaTensor, CudaTensorSet, Error, Result,
    backend::{
        CudaBackend,
        vision::linear::{VisionLinear, required},
    },
    kernels::{SpatialMergeKernels, VisionElementwise},
};

#[derive(Debug)]
pub(super) struct SpatialMergeRunner {
    backend: CudaBackend,
    config: SpatialMergeVisionConfig,
    patch_staging: PinnedBuffer<f32>,
    patch_values: DeviceBuffer<f32>,
    _position_staging: PinnedBuffer<u32>,
    patch_layout: PatchProjectionLayout,
    positions: DeviceBuffer<u32>,
    patch_linear: VisionLinear,
    position_table: CudaTensor,
    embedding: VisionElementwise,
    conversion: VisionElementwise,
    kernels: SpatialMergeKernels,
    layers: Vec<SpatialMergeLayer>,
    merger_norm_weight: CudaTensor,
    merger_norm_bias: CudaTensor,
    merger_fc1: VisionLinear,
    merger_fc2: VisionLinear,
    merger_norm: VisionElementwise,
    merger_activation: VisionElementwise,
    patch_reorder: crate::kernels::VisionPatchLayout,
    scratch: SpatialMergeScratch,
    tokens: usize,
    patch_width: usize,
    soft_tokens: usize,
    source_side: usize,
    grid_height: usize,
    grid_width: usize,
}

impl SpatialMergeRunner {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &SpatialMergeVisionConfig,
        tensors: &CudaTensorSet,
        image: &SpatialMergePreprocessedImage,
    ) -> Result<Self> {
        let input = SpatialInput::prepare(backend, config, image)?;
        let tokens = input.tokens;
        let patch_width = input.patch_width;
        let prefix = vision_prefix(tensors)?;
        let patch_name = format!("{prefix}.patch_embed.proj.weight");
        let patch_layout = patch_projection_layout(&required(tensors, &patch_name)?, config)?;
        let kernels = SpatialMergeKernels::compile(&backend.inner.compiler)?;
        let merged_hidden = checked(
            config.hidden_size,
            checked(config.spatial_merge_size, config.spatial_merge_size)?,
        )?;
        let source_side = position_side(config.num_position_embeddings)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| {
                SpatialMergeLayer::new(
                    backend,
                    config,
                    tensors,
                    &prefix,
                    index,
                    tokens,
                    kernels.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let merger = format!("{prefix}.merger");
        Ok(Self {
            backend: backend.clone(),
            config: config.clone(),
            patch_staging: input.patch_staging,
            patch_values: input.patches,
            _position_staging: input.position_staging,
            patch_layout,
            positions: input.positions,
            patch_linear: VisionLinear::new_flattened(
                backend,
                tensors,
                &format!("{prefix}.patch_embed.proj"),
                tokens,
                patch_width,
                config.hidden_size,
            )?,
            position_table: required(tensors, &format!("{prefix}.pos_embed.weight"))?,
            embedding: elementwise(backend, tokens, config.hidden_size, 0.0)?,
            conversion: elementwise(backend, tokens, patch_width, 0.0)?,
            kernels,
            layers,
            merger_norm_weight: required(tensors, &format!("{merger}.norm.weight"))?,
            merger_norm_bias: required(tensors, &format!("{merger}.norm.bias"))?,
            merger_fc1: VisionLinear::new(
                backend,
                tensors,
                &format!("{merger}.linear_fc1"),
                input.soft_tokens,
                merged_hidden,
                merged_hidden,
                false,
            )?,
            merger_fc2: VisionLinear::new(
                backend,
                tensors,
                &format!("{merger}.linear_fc2"),
                input.soft_tokens,
                merged_hidden,
                config.output_hidden_size,
                false,
            )?,
            merger_norm: elementwise(backend, tokens, config.hidden_size, 1.0e-6)?,
            merger_activation: elementwise(backend, input.soft_tokens, merged_hidden, 0.0)?,
            patch_reorder: crate::kernels::VisionPatchLayout::compile(&backend.inner.compiler)?,
            scratch: SpatialMergeScratch::new(backend, config, tokens, input.soft_tokens)?,
            tokens,
            patch_width,
            soft_tokens: input.soft_tokens,
            source_side,
            grid_height: input.grid_height,
            grid_width: input.grid_width,
        })
    }

    pub(super) fn update_input(&mut self, image: &SpatialMergePreprocessedImage) -> Result<()> {
        validate(image, self.config.spatial_merge_size, self.tokens, self.patch_width)?;
        if image.grid_height != self.grid_height || image.grid_width != self.grid_width {
            return Err(Error::InvalidVisionKernel(
                "spatial-merge runner geometry differs from prepared input",
            ));
        }
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
        self.execute_merger()
    }

    pub(super) fn execute_input(&mut self) -> Result<()> {
        let stream = &self.backend.inner.stream;
        match self.patch_layout {
            PatchProjectionLayout::ChannelFirst => {
                self.conversion.convert(
                    stream,
                    &self.patch_values,
                    &mut self.scratch.patches,
                    1.0,
                    0.0,
                )?;
            },
            PatchProjectionLayout::ChannelLast => self.patch_reorder.cthw_to_thwc(
                stream,
                &self.patch_values,
                &mut self.scratch.patches,
                [
                    self.tokens,
                    self.config.in_channels,
                    self.config.temporal_patch_size,
                    checked(self.config.patch_size, self.config.patch_size)?,
                ],
            )?,
        }
        self.patch_linear.execute(&self.scratch.patches, &mut self.scratch.hidden_a)?;
        let table = bf16(&self.position_table)?;
        self.kernels.interpolate(
            stream,
            table,
            &mut self.scratch.normalized,
            self.grid_height,
            self.grid_width,
            self.source_side,
            self.config.spatial_merge_size,
            self.config.hidden_size,
        )?;
        self.embedding.add(
            stream,
            &self.scratch.hidden_a,
            &self.scratch.normalized,
            &mut self.scratch.hidden_b,
        )?;
        std::mem::swap(&mut self.scratch.hidden_a, &mut self.scratch.hidden_b);
        Ok(())
    }

    pub(super) fn execute_layer(&mut self, index: usize) -> Result<()> {
        let layer = self
            .layers
            .get_mut(index)
            .ok_or(Error::InvalidVisionKernel("spatial-merge layer index is out of range"))?;
        layer.execute(&mut self.scratch, &self.positions)
    }

    pub(super) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(super) fn output(&self) -> (DeviceBuffer<bf16>, usize, usize) {
        (self.scratch.output.clone(), self.soft_tokens, self.config.output_hidden_size)
    }

    pub(super) fn execute_merger(&mut self) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.merger_norm.layer_norm(
            stream,
            &self.scratch.hidden_a,
            bf16(&self.merger_norm_weight)?,
            bf16(&self.merger_norm_bias)?,
            &mut self.scratch.normalized,
        )?;
        self.merger_fc1.execute(&self.scratch.normalized, &mut self.scratch.hidden_b)?;
        self.merger_activation.gelu(
            stream,
            &self.scratch.hidden_b,
            &mut self.scratch.hidden_a,
            false,
        )?;
        self.merger_fc2.execute(&self.scratch.hidden_a, &mut self.scratch.output)
    }
}
