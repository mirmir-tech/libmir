use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use models::{layout::PooledVisionConfig, vision::PooledPreprocessedImage};

use super::{
    super::{
        super::CudaBackend,
        linear::{VisionLinear, required},
    },
    input::{PooledInput, validate},
    layer::PooledLayer,
    scratch::PooledScratch,
};
use crate::{
    CudaTensor, CudaTensorSet, Error, Result,
    kernels::{RmsNormUnit, VisionElementwise, VisionElementwiseSpec, VisionPool, VisionPoolSpec},
};

#[derive(Debug)]
pub(super) struct PooledRunner {
    backend: CudaBackend,
    config: PooledVisionConfig,
    patch_staging: PinnedBuffer<f32>,
    patch_values: DeviceBuffer<f32>,
    position_staging: PinnedBuffer<u32>,
    positions: DeviceBuffer<u32>,
    patch_bf16: DeviceBuffer<bf16>,
    conversion: VisionElementwise,
    patch_linear: VisionLinear,
    position_table: CudaTensor,
    embedding: VisionElementwise,
    layers: Vec<PooledLayer>,
    pool: VisionPool,
    standardization: Option<(CudaTensor, CudaTensor)>,
    pooled_norm: RmsNormUnit,
    output_linear: VisionLinear,
    scratch: PooledScratch,
    tokens: usize,
    patch_width: usize,
    pooled_tokens: usize,
}

impl PooledRunner {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &PooledVisionConfig,
        tensors: &CudaTensorSet,
        image: &PooledPreprocessedImage,
    ) -> Result<Self> {
        let input = PooledInput::prepare(backend, config, image)?;
        let tokens = input.tokens;
        let patch_width = input.patch_width;
        let pooled_tokens = input.pooled_tokens;
        let patch_elements = input.patches.len();
        let eps = config.rms_norm_eps.to_string().parse::<f32>()?;
        let compiler = &backend.inner.compiler;
        let layers = (0..config.num_hidden_layers)
            .map(|index| PooledLayer::new(backend, config, tensors, index, tokens))
            .collect::<Result<Vec<_>>>()?;
        let standardization = if config.standardize {
            Some((
                required(tensors, "model.vision_tower.std_bias")?,
                required(tensors, "model.vision_tower.std_scale")?,
            ))
        } else {
            None
        };
        Ok(Self {
            backend: backend.clone(),
            config: config.clone(),
            patch_staging: input.patch_staging,
            patch_values: input.patches,
            position_staging: input.position_staging,
            positions: input.positions,
            patch_bf16: backend.inner.pool.allocate(&backend.inner.stream, patch_elements)?,
            conversion: VisionElementwise::compile(
                compiler,
                VisionElementwiseSpec {
                    rows: tokens,
                    columns: patch_width,
                    epsilon: 0.0,
                },
            )?,
            patch_linear: VisionLinear::new(
                backend,
                tensors,
                "model.vision_tower.patch_embedder.input_proj",
                tokens,
                patch_width,
                config.hidden_size,
                false,
            )?,
            position_table: required(
                tensors,
                "model.vision_tower.patch_embedder.position_embedding_table",
            )?,
            embedding: VisionElementwise::compile(
                compiler,
                VisionElementwiseSpec {
                    rows: tokens,
                    columns: config.hidden_size,
                    epsilon: eps,
                },
            )?,
            layers,
            pool: VisionPool::compile(
                compiler,
                VisionPoolSpec {
                    grid_height: image.grid_height,
                    grid_width: image.grid_width,
                    hidden: config.hidden_size,
                    kernel: config.pooling_kernel_size,
                },
            )?,
            standardization,
            pooled_norm: RmsNormUnit::compile(compiler, pooled_tokens, config.hidden_size, eps)?,
            output_linear: VisionLinear::new(
                backend,
                tensors,
                "model.embed_vision.embedding_projection",
                pooled_tokens,
                config.hidden_size,
                config.output_hidden_size,
                false,
            )?,
            scratch: PooledScratch::new(backend, config, tokens, pooled_tokens)?,
            tokens,
            patch_width,
            pooled_tokens,
        })
    }

    pub(super) fn update_input(&mut self, image: &PooledPreprocessedImage) -> Result<()> {
        validate(image, self.tokens, self.patch_width, self.config.pooling_kernel_size)?;
        self.patch_staging.copy_from_slice(&image.patches)?;
        self.position_staging.copy_from_slice(&image.position_ids)?;
        let stream = &self.backend.inner.stream;
        stream.copy_to_device(&mut self.patch_staging, &mut self.patch_values)?;
        stream.copy_to_device(&mut self.position_staging, &mut self.positions)?;
        Ok(())
    }

    pub(super) fn execute_input(&mut self) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.conversion
            .convert(stream, &self.patch_values, &mut self.patch_bf16, 2.0, -1.0)?;
        self.patch_linear.execute(&self.patch_bf16, &mut self.scratch.hidden_a)?;
        let position_table = self.position_table.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: self.position_table.name().into(),
            expected: "BF16",
        })?;
        self.embedding.add_positions(
            stream,
            &self.scratch.hidden_a,
            position_table,
            &self.positions,
            self.config.position_embedding_size,
            &mut self.scratch.hidden_b,
        )?;
        std::mem::swap(&mut self.scratch.hidden_a, &mut self.scratch.hidden_b);
        Ok(())
    }

    pub(super) fn execute_layer(&mut self, index: usize) -> Result<()> {
        let layer = self
            .layers
            .get_mut(index)
            .ok_or(Error::InvalidVisionKernel("pooled layer index is out of range"))?;
        layer.execute(&mut self.scratch, &self.positions)
    }

    pub(super) fn execute_output(&mut self) -> Result<()> {
        let stream = &self.backend.inner.stream;
        self.pool.execute(stream, &self.scratch.hidden_a, &mut self.scratch.pooled_a)?;
        let normalized = if let Some((bias, scale)) = self.standardization.as_ref() {
            self.pool.standardize_tensors(
                stream,
                &self.scratch.pooled_a,
                bias,
                scale,
                &mut self.scratch.pooled_b,
            )?;
            self.pooled_norm
                .execute(stream, &self.scratch.pooled_b, &mut self.scratch.pooled_a)?;
            &self.scratch.pooled_a
        } else {
            self.pooled_norm
                .execute(stream, &self.scratch.pooled_a, &mut self.scratch.pooled_b)?;
            &self.scratch.pooled_b
        };
        self.output_linear.execute(normalized, &mut self.scratch.output)?;
        Ok(())
    }

    pub(super) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub(super) fn output(&self) -> (DeviceBuffer<bf16>, usize, usize) {
        (self.scratch.output.clone(), self.pooled_tokens, self.config.output_hidden_size)
    }
}
