//! The model half of Gemma 4's per-layer inputs (Hugging Face
//! `Gemma4TextModel.get_per_layer_inputs` and `project_per_layer_inputs`):
//! every layer's input from the step's token ids and their scaled token
//! embeddings, computed right after the embedding lookup.

use std::collections::HashMap;

use mircuda::{DeviceBuffer, bf16};

use super::boundary::{ModelEmbedding, ModelEmbeddingTemplate};
use crate::{
    Bf16Projection, CudaBackend, CudaTensor, DensePlanRequest, DenseRole, Error, ExecutionPhase,
    Result,
    backend::linear::CheckpointProjectionWeight,
    kernels::{PerLayerInputBf16, PerLayerInputSpec},
};

#[cfg(test)]
mod tests;

/// The row counts whose fixed-shape operations a [`PerLayerInputs`] keeps.
const MAX_RESIDENT_STEPS: usize = 8;

/// The model's per-layer input tensors and geometry.
#[derive(Clone)]
pub struct PerLayerInputsTemplate {
    /// `[vocab, layers * width]`, scaled by `sqrt(width)` at lookup.
    embedding: ModelEmbeddingTemplate,
    /// `[layers * width, hidden]` BF16.
    model_projection: CudaTensor,
    /// `[width]` BF16.
    projection_norm: CudaTensor,
    layers: usize,
    width: usize,
    hidden: usize,
    epsilon: f32,
}

/// The tensors [`PerLayerInputsTemplate::new`] takes.
pub struct PerLayerInputTensors {
    pub embedding: CheckpointProjectionWeight,
    pub model_projection: CudaTensor,
    pub projection_norm: CudaTensor,
}

impl PerLayerInputsTemplate {
    pub fn new(
        tensors: PerLayerInputTensors,
        [vocab, layers, width, hidden]: [usize; 4],
        epsilon: f32,
    ) -> Result<Self> {
        let total = layers
            .checked_mul(width)
            .ok_or(Error::InvalidDecoderKernel("per-layer input width overflow"))?;
        if tensors.model_projection.shape() != [total, hidden]
            || tensors.projection_norm.shape() != [width]
        {
            return Err(Error::UnsupportedDecoderLayer(
                "per-layer input tensors differ from the decoder geometry".into(),
            ));
        }
        Ok(Self {
            embedding: ModelEmbeddingTemplate::new(tensors.embedding, vocab, total, scale(width)?)?,
            model_projection: tensors.model_projection,
            projection_norm: tensors.projection_norm,
            layers,
            width,
            hidden,
            epsilon,
        })
    }
}

/// One step's per-layer inputs for up to `capacity` token rows, layer-major:
/// layer `l` of a `rows`-row step reads `[rows, width]` at `l * rows * width`.
pub(in crate::backend::model) struct PerLayerInputs {
    template: PerLayerInputsTemplate,
    backend: CudaBackend,
    embedding: ModelEmbedding,
    steps: HashMap<usize, Step>,
    capacity: usize,
    embedded: DeviceBuffer<bf16>,
    projected: DeviceBuffer<bf16>,
    inputs: DeviceBuffer<bf16>,
}

/// A session's per-layer inputs: its decode token's, whose layer views its
/// layers keep, and its prefill chunks'.
pub(in crate::backend::model) struct SessionPerLayerInputs {
    pub(in crate::backend::model) decode: PerLayerInputs,
    pub(in crate::backend::model) prefill: PerLayerInputs,
}

impl SessionPerLayerInputs {
    pub(in crate::backend::model) fn new(
        backend: &CudaBackend,
        template: &PerLayerInputsTemplate,
        prefill: usize,
    ) -> Result<Self> {
        Ok(Self {
            decode: PerLayerInputs::new(backend, template, 1)?,
            prefill: PerLayerInputs::new(backend, template, prefill)?,
        })
    }
}

/// The fixed-shape operations of one row count.
struct Step {
    projection: Bf16Projection,
    combine: PerLayerInputBf16,
}

impl PerLayerInputs {
    pub(in crate::backend::model) fn new(
        backend: &CudaBackend,
        template: &PerLayerInputsTemplate,
        capacity: usize,
    ) -> Result<Self> {
        let allocate = || allocate(backend, template, capacity);
        Ok(Self {
            embedding: template.embedding.instantiate(backend)?,
            template: template.clone(),
            backend: backend.clone(),
            steps: HashMap::new(),
            capacity,
            embedded: allocate()?,
            projected: allocate()?,
            inputs: allocate()?,
        })
    }

    /// Grows the buffers to hold `rows` rows; layer views taken before stop
    /// aliasing the inputs.
    pub(in crate::backend::model) fn ensure_capacity(&mut self, rows: usize) -> Result<()> {
        if rows > self.capacity {
            *self = Self::new(&self.backend, &self.template, rows)?;
        }
        Ok(())
    }

    /// Computes the inputs of the `rows` tokens at `selected[start..]`, whose
    /// scaled token embeddings are the first `rows` rows of `embeddings`.
    pub(in crate::backend::model) fn compute(
        &mut self,
        selected: &DeviceBuffer<u32>,
        start: usize,
        rows: usize,
        embeddings: &DeviceBuffer<bf16>,
    ) -> Result<()> {
        if rows == 0 || rows > self.capacity {
            return Err(Error::InvalidDecoderKernel("per-layer input rows exceed the buffers"));
        }
        if !self.steps.contains_key(&rows) {
            // Packed prefill passes take any row count: keep a few shapes.
            if self.steps.len() >= MAX_RESIDENT_STEPS
                && let Some(other) = self.steps.keys().next().copied()
            {
                self.steps.remove(&other);
            }
            let step = Step::new(&self.backend, &self.template, rows)?;
            self.steps.insert(rows, step);
        }
        let step = self
            .steps
            .get_mut(&rows)
            .ok_or(Error::InvalidDecoderKernel("missing per-layer input step"))?;
        self.embedding.execute_batch(selected, start, rows, &mut self.embedded)?;
        // Fixed-shape projections take exactly their rows.
        let embeddings = embeddings.slice(0..rows * self.template.hidden)?;
        let mut projected =
            self.projected.slice(0..rows * self.template.layers * self.template.width)?;
        step.projection
            .execute(&embeddings, &self.template.model_projection, &mut projected)?;
        let norm = self.template.projection_norm.as_bf16().ok_or_else(|| Error::DTypeMismatch {
            name: self.template.projection_norm.name().into(),
            expected: "BF16",
        })?;
        step.combine.execute(
            self.backend.stream(),
            &self.projected,
            &self.embedded,
            norm,
            &mut self.inputs,
        )
    }

    /// Layer `layer`'s inputs of a `rows`-row step.
    pub(in crate::backend::model) fn layer(
        &self,
        layer: usize,
        rows: usize,
    ) -> Result<DeviceBuffer<bf16>> {
        let width = rows * self.template.width;
        let start = layer * width;
        if layer >= self.template.layers || rows > self.capacity {
            return Err(Error::InvalidDecoderKernel("per-layer input view is out of range"));
        }
        Ok(self.inputs.slice(start..start + width)?)
    }
}

impl Step {
    fn new(backend: &CudaBackend, template: &PerLayerInputsTemplate, rows: usize) -> Result<Self> {
        let request = DensePlanRequest {
            phase: if rows == 1 {
                ExecutionPhase::Decode
            } else {
                ExecutionPhase::Prefill
            },
            role: DenseRole::DenseGateUp,
            tokens: rows,
            input_features: template.hidden,
            output_features: template.layers * template.width,
        };
        let spec = PerLayerInputSpec {
            rows,
            layers: template.layers,
            width: template.width,
            projection_scale: scale(template.hidden)?.recip(),
            input_scale: std::f32::consts::FRAC_1_SQRT_2,
            epsilon: template.epsilon,
        };
        Ok(Self {
            projection: backend.prepare_bf16_projection(request)?,
            combine: PerLayerInputBf16::compile(backend.compiler(), spec)?,
        })
    }
}

fn allocate(
    backend: &CudaBackend,
    template: &PerLayerInputsTemplate,
    rows: usize,
) -> Result<DeviceBuffer<bf16>> {
    let elements = rows
        .checked_mul(template.layers * template.width)
        .ok_or(Error::InvalidDecoderKernel("per-layer input size overflow"))?;
    Ok(backend.pool().allocate(backend.stream(), elements)?)
}

/// `sqrt(value)` as an `f32`.
fn scale(value: usize) -> Result<f32> {
    #[allow(clippy::cast_possible_truncation)]
    Ok(f64::from(u32::try_from(value)?).sqrt() as f32)
}
