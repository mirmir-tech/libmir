use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::backend::SamplingLogits;

use crate::{
    CudaBackend, Result,
    backend::{
        linear::{CheckpointProjection, CheckpointProjectionWeight},
        output::{CudaBatchOutputHead, CudaOutputHead, CudaOutputHeadTemplate},
    },
};

/// An output head over dense BF16 weights, which owns a tuned vocabulary
/// plan, or over any other checkpoint projection.
#[derive(Clone)]
pub(in crate::backend::model) enum ModelOutputHeadTemplate {
    Dense(Box<CudaOutputHeadTemplate>),
    Checkpoint {
        weight: Box<CheckpointProjectionWeight>,
        hidden: usize,
        vocab: usize,
    },
}

#[expect(clippy::large_enum_variant, reason = "one head per model, built once")]
pub(in crate::backend::model) enum ModelOutputHead {
    Dense(CudaOutputHead),
    Checkpoint(CheckpointProjection),
}

#[expect(clippy::large_enum_variant, reason = "one head per model, built once")]
pub(in crate::backend::model) enum ModelBatchOutputHead {
    Dense(CudaBatchOutputHead),
    Checkpoint {
        projection: CheckpointProjection,
        stream: Stream,
    },
}

impl ModelOutputHeadTemplate {
    pub(in crate::backend::model) fn prepare(
        backend: &CudaBackend,
        weight: CheckpointProjectionWeight,
        hidden: usize,
        vocab: usize,
    ) -> Result<Self> {
        if let CheckpointProjectionWeight::Dense(source) = weight {
            return CudaOutputHeadTemplate::prepare(backend, source, hidden, vocab)
                .map(Box::new)
                .map(Self::Dense);
        }
        weight.affine_format(1, hidden, vocab)?;
        Ok(Self::Checkpoint { weight: Box::new(weight), hidden, vocab })
    }

    pub(in crate::backend::model) fn instantiate(
        &self,
        backend: &CudaBackend,
    ) -> Result<ModelOutputHead> {
        match self {
            Self::Dense(template) => template.instantiate(backend).map(ModelOutputHead::Dense),
            Self::Checkpoint { weight, hidden, vocab } => {
                projection(backend, 1, *hidden, *vocab, weight).map(ModelOutputHead::Checkpoint)
            },
        }
    }

    pub(in crate::backend::model) fn instantiate_batch(
        &self,
        backend: &CudaBackend,
        rows: usize,
    ) -> Result<ModelBatchOutputHead> {
        match self {
            Self::Dense(template) => {
                CudaBatchOutputHead::new(backend, template, rows).map(ModelBatchOutputHead::Dense)
            },
            Self::Checkpoint { weight, hidden, vocab } => Ok(ModelBatchOutputHead::Checkpoint {
                projection: projection(backend, rows, *hidden, *vocab, weight)?,
                stream: backend.inner.stream.clone(),
            }),
        }
    }
}

impl ModelOutputHead {
    pub(in crate::backend::model) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        sampling: &SamplingLogits,
    ) -> Result<()> {
        match self {
            Self::Dense(head) => head.execute(input, output, sampling),
            Self::Checkpoint(projection) => projection.execute(input, output),
        }
    }
}

impl ModelBatchOutputHead {
    pub(in crate::backend::model) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        match self {
            Self::Dense(head) => head.execute(input, output),
            Self::Checkpoint { projection, .. } => projection.execute(input, output),
        }
    }

    pub(in crate::backend::model) fn stream(&self) -> &Stream {
        match self {
            Self::Dense(head) => head.stream(),
            Self::Checkpoint { stream, .. } => stream,
        }
    }
}

fn projection(
    backend: &CudaBackend,
    tokens: usize,
    hidden: usize,
    vocab: usize,
    weight: &CheckpointProjectionWeight,
) -> Result<CheckpointProjection> {
    CheckpointProjection::new(backend, tokens, hidden, vocab, crate::DenseRole::OutputHead, weight)
}
