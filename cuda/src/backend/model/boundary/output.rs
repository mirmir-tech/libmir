use mircuda::{DeviceBuffer, Stream, bf16};
use runtime::backend::SamplingLogits;

use crate::{
    CudaBackend, Result,
    backend::{
        linear::{CheckpointProjection, CheckpointProjectionWeight},
        output::{CudaBatchOutputHead, CudaOutputHead, CudaOutputHeadTemplate},
    },
};

#[derive(Clone)]
pub(in crate::backend::model) enum ModelOutputHeadTemplate {
    Affine {
        weight: Box<CheckpointProjectionWeight>,
        hidden: usize,
        vocab: usize,
    },
    Dense(Box<CudaOutputHeadTemplate>),
    DirectFp8 {
        weight: Box<CheckpointProjectionWeight>,
        hidden: usize,
        vocab: usize,
    },
    MxFp4 {
        weight: Box<CheckpointProjectionWeight>,
        hidden: usize,
        vocab: usize,
    },
    MxFp8 {
        weight: Box<CheckpointProjectionWeight>,
        hidden: usize,
        vocab: usize,
    },
    PackedInteger {
        weight: Box<CheckpointProjectionWeight>,
        hidden: usize,
        vocab: usize,
    },
}

pub(in crate::backend::model) enum ModelOutputHead {
    Affine(CheckpointProjection),
    Dense(CudaOutputHead),
    DirectFp8(CheckpointProjection),
    MxFp4(CheckpointProjection),
    MxFp8(CheckpointProjection),
    PackedInteger(CheckpointProjection),
}

pub(in crate::backend::model) enum ModelBatchOutputHead {
    Affine {
        projection: CheckpointProjection,
        stream: Stream,
    },
    Dense(CudaBatchOutputHead),
    DirectFp8 {
        projection: CheckpointProjection,
        stream: Stream,
    },
    MxFp4 {
        projection: CheckpointProjection,
        stream: Stream,
    },
    MxFp8 {
        projection: CheckpointProjection,
        stream: Stream,
    },
    PackedInteger {
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
        match weight {
            CheckpointProjectionWeight::Affine(_) => {
                weight.affine_format(1, hidden, vocab)?;
                Ok(Self::Affine { weight: Box::new(weight), hidden, vocab })
            },
            CheckpointProjectionWeight::Dense(source) => {
                CudaOutputHeadTemplate::prepare(backend, source, hidden, vocab)
                    .map(Box::new)
                    .map(Self::Dense)
            },
            CheckpointProjectionWeight::DirectFp8(_)
            | CheckpointProjectionWeight::NvFp4(_)
            | CheckpointProjectionWeight::NvFp4WeightOnly(_) => {
                weight.affine_format(1, hidden, vocab)?;
                Ok(Self::DirectFp8 { weight: Box::new(weight), hidden, vocab })
            },
            CheckpointProjectionWeight::MxFp4(_) => {
                weight.affine_format(1, hidden, vocab)?;
                Ok(Self::MxFp4 { weight: Box::new(weight), hidden, vocab })
            },
            CheckpointProjectionWeight::MxFp8(_) => {
                weight.affine_format(1, hidden, vocab)?;
                Ok(Self::MxFp8 { weight: Box::new(weight), hidden, vocab })
            },
            CheckpointProjectionWeight::PackedInteger(_) => {
                weight.affine_format(1, hidden, vocab)?;
                Ok(Self::PackedInteger { weight: Box::new(weight), hidden, vocab })
            },
        }
    }

    pub(in crate::backend::model) fn instantiate(
        &self,
        backend: &CudaBackend,
    ) -> Result<ModelOutputHead> {
        match self {
            Self::Affine { weight, hidden, vocab } => {
                projection(backend, 1, *hidden, *vocab, weight).map(ModelOutputHead::Affine)
            },
            Self::Dense(template) => template.instantiate(backend).map(ModelOutputHead::Dense),
            Self::DirectFp8 { weight, hidden, vocab } => {
                projection(backend, 1, *hidden, *vocab, weight).map(ModelOutputHead::DirectFp8)
            },
            Self::MxFp4 { weight, hidden, vocab } => {
                projection(backend, 1, *hidden, *vocab, weight).map(ModelOutputHead::MxFp4)
            },
            Self::MxFp8 { weight, hidden, vocab } => {
                projection(backend, 1, *hidden, *vocab, weight).map(ModelOutputHead::MxFp8)
            },
            Self::PackedInteger { weight, hidden, vocab } => {
                projection(backend, 1, *hidden, *vocab, weight).map(ModelOutputHead::PackedInteger)
            },
        }
    }

    pub(in crate::backend::model) fn instantiate_batch(
        &self,
        backend: &CudaBackend,
        rows: usize,
    ) -> Result<ModelBatchOutputHead> {
        match self {
            Self::Affine { weight, hidden, vocab } => Ok(ModelBatchOutputHead::Affine {
                projection: projection(backend, rows, *hidden, *vocab, weight)?,
                stream: backend.inner.stream.clone(),
            }),
            Self::Dense(template) => {
                CudaBatchOutputHead::new(backend, template, rows).map(ModelBatchOutputHead::Dense)
            },
            Self::DirectFp8 { weight, hidden, vocab } => Ok(ModelBatchOutputHead::DirectFp8 {
                projection: projection(backend, rows, *hidden, *vocab, weight)?,
                stream: backend.inner.stream.clone(),
            }),
            Self::MxFp4 { weight, hidden, vocab } => Ok(ModelBatchOutputHead::MxFp4 {
                projection: projection(backend, rows, *hidden, *vocab, weight)?,
                stream: backend.inner.stream.clone(),
            }),
            Self::MxFp8 { weight, hidden, vocab } => Ok(ModelBatchOutputHead::MxFp8 {
                projection: projection(backend, rows, *hidden, *vocab, weight)?,
                stream: backend.inner.stream.clone(),
            }),
            Self::PackedInteger { weight, hidden, vocab } => {
                Ok(ModelBatchOutputHead::PackedInteger {
                    projection: projection(backend, rows, *hidden, *vocab, weight)?,
                    stream: backend.inner.stream.clone(),
                })
            },
        }
    }
}

impl ModelOutputHead {
    pub(in crate::backend::model) fn execute(
        &mut self,
        input: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        sampling: SamplingLogits,
    ) -> Result<()> {
        match self {
            Self::Dense(head) => head.execute(input, output, sampling),
            Self::Affine(projection)
            | Self::DirectFp8(projection)
            | Self::MxFp4(projection)
            | Self::MxFp8(projection)
            | Self::PackedInteger(projection) => projection.execute(input, output),
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
            Self::Affine { projection, .. }
            | Self::DirectFp8 { projection, .. }
            | Self::MxFp4 { projection, .. }
            | Self::MxFp8 { projection, .. }
            | Self::PackedInteger { projection, .. } => projection.execute(input, output),
        }
    }

    pub(in crate::backend::model) fn stream(&self) -> &Stream {
        match self {
            Self::Dense(head) => head.stream(),
            Self::Affine { stream, .. }
            | Self::DirectFp8 { stream, .. }
            | Self::MxFp4 { stream, .. }
            | Self::MxFp8 { stream, .. }
            | Self::PackedInteger { stream, .. } => stream,
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
