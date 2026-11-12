use mircuda::{DeviceBuffer, bf16};
use models::layout::PixelShuffleVisionConfig;

use super::super::super::CudaBackend;
use crate::{CudaTensor, Error, Result, kernels::PixelShuffleGeometry};

/// Device buffers of one image geometry; the encoder's residual stream
/// alternates between `hidden_a` and `hidden_b`.
#[derive(Debug)]
pub(super) struct PixelShuffleScratch {
    pub patches: DeviceBuffer<bf16>,
    pub projected: DeviceBuffer<bf16>,
    pub hidden_a: DeviceBuffer<bf16>,
    pub hidden_b: DeviceBuffer<bf16>,
    pub normalized: DeviceBuffer<bf16>,
    pub qkv: DeviceBuffer<bf16>,
    pub query: DeviceBuffer<bf16>,
    pub key: DeviceBuffer<bf16>,
    pub value: DeviceBuffer<bf16>,
    pub intermediate_a: DeviceBuffer<bf16>,
    pub intermediate_b: DeviceBuffer<bf16>,
    pub folded: DeviceBuffer<bf16>,
    pub folded_norm: DeviceBuffer<bf16>,
    pub projector_hidden: DeviceBuffer<bf16>,
    pub projector_activation: DeviceBuffer<bf16>,
    pub output: DeviceBuffer<bf16>,
}

impl PixelShuffleScratch {
    pub(super) fn new(
        backend: &CudaBackend,
        config: &PixelShuffleVisionConfig,
        geometry: PixelShuffleGeometry,
        patch_width: usize,
    ) -> Result<Self> {
        let patches = elements(geometry.rows, geometry.cols)?;
        let tokens = geometry.tokens()?;
        let hidden = elements(tokens, config.hidden_size)?;
        let intermediate = elements(tokens, config.intermediate_size)?;
        let folded = elements(geometry.folded_tokens()?, geometry.folded_width()?)?;
        let projector = elements(geometry.folded_tokens()?, config.projector_hidden_size)?;
        let allocate = |size| backend.inner.pool.allocate(&backend.inner.stream, size);
        Ok(Self {
            patches: allocate(elements(patches, patch_width)?)?,
            projected: allocate(elements(patches, config.hidden_size)?)?,
            hidden_a: allocate(hidden)?,
            hidden_b: allocate(hidden)?,
            normalized: allocate(hidden)?,
            qkv: allocate(elements(hidden, 3)?)?,
            query: allocate(hidden)?,
            key: allocate(hidden)?,
            value: allocate(hidden)?,
            intermediate_a: allocate(intermediate)?,
            intermediate_b: allocate(intermediate)?,
            folded: allocate(folded)?,
            folded_norm: allocate(folded)?,
            projector_hidden: allocate(projector)?,
            projector_activation: allocate(projector)?,
            output: allocate(elements(geometry.folded_tokens()?, config.output_hidden_size)?)?,
        })
    }
}

pub(super) fn elements(rows: usize, columns: usize) -> Result<usize> {
    rows.checked_mul(columns)
        .ok_or(Error::InvalidVisionKernel("pixel-shuffle scratch overflow"))
}

/// A configuration float in the precision the kernels take.
pub(super) fn single(value: f64) -> Result<f32> {
    Ok(value.to_string().parse()?)
}

pub(super) fn bf16(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
