//! llama.cpp's GGUF Gated `DeltaNet` layout (`LinearAttentionLayout::LlamaCpp`)
//! brought back to the Hugging Face one the kernels read, as the Metal
//! runtime does: load takes `A_log` back from the stored `-exp(A_log)` and,
//! with more value than key heads, moves every value-head row (the `v` part
//! of the QKV projection and convolution, `z`, `a`, `b`, `A`, `dt_bias`) to
//! grouped order. The output projection's columns sit inside GGML blocks, so
//! its input goes back to tiled order instead.

use mircuda::{DeviceBuffer, Stream, bf16};

use super::{AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights};
use crate::{
    CudaBackend, CudaTensor, CudaTensorDType, Error, Result,
    backend::linear::{CheckpointProjection, CheckpointProjectionWeight},
    kernels::{GatedDeltaLlamaCpp, GatedDeltaTiledValues, GgmlBlocks},
};

impl AffineGatedDeltaLayerWeights {
    /// Restores the grouped layout of a llama.cpp GGUF layer.
    pub(in crate::backend) fn restore_llama_cpp_layout(
        &mut self,
        backend: &CudaBackend,
        config: AffineGatedDeltaLayerConfig,
    ) -> Result<()> {
        let kernels = GatedDeltaLlamaCpp::compile(&backend.inner.compiler)?;
        let reorder = Reorder { backend, kernels: &kernels };
        let heads = config.value_heads;
        let tiled = config.tiled_values();
        let rows = |offset: usize, width: usize| -> Result<Vec<u32>> {
            match tiled {
                Some(tiled) => tiled.rows(offset, width),
                None => Ok((0..offset + heads * width)
                    .map(u32::try_from)
                    .collect::<std::result::Result<_, _>>()?),
            }
        };
        // The decay log, in grouped order: a fresh copy, then the log in place.
        self.a_log = reorder.tensor(&self.a_log, &rows(0, 1)?)?;
        let mut decay = self.a_log.as_bf16().cloned().ok_or_else(|| bf16_required(&self.a_log))?;
        kernels.decay_log(&backend.inner.stream, &mut decay)?;
        if tiled.is_none() {
            return Ok(());
        }
        let keys = 2 * config.key_heads * config.key_dim;
        let value_dim = config.value_dim;
        self.qkv = reorder.projection(&self.qkv, &rows(keys, value_dim)?)?;
        self.gate = reorder.projection(&self.gate, &rows(0, value_dim)?)?;
        self.alpha = reorder.projection(&self.alpha, &rows(0, 1)?)?;
        self.beta = reorder.projection(&self.beta, &rows(0, 1)?)?;
        self.convolution = reorder.tensor(&self.convolution, &rows(keys, value_dim)?)?;
        self.dt_bias = reorder.tensor(&self.dt_bias, &rows(0, 1)?)?;
        Ok(())
    }
}

/// Gathers stored rows into grouped order.
struct Reorder<'a> {
    backend: &'a CudaBackend,
    kernels: &'a GatedDeltaLlamaCpp,
}

impl Reorder<'_> {
    /// Gathers `rows` of `row_bytes` from `input` into `output`.
    fn rows(
        &self,
        input: &DeviceBuffer<u8>,
        rows: &[u32],
        row_bytes: usize,
        output: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let backend = &self.backend.inner;
        let mut host = backend.context.allocate_pinned::<u32>(rows.len())?;
        host.copy_from_slice(rows)?;
        let mut indices = backend.pool.allocate::<u32>(&backend.stream, rows.len())?;
        backend.stream.copy_to_device(&mut host, &mut indices)?;
        self.kernels.gather_rows(&backend.stream, input, output, &indices, row_bytes)?;
        // The pinned indices must outlive the transfer.
        backend.stream.synchronize()?;
        Ok(())
    }

    /// A BF16 tensor whose leading axis is reordered.
    fn tensor(&self, tensor: &CudaTensor, rows: &[u32]) -> Result<CudaTensor> {
        let values = tensor.as_bf16().filter(|_| tensor.dtype() == CudaTensorDType::Bf16);
        let values = values.ok_or_else(|| bf16_required(tensor))?;
        let shape = tensor.shape().to_vec();
        let leading = shape.first().copied().unwrap_or(0);
        if leading != rows.len() {
            return Err(Error::InvalidQuantizedTensor {
                name: tensor.name().into(),
                expected: vec![rows.len()],
                actual: shape,
            });
        }
        let row_bytes = 2 * shape[1..].iter().product::<usize>();
        let backend = &self.backend.inner;
        let mut output = backend.pool.allocate::<u8>(&backend.stream, rows.len() * row_bytes)?;
        self.rows(&values.reinterpret::<u8>()?, rows, row_bytes, &mut output)?;
        Ok(CudaTensor::from_bf16(
            tensor.name().into(),
            shape,
            output.reinterpret::<bf16>()?,
        ))
    }

    /// A projection whose output rows are reordered: GGML blocks or BF16.
    fn projection(
        &self,
        weight: &CheckpointProjectionWeight,
        rows: &[u32],
    ) -> Result<CheckpointProjectionWeight> {
        match weight {
            CheckpointProjectionWeight::Ggml(weight) => {
                let row_bytes = weight.kind().bytes(1, weight.input_features())?;
                let backend = &self.backend.inner;
                let mut blocks =
                    GgmlBlocks::allocate(&backend.pool, &backend.stream, rows.len() * row_bytes)?;
                self.rows(weight.blocks(), rows, row_bytes, blocks.view_mut())?;
                Ok(CheckpointProjectionWeight::Ggml(weight.with_blocks(blocks)?))
            },
            CheckpointProjectionWeight::Dense(tensor) => {
                Ok(CheckpointProjectionWeight::Dense(self.tensor(tensor, rows)?))
            },
            _ => Err(Error::UnsupportedDecoderLayer(
                "llama.cpp Gated Delta layouts need GGML or BF16 projections".into(),
            )),
        }
    }
}

fn bf16_required(tensor: &CudaTensor) -> Error {
    Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    }
}

/// The grouped attention output tiled back for a llama.cpp output projection.
#[derive(Debug)]
pub(super) struct TiledOutput {
    kernels: GatedDeltaLlamaCpp,
    values: GatedDeltaTiledValues,
    tokens: usize,
}

impl TiledOutput {
    pub(super) fn new(
        backend: &CudaBackend,
        config: AffineGatedDeltaLayerConfig,
        tokens: usize,
    ) -> Result<Option<Self>> {
        config
            .tiled_values()
            .map(|values| {
                let kernels = GatedDeltaLlamaCpp::compile(&backend.inner.compiler)?;
                Ok(Self { kernels, values, tokens })
            })
            .transpose()
    }
}

/// Projects the grouped `gated` output, tiled into `spare` first when the
/// stored projection expects tiled columns.
pub(super) fn project(
    tiled: Option<&TiledOutput>,
    stream: &Stream,
    projection: &mut CheckpointProjection,
    (gated, spare): (&DeviceBuffer<bf16>, &mut DeviceBuffer<bf16>),
    output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    match tiled {
        Some(tiled) => {
            tiled.kernels.tile_values(stream, tiled.values, (gated, spare), tiled.tokens)?;
            projection.execute(spare, output)
        },
        None => projection.execute(gated, output),
    }
}
