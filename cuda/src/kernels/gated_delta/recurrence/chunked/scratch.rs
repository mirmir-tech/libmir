use mircuda::{Context, DeviceBuffer, DeviceElement, MemoryPool, Stream, bf16};

use super::{CHUNK, validate};
use crate::{
    Result,
    kernels::{GatedDeltaSpec, geometry::product},
};

#[derive(Debug)]
pub struct GatedDeltaChunkedScratch {
    pub(super) cumulative_decay: DeviceBuffer<f32>,
    pub(super) matrix: DeviceBuffer<f32>,
    pub(super) inverse: DeviceBuffer<bf16>,
    pub(super) w: DeviceBuffer<bf16>,
    pub(super) u: DeviceBuffer<bf16>,
    pub(super) chunks: DeviceBuffer<bf16>,
    pub(super) value: DeviceBuffer<bf16>,
    pub(super) cu_seqlens: DeviceBuffer<i32>,
    pub(super) chunk_indices: DeviceBuffer<i32>,
    pub(super) chunk_offsets: DeviceBuffer<i32>,
}

impl GatedDeltaChunkedScratch {
    pub fn new(
        context: &Context,
        pool: &MemoryPool,
        stream: &Stream,
        spec: GatedDeltaSpec,
    ) -> Result<Self> {
        validate(spec)?;
        let tokens_heads = product(spec.tokens, spec.value_heads)?;
        let matrix = product(tokens_heads, CHUNK)?;
        let values = product(tokens_heads, spec.value_dim)?;
        let chunks = spec.tokens.div_ceil(CHUNK);
        let states =
            product(product(product(chunks, spec.value_heads)?, spec.value_dim)?, spec.key_dim)?;
        let mut indices = Vec::with_capacity(product(chunks, 2)?);
        for chunk in 0..chunks {
            indices.push(0);
            indices.push(i32::try_from(chunk)?);
        }
        Ok(Self {
            cumulative_decay: pool.allocate(stream, tokens_heads)?,
            matrix: pool.allocate(stream, matrix)?,
            inverse: pool.allocate(stream, matrix)?,
            w: pool.allocate(stream, product(tokens_heads, spec.key_dim)?)?,
            u: pool.allocate(stream, values)?,
            chunks: pool.allocate(stream, states)?,
            value: pool.allocate(stream, values)?,
            cu_seqlens: upload(context, pool, stream, &[0, i32::try_from(spec.tokens)?])?,
            chunk_indices: upload(context, pool, stream, &indices)?,
            chunk_offsets: upload(context, pool, stream, &[0, i32::try_from(chunks)?])?,
        })
    }
}

fn upload<T: DeviceElement>(
    context: &Context,
    pool: &MemoryPool,
    stream: &Stream,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut host = context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = pool.allocate(stream, values.len())?;
    stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}
