use std::path::PathBuf;

use mircuda::{Compiler, Context, DeviceBuffer, DeviceElement, Driver, MemoryPool, Stream, bf16};

use super::*;
use crate::kernels::{RmsNorm, RmsNormUnit, Rope, RopeSpec};

const ROTARY: RotaryEmbedding = RotaryEmbedding {
    dimensions: 6,
    pairing: 8,
    theta: 10_000.0,
};

#[test]
fn fused_qkv_matches_individual_norm_and_rope_kernels() -> Result<()> {
    let runtime = Runtime::new()?;
    let spec = QkvPostprocessSpec {
        tokens: 2,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 8,
        value_head_dim: 8,
        rotary: Some(ROTARY),
        epsilon: 1.0e-6,
        normalization: QkvNormalization::ALL,
    };
    let query = test_values(32, 0)?;
    let key = test_values(16, 3)?;
    let value = test_values(16, 7)?;
    let mut packed = Vec::with_capacity(64);
    for token in 0..2 {
        packed.extend_from_slice(&query[token * 16..(token + 1) * 16]);
        packed.extend_from_slice(&key[token * 8..(token + 1) * 8]);
        packed.extend_from_slice(&value[token * 8..(token + 1) * 8]);
    }
    let query_weight = test_values(8, 2)?;
    let key_weight = test_values(8, 5)?;
    let actual = fused(&runtime, spec, &packed, &query_weight, &key_weight, 3)?;
    let expected = composed(&runtime, spec, &query, &key, &value, &query_weight, &key_weight)?;
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn batched_packed_qkv_matches_independent_rows() -> Result<()> {
    let runtime = Runtime::new()?;
    let spec = QkvPostprocessSpec {
        tokens: 2,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 8,
        value_head_dim: 8,
        rotary: Some(ROTARY),
        epsilon: 1.0e-6,
        normalization: QkvNormalization::ALL,
    };
    let packed = test_values(64, 0)?;
    let query_weight = test_values(8, 2)?;
    let key_weight = test_values(8, 5)?;
    let actual = batched(&runtime, spec, &packed, &query_weight, &key_weight)?;
    let expected = fused(&runtime, spec, &packed, &query_weight, &key_weight, 3)?;
    assert_eq!(actual, expected);
    Ok(())
}

/// Heads without positional encoding equal rotary heads at zero angles.
#[test]
fn unrotated_qkv_matches_rotary_at_position_zero() -> Result<()> {
    let runtime = Runtime::new()?;
    let rotary = QkvPostprocessSpec {
        tokens: 1,
        query_heads: 2,
        kv_heads: 1,
        head_dim: 8,
        value_head_dim: 8,
        rotary: Some(ROTARY),
        epsilon: 1.0e-6,
        normalization: QkvNormalization::QUERY_KEY,
    };
    let unrotated = QkvPostprocessSpec { rotary: None, ..rotary };
    let packed = test_values(32, 0)?;
    let (query_weight, key_weight) = (test_values(8, 2)?, test_values(8, 5)?);
    let actual = fused(&runtime, unrotated, &packed, &query_weight, &key_weight, 9)?;
    let expected = fused(&runtime, rotary, &packed, &query_weight, &key_weight, 0)?;
    assert_eq!(actual, expected);
    Ok(())
}

fn test_values(elements: usize, offset: usize) -> Result<Vec<bf16>> {
    (0..elements)
        .map(|index| {
            let value = f32::from(u16::try_from((index + offset) % 17)?) / 8.0 - 1.0;
            Ok(bf16::from_f32(value))
        })
        .collect()
}

fn batched(
    runtime: &Runtime,
    spec: QkvPostprocessSpec,
    packed: &[bf16],
    query_weight: &[bf16],
    key_weight: &[bf16],
) -> Result<Outputs> {
    let packed = runtime.copy(packed)?;
    let query_weight = runtime.copy(query_weight)?;
    let key_weight = runtime.copy(key_weight)?;
    let positions = runtime.copy(&[3_u32, 4])?;
    let mut query = runtime.allocate(32)?;
    let mut key = runtime.allocate(16)?;
    let mut value = runtime.allocate(16)?;
    BatchedQkvPostprocess::compile(&runtime.compiler, spec)?.execute(
        &runtime.stream,
        [&packed, &packed, &packed],
        false,
        &query_weight,
        &key_weight,
        &positions,
        &mut query,
        &mut key,
        &mut value,
    )?;
    Ok((runtime.read(&query)?, runtime.read(&key)?, runtime.read(&value)?))
}

type Outputs = (Vec<bf16>, Vec<bf16>, Vec<bf16>);

fn fused(
    runtime: &Runtime,
    spec: QkvPostprocessSpec,
    packed: &[bf16],
    query_weight: &[bf16],
    key_weight: &[bf16],
    position: usize,
) -> Result<Outputs> {
    let packed = runtime.copy(packed)?;
    let query_weight = runtime.copy(query_weight)?;
    let key_weight = runtime.copy(key_weight)?;
    let mut query = runtime.allocate(spec.tokens * spec.query_heads * spec.head_dim)?;
    let mut key = runtime.allocate(spec.tokens * spec.kv_heads * spec.head_dim)?;
    let mut value = runtime.allocate(spec.tokens * spec.kv_heads * spec.value_head_dim)?;
    QkvPostprocess::compile(&runtime.compiler, spec)?.execute(
        &runtime.stream, &packed, &query_weight, &key_weight, &mut query, &mut key, &mut value,
        position,
    )?;
    Ok((runtime.read(&query)?, runtime.read(&key)?, runtime.read(&value)?))
}

#[allow(clippy::too_many_arguments)]
fn composed(
    runtime: &Runtime,
    spec: QkvPostprocessSpec,
    query: &[bf16],
    key: &[bf16],
    value: &[bf16],
    query_weight: &[bf16],
    key_weight: &[bf16],
) -> Result<Outputs> {
    let query = runtime.copy(query)?;
    let key = runtime.copy(key)?;
    let value = runtime.copy(value)?;
    let query_weight = runtime.copy(query_weight)?;
    let key_weight = runtime.copy(key_weight)?;
    let mut query_norm = runtime.allocate(32)?;
    let mut key_norm = runtime.allocate(16)?;
    let mut value_norm = runtime.allocate(16)?;
    RmsNorm::compile(&runtime.compiler, 4, 8, spec.epsilon)?
        .execute(&runtime.stream, &query, &query_weight, &mut query_norm)?;
    RmsNorm::compile(&runtime.compiler, 2, 8, spec.epsilon)?
        .execute(&runtime.stream, &key, &key_weight, &mut key_norm)?;
    RmsNormUnit::compile(&runtime.compiler, 2, 8, spec.epsilon)?
        .execute(&runtime.stream, &value, &mut value_norm)?;
    let mut query_output = runtime.allocate(32)?;
    let mut key_output = runtime.allocate(16)?;
    Rope::compile(&runtime.compiler, rope(spec, 2))?.execute(
        &runtime.stream,
        &query_norm,
        &mut query_output,
        3,
    )?;
    Rope::compile(&runtime.compiler, rope(spec, 1))?
        .execute(&runtime.stream, &key_norm, &mut key_output, 3)?;
    Ok((
        runtime.read(&query_output)?,
        runtime.read(&key_output)?,
        runtime.read(&value_norm)?,
    ))
}

const fn rope(spec: QkvPostprocessSpec, heads: usize) -> RopeSpec {
    RopeSpec {
        tokens: spec.tokens,
        heads,
        head_dim: spec.head_dim,
        rotary_dim: ROTARY.dimensions,
        pairing_dim: ROTARY.pairing,
        theta: ROTARY.theta,
    }
}

struct Runtime {
    context: Context,
    compiler: Compiler,
    pool: MemoryPool,
    stream: Stream,
}

impl Runtime {
    fn new() -> Result<Self> {
        let driver = Driver::initialize()?;
        let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
        let context = driver.create_context(device)?;
        let stream = context.create_stream()?;
        let pool = context.default_memory_pool()?;
        let compiler = Compiler::with_include_paths(
            context.clone(),
            [PathBuf::from("/usr/local/cuda/include")],
        )?;
        Ok(Self { context, compiler, pool, stream })
    }

    fn allocate<T: DeviceElement>(&self, elements: usize) -> Result<DeviceBuffer<T>> {
        Ok(self.pool.allocate(&self.stream, elements)?)
    }

    fn copy<T: DeviceElement + Copy>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = self.allocate(values.len())?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    }

    fn read<T: DeviceElement + Copy>(&self, device: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned::<T>(device.len())?;
        self.stream.copy_to_host(device, &mut host)?;
        Ok(host.to_vec()?)
    }
}
