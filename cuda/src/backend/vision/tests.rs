use mircuda::{DeviceBuffer, DeviceElement, bf16};

use super::super::CudaBackend;
use crate::{
    CudaConfig, Result,
    kernels::{
        SpatialMergeKernels, VisionAttention, VisionAttentionSpec, VisionClip, VisionClipSpec,
        VisionElementwise, VisionElementwiseSpec, VisionEmbeddingSplice, VisionPool,
        VisionPoolSpec, VisionSpatialRope,
    },
};

#[test]
fn splices_image_embeddings_without_host_roundtrip()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let image = copy(&backend, &bf16s(&[10.0, 11.0, 20.0, 21.0]))?;
    let mut hidden = copy(&backend, &bf16s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))?;
    VisionEmbeddingSplice::compile(&backend.inner.compiler, 2, 2, 1)?.execute(
        &backend.inner.stream,
        &image,
        &mut hidden,
    )?;
    assert_eq!(
        read(&backend, &hidden)?.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
        [1.0, 2.0, 10.0, 11.0, 20.0, 21.0, 7.0, 8.0]
    );
    Ok(())
}

#[test]
fn pools_and_standardizes_in_the_vision_kernel_module()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let input = copy(&backend, &bf16s(&[1.0, 2.0, 3.0, 4.0]))?;
    let bias = copy(&backend, &bf16s(&[0.0]))?;
    let scale = copy(&backend, &bf16s(&[2.0]))?;
    let mut pooled = backend.inner.pool.allocate(&backend.inner.stream, 1)?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 1)?;
    let operation = VisionPool::compile(
        &backend.inner.compiler,
        VisionPoolSpec {
            grid_height: 2,
            grid_width: 2,
            hidden: 1,
            kernel: 2,
        },
    )?;
    operation.execute(&backend.inner.stream, &input, &mut pooled)?;
    operation.standardize(&backend.inner.stream, &pooled, &bias, &scale, &mut output)?;
    assert!((read(&backend, &output)?[0].to_f32() - 5.0).abs() < 0.05);
    Ok(())
}

#[test]
fn applies_vision_bias_in_place_without_a_second_device_buffer()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut values = copy(&backend, &bf16s(&[1.0, 2.0, 3.0, 4.0]))?;
    let input = values.clone();
    let bias = copy(&backend, &bf16s(&[10.0, 20.0]))?;
    VisionElementwise::compile(
        &backend.inner.compiler,
        VisionElementwiseSpec { rows: 2, columns: 2, epsilon: 0.0 },
    )?
    .add_bias(&backend.inner.stream, &input, &bias, &mut values)?;
    assert_eq!(as_f32(&read(&backend, &values)?), [11.0, 22.0, 13.0, 24.0]);
    Ok(())
}

#[test]
fn compiles_the_complete_vision_primitive_set()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let compiler = &backend.inner.compiler;
    VisionElementwise::compile(
        compiler,
        VisionElementwiseSpec { rows: 2, columns: 4, epsilon: 1.0e-6 },
    )?;
    VisionClip::compile(compiler, VisionClipSpec { rows: 2, columns: 4 })?;
    let attention = VisionAttention::compile(
        compiler,
        VisionAttentionSpec {
            tokens: 2,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 4,
            scale: 1.0,
        },
    )?;
    VisionSpatialRope::compile(compiler, 2, 2, 4, 10_000.0)?;
    SpatialMergeKernels::compile(compiler)?;
    let query = copy(&backend, &bf16s(&[0.0; 16]))?;
    let key = copy(&backend, &bf16s(&[0.0; 8]))?;
    let value = copy(&backend, &bf16s(&[1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0]))?;
    let mut output = backend.inner.pool.allocate(&backend.inner.stream, 16)?;
    attention.execute(&backend.inner.stream, &query, &key, &value, &mut output)?;
    let actual = read(&backend, &output)?;
    for row in actual.as_chunks::<4>().0 {
        for (actual, expected) in row.iter().zip([2.0, 3.0, 4.0, 5.0]) {
            assert!((actual.to_f32() - expected).abs() < 0.05);
        }
    }
    Ok(())
}

#[test]
fn executes_spatial_merge_interpolation_split_and_rope()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let kernels = SpatialMergeKernels::compile(&backend.inner.compiler)?;
    let table = copy(&backend, &bf16s(&[0.0, 10.0, 20.0, 30.0]))?;
    let mut interpolated = backend.inner.pool.allocate(&backend.inner.stream, 9)?;
    kernels.interpolate(&backend.inner.stream, &table, &mut interpolated, 3, 3, 2, 1, 1)?;
    let actual = read(&backend, &interpolated)?
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    assert_eq!(actual, [0.0, 5.0, 10.0, 10.0, 15.0, 20.0, 20.0, 25.0, 30.0]);

    let qkv_values = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ];
    let qkv = copy(&backend, &bf16s(&qkv_values))?;
    let mut query = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    let mut key = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    let mut value = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    kernels.split_qkv(&backend.inner.stream, &qkv, &mut query, &mut key, &mut value, 1, 8)?;
    assert_eq!(as_f32(&read(&backend, &query)?), qkv_values[..8]);
    assert_eq!(as_f32(&read(&backend, &key)?), qkv_values[8..16]);
    assert_eq!(as_f32(&read(&backend, &value)?), qkv_values[16..]);

    let positions = copy(&backend, &[1_u32, 2])?;
    let mut rotated = backend.inner.pool.allocate(&backend.inner.stream, 8)?;
    kernels.rope(&backend.inner.stream, &query, &positions, &mut rotated, 1, 1, 8)?;
    let angles = [1.0_f32, 0.01, 2.0, 0.02, 1.0, 0.01, 2.0, 0.02];
    let paired = [-5.0_f32, -6.0, -7.0, -8.0, 1.0, 2.0, 3.0, 4.0];
    for (((actual, input), paired), angle) in read(&backend, &rotated)?
        .iter()
        .zip(qkv_values[..8].iter().copied())
        .zip(paired)
        .zip(angles)
    {
        let expected = paired.mul_add(angle.sin(), input * angle.cos());
        assert!((actual.to_f32() - expected).abs() < 0.05);
    }
    Ok(())
}

fn bf16s(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn as_f32(values: &[bf16]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(backend: &CudaBackend, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
