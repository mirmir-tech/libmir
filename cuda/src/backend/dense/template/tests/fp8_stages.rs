use std::collections::BTreeSet;

use mircuda::{DeviceBuffer, DeviceElement, bf16};
use models::weights::{DenseDecoderLayerBindings, TensorBinding, TensorCatalog};

use super::{TestResult, required};
use crate::{
    CudaBackend, CudaTensorSet, DirectFp8CheckpointWeight, GatedActivation,
    kernels::ElementwiseBf16,
};

pub(super) fn validate(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    bindings: DenseDecoderLayerBindings<'_>,
    input: &DeviceBuffer<bf16>,
    hidden: usize,
) -> TestResult<()> {
    let tensors = upload(backend, catalog, bindings)?;
    let norm = tensors.get(&bindings.input_norm.source).ok_or("missing layer 0 input norm")?;
    let mut normalized = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, hidden)?;
    backend
        .prepare_rms_norm_bf16(1, hidden, 1.0e-6)?
        .execute(input, norm, &mut normalized)?;
    let _query = projection(backend, &tensors, bindings.attention.query, &normalized)?;
    let value = projection(backend, &tensors, bindings.attention.value, &normalized)?;
    let values = read(backend, &value)?;
    let mut attention = Vec::with_capacity(hidden);
    for query_head in 0..14 {
        let kv_head = query_head / 7;
        attention.extend_from_slice(&values[kv_head * 64..(kv_head + 1) * 64]);
    }
    let attention = copy(backend, &attention)?;
    let output = projection(backend, &tensors, bindings.attention.output, &attention)?;
    assert_values(
        &read(backend, &output)?,
        &[
            -0.017_456_055, 0.003_906_25, 0.011_962_891, 0.011_291_504, -0.003_768_921,
            -0.004_089_355_5, -0.003_814_697_3, 0.002_975_463_9,
        ],
        "attention output",
    );
    validate_mlp(backend, &tensors, bindings, input, &output, hidden)?;
    Ok(())
}

fn validate_mlp(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    bindings: DenseDecoderLayerBindings<'_>,
    input: &DeviceBuffer<bf16>,
    attention: &DeviceBuffer<bf16>,
    hidden: usize,
) -> TestResult<()> {
    let hidden_ops = ElementwiseBf16::compile(&backend.inner.compiler, hidden)?;
    let mut residual = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, hidden)?;
    hidden_ops.add(&backend.inner.stream, input, attention, &mut residual)?;
    assert_values(
        &read(backend, &residual)?,
        &[
            -0.049_804_688, 0.002_380_371, 0.024_047_852, -0.004_821_777_3, 0.012_329_102,
            -0.034_423_828, -0.001_831_054_7, -0.012_756_348,
        ],
        "attention residual",
    );

    let norm_weight = tensors
        .get(&bindings.post_attention_norm.source)
        .ok_or("missing post-attention norm")?;
    let mut normalized = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, hidden)?;
    backend
        .prepare_rms_norm_bf16(1, hidden, 1.0e-6)?
        .execute(&residual, norm_weight, &mut normalized)?;
    assert_values(
        &read(backend, &normalized)?,
        &[
            -1.789_062_5, 0.091_308_594, 0.816_406_25, -0.194_335_94, 0.445_312_5, -1.210_937_5,
            -0.069_824_22, -0.455_078_13,
        ],
        "post-attention norm",
    );

    let gate = projection(backend, tensors, bindings.gate, &normalized)?;
    let up = projection(backend, tensors, bindings.up, &normalized)?;
    assert_values(
        &read(backend, &gate)?,
        &[-0.304_687_5, -1.234_375, 0.341_796_88, -0.060_058_594],
        "gate projection",
    );
    let intermediate = gate.len();
    let intermediate_ops = ElementwiseBf16::compile(&backend.inner.compiler, intermediate)?;
    let mut activated = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, intermediate)?;
    intermediate_ops.gated(
        &backend.inner.stream,
        &gate,
        &up,
        &mut activated,
        GatedActivation::Silu.into(),
    )?;
    assert_values(
        &read(backend, &activated)?,
        &[
            -0.045_654_297, 0.000_774_383_54, -0.101_562_5, -0.006_134_033, -0.008_544_922,
            0.008_422_852, -0.010_009_766, 0.002_151_489_3,
        ],
        "SwiGLU activation",
    );
    let down = projection(backend, tensors, bindings.down, &activated)?;
    assert_values(
        &read(backend, &down)?,
        &[-0.279_296_88, -0.190_429_69, -0.089_355_47, -0.171_875],
        "down projection",
    );
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, hidden)?;
    hidden_ops.add(&backend.inner.stream, &residual, &down, &mut output)?;
    assert_values(
        &read(backend, &output)?,
        &[
            -0.328_125, -0.188_476_56, -0.065_429_69, -0.176_757_81, 0.046_386_72, -0.013_061_523,
            0.056_640_625, 0.019_775_39,
        ],
        "layer output",
    );
    Ok(())
}

fn assert_values(actual: &[bf16], expected: &[f32], label: &str) {
    let actual = actual[..expected.len()].iter().map(|value| value.to_f32()).collect::<Vec<_>>();
    assert_eq!(actual, expected, "layer 0 {label} differs from vLLM");
}

fn upload(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    bindings: DenseDecoderLayerBindings<'_>,
) -> TestResult<CudaTensorSet> {
    let mut upload = backend.begin_tensor_upload();
    let mut names = BTreeSet::new();
    for binding in [
        bindings.input_norm,
        bindings.attention.query,
        bindings.attention.value,
        bindings.attention.output,
        bindings.post_attention_norm,
        bindings.gate,
        bindings.up,
        bindings.down,
    ] {
        names.extend(binding.physical_sources());
    }
    for name in names {
        upload.enqueue(required(catalog, name)?)?;
    }
    Ok(upload.finish()?)
}

fn projection(
    backend: &CudaBackend,
    tensors: &CudaTensorSet,
    binding: &TensorBinding,
    input: &DeviceBuffer<bf16>,
) -> TestResult<DeviceBuffer<bf16>> {
    let weight = DirectFp8CheckpointWeight::load_binding(tensors, binding)?;
    let output_features = binding.logical_shape.as_ref().ok_or("missing shape")?[0];
    let mut output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, output_features)?;
    weight.prepare(backend, 1)?.execute(input, &weight, &mut output)?;
    Ok(output)
}

fn copy<T: DeviceElement>(backend: &CudaBackend, values: &[T]) -> crate::Result<DeviceBuffer<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.inner.pool.allocate::<T>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}

fn read<T: DeviceElement>(
    backend: &CudaBackend,
    values: &DeviceBuffer<T>,
) -> crate::Result<Vec<T>> {
    let mut host = backend.inner.context.allocate_pinned::<T>(values.len())?;
    backend.inner.stream.copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}
