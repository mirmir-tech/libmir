use mircuda::{DeviceBuffer, bf16};

use super::{moe_test_support::*, *};
use crate::CudaConfig;

#[test]
fn checkpoint_tensor_core_moe_matches_w4a16() -> Result<()> {
    let Some(catalog) = catalog()? else {
        return Ok(());
    };
    let backend = CudaBackend::new(CudaConfig::default())?;
    let MoeCandidates {
        mut reference,
        mut tensor_core,
        mut grouped,
        mut fused,
        mut direct,
        mut direct_down,
        mut hybrid,
    } = candidates(&backend, &catalog)?;
    let input = copy(&backend, &values(HIDDEN)?)?;
    let selected = copy(&backend, &(0..u32::try_from(EXPERTS)?).collect::<Vec<_>>())?;
    let routing = copy(&backend, &[bf16::from_f32(0.125); EXPERTS])?;
    let mut reference_output =
        backend.inner.pool.allocate::<bf16>(&backend.inner.stream, HIDDEN)?;
    let mut actual_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, HIDDEN)?;
    let mut grouped_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, HIDDEN)?;
    let mut fused_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, HIDDEN)?;
    let mut direct_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, HIDDEN)?;
    let mut direct_down_output =
        backend.inner.pool.allocate::<bf16>(&backend.inner.stream, HIDDEN)?;
    let mut hybrid_output = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, HIDDEN)?;
    reference.execute(&input, &selected, &routing, &mut reference_output)?;
    tensor_core.execute(&input, &selected, &routing, &mut actual_output)?;
    grouped.execute(&input, &selected, &routing, &mut grouped_output)?;
    fused.execute(&input, &selected, &routing, &mut fused_output)?;
    direct.execute(&input, &selected, &routing, &mut direct_output)?;
    direct_down.execute(&input, &selected, &routing, &mut direct_down_output)?;
    hybrid.execute(&input, &selected, &routing, &mut hybrid_output)?;
    let expected = read(&backend, &reference_output)?;
    let actual = read(&backend, &actual_output)?;
    let grouped_values = read(&backend, &grouped_output)?;
    let fused_values = read(&backend, &fused_output)?;
    let direct_values = read(&backend, &direct_output)?;
    let direct_down_values = read(&backend, &direct_down_output)?;
    let hybrid_values = read(&backend, &hybrid_output)?;
    assert_reference(&expected, &actual);
    assert_close("grouped W4A4", &actual, &grouped_values);
    assert_eq!(fused_values, grouped_values, "fused grouped W4A4 changed output");
    assert_close("direct W4A4", &grouped_values, &direct_values);
    assert_close("hybrid W4A4", &grouped_values, &hybrid_values);
    assert_close("direct-down W4A4", &grouped_values, &direct_down_values);
    if std::env::var_os("LIBMIR_CUDA_PROFILE_SELECTED_NVFP4").is_some() {
        profile(
            &backend,
            &mut reference,
            &mut tensor_core,
            &mut grouped,
            &mut fused,
            &mut direct,
            &mut direct_down,
            &mut hybrid,
            &input,
            &selected,
            &routing,
            &mut reference_output,
            &mut actual_output,
            &mut grouped_output,
            &mut fused_output,
            &mut direct_output,
            &mut direct_down_output,
            &mut hybrid_output,
        )?;
    }
    Ok(())
}

fn assert_reference(expected: &[bf16], actual: &[bf16]) {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let expected = expected.to_f32();
        let tolerance = (expected.abs() * 0.12).max(2.0);
        assert!((actual.to_f32() - expected).abs() <= tolerance, "output {index}");
    }
}

fn assert_close(name: &str, expected: &[bf16], actual: &[bf16]) {
    for (index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
        let difference = (actual.to_f32() - expected.to_f32()).abs();
        assert!(difference <= 0.5, "{name} output {index}: difference={difference}");
    }
}

#[allow(clippy::print_stderr, clippy::too_many_arguments)]
fn profile(
    backend: &CudaBackend,
    reference: &mut SelectedNvFp4MoeBf16,
    tensor_core: &mut SelectedNvFp4TensorCoreMoeBf16,
    grouped: &mut GroupedNvFp4MoeBf16,
    fused: &mut GroupedNvFp4MoeBf16,
    direct: &mut DirectNvFp4MoeBf16,
    direct_down: &mut DirectDownNvFp4MoeBf16,
    hybrid: &mut HybridNvFp4MoeBf16,
    input: &DeviceBuffer<bf16>,
    selected: &DeviceBuffer<u32>,
    routing: &DeviceBuffer<bf16>,
    reference_output: &mut DeviceBuffer<bf16>,
    tensor_core_output: &mut DeviceBuffer<bf16>,
    grouped_output: &mut DeviceBuffer<bf16>,
    fused_output: &mut DeviceBuffer<bf16>,
    direct_output: &mut DeviceBuffer<bf16>,
    direct_down_output: &mut DeviceBuffer<bf16>,
    hybrid_output: &mut DeviceBuffer<bf16>,
) -> Result<()> {
    const REPEATS: usize = 50;
    let reference_ms = measure(backend, REPEATS, || {
        reference.execute(input, selected, routing, reference_output)
    })?;
    let tensor_core_ms = measure(backend, REPEATS, || {
        tensor_core.execute(input, selected, routing, tensor_core_output)
    })?;
    let grouped_ms =
        measure(backend, REPEATS, || grouped.execute(input, selected, routing, grouped_output))?;
    let fused_ms =
        measure(backend, REPEATS, || fused.execute(input, selected, routing, fused_output))?;
    let direct_ms =
        measure(backend, REPEATS, || direct.execute(input, selected, routing, direct_output))?;
    let direct_down_ms = measure(backend, REPEATS, || {
        direct_down.execute(input, selected, routing, direct_down_output)
    })?;
    let hybrid_ms =
        measure(backend, REPEATS, || hybrid.execute(input, selected, routing, hybrid_output))?;
    eprintln!("selected NVFP4 W4A16: {reference_ms:.3} ms/token");
    eprintln!("selected NVFP4 W4A4:  {tensor_core_ms:.3} ms/token");
    eprintln!("grouped NVFP4 W4A4:   {grouped_ms:.3} ms/token");
    eprintln!("fused grouped W4A4:   {fused_ms:.3} ms/token");
    eprintln!("direct micro W4A4:    {direct_ms:.3} ms/token");
    eprintln!("direct-down W4A4:     {direct_down_ms:.3} ms/token");
    eprintln!("direct-gate W4A4:     {hybrid_ms:.3} ms/token");
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn measure(
    backend: &CudaBackend,
    repeats: usize,
    mut execute: impl FnMut() -> Result<()>,
) -> Result<f32> {
    execute()?;
    backend.synchronize()?;
    let started = backend.inner.context.create_event(true)?;
    let completed = backend.inner.context.create_event(true)?;
    started.record(&backend.inner.stream)?;
    for _ in 0..repeats {
        execute()?;
    }
    completed.record(&backend.inner.stream)?;
    completed.synchronize()?;
    Ok(started.elapsed_ms(&completed)? / repeats as f32)
}
