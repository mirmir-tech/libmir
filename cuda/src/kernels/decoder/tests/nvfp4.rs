use mircuda::bf16;

use super::{Result, RmsNorm, runtime::Runtime};
use crate::kernels::{
    BucketGeometry, BucketQuantize, GatedActivation, NvFp4BucketPreparation, NvFp4Gated,
    NvFp4Preparation, NvFp4RmsNorm, PackedGatedBf16, scale_elements,
};

const ROWS: usize = 2;
const COLUMNS: usize = 128;
const EPSILON: f32 = 1.0e-6;

#[test]
fn fused_rms_norm_nvfp4_matches_bf16_intermediate() -> Result<()> {
    let runtime = Runtime::new()?;
    let values = (0..ROWS * COLUMNS)
        .map(|index| {
            let value = f32::from(u16::try_from(index % 29)?) / 16.0 - 0.875;
            Ok(bf16::from_f32(value))
        })
        .collect::<Result<Vec<_>>>()?;
    let weights = (0..COLUMNS)
        .map(|index| {
            let value = f32::from(u16::try_from(index % 11)?) / 32.0 + 0.75;
            Ok(bf16::from_f32(value))
        })
        .collect::<Result<Vec<_>>>()?;
    let input = runtime.copy(&values)?;
    let weight = runtime.copy(&weights)?;
    let global_scale = runtime.copy(&[0.25_f32])?;
    let mut normalized = runtime.pool.allocate(&runtime.stream, values.len())?;
    let mut expected_packed = runtime.pool.allocate(&runtime.stream, values.len() / 2)?;
    let scale_count = scale_elements(ROWS, COLUMNS)?;
    let mut expected_scales = runtime.pool.allocate_zeroed(&runtime.stream, scale_count)?;
    RmsNorm::compile(&runtime.compiler, ROWS, COLUMNS, EPSILON)?
        .execute(&runtime.stream, &input, &weight, &mut normalized)?;
    NvFp4Preparation::compile(&runtime.compiler)?.quantize(
        &runtime.stream,
        ROWS,
        COLUMNS,
        &normalized,
        &global_scale,
        &mut expected_packed,
        &mut expected_scales,
    )?;
    let mut actual_packed = runtime.pool.allocate(&runtime.stream, values.len() / 2)?;
    let mut actual_scales = runtime.pool.allocate_zeroed(&runtime.stream, scale_count)?;
    let mut inverse = runtime.pool.allocate(&runtime.stream, ROWS)?;
    NvFp4RmsNorm::compile(&runtime.compiler, ROWS, COLUMNS)?.execute(
        &runtime.stream,
        &input,
        &weight,
        &global_scale,
        &mut inverse,
        &mut actual_packed,
        &mut actual_scales,
        EPSILON,
    )?;
    assert_eq!(runtime.read(&actual_packed)?, runtime.read(&expected_packed)?);
    assert_eq!(runtime.read(&actual_scales)?, runtime.read(&expected_scales)?);
    Ok(())
}
#[test]
fn fused_gated_nvfp4_matches_bf16_intermediate() -> Result<()> {
    let runtime = Runtime::new()?;
    let gate = (0..ROWS * COLUMNS)
        .map(|index| {
            let value = f32::from(u16::try_from(index % 31)?) / 16.0 - 0.9375;
            Ok(bf16::from_f32(value))
        })
        .collect::<Result<Vec<_>>>()?;
    let up = (0..ROWS * COLUMNS)
        .map(|index| {
            let value = f32::from(u16::try_from(index % 17)?) / 8.0 - 1.0;
            Ok(bf16::from_f32(value))
        })
        .collect::<Result<Vec<_>>>()?;
    let gate = runtime.copy(&gate)?;
    let up = runtime.copy(&up)?;
    let global_scale = runtime.copy(&[0.25_f32])?;
    let mut activated = runtime.pool.allocate(&runtime.stream, ROWS * COLUMNS)?;
    let mut expected_packed = runtime.pool.allocate(&runtime.stream, ROWS * COLUMNS / 2)?;
    let scale_count = scale_elements(ROWS, COLUMNS)?;
    let mut expected_scales = runtime.pool.allocate_zeroed(&runtime.stream, scale_count)?;
    PackedGatedBf16::compile(&runtime.compiler, ROWS, COLUMNS)?.execute_separate(
        &runtime.stream,
        &gate,
        &up,
        &mut activated,
        GatedActivation::Silu,
    )?;
    NvFp4Preparation::compile(&runtime.compiler)?.quantize(
        &runtime.stream,
        ROWS,
        COLUMNS,
        &activated,
        &global_scale,
        &mut expected_packed,
        &mut expected_scales,
    )?;
    let mut actual_packed = runtime.pool.allocate(&runtime.stream, ROWS * COLUMNS / 2)?;
    let mut actual_scales = runtime.pool.allocate_zeroed(&runtime.stream, scale_count)?;
    NvFp4Gated::compile(&runtime.compiler, ROWS, COLUMNS)?.execute(
        &runtime.stream,
        &gate,
        &up,
        &global_scale,
        &mut actual_packed,
        &mut actual_scales,
        GatedActivation::Silu,
    )?;
    assert_eq!(runtime.read(&actual_packed)?, runtime.read(&expected_packed)?);
    assert_eq!(runtime.read(&actual_scales)?, runtime.read(&expected_scales)?);
    Ok(())
}
#[test]
fn unique_quantization_gather_matches_bucket_quantization() -> Result<()> {
    const TOKENS: usize = 3;
    const SELECTED: usize = 2;
    const EXPERTS: usize = 3;
    const COLUMNS: usize = 64;
    let runtime = Runtime::new()?;
    let values = (0..TOKENS * COLUMNS)
        .map(|index| {
            let value = f32::from(u16::try_from(index % 37)?) / 16.0 - 1.0;
            Ok(bf16::from_f32(value))
        })
        .collect::<Result<Vec<_>>>()?;
    let input = runtime.copy(&values)?;
    let selected = runtime.copy(&[0_u32, 1, 2, 0, 1, 2])?;
    let globals = runtime.copy(&[0.25_f32; EXPERTS])?;
    let assignments = TOKENS * SELECTED;
    let mut counts = runtime.pool.allocate_zeroed(&runtime.stream, EXPERTS)?;
    let mut offsets = runtime.pool.allocate_zeroed(&runtime.stream, EXPERTS)?;
    let mut scale_offsets = runtime.pool.allocate_zeroed(&runtime.stream, EXPERTS)?;
    let mut order = runtime.pool.allocate_zeroed(&runtime.stream, assignments)?;
    let mut positions = runtime.pool.allocate_zeroed(&runtime.stream, assignments)?;
    let mut indices = runtime.pool.allocate_zeroed(&runtime.stream, EXPERTS)?;
    let bucket = NvFp4BucketPreparation::compile(&runtime.compiler)?;
    bucket.prepare(
        &runtime.stream,
        &selected,
        &mut counts,
        &mut offsets,
        &mut scale_offsets,
        &mut order,
        &mut positions,
        &mut indices,
        BucketGeometry { assignments, experts: EXPERTS },
    )?;
    let geometry = BucketQuantize {
        assignments,
        experts: EXPERTS,
        selected: SELECTED,
        input_rows: TOKENS,
        columns: COLUMNS,
        ranked: false,
    };
    let target_scales = scale_elements(EXPERTS * 128, COLUMNS)?;
    let mut expected = runtime.pool.allocate_zeroed(&runtime.stream, assignments * COLUMNS / 2)?;
    let mut expected_scales = runtime.pool.allocate_zeroed(&runtime.stream, target_scales)?;
    bucket.quantize(
        &runtime.stream,
        &input,
        &selected,
        &order,
        &offsets,
        &scale_offsets,
        &globals,
        &mut expected,
        &mut expected_scales,
        geometry,
    )?;
    let mut actual = runtime.pool.allocate_zeroed(&runtime.stream, assignments * COLUMNS / 2)?;
    let mut actual_scales = runtime.pool.allocate_zeroed(&runtime.stream, target_scales)?;
    let mut unique = runtime.pool.allocate_zeroed(&runtime.stream, TOKENS * COLUMNS / 2)?;
    let mut unique_scales = runtime
        .pool
        .allocate_zeroed(&runtime.stream, scale_elements(TOKENS, COLUMNS)?)?;
    NvFp4Preparation::compile(&runtime.compiler)?.quantize(
        &runtime.stream,
        TOKENS,
        COLUMNS,
        &input,
        &globals,
        &mut unique,
        &mut unique_scales,
    )?;
    bucket.gather_quantized(
        &runtime.stream,
        &selected,
        &order,
        &offsets,
        &scale_offsets,
        &unique,
        &unique_scales,
        &mut actual,
        &mut actual_scales,
        geometry,
    )?;
    assert_eq!(runtime.read(&actual)?, runtime.read(&expected)?);
    assert_eq!(runtime.read(&actual_scales)?, runtime.read(&expected_scales)?);
    Ok(())
}
#[test]
#[allow(clippy::print_stderr)]
fn profile_fused_rms_norm_nvfp4() -> Result<()> {
    if std::env::var_os("LIBMIR_CUDA_PROFILE_RMS_NVFP4").is_none() {
        return Ok(());
    }
    let runtime = Runtime::new()?;
    for (rows, columns) in [(1, 4_096), (2, 4_096), (1, 2_816), (2, 2_816)] {
        let (separate, fused) = profile(&runtime, rows, columns)?;
        eprintln!(
            "rows={rows} columns={columns} separate={separate:.3} us fused={fused:.3} us speedup={:.3}x",
            separate / fused,
        );
    }
    Ok(())
}
fn profile(runtime: &Runtime, rows: usize, columns: usize) -> Result<(f32, f32)> {
    const CYCLES: usize = 512;
    let values = vec![bf16::from_f32(0.25); rows * columns];
    let weights = vec![bf16::from_f32(0.875); columns];
    let input = runtime.copy(&values)?;
    let weight = runtime.copy(&weights)?;
    let global_scale = runtime.copy(&[0.25_f32])?;
    let mut normalized = runtime.pool.allocate(&runtime.stream, values.len())?;
    let mut packed = runtime.pool.allocate(&runtime.stream, values.len() / 2)?;
    let mut scales =
        runtime.pool.allocate_zeroed(&runtime.stream, scale_elements(rows, columns)?)?;
    let mut inverse = runtime.pool.allocate(&runtime.stream, rows)?;
    let norm = RmsNorm::compile(&runtime.compiler, rows, columns, EPSILON)?;
    let preparation = NvFp4Preparation::compile(&runtime.compiler)?;
    let separate = runtime.measure(CYCLES, || {
        norm.execute(&runtime.stream, &input, &weight, &mut normalized)?;
        preparation.quantize(
            &runtime.stream, rows, columns, &normalized, &global_scale, &mut packed, &mut scales,
        )
    })?;
    let fused = NvFp4RmsNorm::compile(&runtime.compiler, rows, columns)?;
    let fused = runtime.measure(CYCLES, || {
        fused.execute(
            &runtime.stream, &input, &weight, &global_scale, &mut inverse, &mut packed,
            &mut scales, EPSILON,
        )
    })?;
    Ok((separate, fused))
}
