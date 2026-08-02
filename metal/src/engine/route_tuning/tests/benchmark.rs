use std::{hint::black_box, time::Instant};

use super::{Result, Stream, mlp, values, weights};
use crate::engine::Array;

const ITERATIONS: u32 = 20;

#[test]
#[ignore = "synthetic GPU benchmark"]
#[allow(clippy::print_stdout)]
fn benchmarks_fused_restore_reduction() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let projections = weights(8, 256, 512, &stream)?;
    for tokens in [16_usize, 64, 256] {
        let input = Array::from_f32(&values(tokens * 256), &[1, i32::try_from(tokens)?, 256])?;
        let routes = tokens * 4;
        let indices = (0..routes)
            .map(|index| Ok(u32::try_from(index % 8)?))
            .collect::<Result<Vec<_>>>()?;
        let indices = Array::from_u32(&indices, &[1, i32::try_from(tokens)?, 4])?;
        let routing = Array::from_f32(&vec![0.25; routes], &[1, i32::try_from(tokens)?, 4])?;
        let sorted = input.sort_expert_inputs(&indices, &stream)?;
        let output = mlp(&projections, &sorted.input, &sorted.indices, true, &stream)?;
        let graph = measure(&stream, || {
            sorted.restore(&output, &stream)?.weighted_sum(&routing, -2, &stream)
        })?;
        let fused = measure(&stream, || sorted.restore_weighted(&output, &routing, &stream))?;
        println!(
            "metal_expert_reduce_profile tokens={tokens} routes={routes} graph_us={graph:.3} fused_us={fused:.3}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "synthetic GPU benchmark"]
#[allow(clippy::print_stdout)]
fn benchmarks_kernel_grouping_against_argsort() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for experts in [8_usize, 32, 128] {
        for tokens in [16_usize, 64, 256] {
            let input = Array::from_f32(&values(tokens * 64), &[1, i32::try_from(tokens)?, 64])?;
            let routes = tokens * 4;
            for skewed in [false, true] {
                let indices = (0..routes)
                    .map(|route| {
                        let expert = if skewed && route % 4 != 0 {
                            0
                        } else {
                            route % experts
                        };
                        Ok(u32::try_from(expert)?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let indices = Array::from_u32(&indices, &[1, i32::try_from(tokens)?, 4])?;
                let sorted = measure_pair(&stream, || {
                    let grouped = input.sort_expert_inputs(&indices, &stream)?;
                    Ok((grouped.input, grouped.indices))
                })?;
                let kernel = measure_pair(&stream, || {
                    let grouped = input.group_expert_inputs(&indices, experts, &stream)?;
                    Ok((grouped.input, grouped.indices))
                })?;
                println!(
                    "metal_expert_group_profile experts={experts} tokens={tokens} routes={routes} \
                     skewed={skewed} argsort_us={sorted:.3} kernel_us={kernel:.3}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "synthetic GPU benchmark"]
#[allow(clippy::print_stdout)]
fn benchmarks_kernel_grouping_in_full_expert_path() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let projections = weights(8, 256, 512, &stream)?;
    for tokens in [16_usize, 64, 256] {
        let input = Array::from_f32(&values(tokens * 256), &[1, i32::try_from(tokens)?, 256])?;
        let routes = tokens * 4;
        let routing = Array::from_f32(&vec![0.25; routes], &[1, i32::try_from(tokens)?, 4])?;
        for skewed in [false, true] {
            let indices = (0..routes)
                .map(|route| {
                    let expert = if skewed && route % 4 != 0 {
                        0
                    } else {
                        route % 8
                    };
                    Ok(u32::try_from(expert)?)
                })
                .collect::<Result<Vec<_>>>()?;
            let indices = Array::from_u32(&indices, &[1, i32::try_from(tokens)?, 4])?;
            let argsort = measure(&stream, || {
                let grouped = input.sort_expert_inputs(&indices, &stream)?;
                let output = mlp(&projections, &grouped.input, &grouped.indices, true, &stream)?;
                grouped.restore_weighted(&output, &routing, &stream)
            })?;
            let kernel = measure(&stream, || {
                let grouped = input.group_expert_inputs(&indices, 8, &stream)?;
                let output = mlp(&projections, &grouped.input, &grouped.indices, true, &stream)?;
                grouped.restore_weighted(&output, &routing, &stream)
            })?;
            println!(
                "metal_expert_kernel_path_profile tokens={tokens} routes={routes} skewed={skewed} \
                 argsort_us={argsort:.3} kernel_us={kernel:.3}"
            );
        }
    }
    Ok(())
}

fn measure(stream: &Stream, operation: impl Fn() -> Result<Array>) -> Result<f64> {
    for _ in 0..3 {
        operation()?.async_eval()?;
    }
    stream.synchronize()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        black_box(operation()?).async_eval()?;
    }
    stream.synchronize()?;
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(ITERATIONS))
}

fn measure_pair(stream: &Stream, operation: impl Fn() -> Result<(Array, Array)>) -> Result<f64> {
    for _ in 0..3 {
        let (first, second) = operation()?;
        first.async_eval()?;
        second.async_eval()?;
    }
    stream.synchronize()?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let (first, second) = black_box(operation()?);
        first.async_eval()?;
        second.async_eval()?;
    }
    stream.synchronize()?;
    Ok(started.elapsed().as_secs_f64() * 1_000_000.0 / f64::from(ITERATIONS))
}
