use mircuda::{DeviceBuffer, bf16};
use runtime::backend::SamplingLogits;

use super::*;
use crate::{CudaConfig, Result};

#[test]
fn samples_greedy_top_k_and_nucleus_on_device() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let logits = copy(&backend, &[3.0, 2.0, 1.0, 0.0])?;
    let mut sampler = backend.prepare_device_sampler_bf16(4)?;
    assert_eq!(sample(&backend, &mut sampler, &logits, SamplingLogits::None)?, 0);
    assert_eq!(
        sample(
            &backend,
            &mut sampler,
            &logits,
            SamplingLogits::SampleTopK {
                k: 2,
                vocab_size: 4,
                temperature: 1.0,
                draw: 0.99,
            },
        )?,
        1
    );
    assert_eq!(
        sample(
            &backend,
            &mut sampler,
            &logits,
            SamplingLogits::Sample {
                vocab_size: 4,
                temperature: 1.0,
                top_p: 0.6,
                top_k: 4,
                draw: 0.99,
            },
        )?,
        0
    );
    assert_eq!(
        sample(
            &backend,
            &mut sampler,
            &logits,
            SamplingLogits::Sample {
                vocab_size: 4,
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                draw: 0.99,
            },
        )?,
        3
    );
    Ok(())
}

#[test]
fn full_sampling_preserves_vocab_order_across_blocks() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let logits = copy(&backend, &vec![0.0; 5_000])?;
    let mut sampler = backend.prepare_device_sampler_bf16(5_000)?;
    for (draw, expected) in [(0.0, 0), (0.5, 2_499), (0.99, 4_949)] {
        assert_eq!(
            sample(
                &backend,
                &mut sampler,
                &logits,
                SamplingLogits::Sample {
                    vocab_size: 5_000,
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: 0,
                    draw,
                },
            )?,
            expected
        );
    }
    Ok(())
}

#[test]
fn full_sampling_skips_non_finite_prefixes() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut values = vec![f32::NAN; 5_000];
    values[3_000] = 1.0;
    values[4_000] = 0.0;
    let logits = copy(&backend, &values)?;
    let mut sampler = backend.prepare_device_sampler_bf16(values.len())?;
    assert_eq!(
        sample(
            &backend,
            &mut sampler,
            &logits,
            SamplingLogits::Sample {
                vocab_size: values.len(),
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                draw: 0.0,
            },
        )?,
        3_000
    );
    Ok(())
}

#[test]
fn hierarchical_sampling_preserves_score_and_token_order() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut values = vec![-100.0; 5_000];
    values[4_097] = 4.0;
    values[123] = 4.0;
    values[3_000] = 3.0;
    values[2_000] = 2.0;
    values[1_000] = 1.0;
    let logits = copy(&backend, &values)?;
    let mut sampler = backend.prepare_device_sampler_bf16(values.len())?;
    assert_eq!(sample(&backend, &mut sampler, &logits, SamplingLogits::None)?, 123);
    assert_eq!(
        sample(
            &backend,
            &mut sampler,
            &logits,
            SamplingLogits::SampleTopK {
                k: 4,
                vocab_size: values.len(),
                temperature: 100.0,
                draw: 0.99,
            },
        )?,
        2_000
    );
    Ok(())
}

fn sample(
    backend: &CudaBackend,
    sampler: &mut DeviceSamplerBf16,
    logits: &DeviceBuffer<bf16>,
    policy: SamplingLogits,
) -> Result<u32> {
    let selected = sampler.sample(logits, policy)?;
    let mut host = backend.inner.context.allocate_pinned::<u32>(1)?;
    backend.inner.stream.copy_to_host(selected, &mut host)?;
    Ok(host.to_vec()?[0])
}

fn copy(backend: &CudaBackend, values: &[f32]) -> Result<DeviceBuffer<bf16>> {
    let values = values.iter().map(|value| bf16::from_f32(*value)).collect::<Vec<_>>();
    let mut host = backend.inner.context.allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(&values)?;
    let mut device = backend.inner.pool.allocate::<bf16>(&backend.inner.stream, values.len())?;
    backend.inner.stream.copy_to_device(&mut host, &mut device)?;
    Ok(device)
}
