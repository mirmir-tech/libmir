use runtime::backend::{DeviceSampling, TokenMask};

use super::*;

fn masked(bits: u32, vocab: usize, sampling: DeviceSampling) -> Result<SamplingLogits> {
    Ok(SamplingLogits::Masked {
        mask: TokenMask::new(vec![bits], vocab)?,
        sampling,
    })
}

#[test]
fn masking_preserves_original_logits_and_changes_each_step() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let logits = copy(&backend, &[100.0, 4.0, 3.0, 2.0])?;
    let mut sampler = backend.prepare_device_sampler_bf16(4)?;
    assert_eq!(
        sample(&backend, &mut sampler, &logits, masked(0b0110, 4, DeviceSampling::Greedy)?)?,
        1
    );
    assert_eq!(
        sample(&backend, &mut sampler, &logits, masked(0b1000, 4, DeviceSampling::Greedy)?)?,
        3
    );
    assert_eq!(sample(&backend, &mut sampler, &logits, SamplingLogits::None)?, 0);
    assert_eq!(
        sample(
            &backend,
            &mut sampler,
            &logits,
            masked(
                0b0110,
                4,
                DeviceSampling::Random {
                    temperature: 100.0,
                    top_p: 1.0,
                    top_k: 4,
                    draw: 0.99
                }
            )?
        )?,
        2
    );
    Ok(())
}

#[test]
fn packed_rows_keep_independent_masks_and_unmasked_requests() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let logits = copy(&backend, &[100.0, 2.0, 3.0, 500.0, 0.0, 4.0, 3.0, 100.0])?;
    let mut sampler = backend.prepare_device_batch_sampler_bf16(4, 2)?;
    let policies = [masked(0b110, 3, DeviceSampling::Greedy)?, SamplingLogits::None];
    let selected = sampler.sample(&logits, &policies)?;
    let mut host = backend.inner.context.allocate_pinned(2)?;
    backend.inner.stream.copy_to_host(selected, &mut host)?;
    assert_eq!(host.to_vec()?, [2, 3]);
    let policies = [SamplingLogits::None, masked(0b010, 4, DeviceSampling::Greedy)?];
    let selected = sampler.sample(&logits, &policies)?;
    backend.inner.stream.copy_to_host(selected, &mut host)?;
    assert_eq!(host.to_vec()?, [3, 1]);
    assert!(TokenMask::new(vec![0], 4).is_err());
    assert!(TokenMask::new(vec![0b1000], 3).is_err());
    assert!(TokenMask::new(vec![1, 0], 4).is_err());
    Ok(())
}
