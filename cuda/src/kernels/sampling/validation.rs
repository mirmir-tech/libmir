use mircuda::LaunchConfig;

use super::{CHUNK, MAX_TOP_K, SamplingSpec, THREADS};
use crate::{Error, Result, kernels::geometry::narrow};

pub(super) fn blocks(elements: usize) -> Result<usize> {
    elements
        .checked_add(CHUNK - 1)
        .map(|padded| padded / CHUNK)
        .ok_or_else(|| Error::InvalidSampling("sampling grid overflow".into()))
}

pub(super) fn launch(block_count: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(block_count)?, 1, 1),
        block: (THREADS, 1, 1),
        shared_memory_bytes: 0,
    })
}

pub(super) fn validate(spec: SamplingSpec) -> Result<()> {
    if spec.vocab == 0
        || spec.top_k > spec.vocab
        || spec.top_k > MAX_TOP_K
        || (spec.top_k == 0 && spec.top_p < 1.0)
        || !spec.top_p.is_finite()
        || spec.top_p <= 0.0
        || spec.top_p > 1.0
        || !spec.temperature.is_finite()
        || spec.temperature <= 0.0
        || !spec.draw.is_finite()
        || !(0.0..1.0).contains(&spec.draw)
    {
        Err(Error::InvalidSampling("invalid bounded CUDA sampling policy".into()))
    } else {
        Ok(())
    }
}
