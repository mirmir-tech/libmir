use std::collections::BTreeMap;

use runtime::{
    backend::{DecodeOutput, LogitsTrace, SamplingLogits, TokenEvent},
    kv::{BlockId, BlockTable, CacheConfig},
};

use super::execution::device_sampling;
use crate::{CudaDecodeBatch, CudaMoeModelTemplate, Error, PagedKvCache, Result};

mod execute;

pub(super) struct DecodeBuckets {
    buckets: BTreeMap<usize, CudaDecodeBatch>,
}

impl DecodeBuckets {
    pub(super) fn prepare(
        template: &CudaMoeModelTemplate,
        caches: &[PagedKvCache],
        maximum: usize,
        cache: CacheConfig,
    ) -> Result<Self> {
        let maximum = maximum.min(usize::try_from(cache.block_count)?);
        let mut buckets = BTreeMap::new();
        for size in bucket_sizes(maximum) {
            let mut bucket = template.instantiate_decode_batch_with_caches(size, caches)?;
            warmup(&mut bucket, cache)?;
            buckets.insert(size, bucket);
        }
        Ok(Self { buckets })
    }

    pub(super) fn largest_at_most(&self, rows: usize) -> Option<usize> {
        self.buckets.range(..=rows).next_back().map(|(size, _)| *size)
    }

    pub(super) fn get_mut(&mut self, rows: usize) -> Result<&mut CudaDecodeBatch> {
        self.buckets
            .get_mut(&rows)
            .ok_or(Error::InvalidDecoderKernel("CUDA decode bucket is not prepared"))
    }
}

fn sample_policies(policies: &[SamplingLogits]) -> Vec<SamplingLogits> {
    if !policies.iter().copied().any(device_sampling) {
        return Vec::new();
    }
    policies
        .iter()
        .copied()
        .map(|policy| {
            if device_sampling(policy) {
                policy
            } else {
                SamplingLogits::None
            }
        })
        .collect()
}

fn build_outputs(
    policies: &[SamplingLogits],
    selected: &[u32],
    logits: Option<&[f32]>,
    vocab: usize,
) -> Result<Vec<DecodeOutput>> {
    policies
        .iter()
        .copied()
        .enumerate()
        .map(|(row, policy)| {
            let token =
                if device_sampling(policy) {
                    Some(*selected.get(row).ok_or_else(|| {
                        Error::InvalidSampling("missing sampled batch row".into())
                    })?)
                } else {
                    None
                };
            let logits = if policy.requires_history() {
                let values = logits
                    .and_then(|values| values.chunks_exact(vocab).nth(row))
                    .ok_or_else(|| Error::InvalidSampling("missing logits batch row".into()))?;
                Some(LogitsTrace {
                    shape: vec![1, 1, i32::try_from(vocab)?],
                    values: values.to_vec(),
                })
            } else {
                None
            };
            Ok(DecodeOutput {
                event: TokenEvent {
                    token_id: token,
                    text: "cuda.decode=batch-device-token-pipeline".into(),
                    finished: false,
                },
                logits,
                candidates: None,
                timings: None,
            })
        })
        .collect()
}

fn bucket_sizes(maximum: usize) -> impl Iterator<Item = usize> {
    let mut sizes = std::iter::successors(Some(2_usize), |size| size.checked_mul(2))
        .take_while(|size| *size <= maximum)
        .collect::<Vec<_>>();
    sizes.extend([5, 10, maximum].into_iter().filter(|size| (2..=maximum).contains(size)));
    sizes.sort_unstable();
    sizes.dedup();
    sizes.into_iter()
}

fn warmup(batch: &mut CudaDecodeBatch, cache: CacheConfig) -> Result<()> {
    let tokens = vec![0; batch.batch_size()];
    let mut tables = Vec::with_capacity(batch.batch_size());
    for index in 0..batch.batch_size() {
        let mut table = BlockTable::with_block_size(cache.block_size);
        table.push(BlockId(u32::try_from(index)?));
        table.set_token_len(1);
        tables.push(table);
    }
    let references = tables.iter().collect::<Vec<_>>();
    batch.decode(&tokens, &references)?;
    batch.sample(&vec![SamplingLogits::None; batch.batch_size()])?;
    Ok(())
}

#[cfg(test)]
mod tests;
