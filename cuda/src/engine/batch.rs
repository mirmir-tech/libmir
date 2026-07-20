use std::{collections::BTreeMap, time::Instant};

use runtime::{
    backend::{
        DecodeBatchOutput, DecodeBatchRequest, DecodeOutput, DecodeRequest, DecodeSequence,
        LogitsTrace, SamplingLogits, TokenEvent,
    },
    kv::{BlockId, BlockTable, CacheConfig},
};

use super::{CudaEngine, execution::device_sampling, model::ModelRunner};
use crate::{CudaDecodeBatch, CudaMoeModelTemplate, Error, PagedKvCache, Result};

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

impl CudaEngine {
    pub fn decode_batch_tokens(&self, request: &DecodeBatchRequest) -> Result<DecodeBatchOutput> {
        let loaded = self.model(&request.model().id)?;
        for sequence in request.sequences() {
            loaded.require_session(sequence.session_id)?;
        }
        let waiting = Instant::now();
        let mut runner = loaded.decode_runner()?;
        let wait = waiting.elapsed();
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(request.sequences().len());
        let mut offset = 0;
        while offset < request.sequences().len() {
            let remaining = request.sequences().len() - offset;
            if let Some(rows) =
                runner.batches.as_ref().and_then(|batches| batches.largest_at_most(remaining))
            {
                outputs.extend(
                    self.execute_bucket(&mut runner, &request.sequences()[offset..offset + rows])?,
                );
                offset += rows;
            } else {
                outputs.push(self.execute_scalar(&mut runner, request, offset)?);
                offset += 1;
            }
        }
        tracing::debug!(
            backend = "cuda",
            rows = request.sequences().len(),
            runner_wait_ms = wait.as_secs_f64() * 1_000.0,
            execution_ms = started.elapsed().as_secs_f64() * 1_000.0,
            "completed CUDA decode batch"
        );
        Ok(DecodeBatchOutput::new(outputs)?)
    }

    fn execute_bucket(
        &self,
        runner: &mut ModelRunner,
        sequences: &[DecodeSequence],
    ) -> Result<Vec<DecodeOutput>> {
        let rows = sequences.len();
        let tokens = sequences.iter().map(|item| item.token_id).collect::<Vec<_>>();
        let tables = sequences.iter().map(|item| &item.block_table).collect::<Vec<_>>();
        let policies = sequences.iter().map(|item| item.sampling_logits).collect::<Vec<_>>();
        let bucket = runner
            .batches
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("CUDA model has no decode batches"))?
            .get_mut(rows)?;
        bucket.decode(&tokens, &tables)?;
        let history = policies.iter().any(|policy| policy.requires_history());
        let logits = history.then(|| self.backend.read_logits(bucket.logits()?)).transpose()?;
        let sampled = sample_policies(&policies);
        let selected = if sampled.is_empty() {
            Vec::new()
        } else {
            self.backend.read_tokens(bucket.sample(&sampled)?)?
        };
        runner.selected = None;
        let vocab = bucket.logits()?.len() / rows;
        tracing::debug!(rows, occupancy = 1.0_f64, "executed warmed CUDA decode bucket");
        build_outputs(&policies, &selected, logits.as_deref(), vocab)
    }

    fn execute_scalar(
        &self,
        runner: &mut ModelRunner,
        request: &DecodeBatchRequest,
        index: usize,
    ) -> Result<DecodeOutput> {
        let sequence = &request.sequences()[index];
        self.decode_with_runner(
            runner,
            &DecodeRequest {
                model: request.model().clone(),
                session_id: sequence.session_id,
                token_id: sequence.token_id,
                block_table: sequence.block_table.clone(),
                sampling_logits: sequence.sampling_logits,
            },
        )
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
            })
        })
        .collect()
}

fn bucket_sizes(maximum: usize) -> impl Iterator<Item = usize> {
    std::iter::successors(Some(2_usize), |size| size.checked_mul(2))
        .take_while(move |size| *size <= maximum)
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
mod tests {
    use runtime::backend::SamplingLogits;

    use super::{bucket_sizes, build_outputs, sample_policies};

    #[test]
    fn prepares_binary_decode_buckets() {
        assert_eq!(bucket_sizes(1).collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(bucket_sizes(7).collect::<Vec<_>>(), [2, 4]);
        assert_eq!(bucket_sizes(16).collect::<Vec<_>>(), [2, 4, 8, 16]);
    }

    #[test]
    fn preserves_mixed_host_and_device_sampling_rows() -> crate::Result<()> {
        let policies = [SamplingLogits::Full, SamplingLogits::None];
        assert_eq!(sample_policies(&policies), [SamplingLogits::None, SamplingLogits::None]);
        let outputs = build_outputs(&policies, &[7, 8], Some(&[1.0, 2.0, 3.0, 4.0]), 2)?;
        assert_eq!(outputs[0].event.token_id, None);
        assert_eq!(
            outputs[0].logits.as_ref().map(|trace| trace.values.as_slice()),
            Some(&[1.0, 2.0][..])
        );
        assert_eq!(outputs[1].event.token_id, Some(8));
        assert!(outputs[1].logits.is_none());
        Ok(())
    }
}
