mod cancellation;
mod churn;
mod fixture;
mod history;
mod lifecycle;
mod parity;
mod recovery;
mod reservation;
mod router;

use fixture::load;
use runtime::{
    backend::{ModelHandle, PrefillRequest, SamplingLogits},
    kv::BlockTable,
};
use uuid::Uuid;

use super::MetalPrefillBatch;
use crate::native::{
    error::{Error, Result},
    model::{LoadedModel, NativeOutput},
};

#[test]
fn preserves_cohort_positions_with_remainder_and_finishes_tiny_budgets() -> Result<()> {
    let (mut model, directory) = load()?;
    let mut reference = None;
    for budget in [15, 17, 1] {
        let requests = (0..5)
            .map(|row| {
                let request = PrefillRequest {
                    model: ModelHandle {
                        id: model.info.manifest.id.clone(),
                        backend: "metal".into(),
                    },
                    session_id: Uuid::new_v4(),
                    prompt_tokens: (0..64u32).map(|index| (index + row) % 64).collect(),
                    cache_checkpoints: Vec::new(),
                    block_table: BlockTable::with_block_size(16),
                    cached_tokens: 0,
                    generation_tokens: None,
                    sampling_logits: SamplingLogits::None,
                };
                (request, SamplingLogits::None)
            })
            .collect();
        let (batch, _) = MetalPrefillBatch::prepare(&mut model, requests, None)?;
        batch.preserve_cohort_for_benchmark()?;
        let step = batch.execute_step(&mut model, budget)?;
        assert!(!step.complete);
        {
            let guard = batch.inner.lock()?;
            let state = guard
                .as_ref()
                .ok_or_else(|| Error::InvalidPrefillBatch("missing batch".into()))?;
            let positions = state.sequences.iter().map(|row| row.position).collect::<Vec<_>>();
            drop(guard);
            if budget == 1 {
                assert_eq!(positions.iter().sum::<usize>(), 1);
            } else {
                assert_eq!(positions, vec![3; 5], "remainder split the cohort");
            }
        }
        let actual = finish(&batch, &mut model, budget)?;
        if budget != 1 {
            if let Some(expected) = &reference {
                assert_eq!(&actual, expected, "deferring the remainder changed outputs");
            }
            reference = Some(actual);
        }
    }
    drop(model);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

fn finish(batch: &MetalPrefillBatch, model: &mut LoadedModel, budget: usize) -> Result<Vec<u32>> {
    for _ in 0..512 {
        if !batch.execute_step(model, budget)?.complete {
            continue;
        }
        return batch
            .finish()?
            .into_iter()
            .map(|row| {
                let token = match row.native.output {
                    NativeOutput::Greedy(token) => token,
                    NativeOutput::Logits(logits) => logits.argmax_u32(model.stream())?,
                };
                model.release_session(row.request.session_id)?;
                Ok(token)
            })
            .collect();
    }
    Err(Error::InvalidPrefillBatch("small budget made no progress".into()))
}
