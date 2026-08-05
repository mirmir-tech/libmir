use std::collections::HashMap;

use uuid::Uuid;

use super::{MixedMixerExecution, required};
use crate::{
    CudaBackend, CudaSharedRoutedModelSession, Error, Result,
    engine::{
        execution::{Output, device_sampling, generation_output},
        model::generation::PrefillChunk,
    },
};

pub(super) fn prefill(
    execution: &mut MixedMixerExecution,
    backend: &CudaBackend,
    chunks: &[PrefillChunk<'_>],
) -> Result<Vec<Option<Output>>> {
    prefill_rows(execution, backend, chunks)
}

pub(super) fn decode(
    execution: &mut MixedMixerExecution,
    backend: &CudaBackend,
    sequences: &[runtime::backend::DecodeSequence],
) -> Result<Option<Vec<Output>>> {
    let mut owned = take_sessions(
        &mut execution.sessions,
        sequences.iter().map(|sequence| sequence.session_id),
    )?;
    let result = (|| {
        let batch = match execution.decode_batches.entry(sequences.len()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(execution.template.prepare_decode_batch(sequences.len())?)
            },
        };
        let mut sessions = owned.iter_mut().map(|(_, session)| session).collect::<Vec<_>>();
        let sampled = batch.execute(&mut sessions, sequences)?;
        if let Some(sampled) = sampled {
            return Ok(Some(
                sampled
                    .into_iter()
                    .map(|token| Output { token: Some(token), logits: None })
                    .collect(),
            ));
        }
        owned
            .iter_mut()
            .zip(sequences)
            .map(|((_, session), sequence)| {
                generation_output(backend, session, sequence.sampling_logits)
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    })();
    restore_sessions(&mut execution.sessions, owned);
    result
}

fn prefill_rows(
    execution: &mut MixedMixerExecution,
    backend: &CudaBackend,
    chunks: &[PrefillChunk<'_>],
) -> Result<Vec<Option<Output>>> {
    if can_pack(chunks) {
        return prefill_packed(execution, chunks);
    }
    chunks
        .iter()
        .map(|chunk| {
            if chunk.offset == 0 {
                let session = execution.template.instantiate_with_caches(&execution.caches)?;
                execution.sessions.insert(chunk.request.session_id, session);
            }
            let session = required(&mut execution.sessions, chunk.request.session_id)?;
            session.prefill(chunk.request.session_id, chunk.tokens, chunk.table)?;
            execution.checkpoint_prefix(chunk.request)?;
            if chunk.final_chunk {
                let session = required(&mut execution.sessions, chunk.request.session_id)?;
                generation_output(backend, session, chunk.request.sampling_logits).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect()
}

fn can_pack(chunks: &[PrefillChunk<'_>]) -> bool {
    let Some(first) = chunks.first() else {
        return false;
    };
    (2..=2).contains(&chunks.len())
        && !first.final_chunk
        && chunks.len().saturating_mul(first.tokens.len()) <= 512
        && chunks.iter().all(|chunk| {
            chunk.tokens.len() == first.tokens.len()
                && chunk.final_chunk == first.final_chunk
                && device_sampling(chunk.request.sampling_logits)
        })
}

fn prefill_packed(
    execution: &mut MixedMixerExecution,
    chunks: &[PrefillChunk<'_>],
) -> Result<Vec<Option<Output>>> {
    for chunk in chunks {
        if chunk.offset == 0 {
            let session = execution.template.instantiate_with_caches(&execution.caches)?;
            execution.sessions.insert(chunk.request.session_id, session);
        }
    }
    let ids = chunks.iter().map(|chunk| chunk.request.session_id).collect::<Vec<_>>();
    let mut owned = take_sessions(&mut execution.sessions, ids.iter().copied())?;
    let result: Result<Option<Vec<Option<Output>>>> = (|| {
        let row_tokens = chunks[0].tokens.len();
        let key = (chunks.len(), row_tokens);
        let batch = match execution.prefill_batches.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(execution.template.prepare_prefill_batch(key.0, key.1)?)
            },
        };
        let tokens =
            chunks.iter().flat_map(|chunk| chunk.tokens.iter().copied()).collect::<Vec<_>>();
        let tables = chunks.iter().map(|chunk| chunk.table).collect::<Vec<_>>();
        let starts = chunks.iter().map(|chunk| chunk.offset).collect::<Vec<_>>();
        let policies = chunks[0]
            .final_chunk
            .then(|| chunks.iter().map(|chunk| chunk.request.sampling_logits).collect::<Vec<_>>());
        let mut sessions = owned.iter_mut().map(|(_, session)| session).collect::<Vec<_>>();
        let sampled =
            batch.execute(&mut sessions, &tokens, &tables, &starts, policies.as_deref())?;
        Ok(sampled.map(|tokens| {
            tokens
                .into_iter()
                .map(|token| Some(Output { token: Some(token), logits: None }))
                .collect()
        }))
    })();
    restore_sessions(&mut execution.sessions, owned);
    let outputs = result?.unwrap_or_else(|| (0..chunks.len()).map(|_| None).collect());
    for chunk in chunks {
        execution.checkpoint_prefix(chunk.request)?;
    }
    Ok(outputs)
}

fn take_sessions(
    sessions: &mut HashMap<Uuid, CudaSharedRoutedModelSession>,
    ids: impl Iterator<Item = Uuid>,
) -> Result<Vec<(Uuid, CudaSharedRoutedModelSession)>> {
    let mut owned = Vec::new();
    for id in ids {
        let Some(session) = sessions.remove(&id) else {
            restore_sessions(sessions, owned);
            return Err(Error::State("CUDA decoder session is not initialized".into()));
        };
        owned.push((id, session));
    }
    Ok(owned)
}

fn restore_sessions(
    sessions: &mut HashMap<Uuid, CudaSharedRoutedModelSession>,
    owned: Vec<(Uuid, CudaSharedRoutedModelSession)>,
) {
    for (id, session) in owned {
        sessions.insert(id, session);
    }
}
