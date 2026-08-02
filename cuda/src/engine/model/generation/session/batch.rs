use runtime::backend::{DecodeSequence, LogitsTrace, SamplingLogits};

use crate::{
    CudaBackend, CudaClampedRoutedModelSession, Error, Result,
    engine::execution::{Output, device_sampling, generation_output},
};

pub(super) fn decode(
    session: &mut CudaClampedRoutedModelSession,
    backend: &CudaBackend,
    sequences: &[DecodeSequence],
) -> Result<Option<Vec<Output>>> {
    if sequences.len() < 2 {
        return Ok(None);
    }
    let sessions = sequences.iter().map(|sequence| sequence.session_id).collect::<Vec<_>>();
    let tokens = sequences.iter().map(|sequence| sequence.token_id).collect::<Vec<_>>();
    let tables = sequences.iter().map(|sequence| &sequence.block_table).collect::<Vec<_>>();
    let policies = sequences.iter().map(|sequence| sequence.sampling_logits).collect::<Vec<_>>();
    session.decode_packed_chunk(&sessions, &tokens, &tables)?;
    let device_policies = policies
        .iter()
        .copied()
        .map(|policy| {
            if device_sampling(policy) {
                policy
            } else {
                SamplingLogits::None
            }
        })
        .collect::<Vec<_>>();
    let rows = (0..sequences.len()).collect::<Vec<_>>();
    let history = policies.iter().any(|policy| policy.requires_history());
    let Some(result) =
        session.finish_packed_device_rows(&rows, sequences.len(), &device_policies, history)?
    else {
        let outputs = policies
            .iter()
            .copied()
            .enumerate()
            .map(|(row, policy)| {
                session.finish_packed_prefill_row(row, sequences.len(), policy)?;
                generation_output(backend, session, policy)
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Some(outputs));
    };
    policies
        .iter()
        .copied()
        .enumerate()
        .map(|(row, policy)| {
            let token = device_sampling(policy).then(|| result.selected[row]);
            let logits = if policy.requires_history() {
                let values = result
                    .logits
                    .as_deref()
                    .and_then(|values| values.chunks_exact(result.vocab).nth(row))
                    .ok_or_else(|| Error::InvalidSampling("missing packed logits row".into()))?;
                Some(LogitsTrace {
                    shape: vec![1, 1, i32::try_from(result.vocab)?],
                    values: values.to_vec(),
                })
            } else {
                None
            };
            Ok(Output { token, logits })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}
