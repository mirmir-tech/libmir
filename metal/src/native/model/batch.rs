use std::collections::HashSet;

use runtime::backend::SamplingLogits;
use uuid::Uuid;

use super::{LoadedModel, NativeOutput};
use crate::{
    engine::Array,
    native::{
        error::{Error, Result},
        session::{PendingDecode, SessionState},
        step,
    },
};

#[derive(Clone, Copy)]
pub struct DecodeInput {
    pub session: Uuid,
    pub token: u32,
    pub sampling: SamplingLogits,
}

impl LoadedModel {
    pub(crate) fn can_decode_batch(&self, inputs: &[DecodeInput]) -> bool {
        inputs.len() > 1
            && self
                .execution
                .decoder()
                .is_ok_and(|model| model.prefers_packed_decode(&self.stream))
            && inputs.iter().all(|input| {
                step::supports_device_token(input.sampling)
                    && self
                        .sessions
                        .get(&input.session)
                        .is_some_and(|state| state.pending.is_some())
            })
    }

    pub(crate) fn decode_batch(&mut self, inputs: &[DecodeInput]) -> Result<Vec<NativeOutput>> {
        self.validate_batch(inputs)?;
        let mut states = self.take_batch_states(inputs)?;
        let result = self
            .execution
            .decoder()
            .and_then(|model| decode_states(model, &self.stream, inputs, &mut states));
        for (session, state) in states {
            self.sessions.insert(session, state);
        }
        result
    }

    fn take_batch_states(&mut self, inputs: &[DecodeInput]) -> Result<Vec<(Uuid, SessionState)>> {
        let mut states = Vec::with_capacity(inputs.len());
        for input in inputs {
            let Some(state) = self.sessions.remove(&input.session) else {
                for (session, state) in states {
                    self.sessions.insert(session, state);
                }
                return Err(Error::Session {
                    model: self.info.manifest.id.clone(),
                    session: input.session,
                });
            };
            states.push((input.session, state));
        }
        Ok(states)
    }

    fn validate_batch(&self, inputs: &[DecodeInput]) -> Result<()> {
        if !self.can_decode_batch(inputs) {
            return Err(Error::InvalidDecodeBatch(
                "packed batch requires device sampling and initialized session state".into(),
            ));
        }
        let mut sessions = HashSet::with_capacity(inputs.len());
        for input in inputs {
            if !sessions.insert(input.session) {
                return Err(Error::InvalidDecodeBatch("session occurs more than once".into()));
            }
            let state = self.session(input.session)?;
            let pending = state.pending.as_ref().ok_or(Error::NoPendingDecode)?;
            if pending.token_id != input.token {
                return Err(Error::PendingToken {
                    expected: pending.token_id,
                    actual: input.token,
                });
            }
        }
        Ok(())
    }

    fn session(&self, session: Uuid) -> Result<&SessionState> {
        self.sessions.get(&session).ok_or_else(|| Error::Session {
            model: self.info.manifest.id.clone(),
            session,
        })
    }
}

fn decode_states(
    model: &crate::engine::DecoderModel,
    stream: &crate::engine::Stream,
    inputs: &[DecodeInput],
    states: &mut [(Uuid, SessionState)],
) -> Result<Vec<NativeOutput>> {
    let token_ids = sample_batch(inputs, states, stream)?;
    token_ids.async_eval()?;
    let tokens = token_ids.to_vec_u32_on_stream(stream)?;
    if tokens.len() != states.len() {
        return Err(Error::InvalidDecodeBatch("sampled token count does not match rows".into()));
    }
    token_ids.detach_graph()?;
    stream.detach_paged_arena_graphs()?;
    for (_, state) in states.iter() {
        state.cache.detach_evaluated_graphs()?;
    }
    let positions = states
        .iter()
        .map(|(_, state)| state.model_position().and_then(|position| Ok(i32::try_from(position)?)))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut caches = states.iter_mut().map(|(_, state)| &mut state.cache).collect::<Vec<_>>();
    let logits = model.forward_packed_decode(&token_ids, &mut caches, &positions, stream)?;
    complete_batch(&logits, tokens, stream, states)
}

fn sample_batch(
    inputs: &[DecodeInput],
    states: &mut [(Uuid, SessionState)],
    stream: &crate::engine::Stream,
) -> Result<Array> {
    let logits = inputs
        .iter()
        .zip(states.iter_mut())
        .map(|(input, (_, state))| step::take_pending(state, input.token))
        .collect::<Result<Vec<_>>>()?;
    if inputs.iter().all(|input| input.sampling == SamplingLogits::None) {
        let refs = logits.iter().collect::<Vec<_>>();
        return Ok(Array::concatenate(&refs, 0, stream)?.argmax(stream)?);
    }
    let sampled = inputs
        .iter()
        .zip(&logits)
        .map(|(input, logits)| {
            step::device_token(logits, input.sampling, stream)?.ok_or_else(|| {
                Error::InvalidDecodeBatch("packed row does not use device sampling".into())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let refs = sampled.iter().collect::<Vec<_>>();
    Ok(Array::concatenate(&refs, 0, stream)?)
}

fn complete_batch(
    logits: &Array,
    tokens: Vec<u32>,
    stream: &crate::engine::Stream,
    states: &mut [(Uuid, SessionState)],
) -> Result<Vec<NativeOutput>> {
    let shape = logits.shape()?;
    let width = usize::try_from(*shape.get(2).ok_or_else(|| {
        Error::InvalidDecodeBatch("packed logits must have [batch, sequence, vocab] shape".into())
    })?)?;
    let rows = (0..states.len())
        .map(|row| Ok(logits.slice(&[row, 0, 0], &[row + 1, 1, width], stream)?))
        .collect::<Result<Vec<_>>>()?;
    logits.async_eval()?;
    Ok(states
        .iter_mut()
        .zip(rows)
        .zip(tokens)
        .map(|(((_, state), logits), token_id)| {
            state.pending = Some(PendingDecode { token_id, logits });
            NativeOutput::Greedy(token_id)
        })
        .collect())
}
