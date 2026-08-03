use super::{
    error::{Error, Result},
    model::NativeOutput,
    session::{PendingDecode, SessionState},
};
use crate::engine::{Array, DecoderModel, DeviceSampling, Stream, sample};

pub(super) struct DeferredToken {
    next_logits: Array,
    token_id: u32,
}

impl DeferredToken {
    pub(super) fn enqueue(&self) -> Result<()> {
        Ok(self.next_logits.async_eval()?)
    }

    pub(super) fn complete(self, state: &mut SessionState) -> NativeOutput {
        state.pending = Some(PendingDecode {
            token_id: self.token_id,
            logits: self.next_logits,
        });
        NativeOutput::Greedy(self.token_id)
    }
}

pub(super) fn forward_token(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    token: u32,
    position: usize,
    greedy: bool,
) -> Result<Array> {
    let token_ids = Array::from_u32(&[token], &[1, 1])?;
    forward_ids(model, stream, state, &token_ids, position, greedy)
}

pub(super) fn forward_prefill_state(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    tokens: &[u32],
    position: usize,
) -> Result<Array> {
    let length = i32::try_from(tokens.len())?;
    let token_ids = Array::from_u32(tokens, &[1, length])?;
    let position = i32::try_from(position)?;
    Ok(model.forward_prefill_state(&token_ids, &mut state.cache, position, stream)?)
}

pub(super) fn forward_packed_prefill_state(
    model: &DecoderModel,
    stream: &Stream,
    states: &mut [&mut SessionState],
    positions: &[usize],
    tokens: &[u32],
    sequence: usize,
) -> Result<Array> {
    let batch = states.len();
    let token_ids = Array::from_u32(tokens, &[i32::try_from(batch)?, i32::try_from(sequence)?])?;
    let positions = positions
        .iter()
        .copied()
        .map(i32::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut caches = states.iter_mut().map(|state| &mut state.cache).collect::<Vec<_>>();
    Ok(model.forward_packed_prefill_state(&token_ids, &mut caches, &positions, stream)?)
}

fn forward_ids(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    token_ids: &Array,
    position: usize,
    greedy: bool,
) -> Result<Array> {
    let position = i32::try_from(position)?;
    if greedy {
        Ok(model.forward_greedy_decode(token_ids, &mut state.cache, position, stream)?)
    } else {
        Ok(model.forward_decode(token_ids, &mut state.cache, position, stream)?)
    }
}

pub(super) fn output(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    logits: Array,
    sampling: SamplingLogits,
) -> Result<NativeOutput> {
    if !supports_device_token(sampling) {
        return Ok(NativeOutput::Logits(logits));
    }
    let deferred = deferred_token(model, stream, state, &logits, sampling)?;
    deferred.enqueue()?;
    Ok(deferred.complete(state))
}

fn deferred_token(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    logits: &Array,
    sampling: SamplingLogits,
) -> Result<DeferredToken> {
    let sampled = device_token(logits, sampling, stream)?.ok_or_else(|| {
        Error::InvalidDecodeBatch("batch row does not use device token sampling".into())
    })?;
    sampled.async_eval()?;
    let token_id = sampled
        .to_vec_u32_on_stream(stream)?
        .into_iter()
        .next()
        .ok_or_else(|| Error::InvalidDecodeBatch("sampled token is empty".into()))?;
    sampled.detach_graph()?;
    stream.detach_paged_arena_graphs()?;
    state.cache.detach_evaluated_graphs()?;
    let next_logits = forward_ids(
        model,
        stream,
        state,
        &sampled,
        state.model_position()?,
        sampling == SamplingLogits::None,
    )?;
    Ok(DeferredToken { next_logits, token_id })
}

pub(super) fn decode_pending(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    token: u32,
    sampling: SamplingLogits,
) -> Result<NativeOutput> {
    let logits = take_pending(state, token)?;
    output(model, stream, state, logits, sampling)
}

pub(super) fn take_pending(state: &mut SessionState, token: u32) -> Result<Array> {
    let pending = state.pending.take().ok_or(Error::NoPendingDecode)?;
    if pending.token_id != token {
        return Err(Error::PendingToken {
            expected: pending.token_id,
            actual: token,
        });
    }
    state.position += 1;
    Ok(pending.logits)
}

pub(super) const fn supports_device_token(sampling: SamplingLogits) -> bool {
    matches!(
        sampling,
        SamplingLogits::None | SamplingLogits::SampleTopK { .. } | SamplingLogits::Sample { .. }
    )
}

pub(super) fn device_token(
    logits: &Array,
    sampling: SamplingLogits,
    stream: &Stream,
) -> Result<Option<Array>> {
    let parameters = match sampling {
        SamplingLogits::None => return Ok(Some(logits.argmax(stream)?)),
        SamplingLogits::SampleTopK { k, vocab_size, temperature, draw } => DeviceSampling {
            vocab_size,
            top_k: k,
            top_p: 1.0,
            temperature,
            draw,
        },
        SamplingLogits::Sample {
            vocab_size,
            temperature,
            top_p,
            top_k,
            draw,
        } => DeviceSampling {
            vocab_size,
            top_k,
            top_p,
            temperature,
            draw,
        },
        SamplingLogits::Full | SamplingLogits::TopK { .. } => return Ok(None),
    };
    Ok(Some(sample(logits, parameters, stream)?))
}
use runtime::backend::SamplingLogits;
