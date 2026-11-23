//! Speculative decode steps: the module drafts one token and one decoder
//! pass verifies it after the fed token. The pass records each row's
//! recurrent state, so the cache commits to the kept rows without
//! recomputing them. Steps that cannot speculate (sampled or unbounded ones)
//! still queue their row for the module, so its K/V covers every position.

use runtime::{
    backend::SamplingLogits,
    sampling::{SampledStep, verify_drafts},
    speculation::DraftBounds,
};

use super::{Cursor, SpeculativeState, advance, queue};
use crate::{
    engine::{Array, DecoderModel, Dtype, Stream},
    native::{
        error::{Error, Result},
        model::NativeOutput,
        session::SessionState,
    },
};

/// The decoder's normalized rows of `tokens` from `position`, queued for
/// the module.
pub(in crate::native) fn forward_rows(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    tokens: &[u32],
    position: usize,
) -> Result<Array> {
    let ids = Array::from_u32(tokens, &[1, i32::try_from(tokens.len())?])?;
    let hidden =
        model.forward_prefill_state(&ids, &mut state.cache, i32::try_from(position)?, stream)?;
    if let Some(speculation) = state.speculation.as_mut() {
        queue(speculation, &hidden, tokens, position, stream)?;
    }
    Ok(hidden)
}

/// Feeds `token`; with greedy or device sampling and `bounds`, verifies the
/// module's drafts after it.
pub(in crate::native) fn decode(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    token: u32,
    sampling: &SamplingLogits,
    bounds: Option<&DraftBounds>,
) -> Result<NativeOutput> {
    let position = state.model_position()?;
    let speculation = state.speculation.as_mut().ok_or(Error::NoSpeculation)?;
    let cursor = advance(model, stream, speculation, Some(token))?;
    let count = stream.config().speculation().map_or(0, |value| value.draft_tokens.get());
    let sampled = SampledStep::of(sampling);
    let greedy = matches!(sampling, SamplingLogits::None);
    let (Some(cursor), Some(bounds), true) = (cursor, bounds, greedy || sampled.is_some()) else {
        return plain(model, stream, state, token, position);
    };
    let count = bounds.max_accepted.map_or(count, |limit| limit.min(count));
    if count == 0 {
        return plain(model, stream, state, token, position);
    }
    let drafts = draft(model, stream, speculation, cursor, count)?;
    let rows = [&[token][..], &drafts].concat();
    let ids = Array::from_u32(&rows, &[1, i32::try_from(rows.len())?])?;
    state.cache.arm_row_trail()?;
    let hidden =
        model.forward_prefill_state(&ids, &mut state.cache, i32::try_from(position)?, stream)?;
    let logits = model.project_hidden(&hidden, stream)?;
    let (accepted, next) = match sampled {
        None => verify_greedy(&logits, &drafts, stream)?,
        Some(step) => verify_sampled(&logits, &drafts, step, position, stream)?,
    };
    let (kept, next) = bounds.apply(&drafts, accepted, next);
    state.cache.commit_rows(position, kept + 1, stream)?;
    if let Some(speculation) = state.speculation.as_mut() {
        queue(speculation, &hidden, &rows[..=kept], position, stream)?;
    }
    state.position += kept + 1;
    Ok(NativeOutput::Verified { next, accepted: rows[1..=kept].to_vec() })
}

/// The drafts the decoder's argmax keeps, and its token after them.
fn verify_greedy(logits: &Array, drafts: &[u32], stream: &Stream) -> Result<(usize, u32)> {
    let targets = logits.argmax_rows_u32(stream)?;
    let accepted = drafts
        .iter()
        .zip(&targets)
        .take_while(|(draft, target)| draft == target)
        .count();
    Ok((accepted, targets[accepted]))
}

/// The drafts rejection sampling keeps against the decoder's distribution
/// `step`, and its token after them, as the CUDA backend draws them.
fn verify_sampled(
    logits: &Array,
    drafts: &[u32],
    step: SampledStep,
    position: usize,
    stream: &Stream,
) -> Result<(usize, u32)> {
    let width = usize::try_from(logits.shape()?[2])?;
    let values = logits.astype(Dtype::Float32, stream)?.to_vec_f32(stream)?;
    let rows = values
        .chunks_exact(width)
        .map(|row| &row[..step.vocab_size.min(width)])
        .collect::<Vec<_>>();
    Ok(verify_drafts(&rows, drafts, step.sampling, step.seed(position)?)?)
}

/// Feeds `token` alone.
fn plain(
    model: &DecoderModel,
    stream: &Stream,
    state: &mut SessionState,
    token: u32,
    position: usize,
) -> Result<NativeOutput> {
    let hidden = forward_rows(model, stream, state, &[token], position)?;
    state.position += 1;
    Ok(NativeOutput::Logits(model.project_hidden(&hidden, stream)?))
}

/// Up to `count` greedy drafts from the module's last row, each chained on
/// the module's own output. The chained rows leave the module's K/V, which
/// verified decoder rows rewrite.
fn draft(
    model: &DecoderModel,
    stream: &Stream,
    speculation: &mut SpeculativeState,
    mut cursor: Cursor,
    count: usize,
) -> Result<Vec<u32>> {
    let start = speculation.cache.cached_tokens()?;
    let mut ids = Vec::with_capacity(count);
    loop {
        let id = model.project_hidden(&cursor.hidden, stream)?.argmax_rows(stream)?;
        if ids.len() + 1 == count {
            ids.push(id);
            break;
        }
        let hidden = model.forward_prediction(
            &cursor.hidden,
            &id,
            &mut speculation.cache,
            i32::try_from(cursor.position + 1)?,
            stream,
        )?;
        ids.push(id);
        cursor = Cursor { hidden, position: cursor.position + 1 };
    }
    let drafts =
        Array::concatenate(&ids.iter().collect::<Vec<_>>(), 1, stream)?.to_vec_u32(stream)?;
    if speculation.cache.cached_tokens()? != start {
        speculation.cache.commit_rows(start, 0, stream)?;
    }
    Ok(drafts)
}
