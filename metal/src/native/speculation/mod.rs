//! Speculative decoding with a checkpoint's multi-token prediction module.
//!
//! The module reads each decoder row `h_i` with the token after it,
//! `t_{i+1}`, and keeps its own K/V over those rows. A row joins the
//! module once its following token is known: prefill rows when the next
//! chunk (or the first generated token) arrives, decode rows after
//! verification. The last joined row drafts the token after the next one.

pub(super) mod step;

use crate::{
    engine::{Array, DecoderCache, DecoderModel, Stream},
    native::error::Result,
};

/// A decoder row awaiting the module, with the token after it once known.
#[derive(Debug)]
struct Row {
    hidden: Array,
    position: usize,
    following: Option<u32>,
}

#[derive(Debug)]
pub(super) struct SpeculativeState {
    cache: DecoderCache,
    rows: Vec<Row>,
}

impl SpeculativeState {
    pub(super) fn snapshot(&self) -> Result<Self> {
        Ok(Self {
            cache: self.cache.snapshot_at(self.cache.cached_tokens()?)?,
            rows: self
                .rows
                .iter()
                .map(|row| {
                    Ok(Row {
                        hidden: row.hidden.snapshot()?,
                        position: row.position,
                        following: row.following,
                    })
                })
                .collect::<Result<_>>()?,
        })
    }
}

/// Starts speculation for a session of a model with a loaded module.
pub(super) fn attach(
    model: &DecoderModel,
    stream: &Stream,
    speculation: &mut Option<SpeculativeState>,
) -> Result<bool> {
    if speculation.is_none()
        && let Some(cache) = model.new_prediction_cache(stream)?
    {
        *speculation = Some(SpeculativeState { cache, rows: Vec::new() });
    }
    Ok(speculation.is_some())
}

/// Queues normalized decoder rows `hidden [1, L, H]` of `tokens` from
/// `position`: the waiting row learns its following token, and every row
/// but the last learns the next one.
pub(super) fn queue(
    speculation: &mut SpeculativeState,
    hidden: &Array,
    tokens: &[u32],
    position: usize,
    stream: &Stream,
) -> Result<()> {
    if let Some(waiting) = speculation.rows.last_mut().filter(|row| row.following.is_none()) {
        waiting.following = tokens.first().copied();
    }
    let width = usize::try_from(hidden.shape()?[2])?;
    for (offset, _) in tokens.iter().enumerate() {
        speculation.rows.push(Row {
            hidden: hidden.slice(&[0, offset, 0], &[1, offset + 1, width], stream)?,
            position: position + offset,
            following: tokens.get(offset + 1).copied(),
        });
    }
    Ok(())
}

/// The module's last row: its normalized output and position, whose logits
/// draft the token after the next one.
pub(super) struct Cursor {
    pub(super) hidden: Array,
    pub(super) position: usize,
}

/// Runs the module over every queued row whose following token is known,
/// after `token` completes the waiting row; the last such row.
pub(super) fn advance(
    model: &DecoderModel,
    stream: &Stream,
    speculation: &mut SpeculativeState,
    token: Option<u32>,
) -> Result<Option<Cursor>> {
    if let (Some(token), Some(waiting)) = (token, speculation.rows.last_mut())
        && waiting.following.is_none()
    {
        waiting.following = Some(token);
    }
    let ready = speculation.rows.iter().take_while(|row| row.following.is_some()).count();
    if ready == 0 {
        return Ok(None);
    }
    let rows = speculation.rows.drain(..ready).collect::<Vec<_>>();
    let hidden =
        Array::concatenate(&rows.iter().map(|row| &row.hidden).collect::<Vec<_>>(), 1, stream)?;
    let following = rows.iter().filter_map(|row| row.following).collect::<Vec<_>>();
    let tokens = Array::from_u32(&following, &[1, i32::try_from(following.len())?])?;
    let output = model.forward_prediction(
        &hidden,
        &tokens,
        &mut speculation.cache,
        i32::try_from(rows[0].position)?,
        stream,
    )?;
    let width = usize::try_from(output.shape()?[2])?;
    Ok(Some(Cursor {
        hidden: output.slice(&[0, ready - 1, 0], &[1, ready, width], stream)?,
        position: rows[ready - 1].position,
    }))
}
