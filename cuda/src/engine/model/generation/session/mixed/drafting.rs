//! Packed steps of a drafting model: prompt chunks and decode rows in one
//! pass that also verifies and drafts (`DraftingRow`).

use runtime::backend::DecodeSequence;

use super::{
    MixedMixerExecution,
    batch::{restore_sessions, take_sessions},
};
use crate::{
    Error, Result,
    backend::DraftingRow,
    engine::{
        execution::{Output, device_sampling},
        model::generation::{CombinedOutputs, PrefillChunk},
    },
};

/// Runs `chunks` and `decode` as one drafting pass, or `None` when the
/// step needs a path the packed pass does not take. Decode rows verify
/// drafts only beside prompt chunks: a verified token costs about what a
/// batched decode row does, so speculation pays only where the pass is long
/// anyway. Every row still feeds the prediction module, so its K/V stays
/// complete for steps that speculate later.
pub(super) fn execute(
    execution: &mut MixedMixerExecution,
    chunks: &[PrefillChunk<'_>],
    decode: &[DecodeSequence],
) -> Result<Option<CombinedOutputs>> {
    let Some(drafting) = execution.drafting else {
        return Ok(None);
    };
    if !supported(execution, chunks, decode) {
        return Ok(None);
    }
    for row in chunks {
        if row.offset == 0 {
            let session = execution.template.instantiate_with_caches(&execution.caches)?;
            execution.sessions.insert(row.request.session_id, session);
        }
        execution.arm_terminal_checkpoint(row.request, row.offset, row.tokens.len())?;
    }
    let speculate = !chunks.is_empty();
    let mut counts = chunks.iter().map(|row| row.tokens.len()).collect::<Vec<_>>();
    for row in decode {
        let session = super::required(&mut execution.sessions, row.session_id)?;
        let room = super::speculation::room(&row.block_table, session.position())?;
        let drafts = session.pending_drafts().min(drafting.draft_tokens()).min(room);
        counts.push(if verifies(row, speculate) {
            1 + drafts
        } else {
            1
        });
    }
    execution.prefill_batches.clear();
    execution.combined.prepare(&execution.template, &counts)?;
    let ids = chunks
        .iter()
        .map(|row| row.request.session_id)
        .chain(decode.iter().map(|row| row.session_id));
    let mut owned = take_sessions(&mut execution.sessions, ids)?;
    let result: Result<CombinedOutputs> = (|| {
        let mut rows = Vec::with_capacity(counts.len());
        for row in chunks {
            let prompt = &row.request.prompt_tokens;
            let end = (row.offset + row.tokens.len()).min(prompt.len() - 1);
            rows.push(DraftingRow::Prompt {
                tokens: row.tokens,
                following: &prompt[row.offset + 1..=end],
                sample: row.final_chunk.then(|| row.request.sampling_logits.clone()),
            });
        }
        rows.extend(decode.iter().map(|row| DraftingRow::Decode {
            token: row.token_id,
            sampling: row.sampling_logits.clone(),
            speculation: row.speculation.clone().filter(|_| verifies(row, speculate)),
        }));
        let tables = chunks
            .iter()
            .map(|row| row.table)
            .chain(decode.iter().map(|row| &row.block_table))
            .collect::<Vec<_>>();
        let mut sessions = owned.iter_mut().map(|(_, session)| session).collect::<Vec<_>>();
        let batch = execution.combined.current()?;
        let template = &execution.template;
        let steps = batch.execute_drafting(
            template,
            &mut sessions,
            &rows,
            &tables,
            drafting.draft_tokens(),
        )?;
        let mut steps = steps.into_iter();
        let output = |step: crate::SpeculativeStep| Output {
            accepted: step.accepted,
            drafts: step.drafts,
            token: Some(step.next),
            logits: None,
        };
        let prefill = chunks
            .iter()
            .map(|_| steps.next().ok_or(Error::InvalidExecutionPlan("drafting rows are missing")))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|step| step.map(output))
            .collect();
        let decode = steps
            .map(|step| {
                step.map(output).ok_or(Error::InvalidExecutionPlan("decode row has no token"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(CombinedOutputs { prefill, decode })
    })();
    restore_sessions(&mut execution.sessions, owned);
    let result = result?;
    for row in chunks {
        execution.checkpoint_prefix(row.request)?;
    }
    Ok(Some(result))
}

/// Whether a decode row verifies drafts: beside prompt chunks, and not
/// under a token mask, whose rows the host masks only in scalar steps.
fn verifies(row: &DecodeSequence, speculate: bool) -> bool {
    speculate
        && row.speculation.is_some()
        && !matches!(row.sampling_logits, runtime::backend::SamplingLogits::Masked { .. })
}

/// Rows the packed pass samples on the device, without vision offsets or a
/// repeated session.
fn supported(
    execution: &MixedMixerExecution,
    chunks: &[PrefillChunk<'_>],
    decode: &[DecodeSequence],
) -> bool {
    let mut ids = chunks
        .iter()
        .map(|row| row.request.session_id)
        .chain(decode.iter().map(|row| row.session_id))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    execution.template.supports_combined_generation()
        && ids.windows(2).all(|pair| pair[0] != pair[1])
        && !ids.is_empty()
        && chunks.iter().all(|row| {
            device_sampling(&row.request.sampling_logits)
                && (row.offset == 0 || execution.sessions.contains_key(&row.request.session_id))
        })
        && decode.iter().all(|row| {
            device_sampling(&row.sampling_logits)
                && execution.sessions.contains_key(&row.session_id)
        })
        && ids.iter().all(|id| {
            execution.sessions.get(id).is_none_or(|session| session.position_delta() == 0)
        })
}
