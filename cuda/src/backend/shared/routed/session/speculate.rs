//! Greedy speculative decoding: one decoder pass verifies the drafts, the
//! recurrent state keeps the accepted prefix, and the prediction module
//! drafts the tokens after it.

use runtime::{
    backend::SamplingLogits,
    kv::BlockTable,
    sampling::{DraftSampling, verify_drafts},
    speculation::DraftBounds,
};
use uuid::Uuid;

use super::{
    super::plan::SharedRoutedExecutionPlan, CudaSharedRoutedModelSession, verify::VerifyPass,
};
use crate::{Error, Result};

/// The outcome of one speculative decode step. The drafts after `next`
/// stay on the device for the next step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeculativeStep {
    /// Drafts the decoder accepted, in order; already part of the sequence.
    pub accepted: Vec<u32>,
    /// The decoder's token after them, fed to the next step.
    pub next: u32,
    /// The drafts after `next`, read for a masked step's host.
    pub drafts: Vec<u32>,
}

impl CudaSharedRoutedModelSession {
    /// Feeds `token` and verifies up to `draft_tokens` of the session's
    /// drafts after it in one pass, then drafts as many after the verified
    /// tokens. `table` holds the session through `token` and has blocks for
    /// the drafts of this step and the next. Greedy steps keep drafts equal
    /// to the decoder's argmax; sampled steps keep them by rejection
    /// sampling, so tokens follow `sampling` exactly. Under a token mask,
    /// each row chooses within its mask and the step verifies only the
    /// host's masked drafts, reading the next drafts back for it.
    ///
    /// On error, discard this session: a pass may already have advanced
    /// recurrent buffers. The runtime retires failed sessions rather than
    /// retrying verification against partially updated state.
    pub fn decode_speculative(
        &mut self,
        session_id: Uuid,
        token: u32,
        draft_tokens: usize,
        sampling: &SamplingLogits,
        bounds: &DraftBounds,
        table: &BlockTable,
    ) -> Result<SpeculativeStep> {
        let start = self.position;
        let mask = match sampling {
            SamplingLogits::Masked { mask, sampling } => Some((mask, *sampling)),
            _ => None,
        };
        let limit = mask.map_or(draft_tokens, |_| draft_tokens.min(bounds.masked.len()));
        let verified = self.drafted.min(limit);
        let rows = verified + 1;
        let mut pass = table.clone();
        pass.set_token_len(start + rows);
        let pass_kind = self.verify_pass(session_id, token, rows, &pass)?;
        let (drafts, accepted, next) = match mask {
            Some((mask, masked)) => {
                self.verify_masked(rows, mask, masked, &bounds.masked, start)?
            },
            None => self.verify(rows, sampling, start)?,
        };
        let (accepted, next) = bounds.apply(&drafts, accepted, next);
        // The token after every row: a kept draft, then the verified token;
        // rows after it are dropped and only fill the shape.
        let following = drafts[..accepted]
            .iter()
            .copied()
            .chain(std::iter::repeat_n(next, rows - accepted))
            .collect::<Vec<_>>();
        self.predict(session_id, rows, &following, &pass)?;
        match pass_kind {
            VerifyPass::Graphed => self.commit_verified(rows, start, accepted + 1)?,
            VerifyPass::Scalar => self.roll_back(start, accepted + 1)?,
        }
        self.draft(session_id, rows, accepted, start + accepted, draft_tokens, table)?;
        let next_drafts = match (&self.drafts, mask) {
            (Some(buffer), Some(_)) if self.drafted > 0 => {
                let mut drafts = self.template.backend.read_tokens(buffer)?;
                drafts.truncate(self.drafted);
                drafts
            },
            _ => Vec::new(),
        };
        Ok(SpeculativeStep {
            accepted: drafts[..accepted].to_vec(),
            next,
            drafts: next_drafts,
        })
    }

    /// The verified drafts of the last pass of `rows` rows, how many it
    /// keeps, and the token after them. This is the step's one wait for the
    /// device.
    fn verify(
        &self,
        rows: usize,
        sampling: &SamplingLogits,
        position: usize,
    ) -> Result<(Vec<u32>, usize, u32)> {
        let plans = self.template.plans.clone();
        let mut plans = plans.lock().map_err(|_| poisoned())?;
        let prediction = plans
            .get_mut(rows)
            .and_then(|plan| plan.prediction.as_mut())
            .ok_or(Error::InvalidDecoderKernel("verification plan is missing"))?;
        let read_drafts = || -> Result<Vec<u32>> {
            match &self.drafts {
                Some(drafts) if rows > 1 => {
                    let mut drafts = self.template.backend.read_tokens(drafts)?;
                    drafts.truncate(rows - 1);
                    Ok(drafts)
                },
                _ => Ok(Vec::new()),
            }
        };
        let Some((sampling, draw, vocab)) = draft_sampling(sampling)? else {
            let target = prediction.decoder_argmax(&self.template)?;
            let drafts = read_drafts()?;
            let kept = drafts.iter().zip(&target).take_while(|(draft, token)| draft == token);
            let accepted = kept.count();
            return Ok((drafts, accepted, target[accepted]));
        };
        let logits = prediction.decoder_logits(&self.template)?;
        drop(plans);
        let drafts = read_drafts()?;
        let width = self.template.decoder.vocab_size;
        let rows = logits
            .chunks_exact(width)
            .map(|row| &row[..vocab.min(width)])
            .collect::<Vec<_>>();
        let (accepted, next) = verify_drafts(&rows, &drafts, sampling, seed(draw, position)?)?;
        Ok((drafts, accepted, next))
    }

    /// Drafts `count` tokens from the prediction output at `row` of the last
    /// pass of `rows` rows, whose position is `position`, into the session's
    /// device drafts. Each further draft runs the module on the previous
    /// output and draft one position later.
    pub fn draft(
        &mut self,
        session_id: Uuid,
        rows: usize,
        row: usize,
        position: usize,
        count: usize,
        table: &BlockTable,
    ) -> Result<()> {
        let block_size =
            table.block_size().ok_or(Error::InvalidPagedKv("table has no block size"))?;
        if position + count > table.capacity(block_size) {
            return Err(Error::InvalidPagedKv("draft positions exceed the reserved blocks"));
        }
        let stream = self.template.backend.inner.stream.clone();
        let plans = self.template.plans.clone();
        let mut plans = plans.lock().map_err(|_| poisoned())?;
        let prediction = plans
            .get_mut(rows)
            .and_then(|plan| plan.prediction.as_ref())
            .ok_or(Error::InvalidDecoderKernel("no drafting pass of this shape"))?;
        self.select_row
            .execute(&stream, &prediction.drafted, &mut self.last_hidden, row, rows)?;
        if count > 1 && plans.get_mut(1).is_none() {
            let plan = SharedRoutedExecutionPlan::new(&self.template, 1)?;
            plans.insert(1, plan);
        }
        self.drafted = 0;
        if self.drafts.as_ref().is_none_or(|drafts| drafts.len() < count) {
            self.drafts = Some(self.template.backend.inner.pool.allocate(&stream, count.max(1))?);
        }
        let drafts = self
            .drafts
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("draft buffer is missing"))?;
        for index in 0..count {
            self.output.execute(&self.last_hidden, &mut self.logits)?;
            let token = self.sampler.sample(&self.logits, SamplingLogits::None)?;
            stream.copy_device_range(token, 0..1, drafts, index)?;
            if index + 1 == count {
                break;
            }
            let prediction = plans
                .get_mut(1)
                .and_then(|plan| plan.prediction.as_mut())
                .ok_or(Error::InvalidDecoderKernel("single-row drafting plan is missing"))?;
            prediction.load_row(&self.last_hidden, token)?;
            let at = position + index + 1;
            let mut step = table.clone();
            step.set_token_len(at + 1);
            let state = self
                .prediction
                .as_mut()
                .ok_or(Error::InvalidDecoderKernel("session has no prediction module state"))?;
            prediction.execute(&self.embedding, state, session_id, &step, at)?;
            stream.copy_device_range(
                &prediction.drafted,
                0..prediction.drafted.len(),
                &mut self.last_hidden,
                0,
            )?;
        }
        drop(plans);
        self.drafted = count;
        Ok(())
    }
}

/// The target distribution of a device-sampled step, its draw and vocabulary
/// bound; `None` for greedy steps.
pub(in crate::backend::shared::routed) fn draft_sampling(
    sampling: &SamplingLogits,
) -> Result<Option<(DraftSampling, f32, usize)>> {
    match *sampling {
        SamplingLogits::None => Ok(None),
        SamplingLogits::Sample {
            vocab_size,
            temperature,
            top_p,
            top_k,
            draw,
        } => Ok(Some((DraftSampling { temperature, top_p, top_k }, draw, vocab_size))),
        SamplingLogits::SampleTopK { k, vocab_size, temperature, draw } => {
            Ok(Some((DraftSampling { temperature, top_p: 1.0, top_k: k }, draw, vocab_size)))
        },
        SamplingLogits::Masked { .. } | SamplingLogits::Full | SamplingLogits::TopK { .. } => {
            Err(Error::InvalidSampling("speculative decoding samples on the device".into()))
        },
    }
}

/// The rejection-sampling seed of a step: the host's draw at `position`.
pub(in crate::backend::shared::routed) fn seed(draw: f32, position: usize) -> Result<u64> {
    Ok(u64::from(draw.to_bits()) | (u64::try_from(position)? << 32))
}

fn poisoned() -> Error {
    Error::State("shared-routed execution plan cache is poisoned".into())
}
