//! Verification under the host's token masks: each row chooses within the
//! mask of the token it follows, as an unspeculated masked step would.

use runtime::{
    backend::{DeviceSampling, TokenMask},
    sampling::{DraftSampling, verify_drafts},
    speculation::MaskedDraft,
};

use super::{CudaSharedRoutedModelSession, speculate::seed};
use crate::{Error, Result};

impl CudaSharedRoutedModelSession {
    /// The masked drafts of the last pass of `rows` rows, how many it keeps,
    /// and the token after them; `first` masks the fed token's row. This is
    /// the step's one wait for the verification.
    pub(super) fn verify_masked(
        &self,
        rows: usize,
        first: &TokenMask,
        sampling: DeviceSampling,
        masked: &[MaskedDraft],
        position: usize,
    ) -> Result<(Vec<u32>, usize, u32)> {
        let plans = self.template.plans.clone();
        let mut plans = plans
            .lock()
            .map_err(|_| Error::State("shared-routed execution plan cache is poisoned".into()))?;
        let prediction = plans
            .get_mut(rows)
            .and_then(|plan| plan.prediction.as_mut())
            .ok_or(Error::InvalidDecoderKernel("verification plan is missing"))?;
        let mut logits = prediction.decoder_logits(&self.template)?;
        drop(plans);
        let masked = masked
            .get(..rows - 1)
            .ok_or(Error::InvalidSampling("fewer masked drafts than verified rows".into()))?;
        let drafts = masked.iter().map(|draft| draft.token).collect::<Vec<_>>();
        let width = self.template.decoder.vocab_size;
        let masks = std::iter::once(first).chain(masked.iter().map(|draft| &draft.mask));
        for (row, mask) in logits.chunks_exact_mut(width).zip(masks) {
            apply(row, mask);
        }
        let rows = logits.chunks_exact(width).collect::<Vec<_>>();
        match sampling {
            DeviceSampling::Greedy => {
                let target = rows.iter().map(|row| argmax(row)).collect::<Result<Vec<_>>>()?;
                let kept = drafts.iter().zip(&target).take_while(|(draft, token)| draft == token);
                let accepted = kept.count();
                Ok((drafts, accepted, target[accepted]))
            },
            DeviceSampling::Random { temperature, top_p, top_k, draw } => {
                let sampling = DraftSampling { temperature, top_p, top_k };
                let (accepted, next) =
                    verify_drafts(&rows, &drafts, sampling, seed(draw, position)?)?;
                Ok((drafts, accepted, next))
            },
        }
    }
}

/// Leaves only the tokens `mask` allows.
fn apply(row: &mut [f32], mask: &TokenMask) {
    let words = mask.words();
    for (token, value) in row.iter_mut().enumerate() {
        let allowed = token < mask.vocab() && words[token / 32] >> (token % 32) & 1 == 1;
        if !allowed {
            *value = f32::NEG_INFINITY;
        }
    }
}

fn argmax(row: &[f32]) -> Result<u32> {
    let best = row
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1).then(right.0.cmp(&left.0)))
        .ok_or(Error::InvalidSampling("the mask leaves no finite logit".into()))?;
    Ok(u32::try_from(best.0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mask_leaves_the_best_allowed_token() -> Result<()> {
        let mask = TokenMask::new(vec![0b1010], 4)?;
        let mut row = vec![3.0, 1.0, 5.0, 2.0];
        apply(&mut row, &mask);
        assert_eq!(argmax(&row)?, 3);
        assert!(row[0].is_infinite() && row[2].is_infinite());
        Ok(())
    }
}
