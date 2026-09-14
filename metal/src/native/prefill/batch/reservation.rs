#[cfg(test)]
use super::super::required_prefill_pages;
pub(super) use super::super::reservation::Plan;
use super::sequence::Sequence;
use crate::native::{error::Result, model::LoadedModel};

#[cfg(test)]
fn required_values(pending: bool, tokens: usize, position: usize, page_size: usize) -> usize {
    usize::from(pending) * required_prefill_pages(tokens, position, page_size)
}

pub(super) fn required(sequence: &Sequence) -> usize {
    usize::from(sequence.page_reservation_pending)
        * sequence.reservation.required(sequence.position)
}

pub(super) fn ensure(sequence: &Sequence, loaded: &mut LoadedModel) -> Result<()> {
    loaded.reserve_prefill_pages(required(sequence))
}

pub(super) fn fit(sequences: &mut [Sequence], loaded: &mut LoadedModel) -> Result<()> {
    if loaded.stream.config().cache.decode_reservation
        != crate::config::DecodeReservation::GenerationBudget
    {
        return loaded.reserve_prefill_pages(sequences.iter().map(required).sum());
    }
    let base = sequences
        .iter()
        .filter(|s| s.page_reservation_pending)
        .map(|s| Plan::baseline(s.request.prompt_tokens.len(), &loaded.stream).required(s.position))
        .sum();
    if loaded.prefill_page_headroom()? < base {
        for sequence in sequences.iter_mut() {
            sequence.reclaim_reservation()?;
        }
    }
    loaded.reserve_prefill_pages(base)?;
    let mut spare = loaded.prefill_page_headroom()?.saturating_sub(base);
    let mut remaining = sequences.iter().filter(|s| s.page_reservation_pending).count();
    for sequence in sequences.iter_mut().filter(|s| s.page_reservation_pending) {
        let baseline = Plan::baseline(sequence.request.prompt_tokens.len(), &loaded.stream);
        sequence.reservation.limit_extra(&baseline, spare / remaining);
        spare = spare.saturating_sub(
            sequence
                .reservation
                .required(sequence.position)
                .saturating_sub(baseline.required(sequence.position)),
        );
        remaining -= 1;
        sequence.plan_reservation()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Plan, required_values};

    #[test]
    fn counts_reserved_tail_and_shared_partial_prefix_once() {
        let plan = Plan { tokens: 2049 + 64, page_size: 16 };
        assert_eq!(plan.tokens(), 2113);
        assert_eq!(plan.required(0), 133);
        assert_eq!(plan.required(2048), 5);
        assert_eq!(plan.required(2049), 5); // Four tail pages plus one COW page.
        for prompt in [1_usize, 16, 129, 2049] {
            let plan = Plan { tokens: prompt + 16, page_size: 16 };
            for position in [0, prompt / 2, prompt] {
                assert_eq!(
                    plan.required(position),
                    super::required_prefill_pages(prompt, position, 16)
                );
            }
        }
    }

    #[test]
    fn rechecks_only_until_physical_reservation_is_installed() {
        assert_eq!(required_values(true, 8_194, 0, 16), 514);
        assert_eq!(required_values(false, 8_194, 0, 16), 0);
    }
}
