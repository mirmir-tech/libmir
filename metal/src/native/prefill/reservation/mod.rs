mod output;
use std::num::NonZeroUsize;

pub(in crate::native::prefill) use output::output;

use crate::{config::DecodeReservation, engine::Stream};

pub(in crate::native::prefill) struct Plan {
    pub(in crate::native::prefill) tokens: usize,
    pub(in crate::native::prefill) page_size: usize,
}

impl Plan {
    pub(in crate::native::prefill) fn admit(
        loaded: &mut crate::native::model::LoadedModel,
        prompt: usize,
        budget: Option<NonZeroUsize>,
        position: usize,
    ) -> crate::native::error::Result<Self> {
        let mut plan = Self::new(prompt, budget, &loaded.stream);
        if loaded.stream.config().cache.decode_reservation == DecodeReservation::GenerationBudget {
            let baseline = Self::baseline(prompt, &loaded.stream);
            let required = baseline.required(position);
            loaded.reserve_prefill_pages(required)?;
            plan.limit_extra(&baseline, loaded.prefill_page_headroom()?.saturating_sub(required));
        } else {
            loaded.reserve_prefill_pages(plan.required(position))?;
        }
        Ok(plan)
    }

    pub(in crate::native::prefill) fn baseline(prompt: usize, stream: &Stream) -> Self {
        let page_size = stream.config().kv_cache.block_size.max(1);
        Self {
            tokens: prompt.saturating_add(page_size),
            page_size,
        }
    }

    pub(in crate::native::prefill) fn new(
        prompt: usize,
        budget: Option<NonZeroUsize>,
        stream: &Stream,
    ) -> Self {
        let mut plan = Self::baseline(prompt, stream);
        let extra = match stream.config().cache.decode_reservation {
            DecodeReservation::OnePage => plan.page_size,
            DecodeReservation::GenerationBudget => budget.map_or(plan.page_size, NonZeroUsize::get),
            #[cfg(test)]
            DecodeReservation::Tokens(tokens) => tokens.get(),
        };
        plan.tokens = prompt.saturating_add(extra.max(plan.page_size));
        plan
    }

    pub(in crate::native::prefill) const fn tokens(&self) -> usize {
        self.tokens
    }

    pub(in crate::native::prefill) fn required(&self, position: usize) -> usize {
        let owned = position.div_ceil(self.page_size);
        let cow = usize::from(position > 0 && !position.is_multiple_of(self.page_size));
        self.tokens.div_ceil(self.page_size).saturating_sub(owned).saturating_add(cow)
    }

    pub(in crate::native::prefill) fn limit_extra(&mut self, base: &Self, pages: usize) {
        let limit = base
            .tokens
            .div_ceil(self.page_size)
            .saturating_add(pages)
            .saturating_mul(self.page_size);
        self.tokens = self.tokens.min(limit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uses_budget_and_page_rounding_without_overflow() -> crate::engine::Result<()> {
        let mut stream = Stream::new_gpu()?;
        stream.set_decode_reservation(DecodeReservation::GenerationBudget);
        let page = stream.config().kv_cache.block_size.max(1);
        for prompt in [1, page, page + 1] {
            let base = Plan::baseline(prompt, &stream);
            for budget in [0, 1, page, page + 1, 64, usize::MAX] {
                let mut plan = Plan::new(prompt, NonZeroUsize::new(budget), &stream);
                assert_eq!(plan.tokens(), prompt.saturating_add(budget.max(page)));
                plan.limit_extra(&base, 0);
                for position in [0, prompt / 2, prompt] {
                    assert_eq!(plan.required(position), base.required(position));
                }
            }
        }
        Ok(())
    }
}
