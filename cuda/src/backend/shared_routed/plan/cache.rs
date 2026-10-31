use std::collections::HashMap;

use super::SharedRoutedExecutionPlan;

const RETAINED_PLAN_SHAPES: usize = 8;
/// Plans at least this long price the retention headroom per token.
const PRICED_PLAN_TOKENS: usize = 256;

#[derive(Debug)]
struct CachedExecutionPlan {
    plan: SharedRoutedExecutionPlan,
    last_used: u64,
}

#[derive(Debug)]
pub(in super::super) struct SharedRoutedExecutionPlanCache {
    plans: HashMap<usize, CachedExecutionPlan>,
    clock: u64,
    /// Pool bytes one plan token cost so far, weighted over built plans.
    bytes_per_token: u64,
    plan_bytes: u64,
    plan_tokens: u64,
}

impl SharedRoutedExecutionPlanCache {
    pub(in super::super) fn new() -> Self {
        Self {
            plans: HashMap::new(),
            clock: 0,
            bytes_per_token: 0,
            plan_bytes: 0,
            plan_tokens: 0,
        }
    }

    /// Pool bytes shape churn may take beyond the plans held now: an evicted
    /// plan's memory stays cached in the pool but seldom fits the next shape,
    /// so under traffic the whole budget can be held twice.
    pub(in super::super) fn headroom_bytes(&self) -> u64 {
        let budget = u64::try_from(
            crate::backend::model::DEFAULT_PREFILL_CHUNK_TOKENS
                .saturating_add(crate::backend::model::RETAINED_CHUNK_SLACK_TOKENS),
        )
        .unwrap_or(u64::MAX);
        budget.saturating_mul(self.bytes_per_token)
    }

    pub(in super::super) const fn bytes_per_token(&self) -> u64 {
        self.bytes_per_token
    }

    pub(in super::super) fn get_mut(
        &mut self,
        tokens: usize,
    ) -> Option<&mut SharedRoutedExecutionPlan> {
        self.clock = self.clock.wrapping_add(1);
        let clock = self.clock;
        self.plans.get_mut(&tokens).map(|cached| {
            cached.last_used = clock;
            &mut cached.plan
        })
    }

    pub(in super::super) fn reserve(&mut self, tokens: usize) {
        if self.plans.contains_key(&tokens) {
            return;
        }
        // Each shape owns scratch for every layer. Bound total token capacity,
        // not just shape count, to retain roughly one prefill chunk plus
        // decode. Evict before constructing the replacement so old
        // scratch is freed on the same CUDA stream before its new
        // allocations are enqueued.
        let budget = crate::backend::model::DEFAULT_PREFILL_CHUNK_TOKENS
            .max(tokens)
            .saturating_add(crate::backend::model::RETAINED_CHUNK_SLACK_TOKENS);
        while self.plans.len() >= RETAINED_PLAN_SHAPES
            || self.retained_tokens().saturating_add(tokens) > budget
        {
            let Some(oldest) = self
                .plans
                .iter()
                .min_by_key(|(_, cached)| cached.last_used)
                .map(|(tokens, _)| *tokens)
            else {
                break;
            };
            self.plans.remove(&oldest);
            tracing::debug!(
                evicted_tokens = oldest,
                incoming_tokens = tokens,
                budget,
                "evicted CUDA mixed-mixer execution scratch"
            );
        }
    }

    /// Token capacity held by chunk-sized shapes. Decode and prompt-tail shapes
    /// own little scratch and are bounded by the shape count alone; counting
    /// them made two chunks of one prompt evict each other on every request.
    fn retained_tokens(&self) -> usize {
        self.plans
            .keys()
            .filter(|tokens| **tokens > crate::backend::model::SMALL_PLAN_TOKENS)
            .fold(0_usize, |total, tokens| total.saturating_add(*tokens))
    }

    pub(in super::super) fn insert(&mut self, tokens: usize, plan: SharedRoutedExecutionPlan) {
        self.reserve(tokens);
        self.clock = self.clock.wrapping_add(1);
        // Only chunk-sized plans price a token; a one-token decode plan is
        // all per-plan overhead.
        if tokens >= PRICED_PLAN_TOKENS {
            self.plan_bytes = self.plan_bytes.saturating_add(plan.bytes);
            self.plan_tokens =
                self.plan_tokens.saturating_add(u64::try_from(tokens).unwrap_or(u64::MAX));
            self.bytes_per_token = self.plan_bytes / self.plan_tokens.max(1);
        }
        tracing::debug!(
            tokens,
            bytes = plan.bytes,
            retained_tokens = self.retained_tokens(),
            "retained CUDA mixed-mixer execution plan"
        );
        self.plans.insert(tokens, CachedExecutionPlan { plan, last_used: self.clock });
    }

    pub(in super::super) fn clear(&mut self) {
        self.plans.clear();
    }

    #[cfg(test)]
    pub(in super::super) fn len(&self) -> usize {
        self.plans.len()
    }
}
