use std::time::Duration;

use super::Stream;
use crate::engine::{DecodePlan, DecodePlanAction, DecodePlanKey};

impl Stream {
    pub(crate) fn decode_plan_action(&self, key: &DecodePlanKey) -> DecodePlanAction {
        self.tuner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .decode_plan_action(key)
    }

    pub(crate) fn record_decode_plan(
        &self,
        key: DecodePlanKey,
        plan: DecodePlan,
        elapsed: Duration,
    ) {
        let mut tuner = self.tuner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        tuner.record_decode_plan(key, plan, elapsed);
        tuner.persist();
    }

    pub(crate) fn with_decode_plan<T>(
        &self,
        plan: DecodePlan,
        suppress_operator_tuning: bool,
        run: impl FnOnce() -> T,
    ) -> T {
        self.tuner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .activate_decode_plan(plan, suppress_operator_tuning);
        let _active = ActiveDecodePlan(self);
        run()
    }
}

struct ActiveDecodePlan<'a>(&'a Stream);

impl Drop for ActiveDecodePlan<'_> {
    fn drop(&mut self) {
        self.0
            .tuner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear_decode_plan();
    }
}
