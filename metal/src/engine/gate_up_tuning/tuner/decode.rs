use runtime::tuning::TuningMode;

use super::MetalTuner;
use crate::engine::{DecodePlan, DecodePlanAction, DecodePlanKey};

impl MetalTuner {
    pub(crate) fn decode_plan_action(&self, key: &DecodePlanKey) -> DecodePlanAction {
        if let Some(plan) = self.decode_plans.get(key) {
            return DecodePlanAction::Execute(*plan);
        }
        if self.config.mode == TuningMode::Startup
            && self.startup_open
            && self.budgets.decode_plan.available()
        {
            DecodePlanAction::Measure
        } else {
            DecodePlanAction::Execute(DecodePlan::SeparateGateUp)
        }
    }

    pub(crate) fn record_decode_plan(
        &mut self,
        key: DecodePlanKey,
        plan: DecodePlan,
        elapsed: std::time::Duration,
    ) {
        self.budgets.decode_plan.consume(elapsed);
        self.decode_plans.insert(key, plan);
    }

    pub(crate) const fn active_decode_plan(&self) -> Option<DecodePlan> {
        self.active_decode_plan
    }

    pub(crate) fn activate_decode_plan(&mut self, plan: DecodePlan, suppress_tuning: bool) {
        self.active_decode_plan = Some(plan);
        self.suppress_operator_tuning = suppress_tuning;
    }

    pub(crate) fn clear_decode_plan(&mut self) {
        self.active_decode_plan = None;
        self.suppress_operator_tuning = false;
    }
}
