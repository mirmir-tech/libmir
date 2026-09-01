use std::{collections::HashMap, path::PathBuf, time::Duration};

use runtime::tuning::{StartupBudget, TuningConfig, TuningMode};

use super::{
    super::{
        attention_batch_tuning::{BatchAttentionExecution, BatchAttentionKey},
        attention_tuning::AttentionKey,
        decode_plan_tuning::{DecodePlan, DecodePlanKey},
        expert_tuning::{ExpertExecution, ExpertKey},
        kernels::PagedExecution,
        route_tuning::{RoutingExecution, RoutingKey},
    },
    GateUpExecution, GateUpKey, TuneAction, storage,
};

#[derive(Debug)]
pub struct MetalTuner {
    config: TuningConfig,
    cache_path: Option<PathBuf>,
    startup_open: bool,
    budgets: TuningBudgets,
    decisions: HashMap<GateUpKey, GateUpExecution>,
    attention: HashMap<AttentionKey, PagedExecution>,
    batch_attention: HashMap<BatchAttentionKey, BatchAttentionExecution>,
    experts: HashMap<ExpertKey, ExpertExecution>,
    routing: HashMap<RoutingKey, RoutingExecution>,
    decode_plans: HashMap<DecodePlanKey, DecodePlan>,
    active_decode_plan: Option<DecodePlan>,
    suppress_operator_tuning: bool,
}

#[derive(Debug)]
struct TuningBudgets {
    gate_up: StartupBudget,
    attention: StartupBudget,
    batch_prefill: StartupBudget,
    batch_decode: StartupBudget,
    experts: StartupBudget,
    routing: StartupBudget,
    decode_plan: StartupBudget,
}

impl TuningBudgets {
    fn new(duration: Duration) -> Self {
        let budget = StartupBudget::new(duration);
        Self {
            gate_up: budget,
            attention: budget,
            batch_prefill: budget,
            batch_decode: budget,
            experts: budget,
            routing: budget,
            decode_plan: budget,
        }
    }

    const fn batch_attention(&self, causal: bool) -> StartupBudget {
        if causal {
            self.batch_prefill
        } else {
            self.batch_decode
        }
    }

    fn consume_batch_attention(&mut self, causal: bool, elapsed: Duration) {
        if causal {
            self.batch_prefill.consume(elapsed);
        } else {
            self.batch_decode.consume(elapsed);
        }
    }
}

impl MetalTuner {
    pub fn new(config: TuningConfig) -> Self {
        let budgets = TuningBudgets::new(Duration::from_millis(config.startup_budget_ms));
        let cache_path = config
            .cache_directory
            .as_ref()
            .map(|directory| directory.join(storage::cache_name()));
        let stored = cache_path.as_deref().and_then(storage::load).unwrap_or_default();
        Self {
            config,
            cache_path,
            startup_open: true,
            budgets,
            decisions: stored.gate_up,
            attention: stored.attention,
            batch_attention: stored.batch_attention,
            experts: stored.experts,
            routing: stored.routing,
            decode_plans: stored.decode_plans,
            active_decode_plan: None,
            suppress_operator_tuning: false,
        }
    }

    pub fn plan(&self, key: GateUpKey) -> TuneAction {
        if self.config.mode == TuningMode::Disabled {
            return TuneAction::Execute(GateUpExecution::Fused);
        }
        if let Some(execution) = self.decisions.get(&key) {
            return TuneAction::Execute(*execution);
        }
        if self.config.mode == TuningMode::Startup
            && self.startup_open
            && self.budgets.gate_up.available()
        {
            TuneAction::Measure
        } else {
            TuneAction::Execute(GateUpExecution::Fused)
        }
    }

    pub fn record(&mut self, key: GateUpKey, execution: GateUpExecution, elapsed: Duration) {
        self.budgets.gate_up.consume(elapsed);
        self.decisions.insert(key, execution);
    }

    pub fn persist(&self) {
        let Some(path) = &self.cache_path else {
            return;
        };
        if let Err(error) = storage::persist(
            path,
            &self.decisions,
            &self.attention,
            &self.batch_attention,
            &self.experts,
            &self.routing,
            &self.decode_plans,
        ) {
            tracing::warn!(
                target: "libmir::metal::tuning",
                path = %path.display(),
                %error,
                "failed to persist Metal tuning profile"
            );
        }
    }

    pub const fn config(&self) -> &TuningConfig {
        &self.config
    }

    pub const fn attention_budget_available(&self) -> bool {
        !self.suppress_operator_tuning && self.startup_open && self.budgets.attention.available()
    }

    pub const fn batch_attention_runtime_budget_available(&self, causal: bool) -> bool {
        !self.suppress_operator_tuning && self.budgets.batch_attention(causal).available()
    }

    pub const fn expert_budget_available(&self) -> bool {
        !self.suppress_operator_tuning && self.startup_open && self.budgets.experts.available()
    }

    pub const fn routing_budget_available(&self) -> bool {
        !self.suppress_operator_tuning && self.startup_open && self.budgets.routing.available()
    }

    pub const fn routing_runtime_budget_available(&self) -> bool {
        !self.suppress_operator_tuning && self.budgets.routing.available()
    }

    pub const fn finish_startup(&mut self) {
        self.startup_open = false;
    }

    pub fn attention_decision(&self, key: AttentionKey) -> Option<PagedExecution> {
        self.attention.get(&key).copied()
    }

    pub fn record_attention(
        &mut self,
        key: AttentionKey,
        execution: PagedExecution,
        elapsed: Duration,
    ) {
        self.budgets.attention.consume(elapsed);
        self.attention.insert(key, execution);
    }

    pub fn batch_attention_decision(
        &self,
        key: BatchAttentionKey,
    ) -> Option<BatchAttentionExecution> {
        self.batch_attention.get(&key).copied()
    }

    pub fn record_batch_attention(
        &mut self,
        key: BatchAttentionKey,
        execution: BatchAttentionExecution,
        elapsed: Duration,
    ) {
        self.budgets.consume_batch_attention(key.causal, elapsed);
        self.batch_attention.insert(key, execution);
    }

    pub fn expert_decision(&self, key: ExpertKey) -> Option<ExpertExecution> {
        self.experts.get(&key).copied()
    }

    pub fn record_expert(&mut self, key: ExpertKey, execution: ExpertExecution, elapsed: Duration) {
        self.budgets.experts.consume(elapsed);
        self.experts.insert(key, execution);
    }

    pub fn routing_decision(&self, key: RoutingKey) -> Option<RoutingExecution> {
        self.routing.get(&key).copied()
    }

    pub fn record_routing(
        &mut self,
        key: RoutingKey,
        execution: RoutingExecution,
        elapsed: Duration,
    ) {
        self.budgets.routing.consume(elapsed);
        self.routing.insert(key, execution);
    }
}

mod decode;
