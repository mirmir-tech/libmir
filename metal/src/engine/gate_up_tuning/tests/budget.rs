use std::time::Duration;

use runtime::tuning::{TuningConfig, TuningMode};

use super::{
    super::{GateUpExecution, MetalTuner, TuneAction},
    batch_attention_key, decode_plan_key, fixture_key,
};
use crate::engine::{
    DecodePlan, DecodePlanAction, attention_batch_tuning::BatchAttentionExecution,
};

#[test]
fn startup_mode_measures_once_then_reuses_the_shape_decision() {
    let mut tuner = MetalTuner::new(TuningConfig::default());
    assert_eq!(tuner.plan(fixture_key()), TuneAction::Measure);
    tuner.record(fixture_key(), GateUpExecution::Separate, Duration::from_millis(1));
    assert_eq!(tuner.plan(fixture_key()), TuneAction::Execute(GateUpExecution::Separate));
}

#[test]
fn disabled_and_cached_modes_retain_the_fused_fallback() {
    for mode in [TuningMode::Disabled, TuningMode::Cached] {
        let tuner = MetalTuner::new(TuningConfig { mode, ..TuningConfig::default() });
        assert_eq!(tuner.plan(fixture_key()), TuneAction::Execute(GateUpExecution::Fused));
    }
}

#[test]
fn exhausted_startup_budget_retains_the_fused_fallback() {
    let tuner = MetalTuner::new(TuningConfig {
        startup_budget_ms: 0,
        ..TuningConfig::default()
    });
    assert_eq!(tuner.plan(fixture_key()), TuneAction::Execute(GateUpExecution::Fused));
}

#[test]
fn decode_plan_defaults_to_safe_separate_execution_outside_startup() {
    let tuner = MetalTuner::new(TuningConfig {
        mode: TuningMode::Cached,
        ..TuningConfig::default()
    });
    assert_eq!(
        tuner.decode_plan_action(&decode_plan_key()),
        DecodePlanAction::Execute(DecodePlan::SeparateGateUp)
    );
}

#[test]
fn complete_plan_measurement_suppresses_nested_operator_tuning() {
    let mut tuner = MetalTuner::new(TuningConfig::default());
    tuner.activate_decode_plan(DecodePlan::FusedGateUp, true);
    assert!(!tuner.attention_budget_available());
    assert!(!tuner.expert_budget_available());
    assert!(!tuner.routing_runtime_budget_available());
    tuner.clear_decode_plan();
    assert!(tuner.attention_budget_available());
}

#[test]
fn prefill_measurement_does_not_starve_decode_or_other_families() {
    let mut tuner = MetalTuner::new(TuningConfig {
        startup_budget_ms: 1,
        ..TuningConfig::default()
    });
    let mut prefill = batch_attention_key();
    prefill.causal = true;
    tuner.record_batch_attention(prefill, BatchAttentionExecution::Rows, Duration::from_millis(1));

    assert!(!tuner.batch_attention_runtime_budget_available(true));
    assert!(tuner.batch_attention_runtime_budget_available(false));
    assert!(tuner.attention_budget_available());
    assert!(tuner.expert_budget_available());
    assert!(tuner.routing_budget_available());
}

#[test]
fn completed_startup_retains_decisions_and_shape_discovery_budgets() {
    let mut tuner = MetalTuner::new(TuningConfig::default());
    let key = fixture_key();
    tuner.record(key, GateUpExecution::Separate, Duration::from_micros(1));
    tuner.finish_startup();

    assert_eq!(tuner.plan(key), TuneAction::Execute(GateUpExecution::Separate));
    assert!(!tuner.attention_budget_available());
    assert!(tuner.batch_attention_runtime_budget_available(false));
    assert!(!tuner.expert_budget_available());
    assert!(!tuner.routing_budget_available());
    assert!(tuner.routing_runtime_budget_available());
}
