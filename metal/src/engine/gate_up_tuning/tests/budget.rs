use std::time::Duration;

use runtime::tuning::{TuningConfig, TuningMode};

use super::{
    super::{GateUpExecution, MetalTuner, TuneAction},
    batch_attention_key, fixture_key,
};
use crate::engine::attention_batch_tuning::BatchAttentionExecution;

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
fn prefill_measurement_does_not_starve_decode_or_other_families() {
    let mut tuner = MetalTuner::new(TuningConfig {
        startup_budget_ms: 1,
        ..TuningConfig::default()
    });
    let mut prefill = batch_attention_key();
    prefill.causal = true;
    tuner.record_batch_attention(prefill, BatchAttentionExecution::Rows, Duration::from_millis(1));

    assert!(!tuner.batch_attention_budget_available(true));
    assert!(tuner.batch_attention_budget_available(false));
    assert!(tuner.attention_budget_available());
    assert!(tuner.expert_budget_available());
    assert!(tuner.routing_budget_available());
}

#[test]
fn completed_startup_retains_cached_decisions_but_closes_new_measurements() {
    let mut tuner = MetalTuner::new(TuningConfig::default());
    let key = fixture_key();
    tuner.record(key, GateUpExecution::Separate, Duration::from_micros(1));
    tuner.finish_startup();

    assert_eq!(tuner.plan(key), TuneAction::Execute(GateUpExecution::Separate));
    assert!(!tuner.attention_budget_available());
    assert!(!tuner.batch_attention_budget_available(false));
    assert!(tuner.batch_attention_runtime_budget_available(false));
    assert!(!tuner.expert_budget_available());
    assert!(!tuner.routing_budget_available());
}
