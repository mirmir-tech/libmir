use serde_json::json;

use super::*;

#[test]
fn resolves_checkpoint_defaults_and_explicit_overrides() -> Result<()> {
    let config = GenerationConfig::from_value(&json!({
        "max_new_tokens": 384,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 32,
        "repetition_penalty": 1.1,
        "eos_token_id": [2, 4]
    }))?;
    let settings = config.resolve(GenerationOverrides {
        max_tokens: Some(12),
        min_tokens: Some(8),
        ignore_eos: Some(true),
        temperature: Some(0.2),
        top_p: Some(0.8),
        top_k: Some(16),
        repetition_penalty: Some(1.2),
    })?;

    assert_eq!(settings.max_tokens, 12);
    assert_eq!(settings.min_tokens, 8);
    assert!(settings.ignore_eos);
    assert!((settings.temperature - 0.2).abs() < f32::EPSILON);
    assert!((settings.top_p - 0.8).abs() < f32::EPSILON);
    assert_eq!(settings.top_k, 16);
    assert!((settings.repetition_penalty - 1.2).abs() < f32::EPSILON);
    assert_eq!(config.stop_token_ids(), [2, 4]);
    Ok(())
}

#[test]
fn disables_sampling_when_checkpoint_requests_greedy_generation() -> Result<()> {
    let config = GenerationConfig::from_value(&json!({ "do_sample": false, "temperature": 0.7 }))?;

    assert!(config.resolve(GenerationOverrides::default())?.temperature.abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn sampling_mode_uses_neutral_temperature_when_checkpoint_omits_it() -> Result<()> {
    let config = GenerationConfig::from_value(&json!({ "do_sample": true }))?;
    let settings = config.resolve(GenerationOverrides::default())?;

    assert!((settings.temperature - 1.0).abs() < f32::EPSILON);
    assert!((settings.top_p - 1.0).abs() < f32::EPSILON);
    assert_eq!(settings.top_k, 0);
    Ok(())
}

#[test]
fn checkpoint_and_user_values_override_generic_sampling_defaults() -> Result<()> {
    let config = GenerationConfig::from_value(&json!({
        "do_sample": true,
        "temperature": 0.8,
        "top_p": 0.9
    }))?;
    let checkpoint = config.resolve(GenerationOverrides::default())?;
    let overridden = config.resolve(GenerationOverrides {
        temperature: Some(0.6),
        ..GenerationOverrides::default()
    })?;

    assert!((checkpoint.temperature - 0.8).abs() < f32::EPSILON);
    assert!((checkpoint.top_p - 0.9).abs() < f32::EPSILON);
    assert!((overridden.temperature - 0.6).abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn request_overrides_preserve_loaded_model_settings() -> Result<()> {
    let loaded = GenerationConfig::default().resolve(GenerationOverrides {
        max_tokens: Some(20_048),
        min_tokens: None,
        ignore_eos: None,
        temperature: Some(1.0),
        top_p: Some(0.95),
        top_k: Some(64),
        repetition_penalty: Some(1.0),
    })?;

    assert_eq!(loaded.with_overrides(GenerationOverrides::default())?, loaded);
    assert_eq!(
        loaded
            .with_overrides(GenerationOverrides {
                max_tokens: Some(512),
                ..GenerationOverrides::default()
            })?
            .max_tokens,
        512
    );
    Ok(())
}

#[test]
fn rejects_minimum_larger_than_generation_limit() {
    let result = GenerationConfig::default().resolve(GenerationOverrides {
        max_tokens: Some(8),
        min_tokens: Some(9),
        ..GenerationOverrides::default()
    });

    assert!(result.is_err());
}
