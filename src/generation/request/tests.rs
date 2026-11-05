use super::*;

#[test]
fn json_content_rejects_forced_tools_but_keeps_optional_tools() -> crate::Result<()> {
    use foundation::conversation::ToolChoice;
    let json = ResponseFormat::JsonSchema(serde_json::json!({"type":"object"}));
    for choice in [ToolChoice::Required, ToolChoice::Function("emit".into())] {
        assert!(json.validate_tool_choice(&choice).is_err());
        ResponseFormat::Text.validate_tool_choice(&choice)?;
    }
    for choice in [ToolChoice::Auto, ToolChoice::None] {
        json.validate_tool_choice(&choice)?;
    }
    Ok(())
}

#[test]
fn reasoning_exit_is_default() {
    assert_eq!(
        ReasoningCyclePolicy::default(),
        ReasoningCyclePolicy::ExitReasoning { min_tokens: 64 }
    );
}

#[test]
fn validates_json_schema_before_loading_a_model() -> crate::Result<()> {
    use serde_json::json;
    for schema in [json!({"type":"nonsense"}), json!({"type":"array","minItems":-1}), json!(false)]
    {
        let request = GenerationRequest {
            response_format: ResponseFormat::JsonSchema(schema),
            ..Default::default()
        };
        assert!(request.validate().is_err());
    }
    let request = GenerationRequest {
        response_format: ResponseFormat::JsonSchema(json!({"type":"array", "uniqueItems":true,
            "items":{"type":"string"}})),
        ..Default::default()
    };
    request.validate()?;
    Ok(())
}

#[test]
fn rejects_explicit_incompatible_sampling_but_overrides_checkpoint_defaults() -> crate::Result<()> {
    use models::generation::GenerationConfig;
    let mut request = GenerationRequest {
        response_format: ResponseFormat::JsonSchema(serde_json::json!({"type":"object"})),
        ..Default::default()
    };
    for options in [
        GenerationOverrides {
            min_tokens: Some(5),
            ..Default::default()
        },
        GenerationOverrides {
            ignore_eos: Some(true),
            ..Default::default()
        },
        GenerationOverrides {
            repetition_penalty: Some(1.1),
            ..Default::default()
        },
        GenerationOverrides {
            repetition_penalty: Some(f32::NAN),
            ..Default::default()
        },
    ] {
        request.options = options;
        assert!(request.validate().is_err());
    }
    request.options = GenerationOverrides::default();
    request.validate()?;
    let defaults = GenerationConfig::from_value(&serde_json::json!({"repetition_penalty":1.1}))?;
    let settings = defaults.resolve(request.sampling_options())?;
    assert!((settings.repetition_penalty - 1.0).abs() < f32::EPSILON);
    request.response_format = ResponseFormat::Text;
    let settings = defaults.resolve(request.sampling_options())?;
    assert!((settings.repetition_penalty - 1.1).abs() < f32::EPSILON);
    Ok(())
}
