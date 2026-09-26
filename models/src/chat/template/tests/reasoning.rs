use super::{request, *};

fn jinja(body: &str) -> ChatTemplate {
    ChatTemplate {
        kind: TemplateKind::ModelJinja,
        source: TemplateSource::ChatTemplateFile,
        tokens: TemplateTokens::default(),
        template: Some(body.into()),
    }
}

#[test]
fn thinking_mode_is_request_scoped_and_default_rendering_is_preserved() -> Result<()> {
    let template = jinja(
        "assistant\n{% if enable_thinking %}<think>\n{% else %}<think>\n\n</think>\n\n{% endif %}",
    );
    let conversation = request("Hello");
    let original = template.render(&conversation)?.text;
    assert_eq!(original, "assistant\n<think>\n");
    assert_eq!(
        template.render_with_reasoning(&conversation, ReasoningMode::Enabled)?.text,
        original
    );
    assert_eq!(
        template.render_with_reasoning(&conversation, ReasoningMode::Disabled)?.text,
        "assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(
        template.render(&conversation)?.text,
        original,
        "override leaked to next request"
    );
    Ok(())
}

#[test]
fn unsupported_templates_reject_explicit_modes_instead_of_ignoring_them() {
    for body in [
        "assistant",
        "literal enable_thinking",
        "{% set enable_thinking = true %}{{ enable_thinking }}",
    ] {
        let template = jinja(body);
        assert!(template.render(&request("Hi")).is_ok());
        for mode in [ReasoningMode::Enabled, ReasoningMode::Disabled] {
            assert!(matches!(
                template.render_with_reasoning(&request("Hi"), mode),
                Err(crate::ModelsError::UnsupportedReasoningControl)
            ));
        }
    }
}

#[test]
fn qwen_fallback_closes_the_thinking_block_only_when_requested() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::QwenChatMl,
        source: TemplateSource::Builtin,
        tokens: TemplateTokens::default(),
        template: None,
    };
    assert!(template.render(&request("Hi"))?.text.ends_with("<think>\n"));
    assert!(
        template
            .render_with_reasoning(&request("Hi"), ReasoningMode::Disabled)?
            .text
            .ends_with("<think>\n\n</think>\n\n")
    );
    Ok(())
}

#[test]
fn gemma_fallback_omits_only_the_reasoning_control_and_preserves_the_system_message() -> Result<()>
{
    let template = ChatTemplate {
        kind: TemplateKind::Gemma4,
        source: TemplateSource::Builtin,
        tokens: TemplateTokens::new("<bos>", "<eos>").with_turns("<|turn>", "<turn|>"),
        template: None,
    };
    let mut conversation = request("Hello");
    let mut system = conversation.messages[0].clone();
    system.role = "system".into();
    system.content = "Be concise".into();
    conversation.messages.insert(0, system);
    let prompt = template.render_with_reasoning(&conversation, ReasoningMode::Disabled)?;
    assert!(!prompt.text.contains("<|think|>"));
    assert!(prompt.text.contains("system\nBe concise<turn|>"));
    assert!(prompt.text.ends_with("<|turn>model\n"));
    Ok(())
}

#[test]
fn generic_fallback_rejects_an_explicit_thinking_switch() {
    let template = ChatTemplate {
        kind: TemplateKind::Plain,
        source: TemplateSource::Builtin,
        tokens: TemplateTokens::default(),
        template: None,
    };
    assert!(matches!(
        template.render_with_reasoning(&request("Hi"), ReasoningMode::Disabled),
        Err(crate::ModelsError::UnsupportedReasoningControl)
    ));
}

#[test]
fn named_xml_tools_prefill_only_in_explicit_no_thinking_mode() -> Result<()> {
    use foundation::conversation::{FunctionDefinition, Tool, ToolChoice};
    let template = ChatTemplate {
        kind: TemplateKind::ModelJinja,
        source: TemplateSource::ChatTemplateFile,
        tokens: TemplateTokens::new("", "<|im_end|>"),
        template: Some(
            "{# <tool_call><function= #}{{ enable_thinking }}<|im_start|>assistant\n".into(),
        ),
    };
    let mut conversation = request("extract");
    conversation.tool_choice = ToolChoice::Function("emit".into());
    conversation.tools = vec![Tool {
        kind: "function".into(),
        function: FunctionDefinition {
            name: "emit".into(),
            description: None,
            parameters: serde_json::json!({"type":"object"}),
        },
    }];
    let disabled = template.render_with_reasoning(&conversation, ReasoningMode::Disabled)?;
    assert!(disabled.text.ends_with("<tool_call>\n<function=emit>\n"));
    assert!(disabled.tool_prefix.is_some());
    assert_eq!(
        template.named_tool_prefix(&conversation)?.map(|p| p.text()),
        Some("<tool_call>\n<function=emit>\n".into())
    );
    assert!(
        template
            .render_with_reasoning(&conversation, ReasoningMode::Enabled)?
            .tool_prefix
            .is_none()
    );
    conversation.tool_choice = ToolChoice::Auto;
    assert!(template.named_tool_prefix(&conversation)?.is_none());
    assert!(
        template
            .render_with_reasoning(&conversation, ReasoningMode::Disabled)?
            .tool_prefix
            .is_none()
    );
    conversation.tool_choice = ToolChoice::Function("missing".into());
    assert!(template.render_with_reasoning(&conversation, ReasoningMode::Disabled).is_err());
    Ok(())
}
