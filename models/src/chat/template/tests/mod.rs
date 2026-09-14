mod reasoning;

use foundation::conversation::Message;

use super::*;

#[test]
fn model_template_uses_configured_bos_without_tokenizer_duplication() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::ModelJinja,
        source: TemplateSource::ChatTemplateFile,
        tokens: TemplateTokens::new("<s>", "</s>"),
        template: Some(
            "{{ bos_token }}{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}".into(),
        ),
    };
    let prompt = template.render(&request("Hello"))?;
    assert_eq!(prompt.text, "<s>user: Hello");
    assert!(!prompt.add_special_tokens);
    Ok(())
}

#[test]
fn builtin_plain_allows_tokenizer_bos() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::Plain,
        source: TemplateSource::Builtin,
        tokens: TemplateTokens::default(),
        template: None,
    };
    let prompt = template.render(&request("ping"))?;
    assert_eq!(prompt.text, "user: ping");
    assert!(prompt.add_special_tokens);
    Ok(())
}

#[test]
fn builtin_chatml_delimits_messages_and_opens_assistant_turn() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::ChatMl,
        source: TemplateSource::Builtin,
        tokens: TemplateTokens::new("", "<|im_end|>"),
        template: None,
    };
    let prompt = template.render(&request("Napisz zdanie."))?;

    assert_eq!(
        prompt.text,
        "<|im_start|>user\nNapisz zdanie.<|im_end|>\n<|im_start|>assistant\n"
    );
    assert!(prompt.add_special_tokens);
    Ok(())
}

#[test]
fn builtin_turn_protocol_uses_declared_checkpoint_tokens() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::TurnDelimited,
        source: TemplateSource::Builtin,
        tokens: TemplateTokens::new("<bos>", "<eos>").with_turns("<|turn>", "<turn|>"),
        template: None,
    };
    let prompt = template.render(&request("Napisz zdanie."))?;

    assert_eq!(prompt.text, "<|turn>user\nNapisz zdanie.<turn|>\n<|turn>assistant\n");
    assert!(prompt.add_special_tokens);
    Ok(())
}

#[test]
fn builtin_qwen_opens_the_thinking_block() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::QwenChatMl,
        source: TemplateSource::Builtin,
        tokens: TemplateTokens::new("", "<|im_end|>"),
        template: None,
    };

    let prompt = template.render(&request("Napisz zdanie."))?;

    assert_eq!(
        prompt.text,
        "<|im_start|>user\nNapisz zdanie.<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
    Ok(())
}

#[test]
fn builtin_gemma_uses_model_role_and_thinking_protocol() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::Gemma4,
        source: TemplateSource::Builtin,
        tokens: TemplateTokens::new("<bos>", "<eos>").with_turns("<|turn>", "<turn|>"),
        template: None,
    };

    let prompt = template.render(&request("Napisz zdanie."))?;

    assert_eq!(
        prompt.text,
        "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nNapisz zdanie.<turn|>\n<|turn>model\n"
    );
    assert!(!prompt.add_special_tokens);
    Ok(())
}

#[test]
fn chooses_family_fallbacks_only_for_matching_protocols() {
    assert_eq!(builtin_kind(Some("gemma4"), true, false), TemplateKind::Gemma4);
    assert_eq!(builtin_kind(Some("qwen3_5_moe"), false, true), TemplateKind::QwenChatMl);
    assert_eq!(builtin_kind(Some("qwen3_5_moe"), false, false), TemplateKind::Plain);
    assert_eq!(builtin_kind(Some("mistral"), false, false), TemplateKind::MistralInst);
}

#[test]
fn model_template_enables_declared_reasoning_by_default() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::ModelJinja,
        source: TemplateSource::ChatTemplateFile,
        tokens: TemplateTokens::default(),
        template: Some("{% if enable_thinking %}<|think|>{% endif %}".into()),
    };
    let prompt = template.render(&request("Hello"))?;

    assert_eq!(prompt.text, "<|think|>");
    Ok(())
}

fn request(content: &str) -> Conversation {
    Conversation {
        messages: vec![Message {
            role: "user".into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: foundation::conversation::ToolChoice::default(),
    }
}
