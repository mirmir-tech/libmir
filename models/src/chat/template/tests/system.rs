use foundation::conversation::Message;

use super::*;

/// Mistral v0.3's GGUF template accepts only alternating user and assistant
/// turns; the system text then leads the first user message.
#[test]
fn template_without_a_system_role_takes_it_in_the_first_user_turn() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::ModelJinja,
        source: TemplateSource::ChatTemplateFile,
        tokens: TemplateTokens::new("<s>", "</s>"),
        template: Some(
            "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% else %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}".into(),
        ),
    };
    let mut conversation = request("Hello");
    conversation.messages.insert(0, message("system", "Be brief."));
    let prompt = template.render(&conversation)?;
    assert_eq!(prompt.text, "<s>[INST] Be brief.\nHello [/INST]");
    Ok(())
}

fn message(role: &str, content: &str) -> Message {
    Message {
        role: role.into(),
        content: content.into(),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
    }
}
