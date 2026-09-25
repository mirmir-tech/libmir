use super::*;

#[test]
fn renders_causal_reranker_roles_and_preserves_scoring_suffix() -> Result<()> {
    let template = ChatTemplate {
        kind: TemplateKind::ModelJinja,
        source: TemplateSource::ChatTemplateFile,
        tokens: TemplateTokens::new("", "<|im_end|>"),
        template: Some(concat!(
            "{%- set instruction = messages | selectattr('role', 'eq', 'system') | map(attribute='content') | first | default('Retrieve passages') -%}",
            "{%- set query = messages | selectattr('role', 'eq', 'query') | map(attribute='content') | first -%}",
            "{%- set document = messages | selectattr('role', 'eq', 'document') | map(attribute='content') | first -%}",
            "<|im_start|>user\n<Instruct>: {{ instruction }}\n<Query>: {{ query }}\n<Document>: {{ document }}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\n"
        ).into()),
    };
    let conversation = Conversation {
        messages: vec![message("query", "capital of France"), message("document", "Paris")],
        ..Default::default()
    };
    let prompt = template.render(&conversation)?;
    assert_eq!(
        prompt.text,
        "<|im_start|>user\n<Instruct>: Retrieve passages\n<Query>: capital of France\n<Document>: Paris<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
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
