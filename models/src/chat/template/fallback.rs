use foundation::protocol::ChatCompletionRequest;

use super::{TemplateKind, config::TemplateTokens};

pub(super) fn render_builtin(
    request: &ChatCompletionRequest,
    kind: TemplateKind,
    tokens: &TemplateTokens,
) -> String {
    match kind {
        TemplateKind::ChatMl => render_chatml(request),
        TemplateKind::QwenChatMl => render_qwen(request),
        TemplateKind::TurnDelimited => render_turns(request, tokens),
        TemplateKind::MistralInst => render_mistral(request, tokens),
        TemplateKind::Gemma4 => render_gemma4(request, tokens),
        TemplateKind::Plain => render_plain(request),
        TemplateKind::ModelJinja => unreachable!("model template has a Jinja body"),
    }
}

fn render_qwen(request: &ChatCompletionRequest) -> String {
    let mut prompt = render_chatml(request);
    prompt.push_str("<think>\n");
    prompt
}

fn render_chatml(request: &ChatCompletionRequest) -> String {
    let mut prompt = request.messages.iter().fold(String::new(), |mut prompt, message| {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
        prompt
    });
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn render_plain(request: &ChatCompletionRequest) -> String {
    request
        .messages
        .iter()
        .map(|message| format!("{}: {}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_mistral(request: &ChatCompletionRequest, tokens: &TemplateTokens) -> String {
    let mut messages = request.messages.as_slice();
    let system = messages
        .first()
        .filter(|message| matches!(message.role.as_str(), "system" | "developer"));
    if system.is_some() {
        messages = &messages[1..];
    }
    let mut prompt = tokens.bos().to_owned();
    for (index, message) in messages.iter().enumerate() {
        if message.role == "assistant" {
            prompt.push(' ');
            prompt.push_str(message.content.trim());
            prompt.push_str(tokens.eos());
            continue;
        }
        prompt.push_str("[INST] ");
        if index == 0
            && let Some(system) = system
        {
            prompt.push_str(system.content.trim());
            prompt.push_str("\n\n");
        }
        prompt.push_str(message.content.trim());
        prompt.push_str("[/INST]");
    }
    prompt
}

fn render_turns(request: &ChatCompletionRequest, tokens: &TemplateTokens) -> String {
    let Some((start, end)) = tokens.turn_tokens() else {
        unreachable!("turn-delimited template requires start and end tokens");
    };
    let mut prompt = request.messages.iter().fold(String::new(), |mut prompt, message| {
        prompt.push_str(start);
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str(end);
        prompt.push('\n');
        prompt
    });
    prompt.push_str(start);
    prompt.push_str("assistant\n");
    prompt
}

fn render_gemma4(request: &ChatCompletionRequest, tokens: &TemplateTokens) -> String {
    let Some((start, end)) = tokens.turn_tokens() else {
        unreachable!("Gemma 4 fallback requires turn tokens");
    };
    let mut messages = request.messages.as_slice();
    let mut prompt = tokens.bos().to_owned();
    prompt.push_str(start);
    prompt.push_str("system\n<|think|>\n");
    if let Some(message) = messages
        .first()
        .filter(|message| matches!(message.role.as_str(), "system" | "developer"))
    {
        prompt.push_str(message.content.trim());
        messages = &messages[1..];
    }
    prompt.push_str(end);
    prompt.push('\n');
    for message in messages {
        prompt.push_str(start);
        let assistant = message.role == "assistant";
        prompt.push_str(if assistant {
            "model"
        } else {
            &message.role
        });
        prompt.push('\n');
        if assistant && let Some(reasoning) = message.reasoning_content.as_deref() {
            prompt.push_str("<|channel>thought\n");
            prompt.push_str(reasoning.trim());
            prompt.push_str("\n<channel|>");
        }
        prompt.push_str(message.content.trim());
        prompt.push_str(end);
        prompt.push('\n');
    }
    prompt.push_str(start);
    prompt.push_str("model\n");
    prompt
}

#[cfg(test)]
mod tests {
    use foundation::protocol::ChatMessage;

    use super::*;

    #[test]
    fn renders_mistral_inst_conversation() {
        let request = ChatCompletionRequest {
            model: "mistral".into(),
            messages: vec![
                message("system", "Be concise."),
                message("user", "Hello"),
                message("assistant", "Hi!"),
                message("user", "Two plus two?"),
            ],
            tools: Vec::new(),
            tool_choice: None,
            stream: false,
            max_tokens: None,
            min_tokens: None,
            ignore_eos: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            seed: None,
        };

        let prompt = render_mistral(&request, &TemplateTokens::new("<s>", "</s>"));

        assert_eq!(
            prompt,
            "<s>[INST] Be concise.\n\nHello[/INST] Hi!</s>[INST] Two plus two?[/INST]"
        );
    }

    fn message(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}
