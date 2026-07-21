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
