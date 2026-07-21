use foundation::protocol::ChatCompletionRequest;

use super::{TemplateKind, config::TemplateTokens};

pub(super) fn render_builtin(
    request: &ChatCompletionRequest,
    kind: TemplateKind,
    tokens: &TemplateTokens,
) -> String {
    match kind {
        TemplateKind::ChatMl => render_chatml(request),
        TemplateKind::TurnDelimited => render_turns(request, tokens),
        TemplateKind::Plain => render_plain(request),
        TemplateKind::ModelJinja => unreachable!("model template has a Jinja body"),
    }
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
