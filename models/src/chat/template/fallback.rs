use foundation::{model::ModelFamily, protocol::ChatCompletionRequest};

use super::{TemplateKind, config::TemplateTokens};

pub(super) fn kind(family: &ModelFamily, tokens: &TemplateTokens) -> TemplateKind {
    match family {
        ModelFamily::Gemma if tokens.bos() == "<bos>" => TemplateKind::Gemma4,
        ModelFamily::Gemma => TemplateKind::Gemma,
        ModelFamily::Bielik | ModelFamily::Llama | ModelFamily::Mistral => {
            TemplateKind::LlamaHeader
        },
        ModelFamily::DeepSeek | ModelFamily::Glm | ModelFamily::Qwen => TemplateKind::ChatMl,
        ModelFamily::Unknown => TemplateKind::Plain,
    }
}

pub(super) fn render_builtin(
    kind: TemplateKind,
    request: &ChatCompletionRequest,
    tokens: &TemplateTokens,
) -> String {
    match kind {
        TemplateKind::ChatMl => render_chatml(request),
        TemplateKind::Gemma => render_gemma(request),
        TemplateKind::Gemma4 => render_gemma4(request, tokens),
        TemplateKind::LlamaHeader => render_llama_header(request, tokens),
        TemplateKind::Plain => render_plain(request),
        TemplateKind::ModelJinja => unreachable!("model template has its own renderer"),
    }
}

fn render_chatml(request: &ChatCompletionRequest) -> String {
    let mut prompt = String::new();
    for message in &request.messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn render_gemma(request: &ChatCompletionRequest) -> String {
    let mut prompt = String::new();
    for message in &request.messages {
        let role = if message.role == "assistant" {
            "model"
        } else {
            &message.role
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

fn render_gemma4(request: &ChatCompletionRequest, tokens: &TemplateTokens) -> String {
    let mut prompt = format!("{}<|turn>system\n<|think|>\n<turn|>\n", tokens.bos());
    for message in &request.messages {
        let role = if message.role == "assistant" {
            "model"
        } else {
            &message.role
        };
        prompt.push_str("<|turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(message.content.trim());
        prompt.push_str("<turn|>\n");
    }
    prompt.push_str("<|turn>model\n");
    prompt
}

fn render_llama_header(request: &ChatCompletionRequest, tokens: &TemplateTokens) -> String {
    let mut prompt = tokens.bos().to_owned();
    for message in &request.messages {
        prompt.push_str("<|start_header_id|>");
        prompt.push_str(&message.role);
        prompt.push_str("<|end_header_id|>\n\n");
        prompt.push_str(&message.content);
        prompt.push_str("<|eot_id|>");
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
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
