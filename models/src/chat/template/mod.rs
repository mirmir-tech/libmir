mod config;
mod fallback;
mod protocol;
mod render;

use foundation::protocol::ChatCompletionRequest;

use self::{
    config::{ModelTemplateConfig, TemplateTokens},
    fallback::render_builtin,
    render::render_model_template,
};
use crate::{error::Result, layout::ModelLayout};

#[derive(Debug, Clone)]
pub struct ChatPrompt {
    pub text: String,
    pub source: TemplateSource,
    pub add_special_tokens: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateSource {
    Builtin,
    ChatTemplateFile,
    TokenizerConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateKind {
    ModelJinja,
    ChatMl,
    QwenChatMl,
    TurnDelimited,
    Gemma4,
    Plain,
}

#[derive(Debug, Clone)]
pub struct ChatTemplate {
    kind: TemplateKind,
    source: TemplateSource,
    tokens: TemplateTokens,
    template: Option<String>,
}

impl ChatTemplate {
    pub fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let config = ModelTemplateConfig::from_layout(layout)?;
        Ok(if let Some(template) = config.template {
            Self {
                kind: TemplateKind::ModelJinja,
                source: config.source,
                tokens: config.tokens,
                template: Some(template),
            }
        } else {
            let has_turns = config.tokens.turn_tokens().is_some();
            let has_chatml = protocol::has_chatml_tokens(layout.tokenizer_path.as_deref())?;
            Self {
                kind: builtin_kind(config.model_type.as_deref(), has_turns, has_chatml),
                source: TemplateSource::Builtin,
                tokens: config.tokens,
                template: None,
            }
        })
    }

    pub fn render(&self, request: &ChatCompletionRequest) -> Result<ChatPrompt> {
        let text = self.template.as_deref().map_or_else(
            || Ok(render_builtin(request, self.kind, &self.tokens)),
            |template| render_model_template(template, request, &self.tokens),
        )?;
        Ok(ChatPrompt {
            add_special_tokens: self.tokens.requires_automatic_bos(&text),
            text,
            source: self.source.clone(),
        })
    }

    #[must_use]
    pub const fn kind(&self) -> TemplateKind {
        self.kind
    }
}

fn builtin_kind(model_type: Option<&str>, has_turns: bool, has_chatml: bool) -> TemplateKind {
    match model_type {
        Some(model) if model.starts_with("gemma4") && has_turns => TemplateKind::Gemma4,
        Some(model) if model.starts_with("qwen") && has_chatml => TemplateKind::QwenChatMl,
        _ if has_turns => TemplateKind::TurnDelimited,
        _ if has_chatml => TemplateKind::ChatMl,
        _ => TemplateKind::Plain,
    }
}

#[cfg(test)]
mod tests {
    use foundation::protocol::ChatMessage;

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

    fn request(content: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: content.into(),
                reasoning_content: None,
            }],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
            seed: None,
        }
    }
}
