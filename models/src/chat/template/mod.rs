mod config;
mod fallback;
mod protocol;
mod reasoning;
mod render;
mod tools;
use foundation::conversation::Conversation;
pub use reasoning::ReasoningMode;
pub use tools::ToolCallPrefix;

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
    pub tool_prefix: Option<ToolCallPrefix>,
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
    MistralInst,
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

    pub fn render(&self, conversation: &Conversation) -> Result<ChatPrompt> {
        self.render_with_reasoning(conversation, ReasoningMode::ModelDefault)
    }

    /// Renders the model's reasoning control; rejects unsupported explicit
    /// modes.
    pub fn render_with_reasoning(
        &self,
        conversation: &Conversation,
        reasoning: ReasoningMode,
    ) -> Result<ChatPrompt> {
        if self.template.is_none()
            && reasoning != ReasoningMode::ModelDefault
            && !matches!(self.kind, TemplateKind::QwenChatMl | TemplateKind::Gemma4)
        {
            return Err(crate::ModelsError::UnsupportedReasoningControl);
        }
        let mut text = self.template.as_deref().map_or_else(
            || Ok(render_builtin(conversation, self.kind, &self.tokens, reasoning)),
            |template| render_model_template(template, conversation, &self.tokens, reasoning),
        )?;
        let tool_prefix = tools::prefix(self.template.as_deref(), conversation, reasoning)?;
        if let Some(prefix) = &tool_prefix {
            text.push_str(&prefix.text());
        }
        Ok(ChatPrompt {
            tool_prefix,
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
        Some(model) if model.starts_with("mistral") => TemplateKind::MistralInst,
        _ if has_turns => TemplateKind::TurnDelimited,
        _ if has_chatml => TemplateKind::ChatMl,
        _ => TemplateKind::Plain,
    }
}

#[cfg(test)]
mod tests;
