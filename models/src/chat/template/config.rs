use std::fs;

use serde_json::Value;

use super::TemplateSource;
use crate::{
    error::{ModelsError, Result},
    layout::ModelLayout,
};

#[derive(Debug, Clone, Default)]
pub(super) struct TemplateTokens {
    bos: String,
    eos: String,
}

impl TemplateTokens {
    pub(super) fn new(bos: impl Into<String>, eos: impl Into<String>) -> Self {
        Self { bos: bos.into(), eos: eos.into() }
    }

    pub(super) fn requires_automatic_bos(&self, text: &str) -> bool {
        self.bos.is_empty() || !text.starts_with(&self.bos)
    }

    pub(super) fn bos(&self) -> &str {
        &self.bos
    }

    pub(super) fn eos(&self) -> &str {
        &self.eos
    }
}

#[derive(Debug, Clone)]
pub(super) struct ModelTemplateConfig {
    pub(super) source: TemplateSource,
    pub(super) template: Option<String>,
    pub(super) tokens: TemplateTokens,
}

impl ModelTemplateConfig {
    pub(super) fn from_layout(layout: &ModelLayout) -> Result<Self> {
        let config = layout
            .tokenizer_config_path
            .as_deref()
            .map(read_tokenizer_config)
            .transpose()?
            .unwrap_or_default();
        let file_template =
            layout.chat_template_path.as_deref().map(fs::read_to_string).transpose()?;
        let (source, template) = file_template.map_or_else(
            || {
                let source = config
                    .template
                    .as_ref()
                    .map_or(TemplateSource::Builtin, |_| TemplateSource::TokenizerConfig);
                (source, config.template)
            },
            |template| (TemplateSource::ChatTemplateFile, Some(template)),
        );
        Ok(Self { source, template, tokens: config.tokens })
    }
}

#[derive(Debug, Default)]
struct TokenizerConfig {
    template: Option<String>,
    tokens: TemplateTokens,
}

fn read_tokenizer_config(path: &std::path::Path) -> Result<TokenizerConfig> {
    let value: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(TokenizerConfig {
        template: template(&value)?,
        tokens: TemplateTokens::new(token(&value, "bos_token")?, token(&value, "eos_token")?),
    })
}

fn template(config: &Value) -> Result<Option<String>> {
    let Some(value) = config.get("chat_template") else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(template) = value.as_str() {
        return Ok(Some(template.into()));
    }
    let templates = value
        .as_array()
        .ok_or_else(|| invalid("chat_template must be string or array"))?;
    templates
        .iter()
        .filter_map(|entry| entry.get("template").and_then(Value::as_str).map(|body| (entry, body)))
        .find(|(entry, _body)| entry.get("name").and_then(Value::as_str) == Some("default"))
        .or_else(|| {
            templates.iter().find_map(|entry| {
                entry.get("template").and_then(Value::as_str).map(|body| (entry, body))
            })
        })
        .map(|(_entry, body)| body.into())
        .map(Some)
        .ok_or_else(|| invalid("chat_template array has no string template"))
}

fn token(config: &Value, name: &str) -> Result<String> {
    let Some(value) = config.get(name) else {
        return Ok(String::new());
    };
    if value.is_null() {
        return Ok(String::new());
    }
    if let Some(token) = value.as_str() {
        return Ok(token.into());
    }
    value
        .get("content")
        .and_then(Value::as_str)
        .map(Into::into)
        .ok_or_else(|| invalid(format!("{name} must be string or added-token object")))
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn prefers_default_template_from_template_list() -> Result<()> {
        let config = json!({
            "chat_template": [
                { "name": "tool_use", "template": "tool" },
                { "name": "default", "template": "default" }
            ]
        });

        assert_eq!(template(&config)?.as_deref(), Some("default"));
        Ok(())
    }

    #[test]
    fn accepts_added_token_objects() -> Result<()> {
        let config = json!({ "bos_token": { "content": "<s>" } });

        assert_eq!(token(&config, "bos_token")?, "<s>");
        Ok(())
    }
}
