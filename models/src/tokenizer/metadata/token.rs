use serde_json::Value;
use tokenizers::AddedToken;

use crate::error::{ModelsError, Result};

pub(super) struct TokenConfig {
    pub(super) content: String,
    flags: u8,
}

impl TokenConfig {
    const LSTRIP: u8 = 1 << 1;
    const NORMALIZED: u8 = 1 << 3;
    const RSTRIP: u8 = 1 << 2;
    const SINGLE_WORD: u8 = 1;
    const SPECIAL: u8 = 1 << 4;

    pub(super) fn legacy(content: String) -> Self {
        Self { content, flags: Self::NORMALIZED }
    }

    pub(super) fn from_value(value: &Value) -> Result<Self> {
        let object = value.as_object().ok_or_else(|| {
            ModelsError::InvalidConfig("added token must be a JSON object".into())
        })?;
        let content = object
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| ModelsError::InvalidConfig("added token has no content".into()))?
            .to_owned();
        let flags = [
            ("single_word", Self::SINGLE_WORD),
            ("lstrip", Self::LSTRIP),
            ("rstrip", Self::RSTRIP),
            ("normalized", Self::NORMALIZED),
            ("special", Self::SPECIAL),
        ]
        .into_iter()
        .try_fold(0, |flags, (name, flag)| {
            token_flag(object, name).map(|enabled| flags | (u8::from(enabled) * flag))
        })?;
        Ok(Self { content, flags })
    }

    pub(super) fn special(mut self) -> Self {
        self.flags |= Self::SPECIAL;
        self.flags &= !Self::NORMALIZED;
        self
    }

    pub(super) fn added_token(&self) -> AddedToken {
        AddedToken::from(self.content.clone(), self.has(Self::SPECIAL))
            .single_word(self.has(Self::SINGLE_WORD))
            .lstrip(self.has(Self::LSTRIP))
            .rstrip(self.has(Self::RSTRIP))
            .normalized(self.has(Self::NORMALIZED))
    }

    fn has(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }
}

fn token_flag(object: &serde_json::Map<String, Value>, name: &str) -> Result<bool> {
    let Some(value) = object.get(name) else {
        return Ok(false);
    };
    value
        .as_bool()
        .ok_or_else(|| ModelsError::InvalidConfig(format!("added token {name} must be a boolean")))
}
