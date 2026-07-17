use std::path::Path;

use crate::{Error, Result};

pub(super) fn model_id(path: &Path) -> Result<String> {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| Error::ModelId(path.to_path_buf()))
}

pub(super) fn validate_context(prompt: usize, max_tokens: usize, context: usize) -> Result<()> {
    let requested = prompt.saturating_add(max_tokens);
    if requested <= context {
        return Ok(());
    }
    Err(Error::Context { requested, context, prompt, max_tokens })
}
