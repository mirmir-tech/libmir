use std::{fs, path::Path};

use super::{SemanticModelSpec, validation};
use crate::error::Result;

pub(super) fn read(path: &Path) -> Result<SemanticModelSpec> {
    let spec = toml::from_str(&fs::read_to_string(path)?)?;
    validation::validate(&spec)?;
    Ok(spec)
}
