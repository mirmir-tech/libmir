use super::invalid;
use crate::{error::Result, weights::TensorCatalog};

pub(super) fn companion(catalog: &TensorCatalog, name: &str) -> Result<()> {
    let tensor = catalog.get(name).ok_or_else(|| invalid(name, "companion tensor is missing"))?;
    if tensor.shape.is_empty() || tensor.shape == [1] {
        Ok(())
    } else {
        Err(invalid(name, &format!("expected one value, found shape {:?}", tensor.shape)))
    }
}
