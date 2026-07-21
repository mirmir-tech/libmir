use std::collections::HashSet;

use models::weights::{DenseDecoderLayerBindings, TensorCatalog, TensorInfo};

use crate::{Error, Result};

pub(super) struct DenseLayerSource<'a> {
    pub tensors: Vec<&'a TensorInfo>,
}

impl<'a> DenseLayerSource<'a> {
    pub fn discover(
        catalog: &'a TensorCatalog,
        bindings: DenseDecoderLayerBindings<'_>,
    ) -> Result<Self> {
        let mut seen = HashSet::new();
        let tensors = bindings
            .physical_sources()
            .into_iter()
            .filter(|name| seen.insert(*name))
            .map(|name| required_tensor(catalog, name))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { tensors })
    }
}

fn required_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
