use std::collections::HashSet;

use models::{
    layout::VisionConfig,
    weights::{TensorCatalog, TensorInfo, VisionTensorSchema},
};

use crate::{CudaBackend, CudaTensorSet, Error, Result};

pub fn load_vision_tensors(
    backend: &CudaBackend,
    config: &VisionConfig,
    catalog: &TensorCatalog,
) -> Result<CudaTensorSet> {
    let schema = VisionTensorSchema::discover(config);
    let mut seen = HashSet::with_capacity(schema.requirements.len());
    let mut tensors = Vec::with_capacity(schema.requirements.len());
    for requirement in schema.requirements {
        let tensor = requirement
            .aliases
            .iter()
            .find_map(|alias| find(catalog, alias))
            .ok_or_else(|| Error::MissingTensor(requirement.missing_label()))?;
        if seen.insert(tensor.name.as_str()) {
            tensors.push(tensor);
        }
    }
    for tensor in &catalog.tensors {
        if is_optional_vision_tensor(config, &tensor.name) && seen.insert(tensor.name.as_str()) {
            tensors.push(tensor);
        }
    }
    let cast = backend.prepare_dense_cast()?;
    let mut upload = backend.begin_tensor_upload();
    for tensor in tensors {
        upload.enqueue_as_bf16(tensor, &cast)?;
    }
    upload.finish()
}

#[allow(clippy::case_sensitive_file_extension_comparisons)]
fn is_optional_vision_tensor(config: &VisionConfig, name: &str) -> bool {
    match config {
        VisionConfig::PooledEncoder(_) => {
            (name.starts_with("model.vision_tower.") || name.starts_with("model.embed_vision."))
                && name.ends_with(".bias")
        },
        VisionConfig::SpatialMergeEncoder(_) => false,
    }
}

fn find<'a>(catalog: &'a TensorCatalog, name: &str) -> Option<&'a TensorInfo> {
    catalog.tensors.iter().find(|tensor| tensor.name == name)
}
