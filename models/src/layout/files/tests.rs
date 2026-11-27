use std::fs;

use super::*;
use crate::weights::TensorCatalog;

#[test]
fn discovers_mistral_params_as_supplemental_configuration() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-model-layout-params-{}", std::process::id()));
    fs::create_dir(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    fs::write(root.join("params.json"), "{}")?;
    fs::write(root.join("model.safetensors"), [])?;

    let layout = ModelLayout::inspect(&root)?;
    fs::remove_dir_all(&root)?;

    assert_eq!(layout.configuration_path, Some(root.join("params.json")));
    Ok(())
}

/// A non-import directory with root `SafeTensors` weights beside `.gguf`
/// files loads only the `SafeTensors` weights and never merges GGUF tensors.
#[test]
fn non_import_directory_with_safetensors_and_gguf_reads_only_safetensors() -> Result<()> {
    let root =
        std::env::temp_dir().join(format!("libmir-mixed-safetensors-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    let header =
        r#"{"model.embed_tokens.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}}"#;
    let mut safetensors = u64::try_from(header.len())?.to_le_bytes().to_vec();
    safetensors.extend_from_slice(header.as_bytes());
    safetensors.extend_from_slice(&[0_u8; 8]);
    fs::write(root.join("model.safetensors"), &safetensors)?;
    fs::write(root.join("model-Q4_0.gguf"), b"not-even-read")?;

    let layout = ModelLayout::inspect(&root)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let detected = WeightContainer::detect(&root)?;
    fs::remove_dir_all(&root)?;

    assert_eq!(layout.container(), Some(WeightContainer::SafeTensors));
    assert_eq!(detected, Some(WeightContainer::SafeTensors));
    assert_eq!(layout.weights.len(), 1);
    assert_eq!(catalog.len(), 1);
    assert_eq!(catalog.get("model.embed_tokens.weight").map(|t| t.dtype.as_str()), Some("F16"));
    Ok(())
}
