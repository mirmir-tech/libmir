use std::{fs, path::Path};

use super::WeightContainer;
use crate::error::Result;

#[test]
fn names_the_container_by_extension_in_any_case() {
    let of = |name| WeightContainer::of(Path::new(name));
    assert_eq!(of("Model-Q4_K_M.GGUF"), Some(WeightContainer::Gguf));
    assert_eq!(of("model.SafeTensors"), Some(WeightContainer::SafeTensors));
    assert_eq!(of("config.json"), None);
    assert_eq!(of("gguf"), None);
}

/// A GGUF import: two text shards, the vision tower's `mmproj` and the
/// `SafeTensors` file its import assembles.
#[test]
fn discovers_each_container_of_a_gguf_import() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-weight-container-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let names = [
        "config.json",
        "mmproj-F16.gguf",
        "mmproj-patch-embedding.safetensors",
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ];
    for name in names {
        fs::write(root.join(name), [])?;
    }
    let gguf = WeightContainer::Gguf.discover(&root);
    let safetensors = WeightContainer::SafeTensors.discover(&root);
    let detected = WeightContainer::detect(&root);
    fs::remove_dir_all(&root)?;
    let names = |paths: Vec<std::path::PathBuf>| {
        paths
            .iter()
            .filter_map(|path| path.file_name()?.to_str().map(str::to_owned))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        names(gguf?),
        ["mmproj-F16.gguf", "model-00001-of-00002.gguf", "model-00002-of-00002.gguf"]
    );
    assert_eq!(names(safetensors?), ["mmproj-patch-embedding.safetensors"]);
    assert_eq!(detected?, Some(WeightContainer::Gguf));
    Ok(())
}
