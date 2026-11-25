//! A previous import's manifest as a later import reads it: one naming a
//! file outside its directory is refused before anything changes, and one
//! an earlier version wrote is upgraded without leaving its files behind.

use std::fs;

use serde_json::json;

use super::{ImportManifest, import, root};
use crate::{
    error::ModelsError,
    gguf::{MANIFEST, files_in, tests::TestResult},
    layout::ModelLayout,
};

/// A manifest naming a path outside its directory, or inside a directory
/// of it, is refused, and neither file is removed.
#[test]
fn a_manifest_naming_other_paths_is_refused() -> TestResult {
    let root = root("escape")?;
    let (source, output) = (root.join("source/model.gguf"), root.join("output"));
    fs::create_dir_all(source.parent().unwrap_or(&root))?;
    fs::create_dir_all(output.join("nested"))?;
    fs::write(&source, b"weights")?;
    fs::write(root.join("outside.txt"), b"keep")?;
    fs::write(output.join("nested/inner.txt"), b"keep too")?;
    for name in ["../outside.txt", "nested/inner.txt", "", "/abs"] {
        let manifest = json!({
            "text": [], "vision": [],
            "files": [{ "name": name, "origin": "derived" }],
        });
        fs::write(output.join(MANIFEST), serde_json::to_vec(&manifest)?)?;
        let refused = import(vec![source.clone()], 1).write(&output);
        assert!(
            matches!(refused, Err(ModelsError::InvalidImportEntry { name: ref refused, .. }) if refused == name),
            "{name:?}: {refused:?}"
        );
        assert!(ImportManifest::read(&output).is_none());
    }
    assert_eq!(fs::read(root.join("outside.txt"))?, b"keep");
    assert_eq!(fs::read(output.join("nested/inner.txt"))?, b"keep too");
    assert!(!output.join("config.json").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A vision import an earlier version recorded by file names alone becomes
/// a text-only import of another model: the loader then finds only the new
/// GGUF file and no processor configuration.
#[test]
fn a_legacy_vision_import_switches_to_text_only() -> TestResult {
    let root = root("legacy")?;
    let (first, mmproj) = (root.join("a/first.gguf"), root.join("a/mmproj.gguf"));
    let second = root.join("b/second.gguf");
    for path in [&first, &mmproj, &second] {
        fs::create_dir_all(path.parent().unwrap_or(&root))?;
        fs::write(path, b"weights")?;
    }
    let output = root.join("output");
    let mut vision = import(vec![first], 1);
    vision.vision_shards = vec![mmproj];
    vision.preprocessor_config = Some(json!({}));
    vision.assembled = Some(("vision.safetensors".to_owned(), b"tensors".to_vec()));
    vision.write(&output)?;
    // As an earlier version recorded it.
    let mut manifest =
        serde_json::from_slice::<serde_json::Value>(&fs::read(output.join(MANIFEST))?)?;
    let names = manifest["files"]
        .as_array()
        .ok_or("no files")?
        .iter()
        .map(|file| file["name"].clone())
        .collect::<Vec<_>>();
    manifest["files"] = names.into();
    fs::write(output.join(MANIFEST), serde_json::to_vec(&manifest)?)?;
    fs::write(output.join("notes.safetensors.txt"), b"mine")?;

    import(vec![second], 2).write(&output)?;
    let layout = ModelLayout::inspect(&output)?;
    let weights = layout.weights.iter().map(|weight| weight.path.clone()).collect::<Vec<_>>();
    assert_eq!(weights, [output.join("second.gguf")]);
    assert_eq!(files_in(&output)?, [output.join("second.gguf")]);
    assert_eq!(layout.preprocessor_config_path, None);
    assert!(output.join("mmproj.gguf").symlink_metadata().is_err());
    assert_eq!(fs::read(output.join("notes.safetensors.txt"))?, b"mine");
    fs::remove_dir_all(root)?;
    Ok(())
}
