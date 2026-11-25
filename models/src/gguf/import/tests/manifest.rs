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
    names_only(&output)?;
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

/// A source an import found in place, recorded by an earlier version by
/// file name alone, is never taken for the import's link: switching the
/// import to another model keeps it, a regular file with its bytes.
#[test]
fn a_legacy_in_place_source_survives_a_switch() -> TestResult {
    let root = root("legacy-in-place")?;
    let (first, second) = (root.join("a/model.gguf"), root.join("b/other.gguf"));
    for (path, bytes) in [(&first, b"first bytes"), (&second, b"other bytes")] {
        fs::create_dir_all(path.parent().unwrap_or(&root))?;
        fs::write(path, bytes)?;
    }
    let directory = root.join("a");
    import(vec![first.clone()], 1).write(&directory)?;
    names_only(&directory)?;

    import(vec![second.clone()], 2).write(&directory)?;
    assert!(!first.symlink_metadata()?.file_type().is_symlink());
    assert_eq!(fs::read(&first)?, b"first bytes");
    assert_eq!(fs::canonicalize(directory.join("other.gguf"))?, fs::canonicalize(&second)?);
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A shard given as a link of another name, `model.gguf` to `next.gguf`, is
/// current once imported and stays so unchanged; the link turned to
/// another file makes it stale.
#[cfg(unix)]
#[test]
fn a_source_linked_under_another_name_is_current() -> TestResult {
    let root = root("renamed-source")?;
    let (source, output) = (root.join("source"), root.join("output"));
    fs::create_dir_all(&source)?;
    fs::write(source.join("next.gguf"), b"next")?;
    fs::write(source.join("later.gguf"), b"later")?;
    let link = source.join("model.gguf");
    std::os::unix::fs::symlink(source.join("next.gguf"), &link)?;
    import(vec![link.clone()], 1).write(&output)?;
    let written = fs::read(output.join(MANIFEST))?;
    let manifest = ImportManifest::read(&output).ok_or("no manifest")?;
    assert!(manifest.is_current(&output));
    // Unchanged, a second lookup finds the same manifest current.
    let again = ImportManifest::read(&output).ok_or("no manifest")?;
    assert!(again.is_current(&output));
    assert_eq!(fs::read(output.join(MANIFEST))?, written);

    fs::remove_file(&link)?;
    std::os::unix::fs::symlink(source.join("later.gguf"), &link)?;
    assert!(!manifest.is_current(&output));
    import(vec![link], 2).write(&output)?;
    let refreshed = ImportManifest::read(&output).ok_or("no manifest")?;
    assert!(refreshed.is_current(&output));
    assert_eq!(fs::read(output.join("model.gguf"))?, b"later");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// Rewrites `directory`'s manifest as an earlier version wrote it: its files
/// by name alone.
fn names_only(directory: &std::path::Path) -> TestResult {
    let mut manifest =
        serde_json::from_slice::<serde_json::Value>(&fs::read(directory.join(MANIFEST))?)?;
    let names = manifest["files"]
        .as_array()
        .ok_or("no files")?
        .iter()
        .map(|file| file["name"].clone())
        .collect::<Vec<_>>();
    manifest["files"] = names.into();
    fs::write(directory.join(MANIFEST), serde_json::to_vec(&manifest)?)?;
    Ok(())
}

/// A manifest an earlier version wrote with origins but no link's source
/// finds each link's source by where it links: it reads current, and a
/// switch removes the old link.
#[test]
fn a_manifest_without_link_sources_finds_them() -> TestResult {
    let root = root("unsourced")?;
    let (first, second) = (root.join("a/first.gguf"), root.join("b/second.gguf"));
    for path in [&first, &second] {
        fs::create_dir_all(path.parent().unwrap_or(&root))?;
        fs::write(path, b"weights")?;
    }
    let output = root.join("output");
    import(vec![first], 1).write(&output)?;
    let mut manifest =
        serde_json::from_slice::<serde_json::Value>(&fs::read(output.join(MANIFEST))?)?;
    for file in manifest["files"].as_array_mut().ok_or("no files")? {
        if file["origin"].get("linked").is_some() {
            file["origin"] = json!("linked");
        }
    }
    fs::write(output.join(MANIFEST), serde_json::to_vec(&manifest)?)?;
    assert!(ImportManifest::read(&output).ok_or("no manifest")?.is_current(&output));
    import(vec![second], 2).write(&output)?;
    assert!(output.join("first.gguf").symlink_metadata().is_err());
    assert_eq!(files_in(&output)?, [output.join("second.gguf")]);
    fs::remove_dir_all(root)?;
    Ok(())
}
