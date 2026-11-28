//! A previous import's manifest as a later import reads it: one naming a
//! file outside its directory is refused before anything changes.

use std::fs;

use serde_json::json;

use super::{ImportManifest, import, root};
use crate::{
    error::ModelsError,
    gguf::{MANIFEST, tests::TestResult},
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
            "format": 2,
            "text": [], "vision": [],
            "files": [{ "name": name, "origin": { "derived": { "written": { "bytes": 0, "digest": "" } } } }],
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
