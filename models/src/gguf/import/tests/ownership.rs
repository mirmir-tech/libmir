//! A later import of the same directory replaces or removes only what an
//! import made: never a source it found in place, nor a user's file.

use std::fs;

use super::{FileOrigin, ImportConflict, ImportManifest, conflict, import, root};
use crate::gguf::{files_in, tests::TestResult};

/// Where links are hard links, the previous import's link is a plain file;
/// its manifest names it as a link, so a refresh replaces it.
#[test]
fn a_previous_hard_link_is_replaced() -> TestResult {
    let root = root("hard")?;
    let (source, output) = (root.join("source/model.gguf"), root.join("output"));
    fs::create_dir_all(source.parent().unwrap_or(&root))?;
    fs::write(&source, b"source bytes")?;
    import(vec![source.clone()], 1).write(&output)?;
    fs::remove_file(output.join("model.gguf"))?;
    fs::hard_link(&source, output.join("model.gguf"))?;
    import(vec![source], 2).write(&output)?;
    assert_eq!(fs::read(output.join("model.gguf"))?, b"source bytes");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A source the first import found in place is not the import's: a second
/// import of another model under its name is refused, and the original
/// stays a regular file with its bytes.
#[test]
fn a_source_found_in_place_is_never_replaced() -> TestResult {
    let root = root("in-place")?;
    let (first, second) = (root.join("a/model.gguf"), root.join("b/model.gguf"));
    for (path, bytes) in [(&first, b"first bytes"), (&second, b"other bytes")] {
        fs::create_dir_all(path.parent().unwrap_or(&root))?;
        fs::write(path, bytes)?;
    }
    let directory = root.join("a");
    import(vec![first.clone()], 1).write(&directory)?;
    let manifest = ImportManifest::read(&directory).ok_or("no manifest")?;
    let origin = manifest.files.iter().find(|file| file.name == "model.gguf");
    assert_eq!(origin.map(|file| file.origin.clone()), Some(FileOrigin::InPlace));

    let refused = conflict(&import(vec![second], 2).write(&directory));
    assert_eq!(refused, Some(ImportConflict::OccupiedLink));
    assert!(!first.symlink_metadata()?.file_type().is_symlink());
    assert_eq!(fs::read(&first)?, b"first bytes");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// Switching an import from model A to model B removes what the import
/// made for A, so a loader finds B alone; a user's file stays.
#[test]
fn switching_models_leaves_only_the_new_one() -> TestResult {
    let root = root("switch")?;
    let (first, second) = (root.join("a/first.gguf"), root.join("b/second.gguf"));
    for path in [&first, &second] {
        fs::create_dir_all(path.parent().unwrap_or(&root))?;
        fs::write(path, b"weights")?;
    }
    let output = root.join("output");
    let mut vision = import(vec![first], 1);
    vision.preprocessor_config = Some(serde_json::json!({}));
    vision.write(&output)?;
    fs::write(output.join("notes.txt"), b"mine")?;
    fs::write(output.join("extra.gguf"), b"mine too")?;

    import(vec![second], 2).write(&output)?;
    assert!(output.join("first.gguf").symlink_metadata().is_err());
    assert!(!output.join("preprocessor_config.json").exists());
    assert_eq!(files_in(&output)?, [output.join("second.gguf")]);
    assert_eq!(fs::read(output.join("notes.txt"))?, b"mine");
    assert_eq!(fs::read(output.join("extra.gguf"))?, b"mine too");
    let manifest = ImportManifest::read(&output).ok_or("no manifest")?;
    assert!(manifest.is_current(&output));
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A file put where the import linked a source is the user's, even with the
/// source's size and time: the import is no longer current, and switching
/// models keeps it and its bytes.
#[test]
fn a_replaced_link_is_kept_through_a_switch() -> TestResult {
    let root = root("replaced")?;
    let (first, second) = (root.join("a/first.gguf"), root.join("b/second.gguf"));
    for path in [&first, &second] {
        fs::create_dir_all(path.parent().unwrap_or(&root))?;
        fs::write(path, b"weights A")?;
    }
    let output = root.join("output");
    import(vec![first.clone()], 1).write(&output)?;
    let replacement = output.join("first.gguf");
    fs::remove_file(&replacement)?;
    fs::write(&replacement, b"weights B")?;
    let modified = fs::metadata(&first)?.modified()?;
    fs::File::options().write(true).open(&replacement)?.set_modified(modified)?;
    let manifest = ImportManifest::read(&output).ok_or("no manifest")?;
    assert!(!manifest.is_current(&output));

    import(vec![second], 2).write(&output)?;
    assert_eq!(fs::read(&replacement)?, b"weights B");
    assert_eq!(files_in(&output)?, [output.join("second.gguf")]);
    fs::remove_dir_all(root)?;
    Ok(())
}
