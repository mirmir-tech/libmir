use std::{fs, path::PathBuf};

use serde_json::json;

mod manifest;
mod ownership;

use super::{FileOrigin, Import, ImportConflict, ImportManifest, files, publish};
use crate::{error::ModelsError, gguf::tests::TestResult};

fn root(label: &str) -> std::io::Result<PathBuf> {
    let root = std::env::temp_dir().join(format!("libmir-import-{label}-{}", std::process::id()));
    drop(fs::remove_dir_all(&root));
    fs::create_dir_all(&root)?;
    Ok(root)
}

fn import(shards: Vec<PathBuf>, version: u32) -> Import {
    Import {
        config: json!({ "version": version }),
        generation_config: json!({}),
        tokenizer: json!({}),
        tokenizer_config: json!({}),
        shards,
        vision_shards: Vec::new(),
        preprocessor_config: None,
        assembled: None,
    }
}

fn conflict(result: &crate::error::Result<()>) -> Option<ImportConflict> {
    match result {
        Err(ModelsError::ImportConflict { conflict, .. }) => Some(*conflict),
        _ => None,
    }
}

/// Importing into the source's own directory keeps the source file, before
/// and after a refresh.
#[test]
fn an_import_into_its_source_directory_keeps_the_source() -> TestResult {
    let root = root("own")?;
    let source = root.join("model.gguf");
    fs::write(&source, b"source bytes")?;
    import(vec![source.clone()], 1).write(&root)?;
    import(vec![source.clone()], 2).write(&root)?;
    assert_eq!(fs::read(&source)?, b"source bytes");
    assert!(!source.symlink_metadata()?.file_type().is_symlink());
    assert_eq!(fs::read_to_string(root.join("config.json"))?, "{\n  \"version\": 2\n}");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A different file under a shard's name is kept, and nothing is written; a
/// previous import's link there is replaced.
#[test]
fn a_different_file_under_a_shard_name_is_kept() -> TestResult {
    let root = root("occupied")?;
    let (source, output) = (root.join("source/model.gguf"), root.join("output"));
    fs::create_dir_all(source.parent().unwrap_or(&root))?;
    fs::create_dir_all(&output)?;
    fs::write(&source, b"source bytes")?;
    fs::write(output.join("model.gguf"), b"other bytes")?;
    let refused = conflict(&import(vec![source.clone()], 1).write(&output));
    assert_eq!(refused, Some(ImportConflict::OccupiedLink));
    assert_eq!(fs::read(output.join("model.gguf"))?, b"other bytes");
    assert!(!output.join("config.json").exists());
    fs::remove_file(output.join("model.gguf"))?;
    import(vec![source.clone()], 1).write(&output)?;
    import(vec![source.clone()], 2).write(&output)?;
    assert_eq!(fs::read(output.join("model.gguf"))?, b"source bytes");
    // Two files of one name are refused before anything is written.
    let mut doubled = import(vec![source], 3);
    doubled.assembled = Some(("Config.json".to_owned(), Vec::new()));
    let empty = root.join("empty");
    assert_eq!(conflict(&doubled.write(&empty)), Some(ImportConflict::DuplicateName));
    assert!(!empty.exists());
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A failed publish leaves the last good import, without a staging
/// directory; a retry replaces it whole. A source inside the published
/// directory is refused.
#[test]
fn a_failed_publish_keeps_the_last_good_import() -> TestResult {
    let root = root("publish")?;
    let (source, output) = (root.join("model.gguf"), root.join("imports/model"));
    fs::write(&source, b"source bytes")?;
    import(vec![source.clone()], 1).publish(&output)?;
    fs::write(output.join("leftover"), b"")?;
    let failed = publish::replace(&output, std::slice::from_ref(&source), |staging| {
        fs::create_dir_all(staging)?;
        fs::write(staging.join("config.json"), b"{}")?;
        Err(ModelsError::InvalidGguf("interrupted".into()))
    });
    assert!(failed.is_err());
    assert_eq!(fs::read_to_string(output.join("config.json"))?, "{\n  \"version\": 1\n}");
    assert!(!files::beside(&output, "staging")?.exists());
    import(vec![source.clone()], 2).publish(&output)?;
    assert_eq!(fs::read_to_string(output.join("config.json"))?, "{\n  \"version\": 2\n}");
    assert_eq!(fs::read(output.join("model.gguf"))?, b"source bytes");
    assert!(!output.join("leftover").exists());
    assert!(!files::beside(&output, "previous")?.exists());
    let inside = conflict(&import(vec![source.clone()], 3).publish(&root));
    assert_eq!(inside, Some(ImportConflict::SourceInside));
    assert_eq!(fs::read(&source)?, b"source bytes");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// The manifest lists every file and the sources as they were; losing a
/// file or touching a source makes the import stale.
#[test]
fn the_manifest_records_files_and_sources() -> TestResult {
    let root = root("manifest")?;
    let (source, output) = (root.join("model.gguf"), root.join("imports/model"));
    fs::write(&source, b"source bytes")?;
    import(vec![source.clone()], 1).publish(&output)?;
    let manifest = ImportManifest::read(&output).ok_or("no manifest")?;
    assert_eq!(manifest.text.len(), 1);
    assert!(manifest.vision.is_empty());
    let origin =
        |name: &str| manifest.files.iter().find(|file| file.name == name).map(|file| file.origin);
    assert_eq!(origin("tokenizer.json"), Some(FileOrigin::Derived));
    assert_eq!(origin("model.gguf"), Some(FileOrigin::Linked));
    assert!(manifest.is_current(&output));
    fs::remove_file(output.join("tokenizer.json"))?;
    assert!(!manifest.is_current(&output));
    import(vec![source.clone()], 1).publish(&output)?;
    fs::write(&source, b"changed source")?;
    assert!(!manifest.is_current(&output));
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A publish interrupted between its two renames leaves the last good import
/// aside; a retry that fails restores it rather than removing it.
#[test]
fn a_failed_retry_after_an_interrupted_swap_restores_the_import() -> TestResult {
    let root = root("interrupted")?;
    let (source, output) = (root.join("model.gguf"), root.join("imports/model"));
    fs::write(&source, b"source bytes")?;
    import(vec![source.clone()], 1).publish(&output)?;
    fs::rename(&output, files::beside(&output, "previous")?)?;
    let failed = publish::replace(&output, std::slice::from_ref(&source), |_| {
        Err(ModelsError::MissingFile(source.clone()))
    });
    assert!(failed.is_err());
    assert_eq!(fs::read_to_string(output.join("config.json"))?, "{\n  \"version\": 1\n}");
    assert!(!files::beside(&output, "previous")?.exists());
    fs::remove_dir_all(root)?;
    Ok(())
}

/// Publishers of one directory take turns: each leaves a complete import and
/// no staging or backup directory.
#[test]
fn publishers_of_one_directory_take_turns() -> TestResult {
    let root = root("concurrent")?;
    let (source, output) = (root.join("model.gguf"), root.join("imports/model"));
    fs::write(&source, b"source bytes")?;
    std::thread::scope(|scope| {
        let mut publishers = Vec::new();
        for version in 0..4 {
            let (source, output) = (&source, &output);
            publishers.push(scope.spawn(move || {
                (0..8).try_for_each(|_| import(vec![source.clone()], version).publish(output))
            }));
        }
        publishers.into_iter().try_for_each(|publisher| {
            publisher
                .join()
                .map_err(|_| ModelsError::InvalidGguf("publisher panicked".into()))?
        })
    })?;
    let manifest = ImportManifest::read(&output).ok_or("no manifest")?;
    assert!(manifest.is_current(&output));
    for role in ["staging", "previous"] {
        assert!(!files::beside(&output, role)?.exists(), "{role} left behind");
    }
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A source inside the staging or backup directory, which publishing
/// removes, is refused like one inside the published directory.
#[test]
fn a_source_inside_staging_is_refused() -> TestResult {
    let root = root("staged")?;
    let output = root.join("imports/model");
    let staging = files::beside(&output, "staging")?;
    fs::create_dir_all(&staging)?;
    let source = staging.join("model.gguf");
    fs::write(&source, b"source bytes")?;
    let refused = conflict(&import(vec![source.clone()], 1).publish(&output));
    assert_eq!(refused, Some(ImportConflict::SourceInside));
    assert_eq!(fs::read(&source)?, b"source bytes");
    fs::remove_dir_all(root)?;
    Ok(())
}
