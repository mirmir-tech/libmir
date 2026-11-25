use std::{fs, path::PathBuf};

use serde_json::json;

use super::{Import, ImportConflict, publish};
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
    assert!(!publish::beside(&output, "staging")?.exists());
    import(vec![source.clone()], 2).publish(&output)?;
    assert_eq!(fs::read_to_string(output.join("config.json"))?, "{\n  \"version\": 2\n}");
    assert_eq!(fs::read(output.join("model.gguf"))?, b"source bytes");
    assert!(!output.join("leftover").exists());
    assert!(!publish::beside(&output, "previous")?.exists());
    let inside = conflict(&import(vec![source.clone()], 3).publish(&root));
    assert_eq!(inside, Some(ImportConflict::SourceInside));
    assert_eq!(fs::read(&source)?, b"source bytes");
    fs::remove_dir_all(root)?;
    Ok(())
}
