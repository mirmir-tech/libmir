//! Each file of an import is tied to its own source: a link or a file
//! found in place turned to another recorded source makes the import
//! stale, and only a link the import is shown to have made is its own,
//! wherever the import directory has moved since.

use std::{fs, os::unix::fs::symlink, path::Path};

use super::{FileOrigin, ImportManifest, import, root};
use crate::gguf::{files_in, tests::TestResult};

/// `one.gguf` and `two.gguf` under `root/src`.
fn two_sources(root: &Path) -> TestResult<(std::path::PathBuf, std::path::PathBuf)> {
    let source = root.join("src");
    fs::create_dir_all(&source)?;
    let (one, two) = (source.join("one.gguf"), source.join("two.gguf"));
    fs::write(&one, b"first shard")?;
    fs::write(&two, b"second shard")?;
    Ok((fs::canonicalize(one)?, fs::canonicalize(two)?))
}

/// A two-shard import of the user's links `a.gguf` and `b.gguf`, found in
/// place, is current; `a.gguf` turned to `b.gguf`'s shard, both sources
/// intact, makes it stale, though it still resolves to a recorded source.
#[test]
fn an_in_place_link_turned_to_another_shard_is_stale() -> TestResult {
    let root = root("aliases-in-place")?;
    let (one, two) = two_sources(&root)?;
    let output = root.join("output");
    fs::create_dir_all(&output)?;
    let (a, b) = (output.join("a.gguf"), output.join("b.gguf"));
    symlink(&one, &a)?;
    symlink(&two, &b)?;
    import(vec![a.clone(), b], 1).write(&output)?;
    let manifest = ImportManifest::read(&output).ok_or("no manifest")?;
    let origin = manifest.files.iter().find(|file| file.name == "a.gguf");
    let source = one.clone();
    assert_eq!(origin.map(|file| file.origin.clone()), Some(FileOrigin::InPlace { source }));
    assert!(manifest.is_current(&output));

    fs::remove_file(&a)?;
    symlink(&two, &a)?;
    assert!(!ImportManifest::read(&output).ok_or("no manifest")?.is_current(&output));
    assert_eq!(fs::read(&one)?, b"first shard");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// The same with the shards given as links elsewhere, which the import
/// links: one turned to the other shard makes the import stale.
#[test]
fn a_named_link_turned_to_another_shard_is_stale() -> TestResult {
    let root = root("aliases-linked")?;
    let (one, two) = two_sources(&root)?;
    let given = root.join("given");
    fs::create_dir_all(&given)?;
    let (a, b) = (given.join("a.gguf"), given.join("b.gguf"));
    symlink(&one, &a)?;
    symlink(&two, &b)?;
    let output = root.join("output");
    import(vec![a.clone(), b], 1).write(&output)?;
    assert!(ImportManifest::read(&output).ok_or("no manifest")?.is_current(&output));
    fs::remove_file(&a)?;
    symlink(&two, &a)?;
    assert!(!ImportManifest::read(&output).ok_or("no manifest")?.is_current(&output));
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A user's link in the output directory under a shard's name, to that
/// shard, though the shard is given elsewhere, is the user's: found in
/// place, and kept through a switch.
#[test]
fn a_users_link_to_the_shard_under_its_name_is_kept() -> TestResult {
    let root = root("users-link")?;
    let (one, two) = two_sources(&root)?;
    let output = root.join("output");
    fs::create_dir_all(&output)?;
    let mine = output.join("one.gguf");
    symlink(&one, &mine)?;
    import(vec![one.clone()], 1).write(&output)?;
    import(vec![two], 2).write(&output)?;
    assert_eq!(fs::read_link(&mine)?, one);
    assert_eq!(files_in(&output)?, [output.join("two.gguf")]);
    fs::remove_dir_all(root)?;
    Ok(())
}
