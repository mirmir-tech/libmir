//! Files the import writes, derived and linked, are its own only while
//! they are still as it wrote them, and a write cut short before its
//! manifest moved into place still tells them.

use std::{fs, path::Path};

use super::{ImportConflict, ImportManifest, conflict, import, root};
use crate::gguf::{MANIFEST, files_in, tests::TestResult};

/// `model.gguf` with `bytes` in `root/<folder>`.
fn shard(root: &Path, folder: &str, bytes: &[u8]) -> TestResult<std::path::PathBuf> {
    let path = root.join(folder).join("model.gguf");
    fs::create_dir_all(root.join(folder))?;
    fs::write(&path, bytes)?;
    Ok(path)
}

/// A user's `tokenizer.json` where an import is written refuses it before
/// anything is written, and stays as it was.
#[test]
fn a_users_derived_file_name_is_refused() -> TestResult {
    let root = root("users-tokenizer")?;
    let source = shard(&root, "models", b"weights")?;
    let directory = root.join("models");
    fs::write(directory.join("tokenizer.json"), b"mine")?;
    let refused = conflict(&import(vec![source], 1).write(&directory));
    assert_eq!(refused, Some(ImportConflict::OccupiedFile));
    assert_eq!(fs::read(directory.join("tokenizer.json"))?, b"mine");
    assert!(!directory.join("config.json").exists() && !directory.join(MANIFEST).exists());
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A derived file the user replaced since is theirs, even of the same size
/// and time: a switch that no longer derives it keeps it, and one that
/// does is refused.
#[test]
fn a_replaced_derived_file_is_kept() -> TestResult {
    let root = root("replaced-derived")?;
    let (first, second) = (shard(&root, "a", b"first")?, shard(&root, "b", b"second")?);
    let output = root.join("output");
    let mut vision = import(vec![first], 1);
    vision.preprocessor_config = Some(serde_json::json!({}));
    vision.write(&output)?;
    let processor = output.join("preprocessor_config.json");
    let modified = fs::metadata(&processor)?.modified()?;
    fs::write(&processor, b"[]")?;
    fs::File::options().write(true).open(&processor)?.set_modified(modified)?;
    assert!(!ImportManifest::read(&output).ok_or("no manifest")?.is_current(&output));
    import(vec![second.clone()], 2).write(&output)?;
    assert_eq!(fs::read(&processor)?, b"[]");

    fs::write(output.join("config.json"), b"also mine")?;
    let refused = conflict(&import(vec![second], 3).write(&output));
    assert_eq!(refused, Some(ImportConflict::OccupiedFile));
    assert_eq!(fs::read(output.join("config.json"))?, b"also mine");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A write cut short after its link replaced the last import's, before its
/// manifest moved into place: the import reads stale, and the next import,
/// of another model under the same name, replaces that link.
#[test]
fn a_link_of_a_write_cut_short_is_the_imports() -> TestResult {
    let root = root("cut-short")?;
    let first = shard(&root, "a", b"first")?;
    let (second, third) = (shard(&root, "b", b"second")?, shard(&root, "c", b"third")?);
    let output = root.join("output");
    import(vec![first], 1).write(&output)?;
    let earlier = fs::read(output.join(MANIFEST))?;
    import(vec![second], 2).write(&output)?;
    // As the second write left it: its link made, its manifest still aside.
    fs::rename(output.join(MANIFEST), output.join(".import.json.partial"))?;
    fs::write(output.join(MANIFEST), earlier)?;
    assert!(!ImportManifest::read(&output).ok_or("no manifest")?.is_current(&output));

    import(vec![third], 3).write(&output)?;
    assert_eq!(fs::read(output.join("model.gguf"))?, b"third");
    assert_eq!(files_in(&output)?, [output.join("model.gguf")]);
    assert!(ImportManifest::read(&output).ok_or("no manifest")?.is_current(&output));
    assert!(!output.join(".import.json.partial").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}

/// With its manifest gone, the import's files cannot be told from a
/// user's: another model is refused, saying so for its derived files and
/// then its link, each until it is removed.
#[test]
fn files_without_their_manifest_are_refused_until_removed() -> TestResult {
    let root = root("manifest-gone")?;
    let (first, second) = (shard(&root, "a", b"first")?, shard(&root, "b", b"second")?);
    let output = root.join("output");
    import(vec![first], 1).write(&output)?;
    fs::remove_file(output.join(MANIFEST))?;
    let message = |result: crate::error::Result<()>| {
        let refused = conflict(&result);
        (refused, result.err().map(|error| error.to_string()).unwrap_or_default())
    };
    let (refused, said) = message(import(vec![second.clone()], 2).write(&output));
    assert_eq!(refused, Some(ImportConflict::OccupiedFile));
    assert!(
        said.contains("cannot tell it wrote") && said.contains("generation_config.json"),
        "{said}"
    );
    for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]
    {
        fs::remove_file(output.join(name))?;
    }
    let (refused, said) = message(import(vec![second.clone()], 2).write(&output));
    assert_eq!(refused, Some(ImportConflict::OccupiedLink));
    assert!(said.contains("cannot tell it did") && said.contains("model.gguf"), "{said}");
    fs::remove_file(output.join("model.gguf"))?;
    import(vec![second], 2).write(&output)?;
    assert_eq!(fs::read(output.join("model.gguf"))?, b"second");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A manifest serialized with its `Serialize` reads back as itself, a file
/// found in place, a link and derived files included.
#[test]
fn a_serialized_manifest_reads_back() -> TestResult {
    let root = root("round-trip")?;
    let (output, other) = (root.join("output"), root.join("other/second.gguf"));
    let source = shard(&root, "output", b"weights")?;
    fs::create_dir_all(root.join("other"))?;
    fs::write(&other, b"more")?;
    import(vec![source, other], 1).write(&output)?;
    let manifest = ImportManifest::read(&output).ok_or("no manifest")?;
    fs::write(output.join(MANIFEST), serde_json::to_vec(&manifest)?)?;
    assert_eq!(ImportManifest::read(&output), Some(manifest));
    fs::remove_dir_all(root)?;
    Ok(())
}

/// Publishing refuses a shard given as a user's link inside the directory
/// it replaces, though the link resolves elsewhere, and keeps the link.
#[cfg(unix)]
#[test]
fn a_shard_linked_inside_the_published_directory_is_refused() -> TestResult {
    let root = root("publish-inside")?;
    let source = shard(&root, "a", b"weights")?;
    let output = root.join("imports/model");
    fs::create_dir_all(&output)?;
    let link = output.join("model.gguf");
    std::os::unix::fs::symlink(&source, &link)?;
    let refused = conflict(&import(vec![link.clone()], 1).publish(&output));
    assert_eq!(refused, Some(ImportConflict::SourceInside));
    assert_eq!(fs::read_link(&link)?, source);
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A published directory that is a symlink is replaced as the link: the
/// import becomes a plain directory, and the directory it linked to keeps
/// every file it held.
#[cfg(unix)]
#[test]
fn a_linked_published_directory_is_never_changed() -> TestResult {
    let root = root("publish-linked")?;
    let source = shard(&root, "a", b"weights")?;
    let (real, linked) = (root.join("disk/model"), root.join("imports/model"));
    fs::create_dir_all(&real)?;
    fs::write(real.join("notes.txt"), b"mine")?;
    fs::create_dir_all(root.join("imports"))?;
    std::os::unix::fs::symlink(&real, &linked)?;
    import(vec![source], 1).publish(&linked)?;
    assert!(!linked.symlink_metadata()?.file_type().is_symlink());
    assert_eq!(fs::read(linked.join("model.gguf"))?, b"weights");
    let kept = fs::read_dir(&real)?
        .map(|entry| Ok(entry?.file_name()))
        .collect::<TestResult<Vec<_>>>()?;
    assert_eq!(kept, ["notes.txt"]);
    assert_eq!(fs::read(real.join("notes.txt"))?, b"mine");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A swap of a linked directory cut short leaves the link aside, and a
/// dangling link stands where its directory went: the next publish
/// recovers from either, never touching what a link pointed to.
#[cfg(unix)]
#[test]
fn a_publish_recovers_from_a_link_left_by_a_swap() -> TestResult {
    let root = root("publish-swap")?;
    let source = shard(&root, "a", b"weights")?;
    let (real, linked) = (root.join("disk/model"), root.join("imports/model"));
    fs::create_dir_all(&real)?;
    fs::write(real.join("notes.txt"), b"mine")?;
    fs::create_dir_all(root.join("imports"))?;
    let previous = root.join("imports/.model.previous");
    std::os::unix::fs::symlink(&real, &previous)?;
    import(vec![source.clone()], 1).publish(&linked)?;
    assert_eq!(fs::read(linked.join("model.gguf"))?, b"weights");
    assert!(previous.symlink_metadata().is_err());

    fs::remove_dir_all(&linked)?;
    std::os::unix::fs::symlink(root.join("disk/gone"), &linked)?;
    import(vec![source], 2).publish(&linked)?;
    assert_eq!(fs::read_to_string(linked.join("config.json"))?, "{\n  \"version\": 2\n}");
    assert_eq!(fs::read(real.join("notes.txt"))?, b"mine");
    fs::remove_dir_all(root)?;
    Ok(())
}
