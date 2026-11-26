//! The paths a write goes through: its own names beside the files, never
//! followed through a link or taken from a directory there; the spellings
//! of a shard's name; the identity that spares reading a derived file; and
//! writers of one directory taking turns.

use std::{fs, os::unix::fs::symlink, path::Path};

use super::{ImportConflict, ImportManifest, conflict, import, root};
use crate::gguf::{MANIFEST, files_in, tests::TestResult};

fn shard(root: &Path, folder: &str, name: &str) -> TestResult<std::path::PathBuf> {
    fs::create_dir_all(root.join(folder))?;
    fs::write(root.join(folder).join(name), name.as_bytes())?;
    Ok(root.join(folder).join(name))
}

fn current(directory: &Path) -> bool {
    ImportManifest::read(directory).is_some_and(|manifest| manifest.is_current(directory))
}

/// A derived file whose time changed, as on a file system reporting times
/// coarsely, still reads current by its contents; one whose contents
/// changed under the same size and time does not.
#[test]
fn a_derived_file_reads_current_by_its_contents() -> TestResult {
    let root = root("identity")?;
    let output = root.join("output");
    import(vec![shard(&root, "a", "model.gguf")?], 1).write(&output)?;
    let config = output.join("config.json");
    let modified = fs::metadata(&config)?.modified()?;
    let coarse = std::time::UNIX_EPOCH
        + std::time::Duration::from_secs(modified.duration_since(std::time::UNIX_EPOCH)?.as_secs());
    fs::File::options().write(true).open(&config)?.set_modified(coarse)?;
    assert!(current(&output));
    let contents = fs::read(&config)?;
    let changed = contents.iter().map(|byte| {
        if *byte == b'1' {
            b'7'
        } else {
            *byte
        }
    });
    fs::write(&config, changed.collect::<Vec<_>>())?;
    fs::File::options().write(true).open(&config)?.set_modified(coarse)?;
    assert!(!current(&output));
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A shard whose name changes only in case replaces the import's link
/// under the old spelling, which a case-sensitive file system keeps apart.
#[test]
fn a_shard_renamed_in_case_replaces_the_old_link() -> TestResult {
    let root = root("case")?;
    let output = root.join("output");
    import(vec![shard(&root, "a", "Model.gguf")?], 1).write(&output)?;
    import(vec![shard(&root, "b", "model.gguf")?], 1).write(&output)?;
    assert_eq!(files_in(&output)?, [output.join("model.gguf")]);
    let names = fs::read_dir(&output)?
        .map(|entry| Ok(entry?.file_name().to_string_lossy().into_owned()))
        .collect::<TestResult<Vec<_>>>()?;
    assert!(!names.iter().any(|name| name == "Model.gguf"), "{names:?}");
    assert!(current(&output));
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A link at a name the write puts its manifest under, aside or next, is
/// replaced, never written through.
#[test]
fn a_link_at_a_manifest_name_is_not_followed() -> TestResult {
    let root = root("manifest-links")?;
    let output = root.join("output");
    fs::create_dir_all(&output)?;
    for (name, victim) in
        [(".import.json.next", "next.txt"), ("..import.json.partial.partial", "aside.txt")]
    {
        fs::write(root.join(victim), b"keep")?;
        symlink(root.join(victim), output.join(name))?;
    }
    import(vec![shard(&root, "a", "model.gguf")?], 1).write(&output)?;
    assert_eq!(fs::read(root.join("next.txt"))?, b"keep");
    assert_eq!(fs::read(root.join("aside.txt"))?, b"keep");
    assert!(current(&output) && output.join(MANIFEST).is_file());
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A directory at the name a shard's partial link takes is no write's: the
/// write is refused and the directory stays with what it holds.
#[test]
fn a_directory_at_a_partial_name_is_kept() -> TestResult {
    let root = root("partial-directory")?;
    let output = root.join("output");
    fs::create_dir_all(output.join(".model.gguf.partial"))?;
    fs::write(output.join(".model.gguf.partial/inside.txt"), b"mine")?;
    let refused = conflict(&import(vec![shard(&root, "a", "model.gguf")?], 1).write(&output));
    assert_eq!(refused, Some(ImportConflict::OccupiedFile));
    assert_eq!(fs::read(output.join(".model.gguf.partial/inside.txt"))?, b"mine");
    fs::remove_dir_all(root)?;
    Ok(())
}

/// Writers of one directory take turns: each leaves a complete import.
#[test]
fn writers_of_one_directory_take_turns() -> TestResult {
    let root = root("writers")?;
    let output = root.join("output");
    let source = shard(&root, "a", "model.gguf")?;
    std::thread::scope(|scope| {
        #[expect(clippy::needless_collect, reason = "every writer starts before any is joined")]
        let writers = (0..4)
            .map(|version| {
                let (source, output) = (&source, &output);
                scope.spawn(move || {
                    (0..8).try_for_each(|_| import(vec![source.clone()], version).write(output))
                })
            })
            .collect::<Vec<_>>();
        writers.into_iter().try_for_each(|writer| {
            writer
                .join()
                .map_err(|_| crate::error::ModelsError::InvalidGguf("writer panicked".into()))?
        })
    })?;
    assert!(current(&output));
    fs::remove_dir_all(root)?;
    Ok(())
}
