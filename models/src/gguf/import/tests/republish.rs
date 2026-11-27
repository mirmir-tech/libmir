//! Republishing an import: by exchanging it with the directory where the
//! system can, else by two renames, a symlink there replaced as the link
//! either way.

use std::{
    fs,
    path::{Path, PathBuf},
};

use super::{files, import, publish, root};
use crate::gguf::{import::publish::Exchange, tests::TestResult};

/// A test's directory, removed however the test ends.
pub(super) struct Scratch(PathBuf);

impl Scratch {
    pub(super) fn new(label: &str) -> std::io::Result<Self> {
        root(label).map(Self)
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        drop(fs::remove_dir_all(&self.0));
    }
}

/// `name` holding its own name, in its own folder of `root`.
pub(super) fn shard(root: &Path, name: &str) -> TestResult<PathBuf> {
    let folder = root.join(name.trim_end_matches(".gguf"));
    fs::create_dir_all(&folder)?;
    fs::write(folder.join(name), name)?;
    Ok(folder.join(name))
}

/// How this thread's exchanges went since last asked.
pub(super) fn exchanges() -> Vec<Exchange> {
    publish::EXCHANGES.take()
}

/// Runs `publish` as where the system refuses every exchange.
fn without_exchange<T>(publish: impl FnOnce() -> T) -> T {
    publish::NO_EXCHANGE.set(true);
    let published = publish();
    publish::NO_EXCHANGE.set(false);
    published
}

/// Where the system refuses the exchange, a republish moves the import
/// aside and the new one into its place, and leaves neither staging nor
/// backup directory.
#[test]
fn a_republish_without_an_exchange_replaces_the_import() -> TestResult {
    let root = Scratch::new("republish-renames")?;
    let (a, b) = (shard(&root, "a.gguf")?, shard(&root, "b.gguf")?);
    let output = root.join("imports/model");
    without_exchange(|| import(vec![a], 1).publish(&output))?;
    without_exchange(|| import(vec![b], 2).publish(&output))?;
    assert_eq!(exchanges(), [Exchange::Refused]);
    assert_eq!(fs::read_to_string(output.join("config.json"))?, "{\n  \"version\": 2\n}");
    assert_eq!(fs::read(output.join("b.gguf"))?, b"b.gguf");
    assert!(output.join("a.gguf").symlink_metadata().is_err());
    for role in ["staging", "previous"] {
        assert!(files::beside(&output, role)?.symlink_metadata().is_err(), "{role} left behind");
    }
    Ok(())
}

/// A staging directory gone before the exchange fails the publish and
/// leaves the import as it was, by an exchange or two renames alike.
#[test]
fn a_publish_whose_staging_is_gone_keeps_the_import() -> TestResult {
    let root = Scratch::new("republish-gone")?;
    let output = root.join("imports/model");
    import(vec![shard(&root, "a.gguf")?], 1).publish(&output)?;
    let unfilled = || publish::replace(&output, &[], |_| Ok(()));
    assert!(unfilled().is_err() && without_exchange(unfilled).is_err());
    #[cfg(any(target_os = "android", target_os = "linux", target_vendor = "apple"))]
    assert_eq!(exchanges(), [Exchange::Vanished, Exchange::Refused]);
    assert_eq!(fs::read_to_string(output.join("config.json"))?, "{\n  \"version\": 1\n}");
    assert!(files::beside(&output, "previous")?.symlink_metadata().is_err());
    Ok(())
}

/// Two renames that cannot move the import aside, here onto a backup
/// directory that is not empty, fail the publish, keep the import and
/// remove the staging directory.
#[test]
fn a_publish_that_cannot_move_the_import_aside_removes_its_staging() -> TestResult {
    let root = Scratch::new("republish-aside")?;
    let output = root.join("imports/model");
    import(vec![shard(&root, "a.gguf")?], 1).publish(&output)?;
    let (staging, previous) =
        (files::beside(&output, "staging")?, files::beside(&output, "previous")?);
    let failed = without_exchange(|| {
        publish::replace(&output, &[], |staging| {
            fs::create_dir_all(staging)?;
            fs::create_dir_all(previous.join("occupied"))?;
            Ok(())
        })
    });
    assert!(failed.is_err());
    assert_eq!(exchanges(), [Exchange::Refused]);
    assert_eq!(fs::read_to_string(output.join("config.json"))?, "{\n  \"version\": 1\n}");
    assert!(staging.symlink_metadata().is_err(), "staging left behind");
    Ok(())
}

/// A published directory that is a symlink is replaced as the link, by an
/// exchange or two renames alike: the link is removed, and the directory it
/// linked to keeps every file it held.
#[cfg(unix)]
#[test]
fn a_linked_import_is_replaced_as_the_link_either_way() -> TestResult {
    for exchanged in [true, false] {
        let root = Scratch::new(if exchanged {
            "linked-exchange"
        } else {
            "linked-renames"
        })?;
        let source = shard(&root, "a.gguf")?;
        let (real, linked) = (root.join("disk/model"), root.join("imports/model"));
        fs::create_dir_all(&real)?;
        fs::write(real.join("notes.txt"), b"mine")?;
        fs::create_dir_all(root.join("imports"))?;
        std::os::unix::fs::symlink(&real, &linked)?;
        let publish = || import(vec![source.clone()], 1).publish(&linked);
        if exchanged {
            publish()
        } else {
            without_exchange(publish)
        }?;
        #[cfg(any(target_os = "android", target_os = "linux", target_vendor = "apple"))]
        assert_eq!(
            exchanges(),
            [if exchanged {
                Exchange::Done
            } else {
                Exchange::Refused
            }]
        );
        assert!(linked.symlink_metadata()?.is_dir());
        assert_eq!(fs::read(linked.join("a.gguf"))?, b"a.gguf");
        let kept = fs::read_dir(&real)?
            .map(|entry| Ok(entry?.file_name()))
            .collect::<TestResult<Vec<_>>>()?;
        assert_eq!(kept, ["notes.txt"]);
        let imports = fs::read_dir(root.join("imports"))?
            .map(|entry| Ok(entry?.file_name()))
            .collect::<TestResult<std::collections::BTreeSet<_>>>()?;
        assert_eq!(imports, [".model.lock", "model"].map(Into::into).into());
    }
    Ok(())
}

/// The file system the tests run on exchanges a directory and a link in
/// one step, links not followed, so the tests above publish by exchanging;
/// an entry that is gone is told apart from a refusal.
#[cfg(any(target_os = "android", target_os = "linux", target_vendor = "apple"))]
#[test]
fn the_file_system_exchanges_a_directory_and_a_link() -> TestResult {
    let root = Scratch::new("exchange")?;
    let (directory, target, link) =
        (root.join("directory"), root.join("target"), root.join("link"));
    fs::create_dir_all(&directory)?;
    fs::create_dir_all(&target)?;
    fs::write(directory.join("config.json"), b"{}")?;
    std::os::unix::fs::symlink(&target, &link)?;
    let exchanged = publish::exchange(&directory, &link);
    assert_eq!(exchanged, Exchange::Done, "the temporary directory's file system refused it");
    assert!(link.symlink_metadata()?.is_dir() && link.join("config.json").is_file());
    assert_eq!(fs::read_link(&directory)?, target);
    assert!(fs::read_dir(&target)?.next().is_none());
    assert_eq!(publish::exchange(&link, &root.join("gone")), Exchange::Vanished);
    Ok(())
}
