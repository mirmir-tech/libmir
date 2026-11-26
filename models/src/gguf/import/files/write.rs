//! Writing a planned import: every new file aside, then the manifest
//! naming them beside its place with the records of what earlier writes
//! left, then the old files removed and the new ones moved over, and the
//! manifest last. Cut short anywhere, every file an import made is still
//! recorded, by the last manifest or the one aside.

use std::{ffi::OsStr, fs, path::Path};

use super::{MANIFEST, Plan, beside, link_file, partial, write_new};
use crate::{
    error::Result,
    gguf::import::{FileOrigin, ImportManifest, ImportedFile, Stamp, Written, identity},
};

/// Where a write may be cut short, for tests to stop it there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::gguf::import) enum Step {
    /// The manifest is aside, nothing else changed.
    Aside,
    /// The earlier import's files are removed.
    Removed,
    /// The new files are in place, the manifest still aside.
    Moved,
}

#[cfg(test)]
thread_local! {
    /// The step this thread's next write stops after, as if cut short.
    pub(in crate::gguf::import) static CUT_SHORT: std::cell::Cell<Option<Step>> =
        const { std::cell::Cell::new(None) };
}

impl Plan {
    pub(in crate::gguf::import) fn write(&self, directory: &Path) -> Result<()> {
        let mut renames = Vec::new();
        let mut files = Vec::new();
        for (name, contents) in &self.derived {
            let partial = partial(&directory.join(name))?;
            write_new(&partial, contents)?;
            let written = Some(Written::of(contents)?);
            renames.push((partial, directory.join(name)));
            files.push(ImportedFile {
                name: name.clone(),
                origin: FileOrigin::Derived { written },
            });
        }
        // Links before derived files, so `config.json` moves in last.
        let mut links = Vec::new();
        for link in &self.links {
            let origin = if link.present {
                FileOrigin::InPlace { source: Some(link.target.clone()) }
            } else {
                let partial = partial(&directory.join(&link.name))?;
                link_file(&link.target, &partial)?;
                links.push((partial, directory.join(&link.name)));
                FileOrigin::Linked {
                    source: link.target.clone(),
                    named: link.named.clone(),
                }
            };
            files.push(ImportedFile { name: link.name.clone(), origin });
        }
        links.append(&mut renames);
        let mut manifest = ImportManifest {
            text: self.text.clone(),
            vision: self.vision.clone(),
            files,
        };
        manifest.write_interrupted(directory, &self.records.standing(directory))?;
        reached(Step::Aside)?;
        // A model switched for another leaves nothing of the old one that a
        // loader would find; a file no import made is never touched.
        for name in &self.stale {
            // Another spelling of a name the import holds, on a file system
            // that tells them apart, goes. Where it is that file, on a
            // case-insensitive one, it goes too when this write makes the
            // file anew, as a rename over it would keep the old spelling;
            // a file the import finds in place stays.
            let path = directory.join(name);
            let held = self.held().any(|held| same_entry(&path, &directory.join(held), held, name));
            let remade = links.iter().any(|(_, target)| {
                target
                    .file_name()
                    .and_then(OsStr::to_str)
                    .is_some_and(|remade| same_entry(&path, target, remade, name))
            });
            if !held || remade {
                self.records.remove_made(OsStr::new(name), &path)?;
            }
            // A write cut short left it aside, by the name only writes use;
            // where that name is another spelling of one this write just
            // made aside, on a case-insensitive file system, it is that one.
            let partial = beside(&path, "partial")?;
            if partial.symlink_metadata().is_ok_and(|metadata| !metadata.is_dir())
                && !links.iter().any(|(made, _)| identity::same_entry(&partial, made))
            {
                fs::remove_file(partial)?;
            }
        }
        reached(Step::Removed)?;
        for (partial, path) in links {
            fs::rename(partial, path)?;
        }
        reached(Step::Moved)?;
        // Their identity in place spares reading them to tell them current.
        for file in &mut manifest.files {
            if let FileOrigin::Derived { written: Some(written) } = &mut file.origin {
                written.stamp = Stamp::of(&directory.join(&file.name));
            }
        }
        let path = directory.join(MANIFEST);
        let next = beside(&path, "next")?;
        write_new(&next, &serde_json::to_vec_pretty(&manifest)?)?;
        fs::rename(&next, &path)?;
        fs::remove_file(super::super::manifest::interrupted(directory)?)?;
        Ok(())
    }
}

impl Plan {
    /// The names of the files the import holds.
    fn held(&self) -> impl Iterator<Item = &str> {
        self.derived
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(self.links.iter().map(|link| link.name.as_str()))
    }
}

/// Whether `path`, named `name`, is the entry at `held_path`, named `held`:
/// another spelling of it on a case-insensitive file system.
fn same_entry(path: &Path, held_path: &Path, held: &str, name: &str) -> bool {
    if held.to_lowercase() != name.to_lowercase() {
        return false;
    }
    identity::same_entry(path, held_path)
}

/// Stops the write after `step` where a test asked for it.
#[cfg(test)]
fn reached(step: Step) -> Result<()> {
    if CUT_SHORT.get() == Some(step) {
        return Err(std::io::Error::other(format!("cut short after {step:?}")).into());
    }
    Ok(())
}

#[cfg(not(test))]
#[expect(clippy::unnecessary_wraps, reason = "fails in tests, to cut a write short")]
const fn reached(_: Step) -> Result<()> {
    Ok(())
}
