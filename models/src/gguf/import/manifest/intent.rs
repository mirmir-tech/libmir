//! The manifest a write of an import puts beside `import.json` before it
//! changes any file, with the records it carries of files earlier writes
//! made and it has not yet removed or replaced: cut short, the write
//! leaves every file an import made recorded, and the import stale.

use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;

use super::{ImportManifest, MANIFEST};
use crate::error::Result;

impl ImportManifest {
    /// The manifest of a write of `directory` cut short before it moved its
    /// manifest into place, which tells the files that write made, then
    /// the records it carried of files earlier writes made and it had not
    /// removed yet.
    pub(in crate::gguf::import) fn load_interrupted(directory: &Path) -> Result<Vec<Self>> {
        let path = interrupted(directory)?;
        let Some(manifest) = Self::load_at(&path)? else {
            return Ok(Vec::new());
        };
        let carried = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Carried>(&bytes).ok())
            .map(|carried| carried.carried)
            .unwrap_or_default();
        for earlier in &carried {
            earlier.check(&path)?;
        }
        Ok(std::iter::once(manifest).chain(carried).collect())
    }

    /// Writes `self` where a write of `directory` puts its manifest before
    /// changing any file, with the `carried` records of files earlier
    /// writes made that are still there.
    pub(in crate::gguf::import) fn write_interrupted(
        &self,
        directory: &Path,
        carried: &[&Self],
    ) -> Result<()> {
        let mut value = serde_json::to_value(self)?;
        if let Some(object) = value.as_object_mut() {
            object.insert("carried".to_owned(), serde_json::to_value(carried)?);
        }
        let path = interrupted(directory)?;
        let aside = super::super::files::beside(&path, "partial")?;
        super::super::files::write_new(&aside, &serde_json::to_vec_pretty(&value)?)?;
        fs::rename(&aside, &path)?;
        Ok(())
    }
}

/// The records an interrupted write carried.
#[derive(Deserialize)]
struct Carried {
    #[serde(default)]
    carried: Vec<ImportManifest>,
}

/// Where a write of `directory` puts its manifest before changing any file
/// it names, until it moves it into place.
pub(in crate::gguf::import) fn interrupted(directory: &Path) -> Result<PathBuf> {
    super::super::files::beside(&directory.join(MANIFEST), "partial")
}
