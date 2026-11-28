//! The manifest as it is stored: with its format, so a manifest serialized
//! by this version reads back as one, and a later format is rejected.

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

use super::{ImportManifest, ImportedFile, SourceFile};

/// The manifest format this version writes: every file found in place
/// names the source it resolved to, every derived file how it was written,
/// and a file recorded as linked is one the import made itself.
pub(super) const FORMAT: u32 = 2;

#[derive(Serialize)]
struct Writing<'a> {
    format: u32,
    text: &'a [SourceFile],
    vision: &'a [SourceFile],
    files: &'a [ImportedFile],
}

#[derive(Deserialize)]
struct Stored {
    format: u32,
    text: Vec<SourceFile>,
    vision: Vec<SourceFile>,
    files: Vec<ImportedFile>,
}

impl Serialize for ImportManifest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Writing {
            format: FORMAT,
            text: &self.text,
            vision: &self.vision,
            files: &self.files,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ImportManifest {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Stored { format, text, vision, files } = Stored::deserialize(deserializer)?;
        if format != FORMAT {
            return Err(D::Error::custom(format!("import manifest format {format}, not {FORMAT}")));
        }
        Ok(Self { text, vision, files })
    }
}
