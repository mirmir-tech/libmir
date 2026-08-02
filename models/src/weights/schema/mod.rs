use crate::{layout::EncoderConfig, weights::TensorCatalog};

mod encoder;
mod text;
mod types;
mod vision;

pub use text::TextTensorLayout;
pub use types::{EncoderTensorSchema, TensorReadiness, TensorRequirement, VisionTensorSchema};

impl EncoderTensorSchema {
    #[must_use]
    pub fn discover(config: &EncoderConfig, catalog: &TensorCatalog) -> Self {
        encoder::discover(config, catalog)
    }

    #[must_use]
    pub fn readiness(&self, catalog: &TensorCatalog) -> TensorReadiness {
        readiness(&self.requirements, catalog)
    }
}

impl TensorRequirement {
    #[must_use]
    pub fn any(label: impl Into<String>, aliases: Vec<String>) -> Self {
        Self {
            label: label.into(),
            aliases,
            include_dense_dtype: true,
        }
    }

    #[must_use]
    pub fn bound(label: impl Into<String>, aliases: Vec<String>) -> Self {
        Self {
            label: label.into(),
            aliases,
            include_dense_dtype: false,
        }
    }

    #[must_use]
    pub fn is_present(&self, catalog: &TensorCatalog) -> bool {
        self.aliases.iter().any(|alias| catalog.contains(alias))
    }

    #[must_use]
    pub fn missing_label(&self) -> String {
        format!("{} [{}]", self.label, self.aliases.join(" | "))
    }
}

impl TensorReadiness {
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.missing.is_empty()
    }

    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_ready() {
            format!("tensors {}/{} ready", self.present, self.required)
        } else {
            format!(
                "tensors {}/{} present, {} missing",
                self.present,
                self.required,
                self.missing.len()
            )
        }
    }
}

pub(super) fn readiness(
    requirements: &[TensorRequirement],
    catalog: &TensorCatalog,
) -> TensorReadiness {
    let missing: Vec<String> = requirements
        .iter()
        .filter(|requirement| !requirement.is_present(catalog))
        .map(TensorRequirement::missing_label)
        .collect();
    let mut dtypes = requirements
        .iter()
        .filter(|requirement| requirement.include_dense_dtype)
        .filter_map(|requirement| {
            requirement
                .aliases
                .iter()
                .find_map(|alias| catalog.get(alias))
                .map(|tensor| tensor.dtype.clone())
        })
        .collect::<Vec<_>>();
    dtypes.sort();
    dtypes.dedup();
    TensorReadiness {
        required: requirements.len(),
        present: requirements.len() - missing.len(),
        missing,
        dtypes,
    }
}
