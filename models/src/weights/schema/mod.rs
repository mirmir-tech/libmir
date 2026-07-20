use crate::{
    layout::{DecoderConfig, EncoderConfig},
    weights::TensorCatalog,
};

mod encoder;
mod layout;
mod text;
mod types;
mod vision;

pub use text::TextTensorLayout;
pub use types::{
    DecoderTensorSchema, EncoderTensorSchema, TensorReadiness, TensorRequirement,
    VisionTensorSchema,
};

impl DecoderTensorSchema {
    #[must_use]
    pub fn discover(config: &DecoderConfig, catalog: &TensorCatalog) -> Self {
        if super::hybrid_linear::uses_layout(config, catalog) {
            super::hybrid_linear::schema(config)
        } else {
            layout::discover(config, catalog)
        }
    }

    #[must_use]
    pub fn readiness(&self, catalog: &TensorCatalog) -> TensorReadiness {
        let missing: Vec<String> = self
            .requirements
            .iter()
            .filter(|requirement| !requirement.is_present(catalog))
            .map(TensorRequirement::missing_label)
            .collect();
        TensorReadiness {
            required: self.requirements.len(),
            present: self.requirements.len() - missing.len(),
            missing,
        }
    }
}

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
        Self { label: label.into(), aliases }
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

fn readiness(requirements: &[TensorRequirement], catalog: &TensorCatalog) -> TensorReadiness {
    let missing: Vec<String> = requirements
        .iter()
        .filter(|requirement| !requirement.is_present(catalog))
        .map(TensorRequirement::missing_label)
        .collect();
    TensorReadiness {
        required: requirements.len(),
        present: requirements.len() - missing.len(),
        missing,
    }
}
