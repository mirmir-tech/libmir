use crate::weights::TensorCatalog;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextTensorLayout {
    root: String,
}

impl TextTensorLayout {
    #[must_use]
    pub fn discover(catalog: &TensorCatalog) -> Option<Self> {
        ["model.", ""].into_iter().find_map(|root| {
            [
                format!("{root}embed_tokens.weight"),
                format!("{root}layers.0.self_attn.q_proj.weight"),
                format!("{root}layers.0.mlp.gate_proj.weight"),
                format!("{root}norm.weight"),
            ]
            .iter()
            .all(|name| catalog.contains(name))
            .then(|| Self { root: root.into() })
        })
    }

    #[must_use]
    pub fn name(&self, suffix: impl AsRef<str>) -> String {
        format!("{}{}", self.root, suffix.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::weights::TensorInfo;

    #[test]
    fn discovers_a_rootless_text_checkpoint_from_tensor_structure() {
        let catalog = TensorCatalog::new(
            [
                "embed_tokens.weight",
                "layers.0.self_attn.q_proj.weight",
                "layers.0.mlp.gate_proj.weight",
                "norm.weight",
            ]
            .into_iter()
            .map(|name| TensorInfo {
                name: name.into(),
                file: PathBuf::new(),
                dtype: "BF16".into(),
                shape: Vec::new(),
                data_start: 0,
                data_offsets: [0, 0],
            })
            .collect(),
        );

        assert_eq!(
            TextTensorLayout::discover(&catalog)
                .map(|layout| layout.name("layers.3.self_attn.k_proj.weight")),
            Some("layers.3.self_attn.k_proj.weight".into())
        );
    }
}
