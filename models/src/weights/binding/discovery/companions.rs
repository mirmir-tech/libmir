use crate::weights::TensorCatalog;

pub(super) fn companion(catalog: &TensorCatalog, candidates: [String; 2]) -> Option<String> {
    candidates.into_iter().find(|name| catalog.contains(name))
}

#[allow(clippy::case_sensitive_file_extension_comparisons)]
pub(super) fn is_companion(name: &str, catalog: &TensorCatalog) -> bool {
    name.ends_with(".scales")
        || name.ends_with(".absmax")
        || name.ends_with(".quant_map")
        || name.ends_with(".nested_absmax")
        || name.ends_with(".nested_quant_map")
        || name.contains(".quant_state.bitsandbytes__")
        || name.ends_with(".biases")
        || name.ends_with("_scales")
        || name.ends_with(".weight_scale")
        || name.ends_with(".weight_scale_inv")
        || name.ends_with(".weight_scale_2")
        || name.ends_with(".input_scale")
        || name.ends_with(".weight_shape")
        || name.ends_with(".weight_zero_point")
        || name.ends_with(".weight_g_idx")
        || name.ends_with(".qzeros")
        || name.ends_with(".g_idx")
        || name.ends_with(".bias")
        || name.strip_suffix("_bias").is_some_and(|prefix| {
            catalog.contains(&format!("{prefix}_blocks")) || catalog.contains(prefix)
        })
}
