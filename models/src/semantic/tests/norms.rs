//! Block norm placement and Q/K norm scope from checkpoint tensors.

use std::path::PathBuf;

use super::configuration;
use crate::{
    error::Result,
    layout::DecoderConfig,
    semantic::{MixerSpec, NormPlacement, QkNormalization, SemanticModelSpec},
    weights::{TensorCatalog, TensorInfo},
};

fn tensors(entries: &[(&str, usize)]) -> TensorCatalog {
    TensorCatalog::new(
        entries
            .iter()
            .map(|(name, width)| TensorInfo {
                name: format!("model.layers.0.{name}"),
                file: PathBuf::new(),
                container: crate::layout::WeightContainer::SafeTensors,
                dtype: crate::weights::DenseDType::F32.into(),
                shape: vec![*width],
                data_start: 0,
                data_offsets: [0, 0],
            })
            .collect(),
    )
}

#[test]
fn output_norms_without_input_norms_make_a_post_norm_block() -> Result<()> {
    let decoder = DecoderConfig::from_value(&configuration("any", &[]))?;
    // Four heads of eight: whole-projection Q/K norms, output norms only.
    let post = tensors(&[
        ("self_attn.q_norm.weight", 32),
        ("self_attn.k_norm.weight", 16),
        ("post_attention_layernorm.weight", 32),
        ("post_feedforward_layernorm.weight", 32),
    ]);
    let spec = SemanticModelSpec::discover(&decoder, &post)?;
    let layer = &spec.decoder.layers[0];
    assert_eq!(layer.norm_placement, NormPlacement::Post);
    let MixerSpec::SoftmaxAttention(attention) = &layer.mixer else {
        return Err(crate::error::ModelsError::InvalidConfig("expected attention".into()));
    };
    assert_eq!(attention.qk_normalization, QkNormalization::QueryKeyRmsAcrossHeads);
    // Gemma's sandwich keeps its input norm and per-head Q/K norms.
    let sandwich = tensors(&[
        ("input_layernorm.weight", 32),
        ("self_attn.q_norm.weight", 8),
        ("self_attn.k_norm.weight", 8),
        ("post_attention_layernorm.weight", 32),
        ("post_feedforward_layernorm.weight", 32),
    ]);
    let spec = SemanticModelSpec::discover(&decoder, &sandwich)?;
    assert_eq!(spec.decoder.layers[0].norm_placement, NormPlacement::Pre);
    Ok(())
}
