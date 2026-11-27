use serde_json::json;

use super::{config, missing};
use crate::{
    error::Result,
    layout::{DecoderConfig, HybridLayer},
};

/// NVIDIA-Nemotron-Nano-9B-v2: dense `relu²` MLP blocks named by
/// `hybrid_override_pattern`, no routed experts.
#[test]
fn pairs_the_dense_override_pattern_with_its_mlp_blocks() -> Result<()> {
    let mut value = config("");
    let decoder = value.get_mut("llm_config").ok_or_else(|| missing("llm_config"))?;
    let object = decoder.as_object_mut().ok_or_else(|| missing("decoder object"))?;
    for key in ["layers_block_type", "n_routed_experts", "num_experts_per_tok", "moe_latent_size"] {
        object.remove(key);
    }
    object.insert("hybrid_override_pattern".into(), json!("M-M*-MM-"));
    let decoder = DecoderConfig::from_value(&value)?;
    let hybrid = decoder.state_space.as_ref().ok_or_else(|| missing("hybrid configuration"))?;

    assert!(hybrid.routed.is_none());
    assert_eq!(
        hybrid.blocks.layers(),
        [
            HybridLayer::MambaWithMlp { block: 0 },
            HybridLayer::Mamba { block: 2 },
            HybridLayer::AttentionWithMlp { block: 3 },
            HybridLayer::Mamba { block: 5 },
            HybridLayer::MambaWithMlp { block: 6 },
        ]
    );
    assert_eq!(decoder.num_hidden_layers, 5);
    Ok(())
}
