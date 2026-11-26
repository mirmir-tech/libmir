use std::num::NonZeroUsize;

use models::{
    execution::TaskExecutionPlan,
    layout::DecoderConfig,
    semantic::{
        FeedForwardSpec, MixerSpec, PositionEncodingSpec, QueryTemperature, SemanticModelSpec,
    },
    weights::TensorCatalog,
};
use serde_json::json;

use super::fixtures::AnyResult;
use crate::{
    ArchitectureError, Unlowerable, cuda,
    lowering::{LatentGeometry, MixerLowering},
};

/// A miniature DeepSeek-V2-Lite decoder: latent attention, one dense layer,
/// then routed experts with shared experts, routed by `scoring`.
pub(super) fn latent_contract(scoring: &str) -> AnyResult<(TaskExecutionPlan, SemanticModelSpec)> {
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 32,
        "intermediate_size": 32,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 64,
        "kv_lora_rank": 64,
        "qk_nope_head_dim": 8,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 2,
        "first_k_dense_replace": 1,
        "scoring_func": scoring,
        "topk_method": if scoring == "sigmoid" { "noaux_tc" } else { "greedy" },
        "norm_topk_prob": scoring == "sigmoid",
        "hidden_act": "silu"
    }))?;
    let semantic = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: Vec::new() })?;
    Ok((TaskExecutionPlan::Generation { decoder }, semantic))
}

#[test]
fn lowers_latent_attention_with_its_geometry_on_cuda() -> AnyResult<()> {
    let (_, semantic) = latent_contract("softmax")?;
    let plan = cuda::CudaDecoderPlan::lower(&semantic, None)?;
    assert_eq!(
        plan.layers()[1].mixer,
        MixerLowering::Latent(LatentGeometry { key_value_rank: 64, rope_head_dim: 8 })
    );
    Ok(())
}

#[test]
fn refuses_a_latent_geometry_without_a_kernel() -> AnyResult<()> {
    let (task, mut semantic) = latent_contract("softmax")?;
    for layer in &mut semantic.decoder.layers {
        if let MixerSpec::LatentAttention(latent) = &mut layer.mixer {
            latent.key_value_rank = 96;
        }
    }
    assert_eq!(
        cuda::admit(&task, Some(&semantic)).err(),
        Some(ArchitectureError::Unlowerable {
            layer: Some(0),
            feature: Unlowerable::LatentGeometry { key_value_rank: 96, rope_head_dim: 8 },
        })
    );
    Ok(())
}

#[test]
fn refuses_more_experts_than_the_cuda_router_ranks() -> AnyResult<()> {
    let (task, mut semantic) = latent_contract("softmax")?;
    if let FeedForwardSpec::Routed { routed, .. } = &mut semantic.decoder.layers[1].feed_forward {
        routed.expert_count = 384;
    }
    assert_eq!(
        cuda::admit(&task, Some(&semantic)).err(),
        Some(ArchitectureError::Unlowerable {
            layer: Some(1),
            feature: Unlowerable::ExpertCount { experts: 384, max: 256 },
        })
    );
    Ok(())
}

/// The latent kernels temper queries counting positions from zero, as
/// Mistral 4 does: admission takes that temperature and refuses one counted
/// from Llama 4's one, which the latent loader would refuse.
#[test]
fn takes_latent_temperature_only_from_position_zero() -> AnyResult<()> {
    let (task, mut semantic) = latent_contract("softmax")?;
    let temper = |semantic: &mut SemanticModelSpec, offset| {
        for layer in &mut semantic.decoder.layers {
            if let MixerSpec::LatentAttention(latent) = &mut layer.mixer {
                latent.query_temperature = Some(QueryTemperature {
                    beta: 0.1,
                    window: NonZeroUsize::MIN.saturating_add(8191),
                    offset,
                });
            }
        }
    };
    temper(&mut semantic, 0);
    assert!(cuda::admit(&task, Some(&semantic)).is_ok());
    temper(&mut semantic, 1);
    assert_eq!(
        cuda::admit(&task, Some(&semantic)).err(),
        Some(ArchitectureError::Unlowerable {
            layer: Some(0),
            feature: Unlowerable::LatentQueryTemperatureOffset { offset: 1 },
        })
    );
    Ok(())
}

/// The latent kernels rotate a key part; latent attention without positions
/// is refused at admission rather than by the loader.
#[test]
fn refuses_latent_attention_without_rotary_keys() -> AnyResult<()> {
    let (task, mut semantic) = latent_contract("softmax")?;
    for layer in &mut semantic.decoder.layers {
        if let MixerSpec::LatentAttention(latent) = &mut layer.mixer {
            latent.position = PositionEncodingSpec::None;
        }
    }
    assert_eq!(
        cuda::admit(&task, Some(&semantic)).err(),
        Some(ArchitectureError::Unlowerable {
            layer: Some(0),
            feature: Unlowerable::LatentRotary
        })
    );
    Ok(())
}
