//! Miniature checkpoints with the `DeepSeek` V2/V3 latent-attention schema.

use serde_json::{Value, json};

use super::*;
use crate::semantic::{FeedForwardSpec, MixerSpec, RouterNormalization};

const HIDDEN: usize = 32;
const HEADS: usize = 4;
const RANK: usize = 16;
const QUERY_RANK: usize = 8;
const NOPE: usize = 8;
const ROPE: usize = 4;
const VALUE: usize = 8;
const EXPERTS: usize = 4;
const EXPERT_WIDTH: usize = 8;

fn config(query_rank: Option<usize>, scoring: &str) -> Value {
    json!({
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "hidden_size": HIDDEN,
        "intermediate_size": 16,
        "moe_intermediate_size": EXPERT_WIDTH,
        "num_hidden_layers": 2,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "vocab_size": 64,
        "q_lora_rank": query_rank,
        "kv_lora_rank": RANK,
        "qk_nope_head_dim": NOPE,
        "qk_rope_head_dim": ROPE,
        "v_head_dim": VALUE,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": 2,
        "n_shared_experts": 2,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_group": 1,
        "topk_group": 1,
        "scoring_func": scoring,
        "topk_method": if scoring == "sigmoid" { "noaux_tc" } else { "greedy" },
        "norm_topk_prob": scoring == "sigmoid",
        "routed_scaling_factor": 2.5,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6
    })
}

fn add(tensors: &mut Vec<TensorInfo>, name: &str, shape: &[usize]) {
    tensors.push(tensor(name, "BF16", shape.to_vec()));
}

fn catalog(query_rank: Option<usize>, correction_bias: bool) -> TensorCatalog {
    let mut tensors = Vec::new();
    add(&mut tensors, "model.embed_tokens.weight", &[64, HIDDEN]);
    add(&mut tensors, "model.norm.weight", &[HIDDEN]);
    add(&mut tensors, "lm_head.weight", &[64, HIDDEN]);
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        let attention = format!("{prefix}.self_attn");
        add(&mut tensors, &format!("{prefix}.input_layernorm.weight"), &[HIDDEN]);
        add(&mut tensors, &format!("{prefix}.post_attention_layernorm.weight"), &[HIDDEN]);
        let query = HEADS * (NOPE + ROPE);
        match query_rank {
            Some(rank) => {
                add(&mut tensors, &format!("{attention}.q_a_proj.weight"), &[rank, HIDDEN]);
                add(&mut tensors, &format!("{attention}.q_a_layernorm.weight"), &[rank]);
                add(&mut tensors, &format!("{attention}.q_b_proj.weight"), &[query, rank]);
            },
            None => add(&mut tensors, &format!("{attention}.q_proj.weight"), &[query, HIDDEN]),
        }
        let down = RANK + ROPE;
        add(&mut tensors, &format!("{attention}.kv_a_proj_with_mqa.weight"), &[down, HIDDEN]);
        add(&mut tensors, &format!("{attention}.kv_a_layernorm.weight"), &[RANK]);
        let up = HEADS * (NOPE + VALUE);
        add(&mut tensors, &format!("{attention}.kv_b_proj.weight"), &[up, RANK]);
        add(&mut tensors, &format!("{attention}.o_proj.weight"), &[HIDDEN, HEADS * VALUE]);
        let mlp = format!("{prefix}.mlp");
        if layer == 0 {
            for (projection, shape) in
                [("gate", [16, HIDDEN]), ("up", [16, HIDDEN]), ("down", [HIDDEN, 16])]
            {
                add(&mut tensors, &format!("{mlp}.{projection}_proj.weight"), &shape);
            }
            continue;
        }
        add(&mut tensors, &format!("{mlp}.gate.weight"), &[EXPERTS, HIDDEN]);
        if correction_bias {
            tensors.push(tensor(
                &format!("{mlp}.gate.e_score_correction_bias"),
                "F32",
                vec![EXPERTS],
            ));
        }
        let shared = 2 * EXPERT_WIDTH;
        for (projection, shape) in
            [("gate", [shared, HIDDEN]), ("up", [shared, HIDDEN]), ("down", [HIDDEN, shared])]
        {
            add(&mut tensors, &format!("{mlp}.shared_experts.{projection}_proj.weight"), &shape);
        }
        for expert in 0..EXPERTS {
            for (projection, shape) in [
                ("gate", [EXPERT_WIDTH, HIDDEN]),
                ("up", [EXPERT_WIDTH, HIDDEN]),
                ("down", [HIDDEN, EXPERT_WIDTH]),
            ] {
                let name = format!("{mlp}.experts.{expert}.{projection}_proj.weight");
                add(&mut tensors, &name, &shape);
            }
        }
    }
    TensorCatalog::new(tensors)
}

fn plan(config: &Value, catalog: &TensorCatalog) -> Result<(SemanticModelSpec, WeightBindingPlan)> {
    let decoder = DecoderConfig::from_value(config)?;
    let spec = SemanticModelSpec::discover(&decoder, catalog)?;
    let plan = WeightBindingPlan::discover(&spec, catalog)?;
    Ok((spec, plan))
}

fn invalid(message: &str) -> crate::ModelsError {
    crate::ModelsError::InvalidConfig(message.into())
}

fn rejected(result: Result<(SemanticModelSpec, WeightBindingPlan)>, needle: &str) -> Result<()> {
    match result {
        Ok(_) => Err(invalid("checkpoint must be rejected")),
        Err(error) => {
            assert!(error.to_string().contains(needle), "unexpected error: {error}");
            Ok(())
        },
    }
}

#[test]
fn binds_the_deepseek_v2_lite_schema() -> Result<()> {
    let catalog = catalog(None, false);
    let (spec, plan) = plan(&config(None, "softmax"), &catalog)?;

    let MixerSpec::LatentAttention(latent) = &spec.decoder.layers[0].mixer else {
        return Err(invalid("expected latent attention"));
    };
    assert_eq!((latent.query_rank, latent.cached_width()), (None, RANK + ROPE));
    assert!(matches!(spec.decoder.layers[0].feed_forward, FeedForwardSpec::Dense { .. }));
    let FeedForwardSpec::Routed { routed, shared: Some(shared) } =
        &spec.decoder.layers[1].feed_forward
    else {
        return Err(invalid("expected routed experts with a shared expert"));
    };
    assert_eq!(routed.router_normalization, RouterNormalization::SoftmaxProbabilityTopK);
    assert!((routed.output_scale - 2.5).abs() < f64::EPSILON);
    assert_eq!((shared.intermediate_size, shared.gated_output), (2 * EXPERT_WIDTH, false));
    assert_eq!(plan.tensors.len(), catalog.tensors.len());

    let dense = plan.latent_decoder_layer(0)?;
    assert!(matches!(dense.attention.query, LatentQueryBindings::Direct(_)));
    assert!(matches!(dense.feed_forward, MixedFeedForwardBindings::Dense(_)));
    let MixedFeedForwardBindings::SharedRouted(routed) = plan.latent_decoder_layer(1)?.feed_forward
    else {
        return Err(invalid("expected shared-routed bindings"));
    };
    assert!(routed.shared_output_gate.is_none());
    Ok(())
}

#[test]
fn binds_a_normalized_query_latent_and_a_corrected_sigmoid_router() -> Result<()> {
    let catalog = catalog(Some(QUERY_RANK), true);
    let (spec, plan) = plan(&config(Some(QUERY_RANK), "sigmoid"), &catalog)?;

    let FeedForwardSpec::Routed { routed, .. } = &spec.decoder.layers[1].feed_forward else {
        return Err(invalid("expected routed experts"));
    };
    assert_eq!(routed.router_normalization, RouterNormalization::SigmoidBiasedTopK);
    assert_eq!(plan.tensors.len(), catalog.tensors.len());
    let layer = plan.latent_decoder_layer(1)?;
    assert!(matches!(layer.attention.query, LatentQueryBindings::Latent { .. }));
    // Norms, query latent, key/value latent, output, router and its bias,
    // three shared and twelve routed expert projections.
    assert_eq!(layer.physical_sources().len(), 2 + 3 + 4 + 2 + 3 + 3 * EXPERTS);
    Ok(())
}

#[test]
fn rejects_a_direct_query_where_the_spec_has_a_query_latent() -> Result<()> {
    rejected(
        plan(&config(Some(QUERY_RANK), "softmax"), &catalog(None, false)),
        "binding grammar",
    )
}

#[test]
fn rejects_a_sigmoid_router_without_its_correction_bias() -> Result<()> {
    rejected(
        plan(&config(Some(QUERY_RANK), "sigmoid"), &catalog(Some(QUERY_RANK), false)),
        "correction bias",
    )
}
