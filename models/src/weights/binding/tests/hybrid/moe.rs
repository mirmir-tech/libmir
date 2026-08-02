use std::path::PathBuf;

use serde_json::json;

use super::*;

const HIDDEN: usize = 32;

#[test]
fn discovers_complete_stacked_hybrid_moe_bindings() -> Result<()> {
    let decoder = decoder()?;
    let catalog = catalog();
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let plan = WeightBindingPlan::discover(&spec, &catalog)?;
    let layer = plan.hybrid_moe_layer(0)?;

    assert!(layer.attention.value.is_none());
    assert_eq!(
        layer.router.expert_scale.source,
        "language_model.model.layers.0.router.per_expert_scale"
    );
    assert!(matches!(layer.experts, HybridMoeExpertBindings::Stacked(_)));
    assert!(layer.physical_sources().contains(&"language_model.model.layers.0.layer_scalar"));
    Ok(())
}

#[test]
fn discovers_fused_stacked_hybrid_moe_bindings() -> Result<()> {
    let decoder = decoder()?;
    let mut catalog = catalog();
    catalog.tensors.retain(|tensor| {
        !tensor.name.contains("experts.switch_glu.gate_proj")
            && !tensor.name.contains("experts.switch_glu.up_proj")
    });
    catalog
        .tensors
        .push(tensor("model.language_model.layers.0.experts.gate_up_proj", &[4, 32, HIDDEN]));
    catalog.tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let plan = WeightBindingPlan::discover(&spec, &catalog)?;
    assert!(matches!(
        plan.hybrid_moe_layer(0)?.experts,
        HybridMoeExpertBindings::FusedStacked { .. }
    ));
    Ok(())
}

#[test]
fn rejects_incomplete_hybrid_moe_grammar() -> Result<()> {
    let decoder = decoder()?;
    let mut catalog = catalog();
    catalog
        .tensors
        .retain(|tensor| !tensor.name.ends_with("post_feedforward_layernorm_2.weight"));
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

#[test]
fn discovers_individual_nvfp4_experts_with_physical_companions() -> Result<()> {
    let decoder = decoder()?;
    let mut catalog = catalog();
    catalog.tensors.retain(|tensor| !tensor.name.contains("experts.switch_glu"));
    for expert in 0..4 {
        nvfp4(&mut catalog.tensors, expert, "gate_proj", &[16, HIDDEN]);
        nvfp4(&mut catalog.tensors, expert, "up_proj", &[16, HIDDEN]);
        nvfp4(&mut catalog.tensors, expert, "down_proj", &[HIDDEN, 16]);
    }
    catalog.tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let plan = WeightBindingPlan::discover(&spec, &catalog)?;
    let layer = plan.hybrid_moe_layer(0)?;

    let HybridMoeExpertBindings::Individual { gate, up, down } = &layer.experts else {
        return Err(invalid("expected individual experts"));
    };
    assert_eq!((gate.len(), up.len(), down.len()), (4, 4, 4));
    assert!(
        layer
            .physical_sources()
            .contains(&"model.layers.0.experts.0.gate_proj.weight_scale_2")
    );
    Ok(())
}

fn decoder() -> Result<DecoderConfig> {
    DecoderConfig::from_value(&json!({
        "hidden_size": HIDDEN,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 96,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "hidden_act": "gelu_pytorch_tanh",
        "attention_k_eq_v": true
    }))
}

fn catalog() -> TensorCatalog {
    let prefix = "language_model.model.layers.0";
    let mut tensors = vec![
        tensor("language_model.model.embed_tokens.weight", &[96, HIDDEN]),
        tensor("language_model.model.norm.weight", &[HIDDEN]),
        tensor(&format!("{prefix}.input_layernorm.weight"), &[HIDDEN]),
        tensor(&format!("{prefix}.self_attn.q_proj.weight"), &[32, HIDDEN]),
        tensor(&format!("{prefix}.self_attn.k_proj.weight"), &[16, HIDDEN]),
        tensor(&format!("{prefix}.self_attn.o_proj.weight"), &[HIDDEN, 32]),
        tensor(&format!("{prefix}.self_attn.q_norm.weight"), &[8]),
        tensor(&format!("{prefix}.self_attn.k_norm.weight"), &[8]),
        tensor(&format!("{prefix}.post_attention_layernorm.weight"), &[HIDDEN]),
        tensor(&format!("{prefix}.pre_feedforward_layernorm.weight"), &[HIDDEN]),
        tensor(&format!("{prefix}.mlp.gate_proj.weight"), &[64, HIDDEN]),
        tensor(&format!("{prefix}.mlp.up_proj.weight"), &[64, HIDDEN]),
        tensor(&format!("{prefix}.mlp.down_proj.weight"), &[HIDDEN, 64]),
        tensor(&format!("{prefix}.post_feedforward_layernorm_1.weight"), &[HIDDEN]),
        tensor(&format!("{prefix}.router.proj.weight"), &[4, HIDDEN]),
        tensor(&format!("{prefix}.router.scale"), &[HIDDEN]),
        tensor(&format!("{prefix}.router.per_expert_scale"), &[4]),
        tensor(&format!("{prefix}.pre_feedforward_layernorm_2.weight"), &[HIDDEN]),
        tensor(&format!("{prefix}.post_feedforward_layernorm_2.weight"), &[HIDDEN]),
        tensor(&format!("{prefix}.post_feedforward_layernorm.weight"), &[HIDDEN]),
        tensor(&format!("{prefix}.layer_scalar"), &[1]),
    ];
    for (projection, shape) in [
        ("gate_proj", vec![4, 16, HIDDEN]),
        ("up_proj", vec![4, 16, HIDDEN]),
        ("down_proj", vec![4, HIDDEN, 16]),
    ] {
        tensors.push(tensor(&format!("{prefix}.experts.switch_glu.{projection}.weight"), &shape));
    }
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    TensorCatalog { tensors }
}

fn tensor(name: &str, shape: &[usize]) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: PathBuf::new(),
        dtype: "BF16".into(),
        shape: shape.to_vec(),
        data_start: 0,
        data_offsets: [0, 0],
    }
}

fn nvfp4(tensors: &mut Vec<TensorInfo>, expert: usize, projection: &str, shape: &[usize]) {
    let prefix = format!("model.layers.0.experts.{expert}.{projection}");
    let (output, input) = (shape[0], shape[1]);
    let mut weight = tensor(&format!("{prefix}.weight"), &[output, input / 2]);
    weight.dtype = "U8".into();
    let mut scale = tensor(&format!("{prefix}.weight_scale"), &[output, input / 16]);
    scale.dtype = "F8_E4M3".into();
    tensors.extend([
        weight,
        scale,
        tensor(&format!("{prefix}.weight_scale_2"), &[]),
        tensor(&format!("{prefix}.input_scale"), &[]),
    ]);
    for tensor in tensors.iter_mut().rev().take(2) {
        tensor.dtype = "F32".into();
    }
}

fn invalid(message: &str) -> crate::ModelsError {
    crate::ModelsError::InvalidConfig(message.into())
}
