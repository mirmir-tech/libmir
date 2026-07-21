use std::path::PathBuf;

use serde_json::json;

use super::*;

mod moe;

const HIDDEN: usize = 32;
const GROUP: usize = 16;

#[test]
fn discovers_complete_hybrid_linear_and_softmax_binding_views() -> Result<()> {
    let decoder = decoder()?;
    let catalog = catalog();
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let plan = WeightBindingPlan::discover(&spec, &catalog)?;
    let boundary = plan.decoder_boundary()?;
    let linear = plan.hybrid_decoder_layer(0)?;
    let softmax = plan.hybrid_decoder_layer(1)?;

    assert_eq!(boundary.output.source, "language_model.lm_head.weight");
    let HybridMixerBindings::Linear(linear) = linear.mixer else {
        return Err(invalid("expected linear mixer bindings"));
    };
    assert_eq!(linear.convolution.logical_shape.as_deref(), Some([32, 4, 1].as_slice()));
    let HybridMixerBindings::Softmax(softmax) = softmax.mixer else {
        return Err(invalid("expected softmax mixer bindings"));
    };
    assert_eq!(softmax.query.logical_shape.as_deref(), Some([64, 32].as_slice()));
    assert!(matches!(
        plan.hybrid_decoder_layer(0)?.feed_forward.routed_gate.transforms.as_slice(),
        [BindingTransform::StackedExperts { count: 8 }]
    ));
    Ok(())
}

#[test]
fn rejects_ambiguous_shared_expert_grammar() -> Result<()> {
    let decoder = decoder()?;
    let mut catalog = catalog();
    affine(
        &mut catalog.tensors,
        "language_model.model.layers.0.mlp.switch_mlp.gate_up_proj",
        &[8, 32, HIDDEN],
    );
    catalog.tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    assert!(WeightBindingPlan::discover(&spec, &catalog).is_err());
    Ok(())
}

fn decoder() -> Result<DecoderConfig> {
    DecoderConfig::from_value(&json!({
        "hidden_size": HIDDEN,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 64,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16,
        "attn_output_gate": true,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4
    }))
}

fn catalog() -> TensorCatalog {
    let mut tensors = Vec::new();
    affine(&mut tensors, "language_model.model.embed_tokens", &[64, HIDDEN]);
    dense(&mut tensors, "language_model.model.norm.weight", &[HIDDEN]);
    affine(&mut tensors, "language_model.lm_head", &[64, HIDDEN]);
    for layer in 0..2 {
        common(&mut tensors, layer);
    }
    linear(&mut tensors, 0);
    softmax(&mut tensors, 1);
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    TensorCatalog { tensors }
}

fn common(tensors: &mut Vec<TensorInfo>, layer: usize) {
    let prefix = format!("language_model.model.layers.{layer}");
    dense(tensors, &format!("{prefix}.input_layernorm.weight"), &[HIDDEN]);
    dense(tensors, &format!("{prefix}.post_attention_layernorm.weight"), &[HIDDEN]);
    let mlp = format!("{prefix}.mlp");
    affine(tensors, &format!("{mlp}.gate"), &[8, HIDDEN]);
    for projection in ["gate_proj", "up_proj"] {
        affine(tensors, &format!("{mlp}.switch_mlp.{projection}"), &[8, 16, HIDDEN]);
        affine(tensors, &format!("{mlp}.shared_expert.{projection}"), &[16, HIDDEN]);
    }
    affine(tensors, &format!("{mlp}.switch_mlp.down_proj"), &[8, HIDDEN, 16]);
    affine(tensors, &format!("{mlp}.shared_expert.down_proj"), &[HIDDEN, 16]);
    affine(tensors, &format!("{mlp}.shared_expert_gate"), &[1, HIDDEN]);
}

fn linear(tensors: &mut Vec<TensorInfo>, layer: usize) {
    let prefix = format!("language_model.model.layers.{layer}.linear_attn");
    for (name, shape) in [
        ("in_proj_qkv", vec![32, HIDDEN]),
        ("in_proj_z", vec![16, HIDDEN]),
        ("in_proj_a", vec![4, HIDDEN]),
        ("in_proj_b", vec![4, HIDDEN]),
        ("out_proj", vec![HIDDEN, 16]),
    ] {
        affine(tensors, &format!("{prefix}.{name}"), &shape);
    }
    dense(tensors, &format!("{prefix}.conv1d.weight"), &[32, 4, 1]);
    dense(tensors, &format!("{prefix}.norm.weight"), &[4]);
    dense(tensors, &format!("{prefix}.A_log"), &[4]);
    dense(tensors, &format!("{prefix}.dt_bias"), &[4]);
}

fn softmax(tensors: &mut Vec<TensorInfo>, layer: usize) {
    let prefix = format!("language_model.model.layers.{layer}.self_attn");
    for (name, shape) in [
        ("q_proj", vec![64, HIDDEN]),
        ("k_proj", vec![16, HIDDEN]),
        ("v_proj", vec![16, HIDDEN]),
        ("o_proj", vec![HIDDEN, HIDDEN]),
    ] {
        affine(tensors, &format!("{prefix}.{name}"), &shape);
    }
    dense(tensors, &format!("{prefix}.q_norm.weight"), &[8]);
    dense(tensors, &format!("{prefix}.k_norm.weight"), &[8]);
}

fn affine(tensors: &mut Vec<TensorInfo>, prefix: &str, logical: &[usize]) {
    let Some((input, output)) = logical.split_last() else {
        return;
    };
    let mut weight = output.to_vec();
    weight.push(input / 8);
    let mut parameters = output.to_vec();
    parameters.push(input / GROUP);
    tensors.push(tensor(&format!("{prefix}.weight"), "U32", &weight));
    tensors.push(tensor(&format!("{prefix}.scales"), "BF16", &parameters));
    tensors.push(tensor(&format!("{prefix}.biases"), "BF16", &parameters));
}

fn dense(tensors: &mut Vec<TensorInfo>, name: &str, shape: &[usize]) {
    tensors.push(tensor(name, "BF16", shape));
}

fn tensor(name: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
    TensorInfo {
        name: name.into(),
        file: PathBuf::new(),
        dtype: dtype.into(),
        shape: shape.to_vec(),
        data_start: 0,
        data_offsets: [0, 0],
    }
}

fn invalid(message: &str) -> crate::ModelsError {
    crate::ModelsError::InvalidConfig(message.into())
}
