use super::*;

#[test]
fn binds_official_mistral_dense_checkpoint_names() -> Result<()> {
    let decoder = DecoderConfig::from_value(&json!({
        "model_type": "mistral",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 64,
        "hidden_act": "silu",
        "tie_word_embeddings": false
    }))?;
    let catalog = TensorCatalog {
        tensors: vec![
            tensor("model.embed_tokens.weight", "BF16", vec![64, 32]),
            tensor("model.norm.weight", "BF16", vec![32]),
            tensor("lm_head.weight", "BF16", vec![64, 32]),
            tensor("model.layers.0.input_layernorm.weight", "BF16", vec![32]),
            tensor("model.layers.0.self_attn.q_proj.weight", "BF16", vec![32, 32]),
            tensor("model.layers.0.self_attn.k_proj.weight", "BF16", vec![16, 32]),
            tensor("model.layers.0.self_attn.v_proj.weight", "BF16", vec![16, 32]),
            tensor("model.layers.0.self_attn.o_proj.weight", "BF16", vec![32, 32]),
            tensor("model.layers.0.post_attention_layernorm.weight", "BF16", vec![32]),
            tensor("model.layers.0.mlp.gate_proj.weight", "BF16", vec![64, 32]),
            tensor("model.layers.0.mlp.up_proj.weight", "BF16", vec![64, 32]),
            tensor("model.layers.0.mlp.down_proj.weight", "BF16", vec![32, 64]),
        ],
    };
    let spec = SemanticModelSpec::discover(&decoder, &catalog)?;

    let bindings = WeightBindingPlan::discover(&spec, &catalog)?;
    let boundary = bindings.decoder_boundary_with_tied_output(false)?;
    let layer = bindings.dense_decoder_layer(0)?;

    assert_eq!(boundary.embedding.source, "model.embed_tokens.weight");
    assert_eq!(boundary.output.source, "lm_head.weight");
    assert_eq!(layer.attention.query.shape, [32, 32]);
    assert_eq!(layer.attention.key.shape, [16, 32]);
    assert_eq!(layer.down.shape, [32, 64]);
    Ok(())
}
