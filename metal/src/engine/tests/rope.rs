use models::{layout::DecoderConfig, semantic::MixerSpec};
use serde_json::json;

use super::*;

/// Scaled `RoPE` periods come from the semantic spec, uploaded as MLX `freqs`.
#[test]
fn uploads_the_specs_scaled_rope_periods() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let decoder = DecoderConfig::from_value(&json!({
        "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
        "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 16, "vocab_size": 8,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "rope_scaling": {
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192
        }
    }))?;
    let catalog = models::weights::TensorCatalog {
        tensors: vec![models::weights::TensorInfo {
            name: "model.layers.0.input_layernorm.weight".into(),
            file: std::path::PathBuf::new(),
            container: models::layout::WeightContainer::SafeTensors,
            dtype: models::weights::DenseDType::Bf16.into(),
            shape: vec![16],
            data_start: 0,
            data_offsets: [0, 0],
        }],
    };
    let spec = models::semantic::SemanticModelSpec::discover(&decoder, &catalog)?;
    let Some(MixerSpec::SoftmaxAttention(attention)) =
        spec.decoder.layers.first().map(|layer| &layer.mixer)
    else {
        return Err(Error::InvalidModel("expected softmax attention".into()));
    };
    let values = Array::rope_periods(attention)?
        .ok_or(Error::InvalidModel("scaled RoPE needs periods".into()))?
        .to_vec_f32(&stream)?;
    assert_eq!(values.len(), 4);
    for (actual, expected) in values.iter().zip([1.0, 10.0, 100.0]) {
        assert!((actual - expected).abs() <= expected * 1e-6);
    }
    assert!(values[3] > 4_000.0 && values[3] < 5_000.0);
    Ok(())
}
