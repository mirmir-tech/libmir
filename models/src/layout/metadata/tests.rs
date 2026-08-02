use serde_json::json;

use super::*;

#[test]
fn reads_nested_text_context_len() -> Result<()> {
    let value = json!({
        "model_type": "gemma4",
        "dtype": "bfloat16",
        "quantization_config": {
            "bits": 4,
            "group_size": 64,
            "mode": "affine"
        },
        "text_config": {
            "max_position_embeddings": 262_144
        }
    });
    assert_eq!(read_context_len(&value)?, Some(262_144));
    assert_eq!(read_dtype(&value).as_deref(), Some("bfloat16"));
    assert_eq!(read_quantization(&value), Quantization::Int4);
    assert_eq!(read_quantization_usize(&value, "group_size")?, Some(64));
    assert_eq!(read_quantization_string(&value, "mode").as_deref(), Some("affine"));
    Ok(())
}

#[test]
fn reads_modelopt_nvfp4_contract() -> Result<()> {
    let value = json!({
        "quantization_config": {
            "quant_algo": "NVFP4",
            "quant_method": "modelopt",
            "config_groups": {
                "group_0": { "weights": { "num_bits": 4, "group_size": 16 } }
            }
        }
    });

    assert_eq!(read_quantization(&value), Quantization::NvFp4);
    assert_eq!(read_quantization_usize(&value, "group_size")?, Some(16));
    Ok(())
}

#[test]
fn reads_gpt_oss_mxfp4_contract() {
    let value = json!({
        "model_type": "gpt_oss",
        "architectures": ["GptOssForCausalLM"],
        "quantization_config": { "quant_method": "mxfp4" }
    });

    assert_eq!(read_quantization(&value), Quantization::MxFp4);
}

#[test]
fn reads_mlx_mxfp8_contract() -> Result<()> {
    let value = json!({
        "quantization": { "group_size": 32, "bits": 8, "mode": "mxfp8" }
    });

    assert_eq!(read_quantization(&value), Quantization::MxFp8);
    assert_eq!(read_quantization_usize(&value, "group_size")?, Some(32));
    Ok(())
}

#[test]
fn reads_bitsandbytes_nf4_contract() {
    let value = json!({
        "quantization_config": {
            "quant_method": "bitsandbytes",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "bfloat16",
            "bnb_4bit_use_double_quant": true
        }
    });
    assert_eq!(read_quantization(&value), Quantization::Nf4);
}

#[test]
fn reads_compressed_tensors_w8a16_contract() {
    let value = json!({
        "dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": { "weights": { "num_bits": 8, "strategy": "channel" } }
            }
        }
    });
    assert_eq!(read_quantization(&value), Quantization::Int8);
}
