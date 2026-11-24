//! Gemma 4 text decoders from a `gemma4` header.

use super::{TestResult, Writer};
use crate::gguf::{GgmlType, GgufModel, config::config};

fn array(kind: u32, values: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = kind.to_le_bytes().to_vec();
    bytes.extend(u64::try_from(values.len()).unwrap_or_default().to_le_bytes());
    bytes.extend(values.iter().flatten());
    bytes
}

#[test]
fn gemma4_declares_its_layer_types_and_head_widths() -> TestResult {
    let mut writer = Writer::new().text("general.architecture", "gemma4")?;
    for (key, value) in [
        ("block_count", 2_u32),
        ("attention.key_length", 512),
        ("attention.key_length_swa", 256),
        ("expert_count", 4),
        ("expert_used_count", 2),
    ] {
        writer = writer.kv(&format!("gemma4.{key}"), 4, value.to_le_bytes().to_vec());
    }
    let writer = writer
        .kv("gemma4.attention.sliding_window_pattern", 9, array(7, &[vec![1], vec![0]]))
        .kv(
            "gemma4.attention.head_count_kv",
            9,
            array(4, &[8_u32.to_le_bytes().to_vec(), 2_u32.to_le_bytes().to_vec()]),
        )
        .kv("gemma4.final_logit_softcapping", 6, 30.0_f32.to_le_bytes().to_vec())
        .tensor("token_embd.weight", &[256, 8], GgmlType::Q8_0)?
        .tensor("blk.0.attn_v.weight", &[256, 8], GgmlType::Q8_0)?;
    let path = std::env::temp_dir().join(format!("libmir-gguf-gemma4-{}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    let derived = derived?;
    assert_eq!(derived["model_type"], "gemma4_text");
    assert_eq!(
        derived["layer_types"],
        serde_json::json!(["sliding_attention", "full_attention"])
    );
    assert_eq!(
        (derived["head_dim"].clone(), derived["global_head_dim"].clone()),
        (256.into(), 512.into())
    );
    assert_eq!(
        (
            derived["num_key_value_heads"].clone(),
            derived["num_global_key_value_heads"].clone()
        ),
        (8.into(), 2.into())
    );
    // The full layer has no value projection.
    assert_eq!(derived["attention_k_eq_v"], true);
    assert_eq!(derived["rope_parameters"]["full_attention"]["rope_type"], "proportional");
    assert_eq!(derived["final_logit_softcapping"], 30.0);
    Ok(())
}

/// The E models give one K/V head count for every layer, and their final
/// layers, which read an earlier layer's keys and values, carry no value
/// projection of their own.
#[test]
fn gemma4_e_models_share_one_head_count_and_key_values() -> TestResult {
    let mut writer = Writer::new().text("general.architecture", "gemma4")?;
    for (key, value) in [
        ("block_count", 2_u32),
        ("attention.key_length", 512),
        ("attention.key_length_swa", 256),
        ("attention.head_count_kv", 2),
        ("attention.shared_kv_layers", 1),
        ("embedding_length_per_layer_input", 256),
    ] {
        writer = writer.kv(&format!("gemma4.{key}"), 4, value.to_le_bytes().to_vec());
    }
    let writer = writer
        .kv("gemma4.attention.sliding_window_pattern", 9, array(7, &[vec![1], vec![0]]))
        .tensor("token_embd.weight", &[256, 8], GgmlType::Q8_0)?
        .tensor("blk.0.attn_v.weight", &[256, 8], GgmlType::Q8_0)?;
    // The shared full layer 1 has no value projection: that is not K = V.
    let path =
        std::env::temp_dir().join(format!("libmir-gguf-gemma4-e-{}.gguf", std::process::id()));
    std::fs::write(&path, writer.bytes()?)?;
    let derived = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    let derived = derived?;
    assert_eq!(derived["num_global_key_value_heads"], 2);
    assert_eq!(derived["attention_k_eq_v"], false);
    assert_eq!(derived["num_kv_shared_layers"], 1);
    assert_eq!(derived["hidden_size_per_layer_input"], 256);
    Ok(())
}

/// A Hub variant is checked from its headers and the `rope_freqs` bytes
/// fetched beside them, as its downloaded file would import.
#[test]
fn headers_with_fetched_vectors_derive_what_the_file_does() -> TestResult {
    use crate::gguf::{ConfigurationVector, GgufHeader};

    let mut writer = Writer::new().text("general.architecture", "gemma4")?;
    for (key, value) in
        [("block_count", 2_u32), ("attention.key_length", 8), ("attention.key_length_swa", 8)]
    {
        writer = writer.kv(&format!("gemma4.{key}"), 4, value.to_le_bytes().to_vec());
    }
    let writer = writer
        .kv("gemma4.attention.sliding_window_pattern", 9, array(7, &[vec![1], vec![0]]))
        .kv("gemma4.attention.head_count_kv", 4, 2_u32.to_le_bytes().to_vec())
        .tensor("token_embd.weight", &[256, 8], GgmlType::Q8_0)?
        .tensor("rope_freqs.weight", &[4], GgmlType::F32)?;
    let mut bytes = writer.bytes()?;
    let header = GgufHeader::parse(&bytes, None)
        .map_err(|_| crate::error::ModelsError::InvalidGguf("unparsed test header".into()))?;
    let ranges = header.configuration_vectors().collect::<Vec<_>>();
    let [(ConfigurationVector::RopeFrequencies, range)] = ranges.as_slice() else {
        return Err("the header names no rope_freqs".into());
    };
    let range = usize::try_from(range.start)?..usize::try_from(range.end)?;
    // One of four pairs rotates.
    let marks = [1.0_f32, 1e30, 1e30, 1e30];
    bytes[range.clone()].copy_from_slice(&marks.map(f32::to_le_bytes).concat());
    let path =
        std::env::temp_dir().join(format!("libmir-gguf-vectors-{}.gguf", std::process::id()));
    std::fs::write(&path, &bytes)?;
    let opened = config(&GgufModel::open(&path)?);
    std::fs::remove_file(&path)?;
    let missing = std::path::PathBuf::from("not-downloaded.gguf");
    let fetched = GgufModel::from_headers(
        vec![(missing.clone(), header.clone())],
        [(ConfigurationVector::RopeFrequencies, bytes[range].to_vec())],
    )?;
    let parameters = |derived: &serde_json::Value| {
        derived["rope_parameters"]["full_attention"]["partial_rotary_factor"].clone()
    };
    let opened = opened?;
    assert_eq!(parameters(&opened), 0.25);
    assert_eq!(parameters(&config(&fetched)?), parameters(&opened));
    // Without the bytes the check refuses rather than opening the file.
    let error = config(&GgufModel::from_headers(vec![(missing, header)], [])?).err();
    assert!(error.is_some_and(|error| error.to_string().contains("not fetched")));
    Ok(())
}
