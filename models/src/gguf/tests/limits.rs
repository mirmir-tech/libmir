//! A crafted header's counts and vectors are bounded before anything is
//! sized by them, and a vector the configuration compares against must hold
//! values it can compare.

use super::{TestResult, Writer};
use crate::gguf::{ConfigurationVector, GgmlType, GgufHeader, GgufModel, config::config};

fn parsed(writer: &Writer) -> TestResult<(Vec<u8>, GgufHeader)> {
    let bytes = writer.bytes()?;
    let header = GgufHeader::parse(&bytes, None).map_err(|error| format!("{error:?}"))?;
    Ok((bytes, header))
}

fn llama(blocks: u64) -> TestResult<Writer> {
    Ok(Writer::new()
        .text("general.architecture", "llama")?
        .kv("llama.block_count", 10, blocks.to_le_bytes().to_vec())
        .kv("llama.rope.freq_base", 6, 500_000.0_f32.to_le_bytes().to_vec())
        .kv("llama.attention.sliding_window", 4, 64_u32.to_le_bytes().to_vec())
        .kv("llama.attention.sliding_window_pattern", 4, 2_u32.to_le_bytes().to_vec()))
}

/// The config of `writer`'s header, with `divisors` as its fetched
/// `rope_freqs`.
fn derive(writer: Writer, divisors: &[f32]) -> TestResult<crate::error::Result<serde_json::Value>> {
    let length = u64::try_from(divisors.len())?;
    let (_, header) = parsed(&writer.tensor("rope_freqs.weight", &[length], GgmlType::F32)?)?;
    let bytes = divisors.iter().flat_map(|value| value.to_le_bytes()).collect();
    let model = GgufModel::from_headers(
        vec![("model.gguf".into(), header)],
        [(ConfigurationVector::RopeFrequencies, bytes)],
    )?;
    Ok(config(&model))
}

#[test]
fn a_block_count_beyond_any_model_is_refused() -> TestResult {
    let (_, header) = parsed(&llama(10_000_000_000)?)?;
    let model = GgufModel::from_headers(vec![("model.gguf".into(), header)], [])?;
    let error = config(&model).err().ok_or("10^10 blocks were laid out")?;
    assert!(error.to_string().contains("block_count"), "{error}");
    let (_, header) = parsed(&llama(4)?)?;
    let model = GgufModel::from_headers(vec![("model.gguf".into(), header)], [])?;
    assert_eq!(config(&model)?["layer_types"].as_array().map(Vec::len), Some(4));
    Ok(())
}

#[test]
fn a_configuration_vector_beyond_its_bound_is_not_read() -> TestResult {
    let writer = llama(4)?.tensor("rope_freqs.weight", &[300_000], GgmlType::F32)?;
    let (_, header) = parsed(&writer)?;
    let model = GgufModel::from_headers(vec![("model.gguf".into(), header)], [])?;
    let error = config(&model).err().ok_or("a large vector was read")?;
    assert!(error.to_string().contains("beyond"), "{error}");
    Ok(())
}

/// `NaN` compares false with everything, so a divisor of it would pass
/// every check of Llama 3 scaling; it, an infinity and a divisor that is
/// not positive are refused.
#[test]
fn rope_frequencies_that_cannot_be_compared_are_refused() -> TestResult {
    let scaled = [1.0_f32, 1.0, 8.0, 8.0];
    assert!(derive(llama(4)?, &scaled)?.is_ok());
    for bad in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        let divisors = [1.0, bad, 8.0, 8.0];
        assert!(derive(llama(4)?, &divisors)?.is_err(), "{bad} was taken for a divisor");
    }
    Ok(())
}
