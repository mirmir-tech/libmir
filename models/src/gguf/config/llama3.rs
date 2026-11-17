//! Llama 3 frequency scaling, which llama.cpp's converter bakes into a
//! `rope_freqs` tensor of per-pair divisors instead of metadata
//! (`convert_hf_to_gguf.py` `LlamaModel.generate_extra_tensors`, v1.0.0).
//!
//! Pairs with a wavelength beyond the low-frequency bound take the full
//! `factor`, those below the high-frequency bound 1, and those between
//! `1 / ((1 - s) / factor + s)` with `s = original / wavelength * a - b`
//! for `a = 1 / (high - low)`, `b = low / (high - low)`. The tensor fixes
//! `factor`, `original * a` and `b`, which is all the scaling depends on;
//! with `low_freq_factor = 1` that gives the Hugging Face parameters, and
//! they must reproduce every divisor.

use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::Path,
};

use serde_json::{Map, Value, json};

use super::super::{GgmlType, GgufModel, GgufTensor};
use crate::error::{ModelsError, Result};

const TENSOR: &str = "rope_freqs.weight";

pub(super) fn rope_scaling(model: &GgufModel, config: &mut Map<String, Value>) -> Result<()> {
    let Some((path, start, tensor)) = model.shards.iter().find_map(|(path, header)| {
        let tensor = header.tensors.iter().find(|tensor| tensor.name == TENSOR)?;
        Some((path, header.data_start, tensor))
    }) else {
        return Ok(());
    };
    let invalid = |detail: &str| ModelsError::InvalidGguf(format!("{TENSOR}: {detail}"));
    let theta = config
        .get("rope_theta")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid("needs rope.freq_base"))?;
    let divisors = read(path, start, tensor)?;
    let dims = f64::from(u32::try_from(divisors.len() * 2)?);
    let frequencies = (0..divisors.len())
        .map(|pair| Ok(theta.powf(-2.0 * f64::from(u32::try_from(pair)?) / dims)))
        .collect::<Result<Vec<_>>>()?;
    let factor = divisors.iter().copied().fold(1.0_f64, f64::max);
    // (1 / wavelength, s) of each pair between the bounds.
    let band = divisors
        .iter()
        .zip(&frequencies)
        .filter(|&(&divisor, _)| divisor > 1.0 + 1e-6 && divisor < factor * (1.0 - 1e-6))
        .map(|(&divisor, &frequency)| {
            let s = (1.0 / divisor - 1.0 / factor) / (1.0 - 1.0 / factor);
            (frequency / std::f64::consts::TAU, s)
        })
        .collect::<Vec<_>>();
    let (Some(&(x0, s0)), Some(&(x1, s1))) = (band.first(), band.last()) else {
        return Err(invalid("has fewer than two pairs between the frequency bounds"));
    };
    if (x1 - x0).abs() < f64::EPSILON {
        return Err(invalid("has fewer than two distinct pairs between the bounds"));
    }
    let slope = (s1 - s0) / (x1 - x0);
    let b = slope.mul_add(x0, -s0);
    if b <= 0.0 {
        return Err(invalid("does not describe Llama 3 frequency scaling"));
    }
    let original = (slope / b).round();
    let (low, high) = (1.0, 1.0 + 1.0 / b);
    let scaled = |frequency: f64| {
        let wavelength = std::f64::consts::TAU / frequency;
        let smooth = (original / wavelength - low) / (high - low);
        if smooth >= 1.0 {
            1.0
        } else if smooth <= 0.0 {
            factor
        } else {
            1.0 / ((1.0 - smooth) / factor + smooth)
        }
    };
    if divisors
        .iter()
        .zip(&frequencies)
        .any(|(&divisor, &frequency)| (scaled(frequency) - divisor).abs() > 1e-3 * divisor)
    {
        return Err(invalid("is not reproduced by Llama 3 frequency scaling"));
    }
    config.insert(
        "rope_scaling".into(),
        json!({
            "rope_type": "llama3", "factor": factor, "low_freq_factor": low,
            "high_freq_factor": (high * 1e6).round() / 1e6,
            "original_max_position_embeddings": format!("{original:.0}").parse::<u64>()?,
        }),
    );
    Ok(())
}

fn read(path: &Path, start: u64, tensor: &GgufTensor) -> Result<Vec<f64>> {
    if tensor.ggml_type != GgmlType::F32 || tensor.dims.len() != 1 {
        return Err(ModelsError::InvalidGguf(format!("{TENSOR} is not a vector of F32")));
    }
    let mut bytes = vec![0_u8; usize::try_from(tensor.bytes)?];
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(start + tensor.offset))?;
    file.read_exact(&mut bytes)?;
    let (words, _) = bytes.as_chunks::<4>();
    Ok(words.iter().map(|word| f64::from(f32::from_le_bytes(*word))).collect())
}
