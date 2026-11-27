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

use serde_json::{Map, Value, json};

use super::super::{ConfigurationVector, GgufModel};
use crate::error::{ModelsError, Result};

const TENSOR: &str = ConfigurationVector::RopeFrequencies.tensor();

pub(super) fn rope_scaling(model: &GgufModel, config: &mut Map<String, Value>) -> Result<()> {
    let Some(divisors) = model.vector(ConfigurationVector::RopeFrequencies) else {
        return Ok(());
    };
    let invalid = |detail: &str| ModelsError::InvalidGguf(format!("{TENSOR}: {detail}"));
    let theta = config
        .get("rope_theta")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid("needs rope.freq_base"))?;
    let divisors = divisors?;
    // Every divisor is finite as read; one not positive divides no
    // frequency, and would slip past the comparisons below.
    if divisors.iter().any(|&divisor| divisor.is_nan() || divisor <= 0.0) {
        return Err(invalid("holds a divisor that is not positive"));
    }
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
    if band.is_empty() {
        return hard_boundary(&divisors, &frequencies, factor, config)
            .map_err(|()| invalid("is neither Llama 3 frequency scaling nor a hard boundary"));
    }
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

/// Llama 4's scaling has no band (`low_freq_factor = high_freq_factor = 1`):
/// pairs with a wavelength beyond the original context take `factor`, the
/// others 1. The divisors fix the context only between the longest unscaled
/// and shortest scaled wavelength; the power of two there is the declared
/// one (meta-llama/Llama-4-Scout-17B-16E `original_max_position_embeddings`
/// 8192).
fn hard_boundary(
    divisors: &[f64],
    frequencies: &[f64],
    factor: f64,
    config: &mut Map<String, Value>,
) -> std::result::Result<(), ()> {
    let wavelength = |frequency: f64| std::f64::consts::TAU / frequency;
    let pairs = divisors.iter().zip(frequencies);
    let unscaled = pairs.clone().filter(|&(&divisor, _)| (divisor - 1.0).abs() < 1e-6);
    let scaled = pairs.filter(|&(&divisor, _)| (divisor - factor).abs() < 1e-6 * factor);
    let longest = unscaled.map(|(_, &frequency)| wavelength(frequency)).fold(0.0, f64::max);
    let shortest = scaled
        .map(|(_, &frequency)| wavelength(frequency))
        .fold(f64::INFINITY, f64::min);
    let classified = divisors
        .iter()
        .all(|&divisor| (divisor - 1.0).abs() < 1e-6 || (divisor - factor).abs() < 1e-6 * factor);
    if !classified || factor <= 1.0 || longest >= shortest || !shortest.is_finite() {
        return Err(());
    }
    let original = longest.log2().floor().exp2() * 2.0;
    if original <= longest || original > shortest {
        return Err(());
    }
    config.insert(
        "rope_scaling".into(),
        json!({
            "rope_type": "llama3", "factor": factor, "low_freq_factor": 1.0,
            "high_freq_factor": 1.0,
            "original_max_position_embeddings": format!("{original:.0}").parse::<u64>().map_err(|_| ())?,
        }),
    );
    Ok(())
}
