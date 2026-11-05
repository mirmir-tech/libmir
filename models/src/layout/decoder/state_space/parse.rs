use std::num::NonZeroUsize;

use serde_json::Value;

use super::{
    HybridBlockKind, HybridBlockPattern, LatentRoutedConfig, Mamba2Config, PredictionModules,
    StateSpaceHybridConfig, StateSpaceStateDType, TimeStepBound, TimeStepLimit,
};
use crate::{
    error::Result,
    layout::decoder::parse::{
        invalid, optional_bool, optional_f64, optional_string, optional_usize, require_usize,
    },
};

/// Parses a state-space hybrid decoder, or returns `None` when the
/// configuration does not declare a physical block sequence.
pub(in crate::layout::decoder) fn state_space_hybrid(
    decoder: &Value,
    hidden_size: usize,
) -> Result<Option<StateSpaceHybridConfig>> {
    let Some(raw) = decoder.get("layers_block_type") else {
        return Ok(None);
    };
    let blocks = HybridBlockPattern::pair(block_kinds(raw)?)?;
    if let Some(declared) = optional_usize(decoder, &["num_hidden_layers"])?
        && declared != blocks.physical().len()
    {
        return Err(invalid(format!(
            "num_hidden_layers is {declared}, but layers_block_type has {} blocks",
            blocks.physical().len()
        )));
    }
    let mamba = mamba(decoder, hidden_size)?;
    let routed = routed(decoder)?;
    let prediction = prediction(decoder)?;
    Ok(Some(StateSpaceHybridConfig { blocks, mamba, routed, prediction }))
}

/// Multi-token prediction modules; each must be one attention block followed
/// by one routed block, the only composition with a checkpoint fixture.
fn prediction(decoder: &Value) -> Result<Option<PredictionModules>> {
    let Some(depth) =
        optional_usize(decoder, &["num_nextn_predict_layers"])?.and_then(NonZeroUsize::new)
    else {
        return Ok(None);
    };
    let raw = decoder
        .get("mtp_layers_block_type")
        .ok_or_else(|| invalid("num_nextn_predict_layers requires mtp_layers_block_type"))?;
    if block_kinds(raw)? != [HybridBlockKind::Attention, HybridBlockKind::Moe] {
        return Err(invalid(
            "prediction modules must be one attention block followed by one routed block",
        ));
    }
    Ok(Some(PredictionModules { depth }))
}

fn block_kinds(raw: &Value) -> Result<Vec<HybridBlockKind>> {
    let items = raw.as_array().ok_or_else(|| invalid("layers_block_type must be an array"))?;
    items
        .iter()
        .map(|item| match item.as_str() {
            Some("mamba") => Ok(HybridBlockKind::Mamba),
            Some("attention") => Ok(HybridBlockKind::Attention),
            Some("moe") => Ok(HybridBlockKind::Moe),
            Some(other) => Err(invalid(format!("unsupported hybrid block type {other}"))),
            None => Err(invalid("layers_block_type entries must be strings")),
        })
        .collect()
}

fn mamba(decoder: &Value, hidden_size: usize) -> Result<Mamba2Config> {
    let config = Mamba2Config {
        heads: positive(decoder, "mamba_num_heads")?,
        head_dim: positive(decoder, "mamba_head_dim")?,
        state_size: positive(decoder, "ssm_state_size")?,
        groups: positive(decoder, "n_groups")?,
        convolution_kernel_size: positive(decoder, "conv_kernel")?,
        convolution_bias: optional_bool(decoder, "use_conv_bias")?.unwrap_or(true),
        projection_bias: optional_bool(decoder, "mamba_proj_bias")?.unwrap_or(false),
        chunk_size: positive(decoder, "chunk_size")?,
        time_step_limit: time_step_limit(decoder)?,
        state_dtype: state_dtype(decoder)?,
    };
    let activation = optional_string(decoder, "mamba_hidden_act")?;
    if activation.as_deref().is_some_and(|name| name != "silu") {
        return Err(invalid("Mamba-2 mixer requires the silu activation"));
    }
    if !config.heads.is_multiple_of(config.groups) {
        return Err(invalid("mamba_num_heads must be divisible by n_groups"));
    }
    if let Some(expand) = optional_usize(decoder, &["expand"])?
        && expand.checked_mul(hidden_size) != Some(config.inner_size())
    {
        return Err(invalid("mamba_num_heads * mamba_head_dim must equal expand * hidden_size"));
    }
    Ok(config)
}

fn routed(decoder: &Value) -> Result<LatentRoutedConfig> {
    let activation = optional_string(decoder, "mlp_hidden_act")?;
    if activation.as_deref() != Some("relu2") {
        return Err(invalid("hybrid routed experts require the relu2 activation"));
    }
    if optional_bool(decoder, "mlp_bias")?.unwrap_or(false) {
        return Err(invalid("hybrid routed experts with projection bias are not supported"));
    }
    if optional_bool(decoder, "norm_topk_prob")? != Some(true) {
        return Err(invalid("hybrid routed experts require renormalized top-k weights"));
    }
    for key in ["n_group", "topk_group"] {
        if optional_usize(decoder, &[key])?.unwrap_or(1) != 1 {
            return Err(invalid(format!("grouped expert selection ({key} > 1) is not supported")));
        }
    }
    let expert_count = positive(decoder, "n_routed_experts")?;
    let top_k = positive(decoder, "num_experts_per_tok")?;
    if top_k > expert_count {
        return Err(invalid("num_experts_per_tok cannot exceed n_routed_experts"));
    }
    let shared_intermediate_size = match optional_usize(decoder, &["n_shared_experts"])? {
        None | Some(0) => None,
        Some(1) => Some(positive(decoder, "moe_shared_expert_intermediate_size")?),
        Some(_) => return Err(invalid("more than one shared expert is not supported")),
    };
    let routed_scale = optional_f64(decoder, &["routed_scaling_factor"])?.unwrap_or(1.0);
    if !routed_scale.is_finite() || routed_scale <= 0.0 {
        return Err(invalid("routed_scaling_factor must be finite and positive"));
    }
    Ok(LatentRoutedConfig {
        expert_count,
        top_k,
        latent_size: positive(decoder, "moe_latent_size")?,
        expert_intermediate_size: positive(decoder, "moe_intermediate_size")?,
        shared_intermediate_size,
        routed_scale,
    })
}

fn time_step_limit(decoder: &Value) -> Result<TimeStepLimit> {
    let Some(raw) = decoder.get("time_step_limit").filter(|value| !value.is_null()) else {
        return Ok(TimeStepLimit {
            minimum: 0.0,
            maximum: TimeStepBound::Unbounded,
        });
    };
    let [minimum, maximum] = raw
        .as_array()
        .map(Vec::as_slice)
        .and_then(|items| <&[Value; 2]>::try_from(items).ok())
        .ok_or_else(|| invalid("time_step_limit must have two entries"))?;
    let minimum = bound(minimum)?;
    let maximum = bound(maximum)?;
    if !minimum.is_finite() || minimum < 0.0 || maximum.is_nan() || maximum < minimum {
        return Err(invalid("time_step_limit must be ordered and non-negative"));
    }
    let maximum = if maximum.is_infinite() {
        TimeStepBound::Unbounded
    } else {
        TimeStepBound::Finite(maximum)
    };
    Ok(TimeStepLimit { minimum, maximum })
}

/// Reads a bound written either as a JSON number or as the
/// `{"__float__": "Infinity"}` encoding used for non-finite values.
fn bound(value: &Value) -> Result<f64> {
    if let Some(number) = value.as_f64() {
        return Ok(number);
    }
    match value.get("__float__").and_then(Value::as_str) {
        Some("Infinity" | "inf") => Ok(f64::INFINITY),
        _ => Err(invalid("time_step_limit entries must be numbers or Infinity")),
    }
}

fn state_dtype(decoder: &Value) -> Result<StateSpaceStateDType> {
    match optional_string(decoder, "mamba_ssm_cache_dtype")?.as_deref() {
        None | Some("float32") => Ok(StateSpaceStateDType::Float32),
        Some("bfloat16") => Ok(StateSpaceStateDType::BFloat16),
        Some("float16") => Ok(StateSpaceStateDType::Float16),
        Some(other) => Err(invalid(format!("unsupported mamba_ssm_cache_dtype {other}"))),
    }
}

fn positive(decoder: &Value, key: &str) -> Result<usize> {
    let value = require_usize(decoder, &[key])?;
    if value == 0 {
        return Err(invalid(format!("{key} must be greater than zero")));
    }
    Ok(value)
}
