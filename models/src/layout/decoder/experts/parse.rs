use std::num::NonZeroUsize;

use serde_json::Value;

use super::{ExpertGroupLimit, ExpertRouting, RoutedExpertsConfig};
use crate::{
    error::{ModelsError, Result},
    layout::decoder::{
        defaults::Defaults,
        parse::{invalid, optional_bool, optional_f64, optional_string, optional_usize},
    },
};

/// The `DeepSeek` expert layout, declared by `n_routed_experts` (or by
/// `num_experts` beside latent attention); other routed decoders keep their
/// own fields.
pub(in crate::layout::decoder) fn routed_experts(
    decoder: &Value,
    latent: bool,
    defaults: &Defaults,
) -> Result<Option<RoutedExpertsConfig>> {
    let declared = optional_usize(decoder, &["n_routed_experts"])?.is_some()
        || (latent && optional_usize(decoder, &["num_experts"])?.is_some());
    if !declared {
        return Ok(None);
    }
    if optional_usize(decoder, &["moe_layer_freq"])?.unwrap_or(1) != 1 {
        return Err(invalid("routed layers interleaved by moe_layer_freq are not supported"));
    }
    let routing = routing(decoder, defaults)?;
    // `noaux_tc` always ranks groups; softmax routers rank them only under
    // `group_limited_greedy`.
    let grouped = match routing {
        ExpertRouting::SigmoidWithCorrectionBias => true,
        ExpertRouting::Softmax { .. } => {
            optional_string(decoder, "topk_method")?.as_deref() == Some("group_limited_greedy")
        },
    };
    let groups = optional_usize(decoder, &["n_group", "num_expert_group"])?.unwrap_or(1);
    let group_limit = (grouped && groups > 1)
        .then(|| {
            Ok::<_, ModelsError>(ExpertGroupLimit {
                groups,
                selected: optional_usize(decoder, &["topk_group"])?.unwrap_or(1),
            })
        })
        .transpose()?;
    let output_scale = optional_f64(decoder, &["routed_scaling_factor"])?.unwrap_or(1.0);
    if !output_scale.is_finite() || output_scale <= 0.0 {
        return Err(invalid("routed_scaling_factor must be finite and greater than zero"));
    }
    Ok(Some(RoutedExpertsConfig {
        first_dense_layers: optional_usize(decoder, &["first_k_dense_replace"])?.unwrap_or(0),
        shared_experts: optional_usize(decoder, &["n_shared_experts", "num_shared_experts"])?
            .and_then(NonZeroUsize::new),
        routing,
        group_limit,
        output_scale,
    }))
}

fn routing(decoder: &Value, defaults: &Defaults) -> Result<ExpertRouting> {
    let scoring = match optional_string(decoder, "scoring_func")? {
        Some(scoring) => Some(scoring),
        None => optional_string(decoder, "moe_router_activation_func")?
            .or_else(|| defaults.expert_scoring.map(str::to_owned)),
    };
    let renormalize = match optional_bool(decoder, "norm_topk_prob")? {
        Some(renormalize) => renormalize,
        None => optional_bool(decoder, "moe_renormalize")?.unwrap_or(false),
    };
    match scoring.as_deref() {
        None | Some("softmax") => {
            let method = optional_string(decoder, "topk_method")?;
            if !matches!(method.as_deref(), None | Some("greedy" | "group_limited_greedy")) {
                return Err(invalid("softmax routing requires topk_method greedy"));
            }
            Ok(ExpertRouting::Softmax { renormalize })
        },
        Some("sigmoid") => {
            let method = optional_string(decoder, "topk_method")?;
            if !matches!(method.as_deref(), None | Some("noaux_tc")) {
                return Err(invalid("sigmoid routing requires topk_method noaux_tc"));
            }
            if !renormalize {
                return Err(invalid("sigmoid routing without renormalization is not supported"));
            }
            Ok(ExpertRouting::SigmoidWithCorrectionBias)
        },
        Some(other) => Err(invalid(format!("unsupported expert scoring function {other}"))),
    }
}
