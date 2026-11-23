//! The `DeepSeek` mixture-of-experts layout: leading dense layers, then routed
//! plus shared experts behind a configured router (`DeepSeek` V2/V3, Kimi,
//! GLM-4.5).

mod parse;

use std::num::NonZeroUsize;

pub(super) use parse::routed_experts;

use crate::semantic::ExpertGroupLimit;

/// Routed feed-forward with leading dense layers: the first layers stay
/// dense, the others combine routed experts with optional always-active
/// shared experts.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedExpertsConfig {
    pub first_dense_layers: usize,
    /// Shared experts, fused into one feed-forward of `count` times the expert
    /// width.
    pub shared_experts: Option<NonZeroUsize>,
    pub routing: ExpertRouting,
    /// Groups the router selects experts from, when more than one.
    pub group_limit: Option<ExpertGroupLimit>,
    /// Factor on the combined routed-expert output (`routed_scaling_factor`).
    pub output_scale: f64,
}

/// How a router scores experts and weighs the ones it selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertRouting {
    /// Select by softmax probability over every expert; `renormalize` makes
    /// the selected weights sum to one.
    Softmax { renormalize: bool },
    /// Select by `sigmoid(logit) + correction_bias`, weigh by the
    /// renormalized `sigmoid(logit)` of the selected experts.
    SigmoidWithCorrectionBias,
}
