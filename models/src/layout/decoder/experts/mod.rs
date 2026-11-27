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
    /// Groups the router selects experts from, when more than one.
    pub group_limit: Option<ExpertGroupLimit>,
    /// Factor on the combined routed-expert output (`routed_scaling_factor`).
    pub output_scale: f64,
}
