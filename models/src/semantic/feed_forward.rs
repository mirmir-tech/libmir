use serde::{Deserialize, Serialize};

use super::validation::{activation_spec, invalid, nonzero};
use crate::error::Result;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeedForwardSpec {
    Dense {
        intermediate_size: usize,
        activation: ActivationSpec,
    },
    Routed {
        routed: RoutedExpertsSpec,
        shared: Option<SharedExpertSpec>,
    },
    DenseAndRouted {
        dense_intermediate_size: usize,
        dense_activation: ActivationSpec,
        routed: RoutedExpertsSpec,
    },
    /// Routed experts operating in a projected latent space when
    /// `latent_size` is set, plus an optional shared expert at full width.
    /// Without a latent size the experts work at the model hidden width.
    LatentRouted {
        latent_size: Option<usize>,
        routed: RoutedExpertsSpec,
        shared: Option<SharedExpertSpec>,
        routed_scale: f64,
    },
    /// The layer has no feed-forward sub-block.
    None,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutedExpertsSpec {
    pub expert_count: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub activation: ActivationSpec,
    pub router_normalization: RouterNormalization,
    /// Experts the router may select only from its best-scoring groups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_limit: Option<ExpertGroupLimit>,
    /// Factor on the weighted sum of the selected experts' outputs.
    #[serde(default = "unit_scale")]
    pub output_scale: f64,
}

/// Group-limited expert selection: top-k experts from the best groups.
///
/// The experts form `groups` contiguous groups, and the top-k come only from
/// the `selected` best-scoring ones. A softmax router scores a group by its
/// best expert's probability, the sigmoid router by the sum of its two best
/// biased scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertGroupLimit {
    pub groups: usize,
    pub selected: usize,
}

impl ExpertGroupLimit {
    /// Experts in each group.
    #[must_use]
    pub const fn group_size(&self, expert_count: usize) -> usize {
        expert_count / self.groups
    }
}

impl RoutedExpertsSpec {
    pub(super) fn validate(&self) -> Result<()> {
        nonzero(self.expert_count)?;
        nonzero(self.top_k)?;
        nonzero(self.intermediate_size)?;
        if self.top_k > self.expert_count
            || !self.output_scale.is_finite()
            || self.output_scale <= 0.0
        {
            return Err(invalid("routed top-k exceeds expert count or output scale is invalid"));
        }
        activation_spec(&self.activation)?;
        let Some(limit) = self.group_limit else {
            return Ok(());
        };
        let size = if limit.groups > 1 && self.expert_count.is_multiple_of(limit.groups) {
            limit.group_size(self.expert_count)
        } else {
            0
        };
        let minimum_size = match self.router_normalization {
            RouterNormalization::SoftmaxTopK | RouterNormalization::SoftmaxProbabilityTopK => 1,
            // The two best biased scores rank a sigmoid router's groups.
            RouterNormalization::SigmoidBiasedTopK => 2,
            RouterNormalization::UnitTopK => {
                return Err(invalid("a unit top-k router does not limit expert groups"));
            },
        };
        if size < minimum_size
            || limit.selected == 0
            || limit.selected > limit.groups
            || self.top_k > limit.selected * size
        {
            return Err(invalid(
                "expert groups must split the experts evenly and hold the selected experts",
            ));
        }
        Ok(())
    }
}

const fn unit_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SharedExpertSpec {
    pub intermediate_size: usize,
    pub activation: ActivationSpec,
    pub gated_output: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActivationSpec {
    SwiGlu {
        alpha: f64,
        clamp: Option<f64>,
        up_shift: f64,
    },
    GeluTanh,
    NamedGated {
        name: String,
    },
    /// Ungated `relu(x)^2`.
    ReluSquared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterNormalization {
    SoftmaxTopK,
    /// Select by softmax probability over every expert and weigh by that
    /// probability, without renormalizing the selected weights.
    SoftmaxProbabilityTopK,
    UnitTopK,
    /// Select by `sigmoid(logit) + correction_bias`, weight by the
    /// renormalized `sigmoid(logit)` of the selected experts.
    SigmoidBiasedTopK,
}
