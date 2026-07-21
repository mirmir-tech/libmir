use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutedExpertsSpec {
    pub expert_count: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    pub activation: ActivationSpec,
    pub router_normalization: RouterNormalization,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterNormalization {
    SoftmaxTopK,
    UnitTopK,
}
