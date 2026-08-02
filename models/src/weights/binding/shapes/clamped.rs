use super::{LayerTensorRole, LogicalTensorRole, SemanticModelSpec, routed};
use crate::semantic::ActivationSpec;

pub(super) fn dense_expert(
    spec: &SemanticModelSpec,
    role: &LogicalTensorRole,
    source: &str,
) -> bool {
    let LogicalTensorRole::Layer {
        index,
        tensor: LayerTensorRole::ExpertProjection { expert: None, .. },
    } = role
    else {
        return false;
    };
    let Some(routed) =
        spec.decoder.layers.get(*index).and_then(|layer| routed(&layer.feed_forward))
    else {
        return false;
    };
    matches!(routed.activation, ActivationSpec::SwiGlu { clamp: Some(_), .. })
        && (source.ends_with("mlp.experts.gate_up_proj")
            || source.ends_with("mlp.experts.down_proj"))
}
