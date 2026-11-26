//! Which projections a residual multiplier folds into.

use models::weights::{
    GgmlBlockQuantization, GgmlCompanion, LayerTensorRole, LogicalTensorRole, RoutedExpertBindings,
    SharedRoutedFeedForwardBindings, TensorBinding, TensorStorage,
};

use super::branch_ends;
use crate::checkpoint::dense::{DenseAttentionBindings, DenseLayerBindings};

fn binding(source: &str, storage: TensorStorage) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::InputNorm,
        },
        source: source.into(),
        shape: vec![4, 4],
        logical_shape: None,
        transforms: Vec::new(),
        storage,
    }
}

fn dense(source: &str) -> TensorBinding {
    binding(source, TensorStorage::Dense { dtype: "BF16".into(), bias: None })
}

/// A shared-routed layer ends its branches in the attention output, the
/// routed experts' down bank and the shared expert's down projection; the
/// gate and up projections and the router keep their scale.
#[test]
fn shared_routed_branches_end_in_every_down_projection() {
    let ggml = TensorStorage::Ggml {
        format: GgmlBlockQuantization { ggml_type: models::gguf::GgmlType::Q8_0 },
        companion: GgmlCompanion::Plain,
    };
    let [query, output, router, gate, up, shared_gate, shared_up, shared_down] =
        ["q", "o", "router", "gate", "up", "shared_gate", "shared_up", "shared_down"].map(dense);
    let down = binding("down", ggml);
    let bindings = DenseLayerBindings {
        input_norm: None,
        attention: DenseAttentionBindings::Shared {
            query: &query,
            output: &output,
            query_norm: None,
        },
        post_attention_norm: None,
        sandwich: None,
        feed_forward: models::weights::DenseLayerFeedForwardBindings::SharedRouted(
            SharedRoutedFeedForwardBindings {
                router: &router,
                router_correction_bias: None,
                routed: RoutedExpertBindings::SeparateGateUp { gate: &gate, up: &up, down: &down },
                shared_gate: &shared_gate,
                shared_up: &shared_up,
                shared_down: &shared_down,
                shared_output_gate: None,
            },
        ),
        per_layer_input: None,
    };
    let ends = branch_ends(bindings)
        .into_iter()
        .map(|end| end.source.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ends, ["o", "down", "shared_down"]);
}
