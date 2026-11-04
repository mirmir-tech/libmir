use crate::weights::binding::{
    LayerTensorRole, LinearAttentionTensorRole, LogicalTensorRole, StateSpaceTensorRole,
    TensorBinding,
};

/// Accepts the `[channels, 1, kernel]` layout that checkpoints use for
/// depthwise convolutions in place of the logical shape.
pub(super) fn alternate(binding: &TensorBinding, expected: &[usize]) -> bool {
    if matches!(
        binding.role,
        LogicalTensorRole::Layer {
            tensor: LayerTensorRole::StateSpace {
                tensor: StateSpaceTensorRole::Convolution
            },
            ..
        }
    ) {
        return matches!(expected, [channels, kernel] if binding.shape == [*channels, 1, *kernel]);
    }
    matches!(
        binding.role,
        LogicalTensorRole::Layer {
            tensor: LayerTensorRole::LinearAttention {
                tensor: LinearAttentionTensorRole::Convolution
            },
            ..
        }
    ) && matches!(expected, [channels, kernel, 1] if binding.shape == [*channels, 1, *kernel])
}
