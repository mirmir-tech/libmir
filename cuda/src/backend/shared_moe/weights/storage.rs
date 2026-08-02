use models::weights::{BlockQuantization, RoutedExpertBindings, TensorBinding, TensorStorage};

pub(super) fn routed_is_dense(bindings: RoutedExpertBindings<'_>) -> bool {
    all(bindings, |binding| matches!(binding.storage, TensorStorage::Dense { .. }))
}

pub(super) fn routed_is_mxfp4(bindings: RoutedExpertBindings<'_>) -> bool {
    let RoutedExpertBindings::SeparateGateUp { gate, up, down } = bindings else {
        return false;
    };
    [gate, up, down].iter().all(|binding| {
        matches!(
            binding.storage,
            TensorStorage::BlockQuantized { format, .. } if format.is_mxfp4()
        )
    })
}

pub(super) fn routed_is_mxfp8(bindings: RoutedExpertBindings<'_>) -> bool {
    all(bindings, |binding| {
        matches!(
            binding.storage,
            TensorStorage::BlockQuantized { format, .. } if format == BlockQuantization::MXFP8
        )
    })
}

fn all(bindings: RoutedExpertBindings<'_>, predicate: impl Fn(&TensorBinding) -> bool) -> bool {
    match bindings {
        RoutedExpertBindings::SeparateGateUp { gate, up, down } => {
            [gate, up, down].into_iter().all(predicate)
        },
        RoutedExpertBindings::InterleavedGateUp { gate_up, down } => {
            [gate_up, down].into_iter().all(predicate)
        },
        RoutedExpertBindings::Individual { .. } => false,
    }
}
