use models::weights::{
    BlockQuantization, ExpertProjectionRole, RoutedExpertBindings, TensorBinding, TensorStorage,
};

/// Stacked or per-expert (`mlp.experts.N.*`) BF16 projections; the dense
/// expert loader stacks the per-expert ones.
pub(super) fn routed_is_dense(bindings: RoutedExpertBindings<'_>) -> bool {
    let dense = |binding: &TensorBinding| matches!(binding.storage, TensorStorage::Dense { .. });
    if let RoutedExpertBindings::Individual { .. } = bindings {
        return [ExpertProjectionRole::Gate, ExpertProjectionRole::Up, ExpertProjectionRole::Down]
            .into_iter()
            .all(|projection| {
                let experts = bindings.individual(projection);
                !experts.is_empty() && experts.into_iter().all(dense)
            });
    }
    all(bindings, dense)
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
        RoutedExpertBindings::UngatedStacked { .. } | RoutedExpertBindings::Individual { .. } => {
            false
        },
    }
}
