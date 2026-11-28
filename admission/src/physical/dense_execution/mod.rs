mod bias;
mod latent;
mod metal;
use std::collections::BTreeSet;

use foundation::model::BackendTarget;
use metal::metal_generation_supported;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    semantic::FeedForwardSpec,
    weights::{
        DenseDType, HybridMoeExpertBindings, LayerTensorRole, LogicalTensorRole,
        RoutedExpertBindings, TensorStorage,
    },
};

use super::{AdmissionCheck, AdmissionStatus, CheckSubject, EncodingKind};

pub(super) fn assess(
    backend: BackendTarget,
    execution: Option<&DecoderExecutionContract>,
    task: &TaskExecutionPlan,
) -> Option<AdmissionCheck> {
    let dtypes = dense_dtypes(execution, task);
    if dtypes.is_empty() {
        return None;
    }
    let actual = dtypes.iter().map(|dtype| dtype.as_str()).collect::<Vec<_>>().join(", ");
    let (status, detail) = match backend {
        BackendTarget::Metal => metal(execution, task, &dtypes, &actual),
        BackendTarget::Cuda => cuda(execution, task, &dtypes, &actual),
    };
    Some(AdmissionCheck {
        subject: CheckSubject::Encoding { kind: EncodingKind::Dense, status },
        detail,
    })
}

fn dense_dtypes<'a>(
    execution: Option<&'a DecoderExecutionContract>,
    task: &'a TaskExecutionPlan,
) -> BTreeSet<&'a DenseDType> {
    let storage: Box<dyn Iterator<Item = &'a TensorStorage> + 'a> = match execution {
        Some(execution) => Box::new(execution.bindings.tensors.iter().map(|item| &item.storage)),
        None => match task {
            TaskExecutionPlan::SequenceScoring { bindings, .. } => {
                Box::new(bindings.tensors.iter().map(|item| &item.storage))
            },
            TaskExecutionPlan::Generation { .. }
            | TaskExecutionPlan::CausalScoring { .. }
            | TaskExecutionPlan::Embedding { .. } => Box::new(std::iter::empty()),
        },
    };
    storage
        .filter_map(|storage| match storage {
            TensorStorage::Dense { dtype, .. } => Some(dtype),
            _ => None,
        })
        .collect()
}

fn metal(
    execution: Option<&DecoderExecutionContract>,
    task: &TaskExecutionPlan,
    dtypes: &BTreeSet<&DenseDType>,
    actual: &str,
) -> (AdmissionStatus, String) {
    if !dtypes.iter().all(|dtype| dtype.is_float()) {
        return (
            AdmissionStatus::Unsupported,
            format!("Metal dense execution does not support storage dtype(s): {actual}"),
        );
    }

    if matches!(
        task,
        TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. }
    ) && !execution.is_some_and(|contract| metal_generation_supported(task, contract))
    {
        return (
            AdmissionStatus::Unsupported,
            "Metal does not yet support dense storage for this routed generation composition"
                .into(),
        );
    }

    (
        AdmissionStatus::Supported,
        format!("Metal preserves admitted dense storage on load ({actual})"),
    )
}

fn cuda(
    execution: Option<&DecoderExecutionContract>,
    task: &TaskExecutionPlan,
    dtypes: &BTreeSet<&DenseDType>,
    actual: &str,
) -> (AdmissionStatus, String) {
    if let Some(binding) = bias::unapplied(execution, task) {
        return (
            AdmissionStatus::Unsupported,
            format!(
                "CUDA adds projection biases only to GGML dense layers; {} has one",
                binding.source
            ),
        );
    }
    if execution.is_some_and(|contract| {
        has_dense_routed_experts(contract)
            && !supports_dense_selected_experts(contract)
            && !latent::supported(contract)
    }) {
        return (
            AdmissionStatus::Unsupported,
            "CUDA dense selected-expert execution does not support this routed composition".into(),
        );
    }
    let (required, path) = match task {
        TaskExecutionPlan::SequenceScoring { .. } => (Some(DenseDType::F16), "sequence scoring"),
        TaskExecutionPlan::Generation { .. }
        | TaskExecutionPlan::CausalScoring { .. }
        | TaskExecutionPlan::Embedding { .. } => {
            let convertible =
                matches!(task, TaskExecutionPlan::Embedding { .. }) || execution.is_some();
            if convertible && dtypes.iter().all(|dtype| dtype.is_float()) {
                let path = if matches!(
                    task,
                    TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. }
                ) {
                    "generation"
                } else {
                    "text embedding"
                };
                return (
                    AdmissionStatus::Supported,
                    format!(
                        "CUDA {path} persistently converts dense storage to BF16 on load ({actual})"
                    ),
                );
            }
            if dtypes.len() == 1 && dtypes.contains(&DenseDType::Bf16) {
                return (
                    AdmissionStatus::Supported,
                    "CUDA generation kernels consume dense BF16 storage directly".into(),
                );
            }
            (None, "generation or text embedding")
        },
    };
    if let Some(expected) = &required
        && dtypes.len() == 1
        && dtypes.contains(expected)
    {
        (
            AdmissionStatus::Supported,
            format!("CUDA {path} kernels consume dense {expected} storage directly"),
        )
    } else {
        let required_label = required.as_ref().map_or("BF16, F16, or F32", DenseDType::as_str);
        (
            AdmissionStatus::Unsupported,
            format!(
                "CUDA {path} requires dense {required_label} storage; checkpoint bindings use {actual}"
            ),
        )
    }
}

fn supports_dense_selected_experts(contract: &DecoderExecutionContract) -> bool {
    contract.semantic.decoder.layers.iter().all(|layer| match &layer.feed_forward {
        FeedForwardSpec::Routed {
            routed:
                models::semantic::RoutedExpertsSpec {
                    activation: models::semantic::ActivationSpec::ClampedSwiGlu { .. },
                    group_limit: None,
                    ..
                },
            shared: None,
        } => contract.bindings.routed_decoder_layer(layer.index).is_ok(),
        FeedForwardSpec::DenseAndRouted {
            dense_activation: models::semantic::ActivationSpec::GeluTanh,
            routed:
                models::semantic::RoutedExpertsSpec {
                    activation: models::semantic::ActivationSpec::GeluTanh,
                    router_normalization: models::semantic::RouterNormalization::SoftmaxTopK,
                    group_limit: None,
                    ..
                },
            ..
        } => contract.bindings.hybrid_moe_layer(layer.index).is_ok_and(|bindings| {
            matches!(
                bindings.experts,
                HybridMoeExpertBindings::Stacked(_) | HybridMoeExpertBindings::FusedStacked { .. }
            )
        }),
        FeedForwardSpec::Routed {
            routed:
                models::semantic::RoutedExpertsSpec {
                    activation: models::semantic::ActivationSpec::SwiGlu,
                    router_normalization: models::semantic::RouterNormalization::SoftmaxTopK,
                    group_limit: None,
                    ..
                },
            shared: Some(shared),
        } if matches!(shared.activation, models::semantic::ActivationSpec::SwiGlu)
            && shared.gated_output =>
        {
            contract
                .bindings
                .hybrid_decoder_layer(layer.index)
                .is_ok_and(|bindings| dense_routed(bindings.feed_forward.routed))
        },
        _ => false,
    })
}

fn dense_routed(bindings: RoutedExpertBindings<'_>) -> bool {
    match bindings {
        RoutedExpertBindings::InterleavedGateUp { gate_up, down } => [gate_up, down]
            .iter()
            .all(|binding| matches!(binding.storage, TensorStorage::Dense { .. })),
        RoutedExpertBindings::SeparateGateUp { gate, up, down } => [gate, up, down]
            .iter()
            .all(|binding| matches!(binding.storage, TensorStorage::Dense { .. })),
        RoutedExpertBindings::UngatedStacked { .. } | RoutedExpertBindings::Individual { .. } => {
            false
        },
    }
}

fn has_dense_routed_experts(contract: &DecoderExecutionContract) -> bool {
    contract.bindings.tensors.iter().any(|binding| {
        matches!(
            binding.role,
            LogicalTensorRole::Layer {
                tensor: LayerTensorRole::ExpertProjection { .. },
                ..
            }
        ) && matches!(binding.storage, TensorStorage::Dense { .. })
    })
}
