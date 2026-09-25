mod metal;
use std::collections::BTreeSet;

use foundation::model::BackendTarget;
use metal::metal_generation_supported;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    semantic::FeedForwardSpec,
    weights::{
        HybridMoeExpertBindings, LayerTensorRole, LogicalTensorRole, RoutedExpertBindings,
        TensorStorage,
    },
};

use super::{AdmissionCheck, AdmissionCheckKind, AdmissionStatus};

pub(super) fn assess(
    backend: &BackendTarget,
    execution: Option<&DecoderExecutionContract>,
    task: &TaskExecutionPlan,
) -> Option<AdmissionCheck> {
    let dtypes = dense_dtypes(execution, task);
    if dtypes.is_empty() {
        return None;
    }
    let actual = dtypes.iter().copied().collect::<Vec<_>>().join(", ");
    let (status, detail) = match backend {
        BackendTarget::Metal => metal(execution, task, &dtypes, &actual),
        BackendTarget::Cuda => cuda(execution, task, &dtypes, &actual),
        BackendTarget::CpuReference => (
            AdmissionStatus::Unsupported,
            "the CPU reference backend does not execute dense model tasks".into(),
        ),
    };
    Some(AdmissionCheck {
        kind: AdmissionCheckKind::Dense,
        status,
        detail,
    })
}

fn dense_dtypes<'a>(
    execution: Option<&'a DecoderExecutionContract>,
    task: &'a TaskExecutionPlan,
) -> BTreeSet<&'a str> {
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
            TensorStorage::Dense { dtype, .. } => Some(dtype.as_str()),
            _ => None,
        })
        .collect()
}

fn metal(
    execution: Option<&DecoderExecutionContract>,
    task: &TaskExecutionPlan,
    dtypes: &BTreeSet<&str>,
    actual: &str,
) -> (AdmissionStatus, String) {
    if !dtypes.iter().all(|dtype| matches!(*dtype, "BF16" | "F16" | "F32")) {
        return (
            AdmissionStatus::Unsupported,
            format!("Metal dense execution does not support storage dtype(s): {actual}"),
        );
    }

    if matches!(
        task,
        TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. }
    ) && !execution.is_some_and(metal_generation_supported)
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
    dtypes: &BTreeSet<&str>,
    actual: &str,
) -> (AdmissionStatus, String) {
    if execution.is_some_and(|contract| {
        has_dense_routed_experts(contract) && !supports_dense_selected_experts(contract)
    }) {
        return (
            AdmissionStatus::Unsupported,
            "CUDA dense selected-expert execution does not support this routed composition".into(),
        );
    }
    let (required, path) = match task {
        TaskExecutionPlan::SequenceScoring { .. } => ("F16", "sequence scoring"),
        TaskExecutionPlan::Generation { .. }
        | TaskExecutionPlan::CausalScoring { .. }
        | TaskExecutionPlan::Embedding { .. } => {
            let convertible =
                matches!(task, TaskExecutionPlan::Embedding { .. }) || execution.is_some();
            if convertible && dtypes.iter().all(|dtype| matches!(*dtype, "BF16" | "F16" | "F32")) {
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
            if dtypes.len() == 1 && dtypes.contains("BF16") {
                return (
                    AdmissionStatus::Supported,
                    "CUDA generation kernels consume dense BF16 storage directly".into(),
                );
            }
            ("BF16, F16, or F32", "generation or text embedding")
        },
    };
    if dtypes.len() == 1 && dtypes.contains(required) {
        (
            AdmissionStatus::Supported,
            format!("CUDA {path} kernels consume dense {required} storage directly"),
        )
    } else {
        (
            AdmissionStatus::Unsupported,
            format!(
                "CUDA {path} requires dense {required} storage; checkpoint bindings use {actual}"
            ),
        )
    }
}

fn supports_dense_selected_experts(contract: &DecoderExecutionContract) -> bool {
    contract.semantic.decoder.layers.iter().all(|layer| match &layer.feed_forward {
        FeedForwardSpec::Routed {
            routed:
                models::semantic::RoutedExpertsSpec {
                    activation: models::semantic::ActivationSpec::SwiGlu { clamp: Some(_), .. },
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
                    activation: models::semantic::ActivationSpec::SwiGlu { clamp: None, .. },
                    router_normalization: models::semantic::RouterNormalization::SoftmaxTopK,
                    ..
                },
            shared: Some(shared),
        } if matches!(
            shared.activation,
            models::semantic::ActivationSpec::SwiGlu { clamp: None, .. }
        ) && shared.gated_output =>
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
        RoutedExpertBindings::Individual { .. } => false,
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
