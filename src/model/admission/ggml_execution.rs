//! Backend-specific GGML execution constraints that depend on the decoder
//! architecture, not only on whether the backend has a kernel for a block type.

use foundation::model::BackendTarget;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    gguf::GgmlType,
    semantic::{ActivationSpec, FeedForwardSpec, RoutedExpertsSpec},
    weights::{LayerTensorRole, LogicalTensorRole, TensorStorage},
};

use super::{AdmissionCheck, AdmissionStatus, CheckSubject};
use crate::EncodingKind;

pub(super) fn assess(
    backend: &BackendTarget,
    execution: Option<&DecoderExecutionContract>,
    task: &TaskExecutionPlan,
) -> Option<AdmissionCheck> {
    let backend_name = match backend {
        BackendTarget::Metal => "Metal",
        BackendTarget::Cuda => "CUDA",
        BackendTarget::CpuReference => return None,
    };
    if !matches!(
        task,
        TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. }
    ) {
        return None;
    }
    let contract = execution?;
    let clamped_routed = contract.semantic.decoder.layers.iter().any(|layer| {
        matches!(
            &layer.feed_forward,
            FeedForwardSpec::Routed {
                routed: RoutedExpertsSpec {
                    activation: ActivationSpec::ClampedSwiGlu { .. },
                    ..
                },
                ..
            }
        )
    });
    if !clamped_routed {
        return None;
    }
    let non_mxfp4 = contract.bindings.tensors.iter().find_map(|binding| {
        matches!(
            binding.role,
            LogicalTensorRole::Layer {
                tensor: LayerTensorRole::ExpertProjection { .. },
                ..
            }
        )
        .then_some(&binding.storage)
        .and_then(|storage| match storage {
            TensorStorage::Ggml { format, .. } if format.ggml_type != GgmlType::MXFP4 => {
                Some(format.ggml_type)
            },
            _ => None,
        })
    })?;
    Some(AdmissionCheck {
        subject: CheckSubject::Encoding {
            kind: EncodingKind::Ggml,
            status: AdmissionStatus::Unsupported,
        },
        detail: format!(
            "{backend_name} clamped-routed GGML expert banks require MXFP4; found {}",
            non_mxfp4.name()
        ),
    })
}

#[cfg(test)]
mod tests {
    use foundation::model::BackendTarget;
    use models::{
        execution::{DecoderExecutionContract, TaskExecutionPlan},
        gguf::GgmlType,
        layout::DecoderConfig,
        semantic::SemanticModelSpec,
        weights::{
            ExpertProjectionRole, GgmlBlockQuantization, GgmlCompanion, LayerTensorRole,
            LogicalTensorRole, TensorBinding, TensorCatalog, TensorStorage, WeightBindingPlan,
        },
    };
    use serde_json::json;

    use super::assess;
    use crate::model::admission::AdmissionStatus;

    fn gpt_oss(
        expert_type: GgmlType,
    ) -> crate::Result<(TaskExecutionPlan, DecoderExecutionContract)> {
        let decoder = DecoderConfig::from_value(&json!({
            "model_type": "gpt_oss",
            "hidden_size": 64,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "max_position_embeddings": 256,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "swiglu_limit": 7.0
        }))?;
        let semantic = SemanticModelSpec::discover(&decoder, &TensorCatalog { tensors: vec![] })?;
        let bindings = WeightBindingPlan {
            tensors: vec![TensorBinding {
                role: LogicalTensorRole::Layer {
                    index: 0,
                    tensor: LayerTensorRole::ExpertProjection {
                        expert: None,
                        projection: ExpertProjectionRole::Gate,
                    },
                },
                source: "blk.0.ffn_gate_exps.weight".into(),
                shape: vec![4, 32, 64],
                logical_shape: None,
                transforms: Vec::new(),
                storage: TensorStorage::Ggml {
                    format: GgmlBlockQuantization { ggml_type: expert_type },
                    companion: GgmlCompanion::Bias("blk.0.ffn_gate_exps.bias".into()),
                },
            }],
        };
        Ok((
            TaskExecutionPlan::Generation { decoder },
            DecoderExecutionContract { semantic, bindings },
        ))
    }

    #[test]
    fn clamped_routed_ggml_experts_admit_only_mxfp4_on_metal_and_cuda() -> crate::Result<()> {
        let (task, mxfp4) = gpt_oss(GgmlType::MXFP4)?;
        let (_, q4_k) = gpt_oss(GgmlType::Q4_K)?;
        for backend in [BackendTarget::Metal, BackendTarget::Cuda] {
            assert!(assess(&backend, Some(&mxfp4), &task).is_none(), "{backend:?}");
            let check = assess(&backend, Some(&q4_k), &task);
            assert!(
                check.as_ref().is_some_and(|c| c.status() == AdmissionStatus::Unsupported
                    && c.detail.contains("MXFP4")
                    && c.detail.contains("Q4_K")),
                "{backend:?}: {check:?}"
            );
        }
        Ok(())
    }
}
