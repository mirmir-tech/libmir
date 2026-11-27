//! Whether a backend's clamped-routed (GPT-OSS) loader assembles every
//! layer's routed expert banks: each layer's gate, up and down projections
//! must share one storage the loader reads, and on CUDA, whose loader picks
//! one layout from the whole plan, every layer the same layout. Metal binds
//! dense and block-quantized banks projection by projection, so it also
//! reads a layer mixing them.

use foundation::model::BackendTarget;
use models::{
    execution::{DecoderExecutionContract, TaskExecutionPlan},
    semantic::{ActivationSpec, FeedForwardSpec, RoutedExpertsSpec},
    weights::ExpertProjectionRole,
};

use super::{AdmissionCheck, AdmissionStatus, CheckSubject};
use crate::EncodingKind;

mod banks;
#[cfg(test)]
mod tests;

use banks::{BankRefusal, ExpertBanks, metal_composes};

pub(super) fn assess(
    backend: &BackendTarget,
    execution: Option<&DecoderExecutionContract>,
    task: &TaskExecutionPlan,
) -> Option<AdmissionCheck> {
    let (name, cuda) = match backend {
        BackendTarget::Metal => ("Metal", false),
        BackendTarget::Cuda => ("CUDA", true),
        BackendTarget::CpuReference => return None,
    };
    if !matches!(
        task,
        TaskExecutionPlan::Generation { .. } | TaskExecutionPlan::CausalScoring { .. }
    ) {
        return None;
    }
    let contract = execution?;
    let (kind, detail) = refusal(contract, cuda)?;
    Some(AdmissionCheck {
        subject: CheckSubject::Encoding {
            kind,
            status: AdmissionStatus::Unsupported,
        },
        detail: format!("{name} {detail}"),
    })
}

/// The first layer whose banks the backend's loader does not assemble.
fn refusal(contract: &DecoderExecutionContract, cuda: bool) -> Option<(EncodingKind, String)> {
    let mut selected: Option<(usize, ExpertBanks)> = None;
    for layer in contract
        .semantic
        .decoder
        .layers
        .iter()
        .filter(|layer| clamped(&layer.feed_forward))
    {
        let index = layer.index;
        // The binding grammar requires complete stacked banks in every
        // clamped-routed layer, so a plan without them was never discovered.
        let Ok(experts) = contract.bindings.routed_experts(index) else {
            continue;
        };
        let banks = match ExpertBanks::of(experts) {
            Ok(banks) => banks,
            Err(BankRefusal::Mixed(_)) if !cuda && metal_composes(experts) => continue,
            Err(refusal) => return Some((refusal.kind(), refused(index, &refusal))),
        };
        if !cuda {
            continue;
        }
        if banks.cuda_layout().is_none() {
            return Some((
                banks.kind(),
                format!(
                    "loads no clamped-routed experts stored as {}, as layer {index} is",
                    banks.label()
                ),
            ));
        }
        match selected {
            Some((first, chosen)) if chosen.cuda_layout() != banks.cuda_layout() => {
                return Some((
                    banks.kind(),
                    format!(
                        "loads every clamped-routed layer's experts in one layout; layer {first} \
                         has {} and layer {index} {}",
                        chosen.label(),
                        banks.label()
                    ),
                ));
            },
            Some(_) => {},
            None => selected = Some((index, banks)),
        }
    }
    None
}

fn refused(layer: usize, refusal: &BankRefusal) -> String {
    match refusal {
        BankRefusal::Mixed(encodings) => {
            let storages = encodings
                .iter()
                .map(|(role, encoding)| format!("{} {}", projection(*role), encoding.label()))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "clamped-routed layer {layer} mixes expert storages ({storages}); its loader reads \
                 a layer's gate, up and down banks in one storage"
            )
        },
        BankRefusal::GgmlType(ggml_type) => {
            format!("clamped-routed GGML expert banks require MXFP4; layer {layer} has {ggml_type}")
        },
        BankRefusal::Unbiased(encoding) => format!(
            "clamped-routed layer {layer}'s {} expert banks lack the output bias its loader adds",
            encoding.label()
        ),
        BankRefusal::Unread(encoding) => format!(
            "clamped-routed loader reads no {} expert banks in layer {layer}'s arrangement",
            encoding.label()
        ),
        BankRefusal::Unstacked(_) => format!(
            "clamped-routed loader requires gated stacked expert banks; layer {layer} binds others"
        ),
    }
}

const fn projection(role: ExpertProjectionRole) -> &'static str {
    match role {
        ExpertProjectionRole::Gate => "gate",
        ExpertProjectionRole::Up => "up",
        ExpertProjectionRole::GateUp => "gate/up",
        ExpertProjectionRole::Down => "down",
    }
}

const fn clamped(feed_forward: &FeedForwardSpec) -> bool {
    matches!(
        feed_forward,
        FeedForwardSpec::Routed {
            routed: RoutedExpertsSpec {
                activation: ActivationSpec::ClampedSwiGlu { .. },
                ..
            },
            ..
        }
    )
}
