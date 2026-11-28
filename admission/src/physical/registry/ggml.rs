use foundation::model::BackendTarget;
use models::weights::GgmlBlockQuantization;

use super::AdmissionStatus;
use crate::physical::EncodingKind;

pub(super) fn assess(
    backend: BackendTarget,
    format: GgmlBlockQuantization,
) -> (EncodingKind, AdmissionStatus, String) {
    let (backend, executed) = match backend {
        BackendTarget::Cuda => ("CUDA", crate::ggml::CUDA),
        BackendTarget::Metal => ("Metal", crate::ggml::METAL),
    };
    let name = format.ggml_type.name();
    if executed.contains(&format.ggml_type) {
        (
            EncodingKind::Ggml,
            AdmissionStatus::Partial,
            format!("{backend} executes GGML {name} blocks in place"),
        )
    } else {
        (
            EncodingKind::Ggml,
            AdmissionStatus::Unsupported,
            format!("{backend} has no kernels for GGML {name} blocks yet"),
        )
    }
}
