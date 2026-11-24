use foundation::model::BackendTarget;
use models::weights::GgmlBlockQuantization;

use super::{AdmissionCheckKind, AdmissionStatus};

pub(super) fn assess(
    backend: &BackendTarget,
    format: GgmlBlockQuantization,
) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let (backend, executed) = match backend {
        BackendTarget::Cuda => ("CUDA", architecture::ggml::CUDA),
        BackendTarget::Metal => ("Metal", architecture::ggml::METAL),
        BackendTarget::CpuReference => ("The CPU reference backend", &[][..]),
    };
    let name = format.ggml_type.name();
    if executed.contains(&format.ggml_type) {
        (
            AdmissionCheckKind::Ggml,
            AdmissionStatus::Partial,
            format!("{backend} executes GGML {name} blocks in place"),
        )
    } else {
        (
            AdmissionCheckKind::Ggml,
            AdmissionStatus::Unsupported,
            format!("{backend} has no kernels for GGML {name} blocks yet"),
        )
    }
}
