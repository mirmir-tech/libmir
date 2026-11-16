use foundation::model::BackendTarget;
use models::weights::GgmlBlockQuantization;

use super::{AdmissionCheckKind, AdmissionStatus};

/// GGML types whose kernels a backend executes; every other GGML type is
/// reported unsupported with its name.
const CUDA: &[&str] = &[
    "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_NL",
    "IQ4_XS", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M", "MXFP4",
];
const METAL: &[&str] = &[];

pub(super) fn assess(
    backend: &BackendTarget,
    format: GgmlBlockQuantization,
) -> (AdmissionCheckKind, AdmissionStatus, String) {
    let (backend, executed) = match backend {
        BackendTarget::Cuda => ("CUDA", CUDA),
        BackendTarget::Metal => ("Metal", METAL),
        BackendTarget::CpuReference => ("The CPU reference backend", &[][..]),
    };
    let name = format.ggml_type.name();
    if executed.contains(&name) {
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
