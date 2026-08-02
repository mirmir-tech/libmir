use models::weights::BitsAndBytes4BitQuantization;

use super::{AdmissionCheckKind, AdmissionStatus};

pub(super) fn assess(
    format: BitsAndBytes4BitQuantization,
) -> (AdmissionCheckKind, AdmissionStatus, String) {
    (
        AdmissionCheckKind::BitsAndBytes4Bit,
        if format.is_supported() {
            AdmissionStatus::Partial
        } else {
            AdmissionStatus::Unsupported
        },
        "bitsandbytes NF4/FP4 requires B64, BF16 compute, and optional nested B256 scales".into(),
    )
}
