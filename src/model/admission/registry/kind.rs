use super::{AdmissionCheckKind, WeightEncoding};

pub(super) const fn kind(encoding: &WeightEncoding) -> AdmissionCheckKind {
    match encoding {
        WeightEncoding::Dense { .. } => AdmissionCheckKind::Dense,
        WeightEncoding::Affine { .. } => AdmissionCheckKind::Affine,
        WeightEncoding::PackedInt8 { .. } => AdmissionCheckKind::PackedInt8,
        WeightEncoding::PackedInt4 { .. } => AdmissionCheckKind::PackedInt4,
        WeightEncoding::Awq { .. } => AdmissionCheckKind::Awq,
        WeightEncoding::Gptq { .. } => AdmissionCheckKind::Gptq,
        WeightEncoding::BitsAndBytes4Bit { .. } => AdmissionCheckKind::BitsAndBytes4Bit,
        WeightEncoding::Float8 { .. } => AdmissionCheckKind::Float8,
        WeightEncoding::MxFp4 { .. } => AdmissionCheckKind::MxFp4,
        WeightEncoding::MxFp8 { .. } => AdmissionCheckKind::MxFp8,
        WeightEncoding::NvFp4 { .. } => AdmissionCheckKind::NvFp4,
    }
}
