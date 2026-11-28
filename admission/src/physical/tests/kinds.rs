//! Every weight encoding names its payload-free kind.

use models::{
    gguf::GgmlType,
    weights::{
        AffineBits, AwqBits, AwqPacking, AwqQuantization, AwqScaleDType, AwqStorageDType,
        BitsAndBytes4BitQuantization, BitsAndBytes4BitType, BitsAndBytesComputeDType,
        BitsAndBytesStorageDType, BlockQuantization, Float8Format, Float8Quantization,
        GgmlBlockQuantization, GptqBits, GptqCheckpointFormat, GptqPacking, GptqQuantization,
        GptqScaleDType, GptqStorageDType,
    },
};

use super::{affine, packed_int8};
use crate::physical::{EncodingKind, WeightEncoding};

#[test]
fn every_encoding_names_its_kind() {
    let awq = AwqQuantization {
        bits: AwqBits::Four,
        group_size: 128,
        packing: AwqPacking::GemmOutputInterleaved,
        storage_dtype: AwqStorageDType::I32,
        scale_dtype: AwqScaleDType::F16,
        packed_zero_points: true,
    };
    let gptq = GptqQuantization {
        bits: GptqBits::Four,
        group_size: 128,
        packing: GptqPacking::InputLittleEndian,
        storage_dtype: GptqStorageDType::I32,
        scale_dtype: GptqScaleDType::F16,
        checkpoint_format: GptqCheckpointFormat::Gptq,
        symmetric: true,
        activation_order: false,
        packed_zero_points: true,
    };
    let bitsandbytes = BitsAndBytes4BitQuantization {
        quant_type: BitsAndBytes4BitType::Nf4,
        block_size: 64,
        compute_dtype: BitsAndBytesComputeDType::Bf16,
        storage_dtype: BitsAndBytesStorageDType::U8,
        nested_block_size: None,
    };
    let cases = [
        (
            WeightEncoding::Dense { dtype: models::weights::DenseDType::Bf16 },
            EncodingKind::Dense,
        ),
        (
            WeightEncoding::Affine { format: affine(AffineBits::Four) },
            EncodingKind::Affine,
        ),
        (
            WeightEncoding::PackedInteger { format: packed_int8() },
            EncodingKind::PackedInteger,
        ),
        (WeightEncoding::Awq { format: awq }, EncodingKind::Awq),
        (WeightEncoding::Gptq { format: gptq }, EncodingKind::Gptq),
        (
            WeightEncoding::BitsAndBytes4Bit { format: bitsandbytes },
            EncodingKind::BitsAndBytes4Bit,
        ),
        (
            WeightEncoding::Float8 {
                format: Float8Quantization::unscaled(Float8Format::E4M3),
            },
            EncodingKind::Float8,
        ),
        (WeightEncoding::MxFp4 { format: BlockQuantization::MXFP4 }, EncodingKind::MxFp4),
        (WeightEncoding::MxFp8 { format: BlockQuantization::MXFP8 }, EncodingKind::MxFp8),
        (WeightEncoding::NvFp4 { format: BlockQuantization::NVFP4 }, EncodingKind::NvFp4),
        (
            WeightEncoding::Ggml {
                format: GgmlBlockQuantization { ggml_type: GgmlType::Q4_K },
            },
            EncodingKind::Ggml,
        ),
    ];
    for (encoding, kind) in cases {
        assert_eq!(encoding.kind(), kind, "{encoding:?}");
    }
}
