use models::gguf::GgmlType;

use super::{
    super::tests::{blocks, weight},
    AffineLayout,
};
use crate::engine::{Array, Result, Stream, kernels::ggml::GgmlMatrix};

/// With float32 scales the repacked groups reproduce the GGML decoder's
/// values exactly: the packing, offsets and minima carry no rounding.
#[test]
fn repacked_groups_dequantize_to_the_ggml_values() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let (rows, columns) = (24, 512);
    for ggml_type in [
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q4_K,
        GgmlType::Q5_K,
    ] {
        let layout = ggml_type.affine32().ok_or(crate::engine::Error::ShapeOverflow)?;
        let layout = AffineLayout(layout);
        let data = blocks(ggml_type, rows, columns, 11)?;
        let weight = weight(&data, rows, &stream)?;
        let matrix = GgmlMatrix::new(ggml_type, columns, &weight)?;
        let expected = stream
            .kernels()
            .ggml_dequantize(&stream, &weight, matrix, mirtal::DType::Float32)?
            .to_vec_f32(&stream)?;
        let [packed, scales, biases] = stream.kernels().ggml_affine(
            &stream,
            &weight,
            matrix,
            layout,
            mirtal::DType::Float32,
        )?;
        let quantized = mirtal::Quantized {
            weight: packed.native(),
            scales: scales.native(),
            biases: biases.native(),
            format: mirtal::Quantization::new(AffineLayout::GROUP_SIZE, layout.bits())?,
        };
        let actual = Array::from_native(stream.native().graph().dequantize(quantized)?)?
            .to_vec_f32(&stream)?;
        assert_eq!(actual.len(), expected.len(), "{ggml_type}");
        let worst = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs() / b.abs().max(1.0e-3))
            .fold(0.0_f32, f32::max);
        assert!(worst <= 1.0e-6, "{ggml_type}: relative error {worst}");
    }
    Ok(())
}

#[test]
fn finer_scales_and_wider_values_keep_their_ggml_layout() {
    for ggml_type in [
        GgmlType::Q2_K,
        GgmlType::Q3_K,
        GgmlType::Q6_K,
        GgmlType::IQ4_NL,
        GgmlType::Q5_K,
        GgmlType::Q8_0,
    ] {
        assert_eq!(AffineLayout::of(ggml_type), None, "{ggml_type}");
    }
}
