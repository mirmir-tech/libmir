use super::*;
use crate::{
    config::RouterPrecision,
    engine::{DenseLinear, QuantizedLinear},
};

#[test]
fn precise_router_preserves_bias_topk_and_model_weight_dtype() -> Result<()> {
    let mut config = crate::MetalConfig::default();
    config.diagnostics.router_precision = RouterPrecision::Float32;
    let stream = Stream::new_gpu_with_config(std::sync::Arc::new(config))?;
    let bias = Array::from_f32(&[2.0, -2.0, 1.0, -1.0], &[4])?;
    let zeros = Array::from_f32(&[0.0; 256], &[4, 64])?;
    let dense = BoundLinear::Dense(DenseLinear::from_arrays(&zeros, Some(bias), &stream)?);
    let weights = (0..4u16)
        .flat_map(|row| vec![f32::from(4 - row) * 0.02; 64])
        .collect::<Vec<_>>();
    let weights = Array::from_f32(&weights, &[4, 64])?;
    let affine = BoundLinear::Affine(QuantizedLinear::from_quantized(
        weights.quantize(64, 8, &stream)?,
        64,
        8,
    ));
    for (router, expected) in [(dense, [0, 2]), (affine, [0, 1])] {
        for rows in [1, 5] {
            let input = Array::from_f32(&vec![1.0; rows * 8 * 64], &[i32::try_from(rows)?, 8, 64])?;
            let input = Array::from_native(
                stream.native().graph().astype(input.native(), mirtal::DType::Bfloat16)?,
            )?;
            let routing = router.route_unit(&input, 2, &stream)?;
            assert_eq!(routing.weights.dtype()?, input.dtype()?);
            let ids = routing.indices.to_vec_u32(&stream)?;
            let weights = routing.weights.to_vec_f32(&stream)?;
            assert_eq!(ids.len(), rows * 8 * 2);
            assert_eq!(weights.len(), ids.len());
            for row in ids.as_chunks::<2>().0 {
                let mut row = row.to_vec();
                row.sort_unstable();
                assert_eq!(row, expected);
            }
            for row in weights.as_chunks::<2>().0 {
                assert!(row.iter().all(|value| value.is_finite() && *value > 0.0));
                assert!((row.iter().sum::<f32>() - 1.0).abs() < 0.01);
            }
        }
    }
    Ok(())
}
