use super::*;
use crate::engine::QuantizedArrays;

#[test]
fn executes_routed_and_shared_experts_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let moe = SharedExpertMoe {
        config: SharedExpertMoeConfig::new(2, 1)?,
        router: linear(&[2, 64], &stream)?,
        routed_gate: linear(&[2, 64, 64], &stream)?,
        routed_up: linear(&[2, 64, 64], &stream)?,
        routed_down: linear(&[2, 64, 64], &stream)?,
        fused_routed_gate_up: None,
        shared_gate: linear(&[64, 64], &stream)?,
        shared_up: linear(&[64, 64], &stream)?,
        fused_shared_gate_up: None,
        shared_down: linear(&[64, 64], &stream)?,
        shared_output_gate: linear(&[1, 64], &stream)?,
        fuse_shared_gate_up: false,
    };
    let input = Array::from_f32(&vec![0.0; 64], &[1, 1, 64])?;
    let output = moe.forward(&input, &stream)?;

    output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 1, 64]);
    assert!(output.to_vec_f32()?.iter().all(|value| *value == 0.0));
    Ok(())
}

fn linear(shape: &[i32], stream: &Stream) -> Result<QuantizedLinear> {
    let elements = shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension)?).ok_or(Error::ShapeOverflow)
    })?;
    let dense = Array::from_f32(&vec![0.0; elements], shape)?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(QuantizedLinear::from_quantized(arrays, 64, 4))
}
