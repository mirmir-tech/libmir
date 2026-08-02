use super::*;
use crate::engine::{DenseLinear, QuantizedArrays, QuantizedLinear};

#[test]
fn executes_routed_and_shared_experts_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for fused in [false, true] {
        execute(fused, &stream)?;
    }
    Ok(())
}

#[test]
fn splits_interleaved_fused_expert_projection() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let weight = Array::from_f32(
        &[1.0, 0.0, 10.0, 0.0, 0.0, 2.0, 0.0, 20.0, 2.0, 0.0, 20.0, 0.0, 0.0, 3.0, 0.0, 30.0],
        &[2, 4, 2],
    )?;
    let gate_up = RoutedGateUp::Fused {
        projection: BoundLinear::Dense(DenseLinear::from_arrays(&weight, None, &stream)?),
        width: 2,
        interleaved: true,
    };
    let input = Array::from_f32(&[3.0, 4.0], &[1, 1, 2])?;
    let indices = Array::from_u32(&[0], &[1])?;

    let (gate, up) = gate_up.gather(&input, &indices, false, &stream)?;
    assert_eq!(gate.to_vec_f32_on_stream(&stream)?, [3.0, 8.0]);
    assert_eq!(up.to_vec_f32_on_stream(&stream)?, [30.0, 80.0]);
    Ok(())
}

fn execute(fused: bool, stream: &Stream) -> Result<()> {
    let routed_gate_up = if fused {
        RoutedGateUp::Fused {
            projection: linear(&[2, 128, 64], stream)?,
            width: 64,
            interleaved: false,
        }
    } else {
        RoutedGateUp::Separate {
            gate: linear(&[2, 64, 64], stream)?,
            up: linear(&[2, 64, 64], stream)?,
            fused: None,
        }
    };
    let moe = SharedExpertMoe {
        config: SharedExpertMoeConfig::new(2, 1, 64)?,
        router: linear(&[2, 64], stream)?,
        routed_gate_up,
        routed_down: linear(&[2, 64, 64], stream)?,
        shared_gate: linear(&[64, 64], stream)?,
        shared_up: linear(&[64, 64], stream)?,
        fused_shared_gate_up: None,
        shared_down: linear(&[64, 64], stream)?,
        shared_output_gate: linear(&[1, 64], stream)?,
        fuse_shared_gate_up: false,
    };
    let input = Array::from_f32(&vec![0.0; 64], &[1, 1, 64])?;
    let output = moe.forward(&input, stream)?;

    output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 1, 64]);
    assert!(output.to_vec_f32()?.iter().all(|value| *value == 0.0));
    Ok(())
}

fn linear(shape: &[i32], stream: &Stream) -> Result<BoundLinear> {
    let elements = shape.iter().try_fold(1_usize, |total, dimension| {
        total.checked_mul(usize::try_from(*dimension)?).ok_or(Error::ShapeOverflow)
    })?;
    let dense = Array::from_f32(&vec![0.0; elements], shape)?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(BoundLinear::Affine(QuantizedLinear::from_quantized(arrays, 64, 4)))
}
