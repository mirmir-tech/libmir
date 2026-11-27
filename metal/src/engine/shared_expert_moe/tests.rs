use models::semantic::{ActivationSpec, ExpertGroupLimit, RoutedExpertsSpec, RouterNormalization};

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
    assert_eq!(gate.to_vec_f32(&stream)?, [3.0, 8.0]);
    assert_eq!(up.to_vec_f32(&stream)?, [30.0, 80.0]);
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
        shared: Some(SharedExpert {
            gate: linear(&[64, 64], stream)?,
            up: linear(&[64, 64], stream)?,
            fused_gate_up: None,
            down: linear(&[64, 64], stream)?,
            output_gate: Some(linear(&[1, 64], stream)?),
            gate_up_mode: crate::FusionMode::Disabled,
        }),
        routing: ExpertRouter::Unit,
    };
    let input = Array::from_f32(&vec![0.0; 64], &[1, 1, 64])?;
    let output = moe.forward(&input, stream)?;

    output.async_eval(stream)?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 1, 64]);
    assert!(output.to_vec_f32(stream)?.iter().all(|value| *value == 0.0));
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

/// The experts and weights a configured router selects, on the host.
#[test]
fn configured_routers_select_from_the_best_groups() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let logits = [0.3, -1.2, 2.0, 0.1, -0.4, 1.5, 0.9, -2.0];
    let bias = [0.0, 0.5, -0.2, 0.0, 0.3, 0.0, 0.0, 1.0];
    let cases = [
        (
            ScoredRouting::SigmoidWithCorrectionBias,
            RouterNormalization::SigmoidBiasedTopK,
            4,
            2,
            2.5,
        ),
        (
            ScoredRouting::Softmax { renormalize: false },
            RouterNormalization::SoftmaxProbabilityTopK,
            2,
            1,
            1.0,
        ),
    ];
    for (routing, normalization, groups, selected, output_scale) in cases {
        let biased = routing == ScoredRouting::SigmoidWithCorrectionBias;
        let router = ScoredRouter {
            routing,
            correction_bias: biased.then(|| Array::from_f32(&bias, &[8])).transpose()?,
            group_limit: Some(ExpertGroupLimit { groups, selected }),
            output_scale,
        };
        let output = router.route(&Array::from_f32(&logits, &[1, 1, 8])?, 2, &stream)?;
        let indices = output.indices.to_vec_u32(&stream)?;
        let weights = output.weights.to_vec_f32(&stream)?;
        let mut actual = indices
            .iter()
            .zip(&weights)
            .map(|(&index, &weight)| (usize::try_from(index).unwrap_or(0), weight))
            .collect::<Vec<_>>();
        actual.sort_by_key(|pair| pair.0);
        // The shared host reference of the same router.
        let spec = RoutedExpertsSpec {
            expert_count: 8,
            top_k: 2,
            intermediate_size: 1,
            activation: ActivationSpec::SwiGlu,
            router_normalization: normalization,
            group_limit: Some(ExpertGroupLimit { groups, selected }),
            output_scale: f64::from(output_scale),
        };
        let mut expected = spec
            .route(&logits, biased.then_some(&bias[..]))?
            .iter()
            .map(|routed| (routed.expert, routed.weight))
            .collect::<Vec<_>>();
        expected.sort_by_key(|pair| pair.0);
        assert_eq!(
            actual.iter().map(|pair| pair.0).collect::<Vec<_>>(),
            expected.iter().map(|pair| pair.0).collect::<Vec<_>>(),
            "{routing:?}"
        );
        for ((_, actual), (_, expected)) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1.0e-5, "{routing:?}: {actual} != {expected}");
        }
    }
    Ok(())
}
