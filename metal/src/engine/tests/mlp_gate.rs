//! The one gate mapping every plain gated MLP runs by: `SwiGLU` and tanh
//! GELU each run their own gate, and any other activation is refused rather
//! than run as one of them.

use models::semantic::ActivationSpec;

use crate::engine::{Array, Result, Stream, mlp_gate::GateActivation};

fn silu_reference(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn gelu_reference(x: f32) -> f32 {
    0.5 * x
        * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * 0.044_715f32.mul_add(x.powi(3), x)).tanh())
}

/// `spec`'s gate of `values` by 2 against `reference`.
fn check(spec: &ActivationSpec, reference: fn(f32) -> f32) -> Result<()> {
    let stream = Stream::new_gpu()?;
    let values = [-2.0_f32, -0.5, 0.0, 0.75, 3.0];
    let gate = Array::from_f32(&values, &[1, 5])?;
    let up = Array::from_f32(&[2.0; 5], &[1, 5])?;
    let output = GateActivation::from_spec(spec)?
        .gate(&gate, &up, &stream)?
        .to_vec_f32(&stream)?;
    for (value, actual) in values.iter().zip(output) {
        let expected = 2.0 * reference(*value);
        assert!((actual - expected).abs() < 1e-3, "{spec:?}({value}): {actual} != {expected}");
    }
    Ok(())
}

#[test]
fn each_gate_runs_its_own_activation() -> Result<()> {
    check(&ActivationSpec::SwiGlu, silu_reference)?;
    check(&ActivationSpec::GeluTanh, gelu_reference)?;
    assert!(GateActivation::from_spec(&ActivationSpec::ReluSquared).is_err());
    Ok(())
}
