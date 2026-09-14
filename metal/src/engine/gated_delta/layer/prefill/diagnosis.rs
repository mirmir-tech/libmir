use std::io::Write;

use super::{Array, GatedDeltaLayer, GatedDeltaState, Result, Stream};

impl GatedDeltaLayer {
    /// First-chunk probe only: compare actual checkpoint projections under the
    /// same normalized input. Explicit host reads are confined to this test.
    pub(crate) fn diagnose_packed_projection(&self, input: &Array, stream: &Stream) -> Result<()> {
        let shape = input.shape()?;
        let batch = usize::try_from(shape[0])?;
        let sequence = usize::try_from(shape[1])?;
        let hidden = usize::try_from(shape[2])?;
        for (name, projection) in [
            (Component::Qkv, &self.in_proj_qkv),
            (Component::Gate, &self.in_proj_z),
            (Component::Alpha, &self.in_proj_a),
            (Component::Beta, &self.in_proj_b),
        ] {
            let packed = projection.forward(input, stream)?;
            let mut rows = Vec::new();
            for row in 0..batch {
                let input = input.slice(&[row, 0, 0], &[row + 1, sequence, hidden], stream)?;
                rows.push(projection.forward(&input, stream)?);
            }
            let reference = Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, stream)?;
            report(name, &packed, &reference, stream)?;
            if matches!(name, Component::Gate) {
                let output = self.out_proj.forward(&packed, stream)?;
                let outputs = rows
                    .iter()
                    .map(|row| self.out_proj.forward(row, stream))
                    .collect::<Result<Vec<_>>>()?;
                let expected = Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 0, stream)?;
                report(Component::OutputProjection, &output, &expected, stream)?;
            }
            if matches!(name, Component::Qkv) {
                let output =
                    GatedDeltaState::new()?.convolve_silu(&packed, &self.conv_weight, stream)?;
                let outputs = rows
                    .iter()
                    .map(|row| {
                        GatedDeltaState::new()?.convolve_silu(row, &self.conv_weight, stream)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let expected = Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 0, stream)?;
                report(Component::Convolution, &output, &expected, stream)?;
            }
        }
        let mut states = (0..batch).map(|_| GatedDeltaState::new()).collect::<Result<Vec<_>>>()?;
        let packed = self
            .forward_packed_prefill(input, &mut states.iter_mut().collect::<Vec<_>>(), stream)?
            .ok_or_else(|| {
                super::Error::InvalidModel("projection probe cannot pack rows".into())
            })?;
        let mut outputs = Vec::new();
        let mut values = Vec::new();
        for row in 0..batch {
            let input = input.slice(&[row, 0, 0], &[row + 1, sequence, hidden], stream)?;
            let mut state = GatedDeltaState::new()?;
            outputs.push(self.forward(&input, &mut state, stream)?);
            values.push(state.values()?);
        }
        let expected = Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 0, stream)?;
        report(Component::Output, &packed, &expected, stream)?;
        let actual = states.iter().map(GatedDeltaState::values).collect::<Result<Vec<_>>>()?;
        let actual = Array::concatenate(&actual.iter().collect::<Vec<_>>(), 0, stream)?;
        let expected = Array::concatenate(&values.iter().collect::<Vec<_>>(), 0, stream)?;
        report(Component::State, &actual, &expected, stream)
    }
}

fn report(component: Component, actual: &Array, reference: &Array, stream: &Stream) -> Result<()> {
    assert_eq!(actual.shape()?, reference.shape()?);
    let a = actual.to_vec_f32(stream)?;
    let b = reference.to_vec_f32(stream)?;
    assert!(a.iter().chain(&b).all(|v| v.is_finite()));
    let max_abs = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
    let differing = a.iter().zip(&b).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    writeln!(
        std::io::stderr().lock(),
        "gdn.projection: {}",
        serde_json::json!({
            "component": component.name(), "shape": actual.shape()?, "differing": differing,
            "elements": a.len(), "max_abs": max_abs,
            "reference_max_abs": b.iter().map(|v| v.abs()).fold(0.0_f32, f32::max),
        })
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
enum Component {
    Qkv,
    Gate,
    Alpha,
    Beta,
    Convolution,
    Output,
    State,
    OutputProjection,
}
impl Component {
    const fn name(self) -> &'static str {
        match self {
            Self::Qkv => "qkv",
            Self::Gate => "z",
            Self::Alpha => "a",
            Self::Beta => "b",
            Self::Convolution => "convolution",
            Self::Output => "layer_output",
            Self::State => "recurrent_state",
            Self::OutputProjection => "output_projection_on_z",
        }
    }
}
