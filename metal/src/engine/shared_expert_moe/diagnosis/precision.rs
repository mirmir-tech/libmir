use std::io::Write;

use super::*;
use crate::engine::binding::BoundLinear;

impl SharedExpertMoe {
    pub(crate) fn probe_shared_projection_cost(
        &self,
        input: &Array,
        stream: &Stream,
    ) -> Result<()> {
        self.probe_shared_cost(input, stream, BoundLinear::probe_mxfp4_cost)
    }

    pub(crate) fn probe_shared_projection_native_cost(
        &self,
        input: &Array,
        stream: &Stream,
    ) -> Result<()> {
        self.probe_shared_cost(input, stream, BoundLinear::probe_mxfp4_native_cost)
    }

    fn probe_shared_cost(
        &self,
        input: &Array,
        stream: &Stream,
        probe: fn(&BoundLinear, &Array, &str, &Stream) -> Result<()>,
    ) -> Result<()> {
        let shape = input.shape()?;
        let sequence = 8;
        let input = input.slice(&[0, 0, 0], &[5, sequence, usize::try_from(shape[2])?], stream)?;
        let mut rows = Vec::new();
        for row in 0..5 {
            let input = input.slice(
                &[row, 0, 0],
                &[row + 1, sequence, usize::try_from(shape[2])?],
                stream,
            )?;
            rows.push(
                self.shared_gate
                    .forward(&input, stream)?
                    .silu_mul(&self.shared_up.forward(&input, stream)?, stream)?,
            );
        }
        let activation = Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0, stream)?;
        probe(&self.shared_down, &activation, "shared.down", stream)?;
        Ok(())
    }

    pub(crate) fn probe_shared_projection_accuracy(
        &self,
        input: &Array,
        stream: &Stream,
    ) -> Result<()> {
        let shape = input.shape()?;
        for sequence in [8, 128] {
            let input =
                input.slice(&[0, 0, 0], &[5, sequence, usize::try_from(shape[2])?], stream)?;
            self.shared_gate.probe_mxfp4_accuracy(&input, "shared.gate", stream)?;
            self.shared_up.probe_mxfp4_accuracy(&input, "shared.up", stream)?;
            let mut activated = Vec::new();
            for row in 0..5 {
                let input = input.slice(
                    &[row, 0, 0],
                    &[row + 1, sequence, usize::try_from(shape[2])?],
                    stream,
                )?;
                activated.push(
                    self.shared_gate
                        .forward(&input, stream)?
                        .silu_mul(&self.shared_up.forward(&input, stream)?, stream)?,
                );
            }
            let activated = Array::concatenate(&activated.iter().collect::<Vec<_>>(), 0, stream)?;
            self.shared_down.probe_mxfp4_accuracy(&activated, "shared.down", stream)?;
        }
        Ok(())
    }

    pub(crate) fn diagnose_effective_rows(&self, input: &Array, stream: &Stream) -> Result<()> {
        let shape = input.shape()?;
        let rows = usize::try_from(shape[0])?;
        let sequence = usize::try_from(shape[1])?;
        let width = usize::try_from(shape[2])?;
        let packed = self.effective_components(
            &input.reshape(&[1, shape[0] * shape[1], shape[2]], stream)?,
            stream,
        )?;
        let scalar = (0..rows)
            .map(|row| {
                self.effective_components(
                    &input.slice(&[row, 0, 0], &[row + 1, sequence, width], stream)?,
                    stream,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        for (component, label) in [
            "indices",
            "weights",
            "routed",
            "shared",
            "shared.gate",
            "shared.up",
            "shared.activation",
            "shared.down",
            "shared.output_gate",
        ]
        .into_iter()
        .enumerate()
        {
            let a = scalar
                .iter()
                .map(|row| row[component].to_vec_f32(stream))
                .collect::<Result<Vec<_>>>()?
                .concat();
            let b = packed[component].to_vec_f32(stream)?;
            assert_eq!(a.len(), b.len());
            assert!(a.iter().chain(&b).all(|v| v.is_finite()));
            writeln!(
                std::io::stderr().lock(),
                "moe.effective_rows: {}",
                serde_json::json!({
                    "component":label,"elements":a.len(),
                    "differing":a.iter().zip(&b).filter(|(a,b)| a.to_bits()!=b.to_bits()).count(),
                    "max_abs":a.iter().zip(&b).map(|(a,b)| (a-b).abs()).fold(0.0,f32::max)
                })
            )?;
        }
        Ok(())
    }

    fn effective_components(&self, input: &Array, stream: &Stream) -> Result<[Array; 9]> {
        let routing = self.router.route_unit(input, i32::try_from(self.config.top_k)?, stream)?;
        let routed = self.routed(input, &routing.indices, &routing.weights, stream)?;
        let shared = self.shared(input, stream)?;
        let gate = self.shared_gate.forward(input, stream)?;
        let up = self.shared_up.forward(input, stream)?;
        let activation = gate.silu_mul(&up, stream)?;
        let down = self.shared_down.forward(&activation, stream)?;
        let output_gate = self.shared_output_gate.forward(input, stream)?;
        let composed = output_gate.sigmoid_mul(&down, stream)?;
        assert_eq!(
            composed.to_vec_f32(stream)?,
            shared.to_vec_f32(stream)?,
            "probe must use the executing shared-expert plan"
        );
        Ok([
            routing.indices, routing.weights, routed, shared, gate, up, activation, down,
            output_gate,
        ])
    }
}
