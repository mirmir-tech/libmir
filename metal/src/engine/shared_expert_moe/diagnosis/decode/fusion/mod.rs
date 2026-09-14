mod drift;
mod measurement;

use std::io::Write;

use super::*;
use crate::engine::{Error, RouterOutput, shared_expert_moe::RoutedGateUp};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Variant {
    Separate,
    Fused,
}

#[derive(Clone, Copy)]
enum Pattern<'a> {
    Actual,
    HotSet(&'a Array),
}

impl serde::Serialize for Pattern<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(match self {
            Self::Actual => "actual",
            Self::HotSet(_) => "hot_set",
        })
    }
}

impl<'a> Pattern<'a> {
    fn indices(self, actual: &'a Array) -> &'a Array {
        match self {
            Self::Actual => actual,
            Self::HotSet(indices) => indices,
        }
    }
}

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum ParityStage {
    Gate,
    Up,
    WholeMoe,
}

impl SharedExpertMoe {
    pub(super) fn compare_decode_fusion(
        &self,
        input: &Array,
        case: Case,
        stream: &Stream,
    ) -> Result<()> {
        let RoutedGateUp::Separate { gate, up, fused: None } = &self.routed_gate_up else {
            return Err(Error::InvalidModel(
                "decode fusion requires unfused separate banks".into(),
            ));
        };
        let before = crate::engine::memory_stats()?;
        let (projection, width) =
            gate.fuse_mxfp4_expert_projection(up, stream)?.ok_or_else(|| {
                Error::InvalidModel("decode fusion requires compatible MXFP4 banks".into())
            })?;
        let fused = RoutedGateUp::Fused { projection, width, interleaved: false };
        let after = crate::engine::memory_stats()?;
        writeln!(
            std::io::stderr().lock(),
            "moe.decode_fusion_memory: {}",
            serde_json::json!({
                "case": case, "additional_active_bytes": after.active.saturating_sub(before.active),
            })
        )?;
        let routing = self.diagnostic_routing(input, stream)?;
        stream.eval_many(&[&routing.indices, &routing.weights])?;
        stream.synchronize()?;
        let hot_ids = (0..5)
            .flat_map(|row| (0..8).map(move |choice| (row + choice) % 8))
            .collect::<Vec<u32>>();
        let hot = Array::from_u32(&hot_ids, &[5, 1, 8])?;
        for pattern in [Pattern::Actual, Pattern::HotSet(&hot)] {
            let indices = pattern.indices(&routing.indices);
            let probe = Probe {
                model: self,
                fused: &fused,
                input,
                case,
                pattern,
                stream,
            };
            let expanded = input.expand_dims(&[-2, -3], stream)?;
            let (a, b) = self.routed_gate_up.gather(&expanded, indices, false, stream)?;
            let (c, d) = fused.gather(&expanded, indices, false, stream)?;
            let mut exact = true;
            for (stage, expected, actual) in [(ParityStage::Gate, a, c), (ParityStage::Up, b, d)] {
                exact &= probe.parity(
                    stage,
                    &actual.to_vec_f32(stream)?,
                    &expected.to_vec_f32(stream)?,
                )?;
            }
            let reference = probe.forward(Variant::Separate)?.to_vec_f32(stream)?;
            let actual = probe.forward(Variant::Fused)?.to_vec_f32(stream)?;
            exact &= probe.parity(ParityStage::WholeMoe, &actual, &reference)?;
            if exact {
                probe.measure()?;
            }
        }
        Ok(())
    }
}

struct Probe<'a> {
    model: &'a SharedExpertMoe,
    fused: &'a RoutedGateUp,
    input: &'a Array,
    case: Case,
    pattern: Pattern<'a>,
    stream: &'a Stream,
}

impl Probe<'_> {
    fn forward(&self, variant: Variant) -> Result<Array> {
        let RouterOutput { indices, weights } =
            self.model.diagnostic_routing(self.input, self.stream)?;
        let indices = self.pattern.indices(&indices);
        let routed = match variant {
            Variant::Separate => self.model.routed(self.input, indices, &weights, self.stream)?,
            Variant::Fused => {
                let expanded = self.input.expand_dims(&[-2, -3], self.stream)?;
                let (gate, up) = self.fused.gather(&expanded, indices, false, self.stream)?;
                let activated = gate.silu_mul(&up, self.stream)?;
                self.model
                    .routed_down
                    .gather(&activated, indices, false, self.stream)?
                    .squeeze_axis(-2, self.stream)?
                    .weighted_sum(&weights, -2, self.stream)?
            },
        };
        routed.add(&self.model.shared(self.input, self.stream)?, self.stream)
    }

    fn parity(&self, stage: ParityStage, actual: &[f32], expected: &[f32]) -> Result<bool> {
        let difference = Difference::between(actual, expected);
        writeln!(
            std::io::stderr().lock(),
            "moe.decode_fusion_parity: {}",
            serde_json::json!({
                "case": self.case, "pattern": self.pattern, "stage": stage,
                "elements": actual.len(), "difference": difference,
            })
        )?;
        Ok(difference.differing == 0)
    }
}
