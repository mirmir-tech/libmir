use std::time::Instant;

use super::{Array, GatedDeltaLayer, GatedDeltaState, Result, Stream};

#[derive(Clone, Copy)]
pub(super) enum Variant {
    Rows,
    Packed,
}

pub(super) struct Outcome {
    pub output: Array,
    pub states: Vec<GatedDeltaState>,
    pub graph_ms: f64,
    pub total_ms: f64,
}

impl GatedDeltaLayer {
    pub(super) fn sample_prefill(
        &self,
        variant: Variant,
        input: &Array,
        original: &[GatedDeltaState],
        stream: &Stream,
    ) -> Result<Outcome> {
        // Snapshot construction and settlement are outside the measured range.
        let mut states =
            original.iter().map(GatedDeltaState::snapshot).collect::<Result<Vec<_>>>()?;
        stream.synchronize()?;
        let started = Instant::now();
        let output = match variant {
            Variant::Packed => self
                .forward_packed_prefill(input, &mut states.iter_mut().collect::<Vec<_>>(), stream)?
                .ok_or_else(|| {
                    super::super::Error::InvalidModel("GDN replay cannot pack".into())
                })?,
            Variant::Rows => {
                let shape = input.shape()?;
                let sequence = usize::try_from(shape[1])?;
                let hidden = usize::try_from(shape[2])?;
                let mut outputs = Vec::new();
                for (row, state) in states.iter_mut().enumerate() {
                    let input = input.slice(&[row, 0, 0], &[row + 1, sequence, hidden], stream)?;
                    outputs.push(self.forward(&input, state, stream)?);
                }
                Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 0, stream)?
            },
        };
        let graph_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut roots = vec![&output];
        roots.extend(states.iter().flat_map(GatedDeltaState::graph_roots));
        stream.eval_many(&roots)?;
        stream.synchronize()?;
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        for state in &states {
            state.detach_evaluated_graphs(stream)?;
        }
        output.detach_graph(stream)?;
        Ok(Outcome { output, states, graph_ms, total_ms })
    }
}

impl Variant {
    pub(super) const fn name(self) -> &'static str {
        match self {
            Self::Rows => "rows",
            Self::Packed => "packed",
        }
    }
}
