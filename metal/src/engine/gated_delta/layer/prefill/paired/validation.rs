use std::io::Write;

use super::{Array, GatedDeltaState, Outcome, Result, Stream};

#[derive(Debug, PartialEq, Eq)]
pub(super) struct StateValues {
    offsets: Vec<usize>,
    values: Vec<Vec<u32>>,
    histories: Vec<Vec<u32>>,
}

impl StateValues {
    pub(super) fn read(states: &[GatedDeltaState], stream: &Stream) -> Result<Self> {
        Ok(Self {
            offsets: states.iter().map(|s| s.offset).collect(),
            values: states
                .iter()
                .map(|s| bits(super::required(s.value.as_deref())?, stream))
                .collect::<Result<_>>()?,
            histories: states
                .iter()
                .map(|s| bits(super::required(s.convolution.as_deref())?, stream))
                .collect::<Result<_>>()?,
        })
    }
}

pub(super) fn compare(a: &Outcome, b: &Outcome, layout: &str, stream: &Stream) -> Result<bool> {
    assert_eq!(a.output.shape()?, b.output.shape()?);
    let output = differing(&bits(&a.output, stream)?, &bits(&b.output, stream)?);
    let a_state = StateValues::read(&a.states, stream)?;
    let b_state = StateValues::read(&b.states, stream)?;
    let values = a_state
        .values
        .iter()
        .zip(&b_state.values)
        .map(|(a, b)| differing(a, b))
        .sum::<usize>();
    let histories = a_state
        .histories
        .iter()
        .zip(&b_state.histories)
        .map(|(a, b)| differing(a, b))
        .sum::<usize>();
    let offsets_match = a_state.offsets == b_state.offsets;
    writeln!(
        std::io::stderr().lock(),
        "gdn.parity: {}",
        serde_json::json!({
            "shape": a.output.shape()?, "layout": layout,
            "output_differing": output, "state_differing": values,
            "history_differing": histories, "offsets_match": offsets_match,
        })
    )?;
    Ok(output == 0 && values == 0 && histories == 0 && offsets_match)
}

fn bits(array: &Array, stream: &Stream) -> Result<Vec<u32>> {
    let values = array.to_vec_f32(stream)?;
    assert!(values.iter().all(|v| v.is_finite()));
    Ok(values.into_iter().map(f32::to_bits).collect())
}

fn differing(a: &[u32], b: &[u32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(a, b)| a != b).count()
}
