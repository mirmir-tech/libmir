use super::{DecodeInput, LoadedModel, NativeOutput};
use crate::native::error::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::native) enum DecodeExecution {
    Scalar,
    Packed { rows: usize },
}

enum Group {
    Scalar(usize),
    Packed(Vec<usize>),
}

impl LoadedModel {
    pub(in crate::native) fn decode_grouped(
        &mut self,
        inputs: &[DecodeInput],
    ) -> Result<Vec<(NativeOutput, DecodeExecution)>> {
        // Validate every row before advancing any cache, including scalar rows.
        self.validate_decode_inputs(inputs)?;
        self.validate_decode_capacity(inputs)?;
        let result = self.decode_grouped_admitted(inputs);
        self.finish_decode(inputs, result)
    }

    fn decode_grouped_admitted(
        &mut self,
        inputs: &[DecodeInput],
    ) -> Result<Vec<(NativeOutput, DecodeExecution)>> {
        let packed = self.execution.decoder()?.prefers_packed_decode(&self.stream);
        let groups =
            partition(inputs.len(), |row| packed && self.can_decode_packed_row(&inputs[row]));
        let mut outputs = Vec::with_capacity(inputs.len());
        for group in groups {
            match group {
                Group::Scalar(row) => {
                    let input = inputs[row].clone();
                    let output =
                        self.decode_admitted(input.session, input.token, input.sampling)?;
                    outputs.push((row, (output, DecodeExecution::Scalar)));
                },
                Group::Packed(rows) => {
                    let selected = rows.iter().map(|row| inputs[*row].clone()).collect::<Vec<_>>();
                    let execution = DecodeExecution::Packed { rows: rows.len() };
                    for (row, output) in rows.into_iter().zip(self.decode_batch_admitted(
                        &selected,
                        #[cfg(test)]
                        None,
                    )?) {
                        outputs.push((row, (output, execution)));
                    }
                },
            }
        }
        outputs.sort_unstable_by_key(|(row, _)| *row);
        Ok(outputs.into_iter().map(|(_, output)| output).collect())
    }
}

// No new admission window: use only the ready rows. Keep the relative order
// inside a packed cohort; restore caller order after execution.
fn partition(rows: usize, mut eligible: impl FnMut(usize) -> bool) -> Vec<Group> {
    let (packed, scalar): (Vec<_>, Vec<_>) = (0..rows).partition(|row| eligible(*row));
    let mut groups = scalar.into_iter().map(Group::Scalar).collect::<Vec<_>>();
    match packed.as_slice() {
        [] => {},
        [row] => groups.push(Group::Scalar(*row)),
        _ => groups.push(Group::Packed(packed)),
    }
    groups.sort_unstable_by_key(|group| match group {
        Group::Scalar(row) => *row,
        Group::Packed(rows) => rows[0],
    });
    groups
}
