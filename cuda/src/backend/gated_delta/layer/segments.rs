use super::CudaGatedDeltaState;
use crate::{Error, Result};

/// Consecutive tokens of one row that reach the recurrence in one launch. A
/// row with an armed checkpoint runs as two segments over the same state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RowSegment {
    pub(super) row: usize,
    pub(super) offset: usize,
    pub(super) count: usize,
    pub(super) checkpoint: bool,
}

pub(super) fn row_segments(
    states: &[&mut CudaGatedDeltaState],
    counts: &[usize],
) -> Result<Vec<RowSegment>> {
    let armed = states.iter().map(|state| state.armed_checkpoint()).collect::<Vec<_>>();
    split_rows(&armed, counts)
}

fn split_rows(armed: &[Option<usize>], counts: &[usize]) -> Result<Vec<RowSegment>> {
    let mut segments = Vec::with_capacity(counts.len() + 1);
    let mut offset = 0;
    for (row, (count, armed)) in counts.iter().copied().zip(armed).enumerate() {
        let Some(before) = *armed else {
            segments.push(RowSegment { row, offset, count, checkpoint: false });
            offset += count;
            continue;
        };
        if before == 0 || before >= count {
            return Err(Error::InvalidExecutionPlan(
                "Gated Delta checkpoint lies outside its prefill row",
            ));
        }
        segments.push(RowSegment {
            row,
            offset,
            count: before,
            checkpoint: true,
        });
        segments.push(RowSegment {
            row,
            offset: offset + before,
            count: count - before,
            checkpoint: false,
        });
        offset += count;
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::{RowSegment, split_rows};

    #[test]
    fn armed_row_runs_as_two_segments() -> crate::Result<()> {
        let segments = split_rows(&[None, Some(5), None], &[3, 8, 1])?;
        assert_eq!(
            segments,
            [
                RowSegment {
                    row: 0,
                    offset: 0,
                    count: 3,
                    checkpoint: false
                },
                RowSegment {
                    row: 1,
                    offset: 3,
                    count: 5,
                    checkpoint: true
                },
                RowSegment {
                    row: 1,
                    offset: 8,
                    count: 3,
                    checkpoint: false
                },
                RowSegment {
                    row: 2,
                    offset: 11,
                    count: 1,
                    checkpoint: false
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn checkpoint_must_lie_inside_the_row() {
        assert!(split_rows(&[Some(0)], &[4]).is_err());
        assert!(split_rows(&[Some(4)], &[4]).is_err());
    }
}
