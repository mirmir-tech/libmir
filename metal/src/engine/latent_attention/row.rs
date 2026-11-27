//! One cached latent attention row, `[latent ‖ k_pe]`, stored as the key
//! and value halves of a single head so the paged cache keeps it unchanged.

use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug, Clone, Copy)]
pub(super) struct LatentRow {
    rank: usize,
    width: usize,
}

impl LatentRow {
    pub(super) fn new(rank: usize, rope: usize) -> Result<Self> {
        let width = rank + rope;
        if rank == 0 || rope == 0 || !width.is_multiple_of(2) {
            return Err(Error::InvalidModel(format!(
                "latent row of rank {rank} and rotary width {rope} does not split in halves"
            )));
        }
        Ok(Self { rank, width })
    }

    /// `[.., width]` as `[.., at]` and `[.., width - at]`.
    fn cut(input: &Array, at: usize, stream: &Stream) -> Result<(Array, Array)> {
        let shape = input
            .shape()?
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let [leading @ .., width] = shape.as_slice() else {
            return Err(Error::InvalidModel("latent row has no feature axis".into()));
        };
        let zeros = vec![0; leading.len()];
        let start = [zeros.as_slice(), &[at]].concat();
        let middle = [leading, &[at]].concat();
        let end = [leading, &[*width]].concat();
        Ok((
            input.slice(&[zeros.as_slice(), &[0]].concat(), &middle, stream)?,
            input.slice(&start, &end, stream)?,
        ))
    }

    /// The latent and the rotary key of rows.
    pub(super) fn split(&self, rows: &Array, stream: &Stream) -> Result<(Array, Array)> {
        Self::cut(rows, self.rank, stream)
    }

    /// The cache key and value halves of rows.
    pub(super) fn halves(&self, rows: &Array, stream: &Stream) -> Result<(Array, Array)> {
        Self::cut(rows, self.width / 2, stream)
    }

    /// Rows from the cache's key and value halves.
    pub(super) fn join(keys: &Array, values: &Array, stream: &Stream) -> Result<Array> {
        Array::concatenate(&[keys, values], 3, stream)
    }
}
