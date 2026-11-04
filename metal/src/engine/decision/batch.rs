use models::decision::DecisionRow;

use crate::engine::{Array, Error, Result};

/// Additive mask value of a blocked key. Finite, so a fully blocked padding
/// query stays finite instead of turning into `NaN` that would leak through
/// later matrix products.
const BLOCKED: f32 = f32::MIN;

/// Decision rows padded to one length, with the attention masks they need.
pub struct PaddedBatch {
    pub rows: i32,
    pub length: i32,
    pub tokens: Array,
    /// `[rows, 1, length, length]`: every valid key of the row.
    pub full: Array,
    pub kinds: Array,
    /// Flat `[rows × length]` positions of every option marker, row-major.
    pub markers: Array,
    pub marker_counts: Vec<usize>,
    lengths: Vec<usize>,
}

impl PaddedBatch {
    pub fn new(rows: &[DecisionRow], positions: usize) -> Result<Self> {
        let length = rows
            .iter()
            .map(|row| row.tokens.len())
            .max()
            .ok_or_else(|| Error::InvalidModel("decision batch is empty".into()))?;
        if length > positions {
            return Err(Error::InvalidModel(format!(
                "decision row of {length} tokens exceeds {positions} positions"
            )));
        }
        let mut tokens = Vec::with_capacity(rows.len() * length);
        let mut markers = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            tokens.extend(&row.tokens);
            tokens.resize(tokens.len() + length - row.tokens.len(), 0);
            for marker in &row.markers {
                markers.push(u32::try_from(index * length + marker)?);
            }
        }
        let lengths: Vec<usize> = rows.iter().map(|row| row.tokens.len()).collect();
        let kinds: Vec<u32> = rows
            .iter()
            .map(|row| u32::try_from(row.kind.index()))
            .collect::<std::result::Result<_, _>>()?;
        let (count, width) = (i32::try_from(rows.len())?, i32::try_from(length)?);
        Ok(Self {
            rows: count,
            length: width,
            tokens: Array::from_u32(&tokens, &[count, width])?,
            full: mask(&lengths, length, None)?,
            kinds: Array::from_u32(&kinds, &[count])?,
            markers: Array::from_u32(&markers, &[i32::try_from(markers.len())?])?,
            marker_counts: rows.iter().map(|row| row.markers.len()).collect(),
            lengths,
        })
    }

    /// `[rows, 1, length, length]` mask of keys within `radius` of a query.
    pub fn band(&self, radius: usize) -> Result<Array> {
        mask(&self.lengths, usize::try_from(self.length)?, Some(radius))
    }
}

fn mask(lengths: &[usize], length: usize, radius: Option<usize>) -> Result<Array> {
    let mut values = Vec::with_capacity(lengths.len() * length * length);
    for &valid in lengths {
        for query in 0..length {
            values.extend((0..length).map(|key| {
                let near = radius.is_none_or(|radius| query.abs_diff(key) <= radius);
                if key < valid && near {
                    0.0
                } else {
                    BLOCKED
                }
            }));
        }
    }
    let (rows, width) = (i32::try_from(lengths.len())?, i32::try_from(length)?);
    Array::from_f32(&values, &[rows, 1, width, width])
}
