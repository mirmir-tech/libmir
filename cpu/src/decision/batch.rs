use models::decision::DecisionRow;

use crate::{Error, Result};

/// Decision rows padded to one length.
pub struct PaddedBatch {
    /// `[rows × length]` token ids; padding repeats id 0 and is masked.
    pub tokens: Vec<u32>,
    pub lengths: Vec<usize>,
    pub length: usize,
    /// Type-embedding row of each question.
    pub kinds: Vec<usize>,
    pub markers: Vec<Vec<usize>>,
}

impl PaddedBatch {
    pub fn new(rows: &[DecisionRow], positions: usize) -> Result<Self> {
        let length = rows.iter().map(|row| row.tokens.len()).max().ok_or(Error::EmptyBatch)?;
        if length > positions {
            return Err(Error::RowTooLong { tokens: length, positions });
        }
        let mut tokens = Vec::with_capacity(rows.len() * length);
        for row in rows {
            tokens.extend(&row.tokens);
            tokens.resize(tokens.len() + length - row.tokens.len(), 0);
        }
        Ok(Self {
            tokens,
            lengths: rows.iter().map(|row| row.tokens.len()).collect(),
            length,
            kinds: rows.iter().map(|row| row.kind.index()).collect(),
            markers: rows.iter().map(|row| row.markers.clone()).collect(),
        })
    }
}
