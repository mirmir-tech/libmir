use super::{PaddingSide, TokenizedPrompt};
use crate::{ModelsError, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedBatch {
    pub token_ids: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub lengths: Vec<usize>,
    pub batch_size: usize,
    pub sequence_length: usize,
}

impl TokenizedBatch {
    pub fn pad(
        sequences: &[TokenizedPrompt],
        pad_token_id: u32,
        padding_side: PaddingSide,
    ) -> Result<Self> {
        let sequence_length = sequences.iter().map(|item| item.token_ids.len()).max().unwrap_or(0);
        if sequences.is_empty() || sequence_length == 0 {
            return Err(ModelsError::InvalidConfig("cannot pad an empty token batch".into()));
        }
        let elements = sequences
            .len()
            .checked_mul(sequence_length)
            .ok_or_else(|| ModelsError::InvalidConfig("token batch size overflow".into()))?;
        let mut batch = Self {
            token_ids: vec![pad_token_id; elements],
            type_ids: vec![0; elements],
            attention_mask: vec![0; elements],
            lengths: sequences.iter().map(|item| item.token_ids.len()).collect(),
            batch_size: sequences.len(),
            sequence_length,
        };
        for (row, sequence) in sequences.iter().enumerate() {
            if sequence.token_ids.len() != sequence.type_ids.len()
                || sequence.token_ids.len() != sequence.attention_mask.len()
            {
                return Err(ModelsError::InvalidConfig(
                    "token ids, type ids, and attention mask lengths differ".into(),
                ));
            }
            let padding = sequence_length - sequence.token_ids.len();
            let column = if padding_side == PaddingSide::Left {
                padding
            } else {
                0
            };
            let start = row * sequence_length + column;
            let end = start + sequence.token_ids.len();
            batch.token_ids[start..end].copy_from_slice(&sequence.token_ids);
            batch.type_ids[start..end].copy_from_slice(&sequence.type_ids);
            batch.attention_mask[start..end].copy_from_slice(&sequence.attention_mask);
        }
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn left_padding_preserves_masks_and_lengths() -> Result<()> {
        let sequences = [prompt(&[4, 5]), prompt(&[6])];
        let batch = TokenizedBatch::pad(&sequences, 1, PaddingSide::Left)?;
        assert_eq!(batch.token_ids, [4, 5, 1, 6]);
        assert_eq!(batch.attention_mask, [1, 1, 0, 1]);
        assert_eq!(batch.lengths, [2, 1]);
        Ok(())
    }

    fn prompt(ids: &[u32]) -> TokenizedPrompt {
        TokenizedPrompt {
            token_ids: ids.to_vec(),
            type_ids: vec![0; ids.len()],
            attention_mask: vec![1; ids.len()],
            bytes: 0,
        }
    }
}
