use std::ops::Range;

use mirtal::{DType, Shape};

use crate::engine::{Array, Error, Result, Stream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageTokenSpan {
    pub start: usize,
    pub end: usize,
}

impl ImageTokenSpan {
    pub fn new(range: Range<usize>, sequence: usize) -> Result<Self> {
        if range.start >= range.end || range.end > sequence {
            return Err(Error::InvalidModel(format!(
                "image token span {}..{} is invalid for sequence length {sequence}",
                range.start, range.end
            )));
        }
        Ok(Self { start: range.start, end: range.end })
    }

    #[must_use]
    pub const fn len(self) -> usize {
        self.end - self.start
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Builds Gemma 4's decoder mask on the accelerator:
/// `(causal OR image_block) AND sliding_window`.
pub fn prefix_attention_mask(
    sequence: usize,
    image: ImageTokenSpan,
    sliding_window: Option<usize>,
    stream: &Stream,
) -> Result<Array> {
    let image = ImageTokenSpan::new(image.start..image.end, sequence)?;
    let graph = stream.native().graph();
    let positions = graph.arange(0.0, as_f32(sequence)?, 1.0, DType::Uint32)?;
    let keys = graph.expand_dims(&positions, &[0])?;
    let queries = graph.expand_dims(&positions, &[-1])?;
    let causal = graph.greater_equal(&queries, &keys)?;
    let scalar = Shape::new([])?;
    let start = graph.full(&scalar, as_f32(image.start)?, DType::Uint32)?;
    let end = graph.full(&scalar, as_f32(image.end)?, DType::Uint32)?;
    let queries_in_image =
        graph.logical_and(&graph.greater_equal(&queries, &start)?, &graph.less(&queries, &end)?)?;
    let keys_in_image =
        graph.logical_and(&graph.greater_equal(&keys, &start)?, &graph.less(&keys, &end)?)?;
    let image_block = graph.logical_and(&queries_in_image, &keys_in_image)?;
    let mask = graph.maximum(&causal, &image_block)?;
    let Some(window) = sliding_window else {
        return Array::from_native(mask);
    };
    let lower = graph.less(&queries, &graph.add_scalar(&keys, as_f32(window)?)?)?;
    Array::from_native(graph.logical_and(&mask, &lower)?)
}

fn as_f32(value: usize) -> Result<f32> {
    Ok(value.to_string().parse()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Dtype;

    #[test]
    fn opens_only_the_image_block_over_the_causal_mask() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let image = ImageTokenSpan::new(2..5, 6)?;
        let mask =
            prefix_attention_mask(6, image, None, &stream)?.astype(Dtype::Uint32, &stream)?;
        assert_eq!(mask.shape()?, [6, 6]);
        assert_eq!(
            mask.to_vec_u32(&stream)?,
            rows(&["100000", "110000", "111110", "111110", "111110", "111111"])
        );
        Ok(())
    }

    #[test]
    fn clamps_the_image_block_to_the_sliding_window() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let image = ImageTokenSpan::new(2..5, 6)?;
        let mask =
            prefix_attention_mask(6, image, Some(3), &stream)?.astype(Dtype::Uint32, &stream)?;
        assert_eq!(
            mask.to_vec_u32(&stream)?,
            rows(&["100000", "110000", "111110", "011110", "001110", "000111"])
        );
        Ok(())
    }

    fn rows(rows: &[&str]) -> Vec<u32> {
        rows.iter()
            .flat_map(|row| row.bytes().map(|value| u32::from(value - b'0')))
            .collect()
    }
}
