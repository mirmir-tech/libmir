use models::layout::RotaryEmbeddingLayout;

use crate::engine::{Array, Dtype, Error, Result, Stream};

pub fn apply_mrope(
    input: &Array,
    positions: &Array,
    dimensions: usize,
    base: f32,
    layout: &RotaryEmbeddingLayout,
    stream: &Stream,
) -> Result<Array> {
    let sections = match layout {
        RotaryEmbeddingLayout::MultiSection(sections)
        | RotaryEmbeddingLayout::InterleavedMultiSection(sections) => sections,
        RotaryEmbeddingLayout::Standard => {
            return Err(Error::InvalidModel("MRoPE requires multimodal sections".into()));
        },
    };
    let half = dimensions / 2;
    if sections.len() != 3 || sections.iter().sum::<usize>() != half {
        return Err(Error::InvalidModel(format!(
            "MRoPE sections {sections:?} do not cover half of {dimensions} rotary dimensions"
        )));
    }
    let selectors = selectors(half, sections, layout)?;
    let selectors = Array::from_f32(&selectors, &[3, 1, i32::try_from(half)?])?;
    let selected = positions
        .astype(Dtype::Float32, stream)?
        .expand_dims(&[2], stream)?
        .multiply(&selectors, stream)?
        .reduce_sum(0, false, stream)?;
    let inverse = inverse_frequency(dimensions, base)?;
    let angles = selected.multiply(&inverse, stream)?;
    let cos = Array::concatenate(&[&angles, &angles], -1, stream)?
        .cos(stream)?
        .expand_dims(&[0, 1], stream)?
        .astype_like(input, stream)?;
    let sin = Array::concatenate(&[&angles, &angles], -1, stream)?
        .sin(stream)?
        .expand_dims(&[0, 1], stream)?
        .astype_like(input, stream)?;
    rotate(input, &cos, &sin, dimensions, stream)
}

fn selectors(half: usize, sections: &[usize], layout: &RotaryEmbeddingLayout) -> Result<Vec<f32>> {
    let mut values = vec![0.0; 3 * half];
    for frequency in 0..half {
        let axis = match layout {
            RotaryEmbeddingLayout::MultiSection(_) => {
                if frequency < sections[0] {
                    0
                } else if frequency < sections[0] + sections[1] {
                    1
                } else {
                    2
                }
            },
            RotaryEmbeddingLayout::InterleavedMultiSection(_) => {
                if frequency % 3 == 1 && frequency < 3 * sections[1] {
                    1
                } else if frequency % 3 == 2 && frequency < 3 * sections[2] {
                    2
                } else {
                    0
                }
            },
            RotaryEmbeddingLayout::Standard => return Err(Error::ShapeOverflow),
        };
        values[axis * half + frequency] = 1.0;
    }
    Ok(values)
}

fn inverse_frequency(dimensions: usize, base: f32) -> Result<Array> {
    let denominator = dimensions.to_string().parse::<f64>()?;
    let base = f64::from(base);
    let values = (0..dimensions)
        .step_by(2)
        .map(|index| {
            base.powf(-index.to_string().parse::<f64>()? / denominator)
                .to_string()
                .parse::<f32>()
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Array::from_f32(&values, &[i32::try_from(values.len())?])
}

fn rotate(
    input: &Array,
    cos: &Array,
    sin: &Array,
    dimensions: usize,
    stream: &Stream,
) -> Result<Array> {
    let width = usize::try_from(*input.shape()?.last().ok_or(Error::ShapeOverflow)?)?;
    let rotary = slice_last(input, 0, dimensions, stream)?;
    let tail = slice_last(input, dimensions, width, stream)?;
    let half = dimensions / 2;
    let first = slice_last(&rotary, 0, half, stream)?;
    let second = slice_last(&rotary, half, dimensions, stream)?;
    let rotated =
        Array::concatenate(&[&second.multiply_scalar(-1.0, stream)?, &first], -1, stream)?;
    let rotary = rotary.multiply(cos, stream)?.add(&rotated.multiply(sin, stream)?, stream)?;
    Array::concatenate(&[&rotary, &tail], -1, stream)
}

fn slice_last(input: &Array, start: usize, stop: usize, stream: &Stream) -> Result<Array> {
    let shape = input.shape()?;
    let mut starts = vec![0; shape.len()];
    let mut stops = shape
        .iter()
        .copied()
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let axis = shape.len() - 1;
    starts[axis] = start;
    stops[axis] = stop;
    input.slice(&starts, &stops, stream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RopeOptions;

    #[test]
    fn matches_standard_rope_when_all_three_axes_share_text_positions() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let input = Array::from_f32(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            &[1, 1, 2, 8],
        )?;
        let positions = Array::from_u32(&[0, 1, 0, 1, 0, 1], &[3, 2])?;
        let layout = RotaryEmbeddingLayout::InterleavedMultiSection(vec![1, 1, 1]);
        let actual = apply_mrope(&input, &positions, 6, 10_000.0, &layout, &stream)?;
        let expected = input.rope(
            RopeOptions {
                dimensions: 6,
                traditional: false,
                base: Some(10_000.0),
                scale: 1.0,
                offset: 0,
            },
            &stream,
        )?;
        let actual = actual.to_vec_f32_on_stream(&stream)?;
        let expected = expected.to_vec_f32_on_stream(&stream)?;
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-5)
        );
        Ok(())
    }
}
