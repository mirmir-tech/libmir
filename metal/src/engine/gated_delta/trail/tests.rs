use super::super::{GatedDeltaInputs, GatedDeltaState};
use crate::engine::{Array, Result, Stream};

const ROWS: usize = 3;
const VALUE_HEADS: usize = 2;
const DIMENSION: usize = 64;
const CHANNELS: usize = 8;
const KERNEL: usize = 4;

/// Deterministic values in `[-scale, scale)`.
fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 40) as f32 / (1_u64 << 24) as f32;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
}

struct Pass {
    convolution: Array,
    weight: Array,
    query: Array,
    key: Array,
    value: Array,
    alpha: Array,
    beta: Array,
    a_log: Array,
    dt_bias: Array,
}

fn rows(array: &Array, rows: usize, stream: &Stream) -> Result<Array> {
    let shape = array
        .shape()?
        .iter()
        .map(|&value| usize::try_from(value))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mut stop = shape.clone();
    stop[1] = rows;
    array.slice(&vec![0; shape.len()], &stop, stream)
}

impl Pass {
    fn new() -> Result<Self> {
        let heads = |seed, width: usize| {
            Array::from_f32(
                &values(ROWS * width * DIMENSION, seed, 0.2),
                &[1, 3, i32::try_from(width)?, 64],
            )
        };
        Ok(Self {
            convolution: Array::from_f32(&values(ROWS * CHANNELS, 1, 1.0), &[1, 3, 8])?,
            weight: Array::from_f32(&values(CHANNELS * KERNEL, 2, 0.5), &[8, 4, 1])?,
            query: heads(3, 1)?,
            key: heads(4, 1)?,
            value: heads(5, VALUE_HEADS)?,
            alpha: Array::from_f32(&values(ROWS * VALUE_HEADS, 6, 1.0), &[1, 3, 2])?,
            beta: Array::from_f32(&values(ROWS * VALUE_HEADS, 7, 1.0), &[1, 3, 2])?,
            a_log: Array::from_f32(&values(VALUE_HEADS, 8, 0.5), &[2])?,
            dt_bias: Array::from_f32(&values(VALUE_HEADS, 9, 0.5), &[2])?,
        })
    }

    /// Feeds the first `count` rows.
    fn feed(&self, state: &mut GatedDeltaState, count: usize, stream: &Stream) -> Result<()> {
        state.convolve_silu(&rows(&self.convolution, count, stream)?, &self.weight, stream)?;
        let [query, key, value, alpha, beta] =
            [&self.query, &self.key, &self.value, &self.alpha, &self.beta]
                .map(|array| rows(array, count, stream));
        state.update(
            GatedDeltaInputs {
                query: &query?,
                key: &key?,
                value: &value?,
                alpha: &alpha?,
                beta: &beta?,
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
            },
            stream,
        )?;
        Ok(())
    }
}

#[test]
fn committing_rows_matches_a_state_that_saw_only_them() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let pass = Pass::new()?;
    for kept in 1..=ROWS {
        let mut trailed = GatedDeltaState::new()?;
        trailed.arm_row_trail();
        pass.feed(&mut trailed, ROWS, &stream)?;
        trailed.commit_rows(kept, &stream)?;
        let mut reference = GatedDeltaState::new()?;
        pass.feed(&mut reference, kept, &stream)?;
        assert_eq!(trailed.offset()?, kept);
        assert_eq!(
            trailed.values()?.to_vec_f32(&stream)?,
            reference.values()?.to_vec_f32(&stream)?
        );
        let history = |state: &GatedDeltaState| -> Result<Vec<f32>> {
            let root = state.convolution_root().ok_or(crate::engine::Error::ShapeOverflow)?;
            root.to_vec_f32(&stream)
        };
        assert_eq!(history(&trailed)?, history(&reference)?);
    }
    Ok(())
}

#[test]
fn commit_requires_an_armed_pass() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut state = GatedDeltaState::new()?;
    Pass::new()?.feed(&mut state, ROWS, &stream)?;
    assert!(state.commit_rows(1, &stream).is_err());
    Ok(())
}
