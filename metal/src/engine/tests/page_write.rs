use std::{hint::black_box, io::Write, time::Instant};

use crate::engine::{
    Array, Result, Stream,
    kernels::{PageWriteOptions, PreparedPageWrite},
};

const HEAD_DIM: usize = 256;
const KV_HEADS: usize = 2;
const PAGE_CAPACITY: usize = 256;
const PAGE_SIZE: usize = 16;
const BUILD_ITERATIONS: usize = 1_000;
const ITERATIONS: usize = 100;
const SAMPLES: usize = 7;

struct Inputs {
    keys: Array,
    values: Array,
    page_keys: Array,
    page_values: Array,
    table: Array,
}

#[test]
#[ignore = "synthetic GPU benchmark"]
fn benchmarks_prepared_page_write_dispatch() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut fresh = Inputs::new()?;
    let mut reused = Inputs::new()?;
    let mut prepared = PreparedPageWrite::default();
    warm(&mut fresh, &mut reused, &mut prepared, &stream)?;
    let mut fresh_samples = Vec::with_capacity(SAMPLES);
    let mut reused_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        if sample.is_multiple_of(2) {
            fresh_samples.push(measure_fresh(&mut fresh, &stream)?);
            reused_samples.push(measure_reused(&mut reused, &mut prepared, &stream)?);
        } else {
            reused_samples.push(measure_reused(&mut reused, &mut prepared, &stream)?);
            fresh_samples.push(measure_fresh(&mut fresh, &stream)?);
        }
    }
    let fresh_build = measure_build(&mut Inputs::new()?, None, &stream)?;
    let prepared_build =
        measure_build(&mut Inputs::new()?, Some(&mut PreparedPageWrite::default()), &stream)?;
    writeln!(
        std::io::stderr().lock(),
        "page_write.benchmark: samples={SAMPLES}, iterations={ITERATIONS}, fresh={:.4}ms, prepared={:.4}ms, fresh_build={fresh_build:.3}us, prepared_build={prepared_build:.3}us",
        median(fresh_samples),
        median(reused_samples),
    )?;
    Ok(())
}

impl Inputs {
    fn new() -> Result<Self> {
        let token_elements = KV_HEADS * HEAD_DIM;
        let page_elements = KV_HEADS * PAGE_CAPACITY * PAGE_SIZE * HEAD_DIM;
        Ok(Self {
            keys: Array::from_f32(
                &vec![0.25; token_elements],
                &[1, i32::try_from(KV_HEADS)?, 1, i32::try_from(HEAD_DIM)?],
            )?,
            values: Array::from_f32(
                &vec![0.5; token_elements],
                &[1, i32::try_from(KV_HEADS)?, 1, i32::try_from(HEAD_DIM)?],
            )?,
            page_keys: Array::from_f32(
                &vec![0.0; page_elements],
                &[
                    i32::try_from(KV_HEADS)?,
                    i32::try_from(PAGE_CAPACITY)?,
                    i32::try_from(PAGE_SIZE)?,
                    i32::try_from(HEAD_DIM)?,
                ],
            )?,
            page_values: Array::from_f32(
                &vec![0.0; page_elements],
                &[
                    i32::try_from(KV_HEADS)?,
                    i32::try_from(PAGE_CAPACITY)?,
                    i32::try_from(PAGE_SIZE)?,
                    i32::try_from(HEAD_DIM)?,
                ],
            )?,
            table: Array::from_u32(
                &(0..u32::try_from(PAGE_CAPACITY)?).collect::<Vec<_>>(),
                &[i32::try_from(PAGE_CAPACITY)?],
            )?,
        })
    }

    fn write(
        &mut self,
        offset: usize,
        prepared: &mut PreparedPageWrite,
        stream: &Stream,
    ) -> Result<()> {
        self.enqueue(offset, prepared, stream)?;
        self.page_values.async_eval()?;
        stream.synchronize()?;
        Ok(())
    }

    fn enqueue(
        &mut self,
        offset: usize,
        prepared: &mut PreparedPageWrite,
        stream: &Stream,
    ) -> Result<()> {
        let [keys, values] = stream.page_write(
            [
                self.keys.native(),
                self.values.native(),
                self.page_keys.native(),
                self.page_values.native(),
                self.table.native(),
            ],
            PageWriteOptions {
                sequence: 1,
                offset,
                kv_heads: KV_HEADS,
                page_capacity: PAGE_CAPACITY,
                page_size: PAGE_SIZE,
                head_dim: HEAD_DIM,
            },
            prepared,
        )?;
        self.page_keys = Array::from_native(keys)?;
        self.page_values = Array::from_native(values)?;
        Ok(())
    }
}

fn warm(
    fresh: &mut Inputs,
    reused: &mut Inputs,
    prepared: &mut PreparedPageWrite,
    stream: &Stream,
) -> Result<()> {
    for offset in 0..4 {
        fresh.write(offset, &mut PreparedPageWrite::default(), stream)?;
        reused.write(offset, prepared, stream)?;
    }
    Ok(())
}

fn measure_fresh(inputs: &mut Inputs, stream: &Stream) -> Result<f64> {
    measure(|offset| inputs.write(offset, &mut PreparedPageWrite::default(), stream))
}

fn measure_reused(
    inputs: &mut Inputs,
    prepared: &mut PreparedPageWrite,
    stream: &Stream,
) -> Result<f64> {
    measure(|offset| inputs.write(offset, prepared, stream))
}

fn measure(mut write: impl FnMut(usize) -> Result<()>) -> Result<f64> {
    let started = Instant::now();
    for iteration in 0..ITERATIONS {
        black_box(write(iteration % (PAGE_CAPACITY * PAGE_SIZE))?);
    }
    Ok(started.elapsed().as_secs_f64() * 1_000.0 / f64::from(u32::try_from(ITERATIONS)?))
}

fn measure_build(
    inputs: &mut Inputs,
    mut prepared: Option<&mut PreparedPageWrite>,
    stream: &Stream,
) -> Result<f64> {
    let started = Instant::now();
    for iteration in 0..BUILD_ITERATIONS {
        let mut fresh = PreparedPageWrite::default();
        let dispatch = prepared.as_deref_mut().unwrap_or(&mut fresh);
        inputs.enqueue(iteration % (PAGE_CAPACITY * PAGE_SIZE), dispatch, stream)?;
    }
    let elapsed = started.elapsed().as_secs_f64() * 1_000_000.0;
    inputs.page_values.async_eval()?;
    stream.synchronize()?;
    Ok(elapsed / f64::from(u32::try_from(BUILD_ITERATIONS)?))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[values.len() / 2]
}
