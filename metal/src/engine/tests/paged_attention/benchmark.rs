use std::{env, hint::black_box, io::Write, time::Instant};

use crate::engine::{Array, PagedAttention, Result, Stream, attention::PagedAttentionScratch};

const PAGE_SIZE: usize = 16;
const QUERY_HEADS: usize = 16;
const ITERATIONS: usize = 20;
const SAMPLES: usize = 7;
const WARMUP: usize = 4;

struct Inputs {
    query: Array,
    keys: Array,
    values: Array,
    key_pages: Array,
    value_pages: Array,
    page_table: Array,
    dependency: Array,
    scratch: PagedAttentionScratch,
    context: usize,
    fragmented: bool,
}

#[test]
#[ignore = "synthetic GPU benchmark"]
fn benchmarks_paged_sdpa_decode_matrix() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let (head_dim, kv_heads) = dimensions();
    let mut report = std::io::stderr().lock();
    for context in contexts()? {
        let inputs = Inputs::new(context, head_dim, kv_heads)?;
        warm(&inputs, &stream)?;
        let (contiguous, paged) = samples(&inputs, &stream)?;
        writeln!(
            report,
            "paged_sdpa.benchmark: context={context}, head_dim={head_dim}, fragmented={}, samples={SAMPLES}, iterations={ITERATIONS}, contiguous={contiguous:.3}ms, paged={paged:.3}ms",
            inputs.fragmented,
        )?;
    }
    Ok(())
}

impl Inputs {
    fn new(context: usize, head_dim: usize, kv_heads: usize) -> Result<Self> {
        let head_dim_i32 = i32::try_from(head_dim)?;
        let kv_heads_i32 = i32::try_from(kv_heads)?;
        let context_i32 = i32::try_from(context)?;
        let pages = context.div_ceil(PAGE_SIZE);
        let pages_i32 = i32::try_from(pages)?;
        let fragmented = fragmented();
        let mut table = (0..u32::try_from(pages)?).collect::<Vec<_>>();
        if fragmented {
            table.reverse();
        }
        let query = super::patterned(QUERY_HEADS * head_dim, 17);
        let page_length = kv_heads * pages * PAGE_SIZE * head_dim;
        let key_pages = super::patterned(page_length, 31);
        let value_pages = super::patterned(page_length, 47);
        let keys = compact_pages(&key_pages, kv_heads, context, head_dim, &table)?;
        let values = compact_pages(&value_pages, kv_heads, context, head_dim, &table)?;
        Ok(Self {
            query: Array::from_f32(&query, &[1, 16, 1, head_dim_i32])?,
            keys: Array::from_f32(&keys, &[1, kv_heads_i32, context_i32, head_dim_i32])?,
            values: Array::from_f32(&values, &[1, kv_heads_i32, context_i32, head_dim_i32])?,
            key_pages: Array::from_f32(&key_pages, &[kv_heads_i32, pages_i32, 16, head_dim_i32])?,
            value_pages: Array::from_f32(
                &value_pages,
                &[kv_heads_i32, pages_i32, 16, head_dim_i32],
            )?,
            page_table: Array::from_u32(&table, &[pages_i32])?,
            dependency: Array::from_u32(&[u32::try_from(context)?], &[1])?,
            scratch: PagedAttentionScratch::default(),
            context,
            fragmented,
        })
    }

    fn paged(&self) -> PagedAttention<'_> {
        PagedAttention {
            key_pages: &self.key_pages,
            value_pages: &self.value_pages,
            key_scales: None,
            value_scales: None,
            page_table: &self.page_table,
            page_dependency: &self.dependency,
            page_size: PAGE_SIZE,
            context_tokens: self.context,
        }
    }

    fn contiguous_context(&self, stream: &Stream) -> Result<(Array, Array)> {
        if !self.fragmented {
            return Ok((
                Array::from_native(self.keys.native().clone())?,
                Array::from_native(self.values.native().clone())?,
            ));
        }
        let graph = stream.native().graph();
        let pages = self.context.div_ceil(PAGE_SIZE);
        let ids = graph.slice(self.page_table.native(), &[0], &[pages])?;
        let keys = graph.take(self.key_pages.native(), &ids, 1)?;
        let values = graph.take(self.value_pages.native(), &ids, 1)?;
        let dimensions = self.keys.native().shape()?.dimensions().to_vec();
        let shape = mirtal::Shape::new([1, dimensions[1], pages * PAGE_SIZE, dimensions[3]])?;
        let stop = [1, dimensions[1], self.context, dimensions[3]];
        let keys = graph.slice(&graph.reshape(&keys, &shape)?, &[0, 0, 0, 0], &stop)?;
        let values = graph.slice(&graph.reshape(&values, &shape)?, &[0, 0, 0, 0], &stop)?;
        Ok((Array::from_native(keys)?, Array::from_native(values)?))
    }
}

fn samples(inputs: &Inputs, stream: &Stream) -> Result<(f64, f64)> {
    let mut contiguous = Vec::with_capacity(SAMPLES);
    let mut paged = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        if sample.is_multiple_of(2) {
            contiguous.push(measure_contiguous(inputs, stream)?);
            paged.push(measure_paged(inputs, stream)?);
        } else {
            paged.push(measure_paged(inputs, stream)?);
            contiguous.push(measure_contiguous(inputs, stream)?);
        }
    }
    Ok((median(contiguous), median(paged)))
}

fn warm(inputs: &Inputs, stream: &Stream) -> Result<()> {
    validate(inputs, stream)?;
    for _ in 0..WARMUP {
        let _ = measure_contiguous(inputs, stream)?;
        let _ = measure_paged(inputs, stream)?;
    }
    Ok(())
}

fn validate(inputs: &Inputs, stream: &Stream) -> Result<()> {
    let expected = inputs
        .query
        .scaled_dot_product_attention(&inputs.keys, &inputs.values, 1.0, false, stream)?;
    let actual = inputs.query.paged_scaled_dot_product_attention(inputs.paged(), 1.0, stream)?;
    actual.async_eval()?;
    stream.synchronize()?;
    super::assert_approx(&expected.to_vec_f32()?, &actual.to_vec_f32()?, 1.0e-4);
    Ok(())
}

fn measure_contiguous(inputs: &Inputs, stream: &Stream) -> Result<f64> {
    measure(stream, || {
        let (keys, values) = inputs.contiguous_context(stream)?;
        inputs.query.scaled_dot_product_attention(&keys, &values, 1.0, false, stream)
    })
}

fn measure_paged(inputs: &Inputs, stream: &Stream) -> Result<f64> {
    measure(stream, || {
        inputs.query.paged_scaled_dot_product_attention_with_scratch(
            inputs.paged(),
            &inputs.scratch,
            1.0,
            stream,
        )
    })
}

fn measure(stream: &Stream, mut operation: impl FnMut() -> Result<Array>) -> Result<f64> {
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        let output = operation()?;
        output.async_eval()?;
        stream.synchronize()?;
        black_box(output);
    }
    Ok(started.elapsed().as_secs_f64() * 1_000.0 / f64::from(u32::try_from(ITERATIONS)?))
}

fn contexts() -> Result<Vec<usize>> {
    env::var("MIRMIR_PAGED_BENCH_CONTEXTS")
        .unwrap_or_else(|_| "128,512,1024,2048".into())
        .split(',')
        .map(|value| Ok(value.trim().parse()?))
        .collect()
}

fn dimensions() -> (usize, usize) {
    if matches!(env::var("MIRMIR_PAGED_BENCH_GLOBAL").as_deref(), Ok("1" | "true" | "TRUE")) {
        return (512, 2);
    }
    let head_dim = env::var("MIRMIR_PAGED_BENCH_HEAD_DIM")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256);
    let kv_heads = env::var("MIRMIR_PAGED_BENCH_KV_HEADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    (head_dim, kv_heads)
}

fn fragmented() -> bool {
    matches!(env::var("MIRMIR_PAGED_BENCH_FRAGMENTED").as_deref(), Ok("1" | "true" | "TRUE"))
}

fn compact_pages(
    values: &[f32],
    heads: usize,
    context: usize,
    head_dim: usize,
    table: &[u32],
) -> Result<Vec<f32>> {
    let mut output = Vec::with_capacity(heads * context * head_dim);
    for head in 0..heads {
        for (logical, physical) in table.iter().copied().enumerate() {
            let tokens = (context - logical * PAGE_SIZE).min(PAGE_SIZE);
            let page = head * table.len() + usize::try_from(physical)?;
            let start = page * PAGE_SIZE * head_dim;
            output.extend_from_slice(&values[start..start + tokens * head_dim]);
        }
    }
    Ok(output)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[values.len() / 2]
}
