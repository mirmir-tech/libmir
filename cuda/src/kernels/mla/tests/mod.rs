mod expanded;
mod fill;
mod fixture;
mod reference;
mod timing;

use std::ops::Range;

use mircuda::bf16;
use models::{layout::RotaryPairing, semantic::QueryTemperature};

use self::{
    fixture::{Runtime, Values},
    reference::{Token, model},
};
use super::{
    LatentAbsorption, LatentAttention, LatentAttentionInputs, LatentAttentionLayer,
    LatentAttentionSpec, LatentCacheView, LatentUpLayout, LatentUpWeights,
};
use crate::Result;

/// Block-table width of every test sequence.
const MAX_BLOCKS: usize = 20;

/// The latent geometry of `DeepSeek` V2/V3 checkpoints. Ten heads run the
/// warp kernel with a partly idle second head group; whole groups of
/// sixteen run the tensor-core kernel.
const fn spec(
    heads: usize,
    rotary_pairing: RotaryPairing,
    max_queries: usize,
    splits: usize,
) -> LatentAttentionSpec {
    LatentAttentionSpec {
        heads,
        latent_rank: 512,
        rope_dim: 64,
        nope_dim: 128,
        value_dim: 128,
        rotary_pairing,
        block_size: 16,
        max_queries,
        splits,
    }
}

/// Launches over `(sequence, positions)` chunks of sequences of `lengths`.
struct Scenario {
    spec: LatentAttentionSpec,
    lengths: Vec<usize>,
    launches: Vec<Vec<(usize, Range<usize>)>>,
    rope_scale: f32,
    temperature: Option<QueryTemperature>,
    layout: LatentUpLayout,
    /// Launches of one sequence each run expanded attention.
    expanded: bool,
}

#[test]
fn prefill_then_decode_matches_expanded_attention() -> Result<()> {
    // One split spans three 32-row tiles, so the running maximum rescales.
    // One split runs the paired tensor-core kernel: sixteen heads pair
    // consecutive tokens, thirty-two the two head groups of one token.
    for (heads, splits) in [(10, 3), (32, 3), (10, 1), (16, 1), (32, 1)] {
        run(&Scenario {
            spec: spec(heads, RotaryPairing::Interleaved, 76, splits),
            lengths: vec![77],
            launches: vec![vec![(0, 0..76)], vec![(0, 76..77)]],
            rope_scale: 1.0,
            temperature: None,
            layout: LatentUpLayout::Fused,
            expanded: false,
        })?;
    }
    Ok(())
}

#[test]
fn chunked_batch_matches_expanded_attention() -> Result<()> {
    for heads in [10, 16] {
        run(&Scenario {
            spec: spec(heads, RotaryPairing::Halves, 20, 1),
            lengths: vec![25, 9],
            launches: vec![vec![(0, 0..20)], vec![(0, 20..25), (1, 0..9)]],
            rope_scale: 0.8,
            // Queries past positions 8, 16 and 24 scale up.
            temperature: std::num::NonZeroUsize::new(8).map(|window| QueryTemperature {
                beta: 0.5,
                window,
                offset: 0,
            }),
            layout: LatentUpLayout::Split,
            expanded: false,
        })?;
    }
    Ok(())
}

fn run(scenario: &Scenario) -> Result<()> {
    let runtime = Runtime::new()?;
    let spec = scenario.spec;
    let mut values = Values::new(0x5eed);
    let model = model(spec, &mut values, scenario.rope_scale, scenario.temperature);
    let sequences = tokens(spec, &scenario.lengths, &mut values);
    let expected = sequences.iter().map(|sequence| model.attend(sequence)).collect::<Vec<_>>();
    let blocks = scenario.lengths.len() * MAX_BLOCKS;
    let table = scattered(blocks)?;

    let attention = LatentAttention::compile(&runtime.compiler, spec)?;
    let mut workspace = attention.workspace(&runtime.pool, &runtime.stream)?;
    let mut pages = runtime.pool.allocate_zeroed(&runtime.stream, spec.page_bytes(blocks)?)?;
    let block_tables = runtime.copy(&table)?;
    let latent_norm = runtime.copy_f32(&model.latent_norm)?;
    let (key_up, value_up) = split_up(spec, &model.key_value_up);
    let (key_value_up, key_up, value_up) = (
        runtime.copy_f32(&model.key_value_up)?,
        runtime.copy_f32(&key_up)?,
        runtime.copy_f32(&value_up)?,
    );
    let inverse_frequencies = runtime.copy(&model.inverse_frequencies)?;
    let layer = LatentAttentionLayer {
        latent_norm: &latent_norm,
        up: match scenario.layout {
            LatentUpLayout::Fused => LatentUpWeights::Fused(&key_value_up),
            LatentUpLayout::Split => LatentUpWeights::Split { key: &key_up, value: &value_up },
        },
        inverse_frequencies: &inverse_frequencies,
        epsilon: model.epsilon,
        rope_scale: model.rope_scale,
        softmax_scale: model.softmax_scale,
        temperature: model.temperature,
    };
    let mut expansion = scenario
        .expanded
        .then(|| expanded::Expansion::new(&runtime, spec, scenario.layout))
        .transpose()?;
    for launch in &scenario.launches {
        let tokens = launch
            .iter()
            .flat_map(|(sequence, positions)| {
                positions.clone().map(|position| (*sequence, position))
            })
            .collect::<Vec<_>>();
        let (mut query, mut latent, mut host) =
            (Vec::new(), Vec::new(), [(); 3].map(|()| Vec::new()));
        for &(sequence, position) in &tokens {
            query.extend_from_slice(&sequences[sequence][position].query);
            latent.extend_from_slice(&sequences[sequence][position].latent);
            let block = table[sequence * MAX_BLOCKS + position / spec.block_size];
            let offset = u32::try_from(position % spec.block_size)?;
            host[0].push(u32::try_from(position)?);
            host[1].push(block * u32::try_from(spec.block_size)? + offset);
            host[2].push(u32::try_from(sequence)?);
        }
        let [positions, slot_mapping, query_sequences] =
            [runtime.copy(&host[0])?, runtime.copy(&host[1])?, runtime.copy(&host[2])?];
        let (query, latent) = (runtime.copy_f32(&query)?, runtime.copy_f32(&latent)?);
        let mut absorption = LatentAbsorption::new(
            &runtime.context,
            &runtime.stream,
            spec,
            scenario.layout,
            tokens.len(),
        )?;
        let width = spec.heads * spec.value_dim;
        let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, tokens.len() * width)?;
        let cache = LatentCacheView {
            pages: &mut pages,
            block_tables: &block_tables,
            table_stride: MAX_BLOCKS,
        };
        let inputs = LatentAttentionInputs {
            query: &query,
            latent: &latent,
            positions: &positions,
            slot_mapping: &slot_mapping,
            query_sequences: &query_sequences,
        };
        if let Some(expansion) = &mut expansion {
            expansion.execute(
                &attention, &runtime, &layer, cache, inputs, launch, &mut workspace, &mut output,
            )?;
        } else {
            attention.execute(
                &runtime.stream, &mut absorption, &layer, cache, inputs, &mut workspace,
                &mut output,
            )?;
        }
        let actual = runtime.read(&output)?;
        for (index, &(sequence, position)) in tokens.iter().enumerate() {
            compare(
                &actual[index * width..(index + 1) * width],
                &expected[sequence][position],
                (sequence, position),
            );
        }
    }
    Ok(())
}

/// Logical blocks scattered over the pool in reverse, so table lookups matter.
fn scattered(blocks: usize) -> Result<Vec<u32>> {
    let table = (0..blocks).map(|index| u32::try_from(blocks - 1 - index));
    Ok(table.collect::<std::result::Result<_, _>>()?)
}

/// Random query and latent rows of every token of sequences of `lengths`.
fn tokens(spec: LatentAttentionSpec, lengths: &[usize], values: &mut Values) -> Vec<Vec<Token>> {
    lengths
        .iter()
        .map(|length| {
            (0..*length)
                .map(|_| Token {
                    query: values.take(spec.heads * (spec.nope_dim + spec.rope_dim), 1.0),
                    latent: values.take(spec.row_width(), 1.0),
                })
                .collect()
        })
        .collect()
}

/// GGUF's split layout of `kv_b_proj`: per head, the key rows transposed to
/// `[latent, nope]` and the value rows `[value, latent]`.
fn split_up(spec: LatentAttentionSpec, fused: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (rank, nope, value) = (spec.latent_rank, spec.nope_dim, spec.value_dim);
    let row = |head: usize, index: usize| &fused[(head * (nope + value) + index) * rank..][..rank];
    let mut key = Vec::with_capacity(spec.heads * rank * nope);
    let mut values = Vec::with_capacity(spec.heads * value * rank);
    for head in 0..spec.heads {
        key.extend((0..rank).flat_map(|c| (0..nope).map(move |d| row(head, d)[c])));
        values.extend((0..value).flat_map(|e| row(head, nope + e).iter().copied()));
    }
    (key, values)
}

/// BF16 latent, query and output rounding stay far below a 3 % error of the
/// output's magnitude; a layout or indexing error does not.
fn compare(actual: &[bf16], expected: &[f32], token: (usize, usize)) {
    let magnitude = expected.iter().fold(0.0_f32, |maximum, value| maximum.max(value.abs()));
    let worst = actual.iter().zip(expected).fold(0.0_f32, |worst, (actual, expected)| {
        worst.max((actual.to_f32() - expected).abs())
    });
    assert!(
        worst <= 0.03 * magnitude,
        "token {token:?}: max error {worst} against magnitude {magnitude}",
    );
}
