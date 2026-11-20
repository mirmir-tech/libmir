mod fixture;
mod reference;
mod timing;

use std::ops::Range;

use mircuda::bf16;
use models::layout::RotaryPairing;

use self::{
    fixture::{Runtime, Values},
    reference::{Model, Token},
};
use super::{
    LatentAbsorption, LatentAttention, LatentAttentionInputs, LatentAttentionLayer,
    LatentAttentionSpec, LatentCacheView,
};
use crate::Result;

/// `DeepSeek` V2/V3 geometry with ten heads, so the second head group of the
/// attention kernel is partly idle.
const fn spec(
    rotary_pairing: RotaryPairing,
    max_queries: usize,
    splits: usize,
) -> LatentAttentionSpec {
    LatentAttentionSpec {
        heads: 10,
        latent_rank: 512,
        rope_dim: 64,
        nope_dim: 128,
        value_dim: 128,
        rotary_pairing,
        block_size: 16,
        max_blocks: 5,
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
}

#[test]
fn prefill_then_decode_matches_expanded_attention() -> Result<()> {
    run(&Scenario {
        spec: spec(RotaryPairing::Interleaved, 76, 3),
        lengths: vec![77],
        launches: vec![vec![(0, 0..76)], vec![(0, 76..77)]],
        rope_scale: 1.0,
    })
}

#[test]
fn chunked_batch_matches_expanded_attention() -> Result<()> {
    run(&Scenario {
        spec: spec(RotaryPairing::Halves, 20, 1),
        lengths: vec![25, 9],
        launches: vec![vec![(0, 0..20)], vec![(0, 20..25), (1, 0..9)]],
        rope_scale: 0.8,
    })
}

fn run(scenario: &Scenario) -> Result<()> {
    let runtime = Runtime::new()?;
    let spec = scenario.spec;
    let mut values = Values::new(0x5eed);
    let model = model(spec, &mut values, scenario.rope_scale);
    let sequences = scenario
        .lengths
        .iter()
        .map(|length| {
            (0..*length)
                .map(|_| Token {
                    query: values.take(spec.heads * (spec.nope_dim + spec.rope_dim), 1.0),
                    latent: values.take(spec.row_width(), 1.0),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected = sequences.iter().map(|sequence| model.attend(sequence)).collect::<Vec<_>>();
    let blocks = scenario.lengths.len() * spec.max_blocks;
    // Scatter logical blocks over the pool in reverse so table lookups matter.
    let table = (0..blocks)
        .map(|index| u32::try_from(blocks - 1 - index))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let slot = |sequence: usize, position: usize| -> Result<u32> {
        let physical = table[sequence * spec.max_blocks + position / spec.block_size];
        Ok(physical * u32::try_from(spec.block_size)? + u32::try_from(position % spec.block_size)?)
    };

    let attention = LatentAttention::compile(&runtime.compiler, spec)?;
    let mut workspace = attention.workspace(&runtime.pool, &runtime.stream)?;
    let mut pages = runtime.pool.allocate_zeroed(&runtime.stream, spec.page_bytes(blocks)?)?;
    let block_tables = runtime.copy(&table)?;
    let latent_norm = runtime.copy_f32(&model.latent_norm)?;
    let key_value_up = runtime.copy_f32(&model.key_value_up)?;
    let inverse_frequencies = runtime.copy(&model.inverse_frequencies)?;
    let layer = LatentAttentionLayer {
        latent_norm: &latent_norm,
        key_value_up: &key_value_up,
        inverse_frequencies: &inverse_frequencies,
        epsilon: model.epsilon,
        rope_scale: model.rope_scale,
        softmax_scale: model.softmax_scale,
    };
    for launch in &scenario.launches {
        let tokens = launch
            .iter()
            .flat_map(|(sequence, positions)| {
                positions.clone().map(|position| (*sequence, position))
            })
            .collect::<Vec<_>>();
        let (mut query, mut latent, mut host) =
            (Vec::new(), Vec::new(), [(); 4].map(|()| Vec::new()));
        for &(sequence, position) in &tokens {
            query.extend_from_slice(&sequences[sequence][position].query);
            latent.extend_from_slice(&sequences[sequence][position].latent);
            host[0].push(u32::try_from(position)?);
            host[1].push(slot(sequence, position)?);
            host[2].push(u32::try_from(sequence)?);
            host[3].push(u32::try_from(position + 1)?);
        }
        let [positions, slot_mapping, query_sequences, query_visible] = [
            runtime.copy(&host[0])?,
            runtime.copy(&host[1])?,
            runtime.copy(&host[2])?,
            runtime.copy(&host[3])?,
        ];
        let (query, latent) = (runtime.copy_f32(&query)?, runtime.copy_f32(&latent)?);
        let mut absorption =
            LatentAbsorption::new(&runtime.context, &runtime.stream, spec, tokens.len())?;
        let width = spec.heads * spec.value_dim;
        let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, tokens.len() * width)?;
        attention.execute(
            &runtime.stream,
            &mut absorption,
            &layer,
            LatentCacheView {
                pages: &mut pages,
                block_tables: &block_tables,
            },
            LatentAttentionInputs {
                query: &query,
                latent: &latent,
                positions: &positions,
                slot_mapping: &slot_mapping,
                query_sequences: &query_sequences,
                query_visible: &query_visible,
            },
            &mut workspace,
            &mut output,
        )?;
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

fn model(spec: LatentAttentionSpec, values: &mut Values, rope_scale: f32) -> Model {
    let latent_norm = values
        .take(spec.latent_rank, 0.2)
        .into_iter()
        .map(|value| bf16::from_f32(1.0 + value).to_f32())
        .collect();
    let rows = spec.heads * (spec.nope_dim + spec.value_dim);
    let rope = f32::from(u16::try_from(spec.rope_dim).unwrap_or(1));
    let inverse_frequencies = (0..spec.rope_dim / 2)
        .map(|pair| 10_000_f32.powf(-2.0 * f32::from(u16::try_from(pair).unwrap_or(0)) / rope))
        .collect();
    Model {
        spec,
        latent_norm,
        key_value_up: values.take(rows * spec.latent_rank, 0.044),
        inverse_frequencies,
        epsilon: 1e-6,
        rope_scale,
        softmax_scale: 3.0
            / f32::from(u16::try_from(spec.nope_dim + spec.rope_dim).unwrap_or(1)).sqrt(),
    }
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
