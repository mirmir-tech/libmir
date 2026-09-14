mod context;
mod report;

use super::{fixture, *};
use crate::{
    engine::Array,
    native::{prefill::evaluation, session::SessionState, step},
};

#[test]
#[ignore = "real-model scalar/packed prefill with matched chunks and logits/state comparison"]
fn diagnoses_matched_prefill() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut loaded = fixture::load_path(std::env::var("MIRMIR_BENCH_MODEL")?, 0)?;
    let mut previous: Option<(Vec<f32>, Vec<f32>)> = None;
    for chunk in [31, 8] {
        let (scalar, packed) = compare(&mut loaded, chunk)?;
        if let Some((old_scalar, old_packed)) = previous {
            report::difference("scalar.chunk31/chunk8", &old_scalar, &scalar)?;
            report::difference("packed.chunk31/chunk8", &old_packed, &packed)?;
        }
        previous = Some((scalar, packed));
    }
    Ok(())
}

#[test]
fn matched_prefill_fixture_is_finite_and_preserves_offsets() -> Result<()> {
    let (mut loaded, directory) = fixture::load_family(fixture::Family::Hybrid, 0)?;
    let (scalar, packed) = compare(&mut loaded, 8)?;
    assert_eq!(scalar, packed, "fixture logits must be identical");
    drop(loaded);
    std::fs::remove_dir_all(directory)?;
    Ok(())
}

fn states(loaded: &LoadedModel) -> Result<Vec<SessionState>> {
    (0..5)
        .map(|_| {
            let mut cache = loaded.execution.decoder()?.new_cache(loaded.stream())?;
            cache.reserve(256)?;
            cache.plan_contiguous(48);
            Ok(SessionState::new(cache))
        })
        .collect()
}

fn compare(loaded: &mut LoadedModel, chunk: usize) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut scalar = states(loaded)?;
    let mut packed = states(loaded)?;
    loaded.reserve_prefill_pages(30)?;
    let mut position = 0;
    while position < 31 {
        let count = chunk.min(31 - position);
        let mut scalar_hidden = Vec::new();
        for (row, state) in scalar.iter_mut().enumerate() {
            let tokens = prompt(row, position, count)?;
            let hidden = step::forward_prefill_state(
                loaded.execution.decoder()?,
                loaded.stream(),
                state,
                &tokens,
                position,
            )?;
            evaluation::materialize(loaded, state, &hidden)?;
            scalar_hidden.extend(hidden.to_vec_f32(loaded.stream())?);
            assert_eq!(state.cache.cached_tokens()?, position + count);
        }
        let tokens = (0..5)
            .map(|row| prompt(row, position, count))
            .collect::<Result<Vec<_>>>()?
            .concat();
        let mut refs = packed.iter_mut().collect::<Vec<_>>();
        let hidden = step::forward_packed_prefill_state(
            loaded.execution.decoder()?,
            loaded.stream(),
            &mut refs,
            &[position; 5],
            &tokens,
            count,
        )?;
        evaluation::materialize_packed(loaded, &refs, &hidden)?;
        report::difference(
            &format!("chunk={chunk},offset={},hidden", position + count),
            &scalar_hidden,
            &hidden.to_vec_f32(loaded.stream())?,
        )?;
        let a = scalar[0].cache.gated_delta_state(0)?.values()?.to_vec_f32(loaded.stream())?;
        let b = packed[0].cache.gated_delta_state(0)?.values()?.to_vec_f32(loaded.stream())?;
        report::difference(
            &format!("chunk={chunk},offset={},layer0.state", position + count),
            &a,
            &b,
        )?;
        for state in &packed {
            assert_eq!(state.cache.cached_tokens()?, position + count);
        }
        position += count;
    }
    let scalar_logits = logits(loaded, &mut scalar)?;
    let packed_logits = logits(loaded, &mut packed)?;
    report::difference(&format!("chunk={chunk},logits"), &scalar_logits, &packed_logits)?;
    drop((scalar, packed));
    assert_eq!(loaded.stream().paged_arenas().resident_arenas()?, 0);
    Ok((scalar_logits, packed_logits))
}

fn logits(loaded: &LoadedModel, states: &mut [SessionState]) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    for (row, state) in states.iter_mut().enumerate() {
        let output: Array = step::forward_token(
            loaded.execution.decoder()?,
            loaded.stream(),
            state,
            u32::try_from((row + 31) % 64)?,
            31,
            false,
        )?;
        evaluation::materialize(loaded, state, &output)?;
        values.extend(output.to_vec_f32(loaded.stream())?);
    }
    Ok(values)
}

fn prompt(row: usize, position: usize, count: usize) -> Result<Vec<u32>> {
    (position..position + count)
        .map(|index| Ok(u32::try_from((row + index) % 64)?))
        .collect()
}
