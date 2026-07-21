use super::*;
use crate::engine::lowering::MixerLowering;

#[test]
fn keeps_recurrent_gated_delta_state_on_the_explicit_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut state = GatedDeltaState::new()?;
    let inputs = inputs(&[2.0, 4.0, 3.0, 5.0], 2)?;
    let output = state.update(inputs.as_borrowed(), &stream)?;
    output.async_eval()?;
    stream.synchronize()?;

    assert_close(&output.to_vec_f32()?, &[1.0, 2.0, 1.75, 3.0]);
    assert_eq!(state.offset()?, 2);
    assert_close(&state.values()?.to_vec_f32_on_stream(&stream)?, &[1.75, 3.0]);
    Ok(())
}

#[test]
fn uses_the_metal_recurrence_kernel_for_supported_key_dimensions() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut state = GatedDeltaState::new()?;
    let mut query_and_key = vec![0.0; 64];
    query_and_key[0] = 1.0;
    query_and_key[32] = 1.0;
    let inputs = GatedDeltaInputs {
        query: &Array::from_f32(&query_and_key, &[1, 2, 1, 32])?,
        key: &Array::from_f32(&query_and_key, &[1, 2, 1, 32])?,
        value: &Array::from_f32(&[2.0, 4.0], &[1, 2, 1, 1])?,
        alpha: &Array::from_f32(&[0.0, 0.0], &[1, 2, 1])?,
        beta: &Array::from_f32(&[0.0, 0.0], &[1, 2, 1])?,
        a_log: &Array::from_f32(&[0.0], &[1])?,
        dt_bias: &Array::from_f32(&[0.0], &[1])?,
    };
    let output = state.update(inputs, &stream)?;

    output.async_eval()?;
    stream.synchronize()?;
    assert_close(&output.to_vec_f32()?, &[1.0, 2.25]);
    assert_close(&state.values()?.to_vec_f32_on_stream(&stream)?[..1], &[2.25]);
    Ok(())
}

#[test]
fn fused_decode_matches_the_general_recurrence_step() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut general = GatedDeltaState::new()?;
    let mut fused = GatedDeltaState::new()?;
    for step in 0..2 {
        let general_inputs = decode_inputs(step)?;
        let fused_inputs = decode_inputs(step)?;
        let general_output = general.update(general_inputs.as_borrowed(), &stream)?;
        let fused_output = fused.update_fused(fused_inputs.as_borrowed(), &stream)?;
        general_output.async_eval()?;
        fused_output.async_eval()?;
        stream.synchronize()?;
        assert_eq!(general_output.to_vec_f32()?, fused_output.to_vec_f32()?);
        assert_eq!(
            general.values()?.to_vec_f32_on_stream(&stream)?,
            fused.values()?.to_vec_f32_on_stream(&stream)?,
        );
    }
    Ok(())
}

#[test]
fn snapshots_recurrent_state_without_a_host_round_trip() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut original = GatedDeltaState::new()?;
    let first = inputs(&[2.0, 4.0], 1)?;
    let _first_output = original.update(first.as_borrowed(), &stream)?;
    let mut snapshot = original.snapshot()?;
    let second = inputs(&[3.0, 5.0], 1)?;
    let original_output = original.update(second.as_borrowed(), &stream)?;
    let snapshot_inputs = inputs(&[3.0, 5.0], 1)?;
    let snapshot_output = snapshot.update(snapshot_inputs.as_borrowed(), &stream)?;

    original_output.async_eval()?;
    snapshot_output.async_eval()?;
    stream.synchronize()?;
    assert_close(&original_output.to_vec_f32()?, &snapshot_output.to_vec_f32()?);
    assert_eq!(original.offset()?, 2);
    assert_eq!(snapshot.offset()?, 2);
    Ok(())
}

#[test]
fn snapshots_mixed_linear_and_key_value_cache_at_a_common_offset() -> Result<()> {
    let mut cache = DecoderCache::new_hybrid_linear(
        &[MixerLowering::Linear, MixerLowering::Softmax { sinks: false, window: None }],
        16,
    )?;

    assert_eq!(cache.cached_tokens()?, 0);
    assert_eq!(cache.snapshot_at(0)?.cached_tokens()?, 0);
    assert!(cache.snapshot_at(1).is_err());
    cache.reset()?;
    Ok(())
}

#[test]
fn keeps_depthwise_convolution_history_in_a_state_snapshot() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut original = GatedDeltaState::new()?;
    let weight = Array::from_f32(&[1.0, 1.0], &[1, 2, 1])?;
    let first = Array::from_f32(&[1.0, 2.0], &[1, 2, 1])?;
    let first_output = original.convolve_silu(&first, &weight, &stream)?;
    let mut snapshot = original.snapshot()?;
    let next = Array::from_f32(&[3.0], &[1, 1, 1])?;
    let original_output = original.convolve_silu(&next, &weight, &stream)?;
    let snapshot_output = snapshot.convolve_silu(&next, &weight, &stream)?;

    first_output.async_eval()?;
    original_output.async_eval()?;
    snapshot_output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(first_output.shape()?, vec![1, 2, 1]);
    assert_close(&original_output.to_vec_f32()?, &snapshot_output.to_vec_f32()?);
    Ok(())
}

struct Inputs {
    query: Array,
    key: Array,
    value: Array,
    alpha: Array,
    beta: Array,
    a_log: Array,
    dt_bias: Array,
}

impl Inputs {
    fn as_borrowed(&self) -> GatedDeltaInputs<'_> {
        GatedDeltaInputs {
            query: &self.query,
            key: &self.key,
            value: &self.value,
            alpha: &self.alpha,
            beta: &self.beta,
            a_log: &self.a_log,
            dt_bias: &self.dt_bias,
        }
    }
}

fn inputs(values: &[f32], tokens: i32) -> Result<Inputs> {
    let heads = 2;
    Ok(Inputs {
        query: Array::from_f32(
            &vec![1.0; usize::try_from(tokens * heads)? / 2],
            &[1, tokens, 1, 1],
        )?,
        key: Array::from_f32(&vec![1.0; usize::try_from(tokens * heads)? / 2], &[1, tokens, 1, 1])?,
        value: Array::from_f32(values, &[1, tokens, heads, 1])?,
        alpha: Array::from_f32(&vec![0.0; usize::try_from(tokens * heads)?], &[1, tokens, heads])?,
        beta: Array::from_f32(&vec![0.0; usize::try_from(tokens * heads)?], &[1, tokens, heads])?,
        a_log: Array::from_f32(&[0.0, 0.0], &[heads])?,
        dt_bias: Array::from_f32(&[0.0, 0.0], &[heads])?,
    })
}

fn decode_inputs(step: usize) -> Result<Inputs> {
    let scale = f32::from(u16::try_from(step)?) + 1.0;
    let mut query = vec![0.0; 32];
    let mut key = vec![0.0; 32];
    for index in 0..32 {
        query[index] = scale * f32::from(u16::try_from(index + 1)?) / 64.0;
        key[index] = f32::from(u16::try_from(33 - index)?) / 64.0;
    }
    Ok(Inputs {
        query: Array::from_f32(&query, &[1, 1, 1, 32])?,
        key: Array::from_f32(&key, &[1, 1, 1, 32])?,
        value: Array::from_f32(&[0.25, -0.5, 0.75, 1.25], &[1, 1, 2, 2])?,
        alpha: Array::from_f32(&[-0.3, 0.2], &[1, 1, 2])?,
        beta: Array::from_f32(&[0.4, -0.7], &[1, 1, 2])?,
        a_log: Array::from_f32(&[0.1, -0.2], &[2])?,
        dt_bias: Array::from_f32(&[0.05, -0.1], &[2])?,
    })
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }
}
