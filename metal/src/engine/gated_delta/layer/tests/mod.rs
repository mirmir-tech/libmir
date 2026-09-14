mod prefill;

use super::*;
use crate::engine::{QuantizedArrays, QuantizedLinear};

#[test]
fn executes_a_complete_gated_delta_layer_on_the_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let layer = layer(&stream)?;
    let input = Array::from_f32(&vec![0.0; 128], &[1, 2, 64])?;
    let mut state = GatedDeltaState::new()?;
    let output = layer.forward(&input, &mut state, &stream)?;

    let mut roots = vec![&output];
    roots.extend(state.graph_roots());
    stream.eval_many(&roots)?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 2, 64]);
    assert!(output.to_vec_f32(&stream)?.iter().all(|value| *value == 0.0));
    assert_eq!(state.offset()?, 2);
    state.detach_evaluated_graphs(&stream)?;
    let decode = Array::from_f32(&vec![0.0; 64], &[1, 1, 64])?;
    let output = layer.forward(&decode, &mut state, &stream)?;

    output.async_eval(&stream)?;
    stream.synchronize()?;
    assert!(output.to_vec_f32(&stream)?.iter().all(|value| *value == 0.0));
    assert_eq!(state.offset()?, 3);
    Ok(())
}

fn linear(output_width: i32, stream: &Stream) -> Result<BoundLinear> {
    let values = vec![0.0; usize::try_from(output_width * 64)?];
    let dense = Array::from_f32(&values, &[output_width, 64])?;
    let arrays: QuantizedArrays = dense.quantize(64, 4, stream)?;
    Ok(BoundLinear::Affine(QuantizedLinear::from_quantized(arrays, 64, 4)))
}

fn layer(stream: &Stream) -> Result<GatedDeltaLayer> {
    let config = GatedDeltaLayerConfig::from_linear_attention(
        &LinearAttentionConfig {
            convolution_kernel_size: 2,
            key_heads: 1,
            value_heads: 1,
            key_head_dim: 64,
            value_head_dim: 64,
        },
        1.0e-6,
    )?;
    let mut layer = GatedDeltaLayer {
        config,
        in_proj_qkv: linear(192, stream)?,
        in_proj_z: linear(64, stream)?,
        in_proj_b: linear(1, stream)?,
        in_proj_a: linear(1, stream)?,
        out_proj: linear(64, stream)?,
        conv_weight: Array::from_f32(&vec![0.0; 384], &[192, 2, 1])?,
        norm_weight: NormWeight::from_weight(Array::from_f32(&vec![1.0; 64], &[64])?),
        a_log: Array::from_f32(&[0.0], &[1])?,
        dt_bias: Array::from_f32(&[0.0], &[1])?,
        compiled_decode: None,
    };
    layer.compiled_decode = CompiledDecode::new(&layer, stream)?;
    Ok(layer)
}

#[test]
fn packed_state_reuse_matches_concatenation_through_cohort_changes() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut layer = layer(&stream)?;
    layer.in_proj_qkv = varying_linear(192, &stream)?;
    layer.in_proj_z = varying_linear(64, &stream)?;
    layer.out_proj = varying_linear(64, &stream)?;
    layer.conv_weight = Array::from_f32(&vec![0.25; 384], &[192, 2, 1])?;
    layer.compiled_decode = CompiledDecode::new(&layer, &stream)?;
    let mut reused = Vec::new();
    for row in 0..3 {
        let mut state = GatedDeltaState::new()?;
        let input =
            Array::from_f32(&vec![f32::from(i16::try_from(row + 1)?) * 0.01; 128], &[1, 2, 64])?;
        let output = layer.forward(&input, &mut state, &stream)?;
        let mut roots = vec![&output];
        roots.extend(state.graph_roots());
        stream.eval_many(&roots)?;
        reused.push(state);
    }
    let mut reference = reused.iter().map(GatedDeltaState::snapshot).collect::<Result<Vec<_>>>()?;
    let saved = reused
        .iter()
        .map(GatedDeltaState::values)
        .map(|value| value?.to_vec_f32(&stream))
        .collect::<Result<Vec<_>>>()?;
    let snapshots = reused.iter().map(GatedDeltaState::snapshot).collect::<Result<Vec<_>>>()?;
    let mut removed = None;
    for step in 0..128 {
        if step == 32 {
            reused.rotate_left(1);
            reference.rotate_left(1);
        }
        if step == 64 {
            removed = Some((
                required(reused.pop(), "third row")?,
                required(reference.pop(), "third reference row")?,
            ));
        }
        if step == 96 {
            let (a, b) = required(removed.take(), "removed row")?;
            reused.push(a);
            reference.push(b);
        }
        // Recreate the old implementation: concatenate independent row handles on every
        // step.
        for state in &mut reference {
            state.discard_packed_source()?;
        }
        let input = Array::from_f32(
            &vec![0.02; reused.len() * 64],
            &[i32::try_from(reused.len())?, 1, 64],
        )?;
        let actual = layer
            .forward_packed(&input, &mut reused.iter_mut().collect::<Vec<_>>(), &stream)?
            .ok_or_else(|| Error::InvalidModel("missing compiled test batch".into()))?;
        let expected = layer
            .forward_packed(&input, &mut reference.iter_mut().collect::<Vec<_>>(), &stream)?
            .ok_or_else(|| Error::InvalidModel("missing compiled test batch".into()))?;
        let mut roots = vec![&actual, &expected];
        for state in reused.iter().chain(&reference) {
            roots.extend(state.graph_roots());
        }
        stream.eval_many(&roots)?;
        stream.synchronize()?;
        assert_eq!(
            actual.to_vec_f32(&stream)?,
            expected.to_vec_f32(&stream)?,
            "output step {step}"
        );
        assert!(actual.to_vec_f32(&stream)?.iter().any(|value| *value != 0.0));
        for (a, b) in reused.iter().zip(&reference) {
            assert_eq!(
                a.values()?.to_vec_f32(&stream)?,
                b.values()?.to_vec_f32(&stream)?,
                "state step {step}"
            );
            assert_eq!(
                required(a.convolution.as_ref(), "history")?.to_vec_f32(&stream)?,
                required(b.convolution.as_ref(), "reference history")?.to_vec_f32(&stream)?,
                "convolution step {step}"
            );
            a.detach_evaluated_graphs(&stream)?;
            b.detach_evaluated_graphs(&stream)?;
        }
    }
    for (snapshot, expected) in snapshots.iter().zip(saved) {
        assert_eq!(snapshot.values()?.to_vec_f32(&stream)?, expected);
    }
    Ok(())
}

fn varying_linear(output_width: i32, stream: &Stream) -> Result<BoundLinear> {
    let values = (0..output_width * 64)
        .map(|index| Ok(f32::from(i16::try_from(index % 17)?) * 0.002 - 0.016))
        .collect::<Result<Vec<_>>>()?;
    let arrays = Array::from_f32(&values, &[output_width, 64])?.quantize(64, 4, stream)?;
    Ok(BoundLinear::Affine(QuantizedLinear::from_quantized(arrays, 64, 4)))
}

fn required<T>(value: Option<T>, field: &str) -> Result<T> {
    value.ok_or_else(|| Error::InvalidModel(format!("missing test {field}")))
}
