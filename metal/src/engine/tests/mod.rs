use super::*;

mod checkpoint;
mod gated_delta;
mod page_write;
mod paged_attention;
mod paged_store;
mod quantized_kv;
mod rope;
mod router;
mod sampling;
mod sliding_cache;
mod vision_primitives;

#[test]
fn executes_addition_on_explicit_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let left = Array::from_f32(&[1.0, 2.0], &[1, 2])?;
    let right = Array::from_f32(&[3.0, 4.0], &[1, 2])?;
    let output = left.add(&right, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(output.to_vec_f32()?, vec![4.0, 6.0]);
    assert!(version()?.starts_with("0.31."));
    Ok(())
}

#[test]
fn executes_graph_transforms_on_explicit_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2])?;
    let weight = Array::from_f32(&[2.0, 0.5], &[2])?;
    let output = input
        .rms_norm(&weight, 0.0, &stream)?
        .transpose(&[0, 2, 1], &stream)?
        .reshape(&[1, 4], &stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(output.shape()?, vec![1, 4]);
    assert!(output.to_vec_f32()?.iter().all(|value| value.is_finite()));
    Ok(())
}

#[test]
fn executes_unscaled_rms_norm_on_explicit_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[3.0, 4.0], &[1, 2])?;
    let output = input.rms_norm_unit(0.0, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    let values = output.to_vec_f32()?;
    assert!((values[0] - 0.848_528_15).abs() < 1.0e-5);
    assert!((values[1] - 1.131_370_9).abs() < 1.0e-5);
    Ok(())
}

#[test]
fn executes_int4_quantized_matmul_on_explicit_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&vec![1.0; 64], &[1, 64])?;
    let dense = Array::from_f32(&vec![1.0; 64], &[1, 64])?;
    let quantized = dense.quantize(64, 4, &stream)?;
    assert_eq!(input.shape()?, vec![1, 64]);
    assert_eq!(input.dtype()?, Dtype::Float32);
    assert_eq!(quantized.weight.dtype()?, Dtype::Uint32);
    let output = input.quantized_matmul(&quantized, true, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    let values = output.to_vec_f32()?;
    assert_eq!(values.len(), 1);
    assert!((values[0] - 64.0).abs() < 1.0e-3);
    Ok(())
}

#[test]
fn gathers_int4_moe_experts_on_explicit_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&vec![1.0; 128], &[1, 1, 2, 1, 64])?;
    let dense = Array::from_f32(&expert_weights(), &[2, 1, 64])?;
    let indices = Array::from_u32(&[0, 1], &[1, 1, 2])?;
    let quantized = dense.quantize(64, 4, &stream)?;
    let output = input.gather_qmm(
        &quantized,
        &indices,
        mirtal::GatherQmmOptions { transpose: true, sorted_indices: false },
        &stream,
    )?;
    output.async_eval()?;
    stream.synchronize()?;
    let values = output.to_vec_f32()?;
    assert_eq!(values.len(), 2);
    assert!((values[0] - 64.0).abs() < 1.0e-3);
    assert!((values[1] - 128.0).abs() < 1.0e-3);
    Ok(())
}

#[test]
fn executes_rope_on_explicit_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4])?;
    let options = RopeOptions {
        dimensions: 4,
        traditional: false,
        base: Some(10_000.0),
        scale: 1.0,
        offset: 0,
    };
    let output = input.rope(options, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(output.to_vec_f32()?, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn executes_decode_sdpa_on_explicit_gpu_stream() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let queries = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let keys = Array::from_f32(&[1.0, 0.0], &[1, 1, 1, 2])?;
    let values = Array::from_f32(&[3.0, 4.0], &[1, 1, 1, 2])?;
    let output = queries.scaled_dot_product_attention(&keys, &values, 1.0, false, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(output.to_vec_f32()?, vec![3.0, 4.0]);
    Ok(())
}

#[test]
fn applies_causal_mask_during_prefill() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let queries = Array::from_f32(&[0.0, 0.0], &[1, 1, 2, 1])?;
    let keys = Array::from_f32(&[0.0, 0.0], &[1, 1, 2, 1])?;
    let values = Array::from_f32(&[3.0, 4.0], &[1, 1, 2, 1])?;
    let output = queries.scaled_dot_product_attention(&keys, &values, 1.0, true, &stream)?;
    output.async_eval()?;
    stream.synchronize()?;
    assert_eq!(output.to_vec_f32()?, vec![3.0, 3.5]);
    Ok(())
}

#[test]
fn keeps_greedy_argmax_on_mlx_until_requested() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let logits = Array::from_f32(&[1.0, 3.0, 2.0], &[1, 1, 3])?;
    let token = logits.argmax(&stream)?;
    token.async_eval()?;
    assert_eq!(token.item_u32()?, 1);
    Ok(())
}

#[test]
fn keeps_top_k_candidates_on_mlx_until_requested() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let logits = Array::from_f32(&[1.0, 4.0, 3.0, 2.0], &[1, 1, 4])?;
    let candidates = logits.top_k(2, 4, &stream)?;

    assert_eq!(candidates.token_ids.to_vec_u32_on_stream(&stream)?, vec![2, 1]);
    assert_eq!(candidates.scores.to_vec_f32_on_stream(&stream)?, vec![3.0, 4.0]);
    Ok(())
}

#[test]
fn snapshots_native_array_handle() -> Result<()> {
    let source = Array::from_f32(&[1.0, 2.0], &[1, 2])?;
    let snapshot = source.snapshot()?;
    drop(source);

    assert_eq!(snapshot.to_vec_f32()?, vec![1.0, 2.0]);
    Ok(())
}

#[test]
fn appends_and_slices_native_kv_cache_on_gpu() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new(2)?;
    let first_keys = Array::from_f32(&[1.0, 2.0], &[1, 1, 1, 2])?;
    let first_values = Array::from_f32(&[3.0, 4.0], &[1, 1, 1, 2])?;
    let _first = cache.update(&first_keys, &first_values, &stream)?;
    let second_keys = Array::from_f32(&[5.0, 6.0], &[1, 1, 1, 2])?;
    let second_values = Array::from_f32(&[7.0, 8.0], &[1, 1, 1, 2])?;
    let second = cache.update(&second_keys, &second_values, &stream)?;
    let third_keys = Array::from_f32(&[9.0, 10.0], &[1, 1, 1, 2])?;
    let third_values = Array::from_f32(&[11.0, 12.0], &[1, 1, 1, 2])?;
    let third = cache.update(&third_keys, &third_values, &stream)?;

    third.keys.async_eval()?;
    stream.synchronize()?;
    assert_eq!(cache.offset()?, 3);
    assert_eq!(second.keys.shape()?, vec![1, 1, 2, 2]);
    assert_eq!(second.keys.to_vec_f32()?, vec![1.0, 2.0, 5.0, 6.0]);
    assert_eq!(third.keys.shape()?, vec![1, 1, 3, 2]);
    assert_eq!(third.keys.to_vec_f32()?, vec![1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
    assert_eq!(third.values.to_vec_f32()?, vec![3.0, 4.0, 7.0, 8.0, 11.0, 12.0]);
    Ok(())
}

#[test]
fn snapshots_native_kv_cache_without_sharing_mutable_state() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new(2)?;
    let first = Array::from_f32(&[1.0, 2.0], &[1, 1, 1, 2])?;
    let initial = cache.update(&first, &first, &stream)?;
    initial.keys.async_eval()?;
    stream.synchronize()?;

    let snapshot = cache.snapshot_at(1)?;
    assert!(cache.snapshot_at(2).is_err());

    let second = Array::from_f32(&[3.0, 4.0], &[1, 1, 1, 2])?;
    let updated = cache.update(&second, &second, &stream)?;
    updated.keys.async_eval()?;
    stream.synchronize()?;

    assert_eq!(cache.offset()?, 2);
    assert_eq!(snapshot.offset()?, 1);
    Ok(())
}

#[test]
fn rotates_sliding_kv_cache_without_growing_context() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut cache = KvCache::new_with_window(2, Some(2))?;
    for values in [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] {
        let keys = Array::from_f32(&values, &[1, 1, 1, 2])?;
        let context = cache.update(&keys, &keys, &stream)?;
        context.keys.async_eval()?;
    }
    stream.synchronize()?;
    let last = Array::from_f32(&[7.0, 8.0], &[1, 1, 1, 2])?;
    let context = cache.update(&last, &last, &stream)?;
    context.keys.async_eval()?;
    stream.synchronize()?;
    assert_eq!(cache.offset()?, 4);
    assert_eq!(context.keys.shape()?, vec![1, 1, 2, 2]);
    assert_eq!(context.keys.to_vec_f32()?, vec![5.0, 6.0, 7.0, 8.0]);
    Ok(())
}

fn expert_weights() -> Vec<f32> {
    let mut values = vec![1.0_f32; 64];
    values.extend(vec![2.0_f32; 64]);
    values
}
