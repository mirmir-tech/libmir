use super::{AttentionBias, AttentionRequest};
use crate::engine::{Array, KvCache, PagedContextMode, Result, Stream};

#[test]
fn paged_capabilities_preserve_sinks_and_explicit_masks() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let query = Array::from_f32(&[0.25; 32], &[1, 1, 1, 32])?;
    let keys = Array::from_f32(&[0.5; 64], &[1, 1, 2, 32])?;
    let values = Array::from_f32(&[vec![0.25; 32], vec![0.75; 32]].concat(), &[1, 1, 2, 32])?;
    let sinks = Array::from_f32(&[0.0], &[1])?;
    let mut cache = KvCache::new_paged(16, 16)?;
    let mut context =
        cache.update_for_attention_mode(&keys, &values, &stream, 0, PagedContextMode::Both)?;
    let plain = AttentionRequest::new(&query, &context, 0.125, false, AttentionBias::None)?
        .execute(&stream)?;
    let plain = plain.to_vec_f32(&stream)?;
    let biased =
        AttentionRequest::new(&query, &context, 0.125, false, AttentionBias::Sinks(&sinks))?
            .execute(&stream)?
            .to_vec_f32(&stream)?;
    assert!(biased.iter().zip(&plain).all(|(biased, plain)| biased > &0.0 && biased < plain));
    context.mask = Some(Array::from_f32(&[0.0, f32::NEG_INFINITY], &[1, 2])?);
    let masked = AttentionRequest::new(&query, &context, 0.125, false, AttentionBias::None)?
        .execute(&stream)?
        .to_vec_f32(&stream)?;
    assert_eq!(masked, vec![0.25; 32]);
    let both = AttentionRequest::new(&query, &context, 0.125, false, AttentionBias::Sinks(&sinks))?
        .execute(&stream)?
        .to_vec_f32(&stream)?;
    assert!(both.iter().all(|value| *value > 0.0 && *value < 0.25));
    // A native-only context has no valid chronological view for this fallback.
    context.keys = Array::from_f32(&[], &[0])?;
    assert!(AttentionRequest::new(&query, &context, 0.125, false, AttentionBias::None).is_err());
    context.mask = None;
    assert!(
        AttentionRequest::new(&query, &context, 0.125, false, AttentionBias::Sinks(&sinks))
            .is_err()
    );
    Ok(())
}
