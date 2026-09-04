use super::{
    super::{RoutingExecution, fallback, key},
    Array, Result, spec,
};

#[test]
fn buckets_route_shapes_without_reading_indices() -> Result<()> {
    let input = Array::from_f32(&vec![0.0; 17 * 64], &[1, 17, 64])?;
    let indices = Array::from_u32(&vec![0; 17 * 4], &[1, 17, 4])?;
    let profile = key(spec(8, 96, false), &input, &indices)?;
    assert_eq!(profile.route_bucket, 128);
    assert_eq!(profile.top_k, 4);
    let below_input = Array::from_f32(&vec![0.0; 9 * 64], &[1, 9, 64])?;
    let below_threshold = Array::from_u32(&[0; 9 * 4], &[1, 9, 4])?;
    assert_eq!(key(spec(8, 96, false), &below_input, &below_threshold)?.route_bucket, 64);
    assert_eq!(fallback(&below_threshold)?, RoutingExecution::Unsorted);
    let grouped = Array::from_u32(&vec![0; 256 * 8], &[1, 256, 8])?;
    assert_eq!(fallback(&grouped)?, RoutingExecution::GroupedFused);
    Ok(())
}
