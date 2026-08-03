use super::{
    super::{key, route_patterns},
    spec,
};
use crate::engine::{Array, Result, Stream};

#[test]
fn tuning_patterns_cover_balanced_and_hot_routes() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let input = Array::from_f32(&vec![0.0; 4 * 64], &[1, 4, 64])?;
    let indices = Array::from_u32(&[7, 6, 5, 4, 3, 2, 1, 0], &[1, 4, 2])?;
    let profile = key(spec(8, 96, false), &input, &indices)?;
    let patterns = route_patterns(profile, &indices)?;

    assert_eq!(patterns.balanced.to_vec_u32_on_stream(&stream)?, [0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(patterns.hot_set.to_vec_u32_on_stream(&stream)?, [0, 1, 0, 1, 0, 1, 0, 1]);
    Ok(())
}
