use super::*;

#[test]
fn selects_largest_router_scores() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let scores = (0_u16..128).map(f32::from).collect::<Vec<_>>();
    let scores = Array::from_f32(&scores, &[1, 1, 128])?;
    let scales = Array::from_f32(&vec![1.0; 128], &[128])?;
    let mut indices = scores.router_top_k(&scales, 8, &stream)?.indices.to_vec_u32(&stream)?;
    indices.sort_unstable();
    assert_eq!(indices, (120_u32..128).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn selects_largest_scores_for_each_router_token() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let mut scores = (0_u16..128).map(f32::from).collect::<Vec<_>>();
    scores.extend((0_u16..128).rev().map(f32::from));
    let scores = Array::from_f32(&scores, &[1, 2, 128])?;
    let scales = Array::from_f32(&vec![1.0; 128], &[128])?;
    let indices = scores.router_top_k(&scales, 8, &stream)?.indices.to_vec_u32(&stream)?;
    assert_eq!(indices.len(), 16);
    let mut first = indices[..8].to_vec();
    let mut second = indices[8..].to_vec();
    first.sort_unstable();
    second.sort_unstable();
    assert_eq!(first, (120_u32..128).collect::<Vec<_>>());
    assert_eq!(second, (0_u32..8).collect::<Vec<_>>());
    Ok(())
}

#[test]
fn unit_router_matches_an_explicit_all_ones_scale() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let scores = (0_u16..128).map(|value| f32::from(value) / 17.0).collect::<Vec<_>>();
    let scores = Array::from_f32(&scores, &[1, 1, 128])?;
    let unit_factors = Array::from_f32(&vec![1.0; 128], &[128])?;
    let explicit_routing = scores.router_top_k(&unit_factors, 8, &stream)?;
    let unit = scores.router_top_k_unit(8, &stream)?;

    explicit_routing.indices.async_eval(&stream)?;
    explicit_routing.weights.async_eval(&stream)?;
    unit.indices.async_eval(&stream)?;
    unit.weights.async_eval(&stream)?;
    stream.synchronize()?;
    assert_eq!(explicit_routing.indices.to_vec_u32(&stream)?, unit.indices.to_vec_u32(&stream)?,);
    assert_eq!(explicit_routing.weights.to_vec_f32(&stream)?, unit.weights.to_vec_f32(&stream)?);
    Ok(())
}
