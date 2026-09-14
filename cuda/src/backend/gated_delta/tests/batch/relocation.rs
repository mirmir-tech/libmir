use super::*;

#[test]
fn resident_rows_survive_in_place_reordering() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let config = GatedDeltaStateConfig {
        key_heads: 1,
        value_heads: 1,
        key_dim: 32,
        value_dim: 2,
        convolution_kernel_size: 2,
    };
    let mut first = backend.prepare_gated_delta_state(config)?;
    let mut second = backend.prepare_gated_delta_state(config)?;
    let first_values = vec![1.0_f32; first.state.len()];
    let second_values = vec![2.0_f32; second.state.len()];
    let first_history = vec![bf16::from_f32(3.0); first.convolution.len()];
    let second_history = vec![bf16::from_f32(4.0); second.convolution.len()];
    first.state = copy(&backend, &first_values)?;
    second.state = copy(&backend, &second_values)?;
    first.convolution = copy(&backend, &first_history)?;
    second.convolution = copy(&backend, &second_history)?;
    let mut batch = CudaGatedDeltaBatchState::new(&backend, config, 2, 1)?;
    batch.pack(&[&mut first, &mut second])?;
    batch.commit(&mut [&mut first, &mut second])?;
    batch.pack(&[&mut second, &mut first])?;
    batch.commit(&mut [&mut second, &mut first])?;
    first.materialize()?;
    second.materialize()?;
    assert_eq!(read(&backend, &first.state)?, first_values);
    assert_eq!(read(&backend, &second.state)?, second_values);
    assert_eq!(read(&backend, &first.convolution)?, first_history);
    assert_eq!(read(&backend, &second.convolution)?, second_history);
    Ok(())
}

#[test]
fn reusing_a_batch_preserves_sessions_outside_the_next_cohort() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut first = seeded(&backend, 1.0)?;
    let mut second = seeded(&backend, 2.0)?;
    let mut third = seeded(&backend, 3.0)?;
    let mut fourth = seeded(&backend, 4.0)?;
    let mut batch = CudaGatedDeltaBatchState::new(&backend, first.config, 2, 1)?;
    batch.pack(&[&mut first, &mut second])?;
    batch.commit(&mut [&mut first, &mut second])?;
    batch.pack(&[&mut third, &mut fourth])?;
    batch.commit(&mut [&mut third, &mut fourth])?;
    for (state, value) in
        [(&mut first, 1.0), (&mut second, 2.0), (&mut third, 3.0), (&mut fourth, 4.0)]
    {
        state.materialize()?;
        assert_eq!(read(&backend, &state.state)?, vec![value; state.state.len()]);
        assert_eq!(
            read(&backend, &state.convolution)?,
            vec![bf16::from_f32(value); state.convolution.len()]
        );
    }
    Ok(())
}

#[test]
fn reusing_an_old_batch_does_not_overwrite_materialized_progress() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut first = seeded(&backend, 1.0)?;
    let mut second = seeded(&backend, 2.0)?;
    let mut third = seeded(&backend, 3.0)?;
    let mut batch = CudaGatedDeltaBatchState::new(&backend, first.config, 2, 1)?;
    batch.pack(&[&mut first, &mut second])?;
    batch.commit(&mut [&mut first, &mut second])?;
    first.materialize()?;
    let updated = vec![7.0_f32; first.state.len()];
    let source = copy(&backend, &updated)?;
    backend
        .inner
        .stream
        .copy_device_range(&source, 0..source.len(), &mut first.state, 0)?;
    first.advance(1)?;
    batch.pack(&[&mut third, &mut second])?;
    batch.commit(&mut [&mut third, &mut second])?;
    assert_eq!(read(&backend, &first.state)?, updated);
    Ok(())
}

fn seeded(backend: &CudaBackend, value: f32) -> Result<CudaGatedDeltaState> {
    let config = GatedDeltaStateConfig {
        key_heads: 1,
        value_heads: 1,
        key_dim: 32,
        value_dim: 2,
        convolution_kernel_size: 2,
    };
    let mut state = backend.prepare_gated_delta_state(config)?;
    state.state = copy(backend, &vec![value; state.state.len()])?;
    state.convolution = copy(backend, &vec![bf16::from_f32(value); state.convolution.len()])?;
    Ok(state)
}
