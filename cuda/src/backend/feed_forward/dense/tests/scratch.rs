use super::*;

#[test]
#[ignore = "requires an idle CUDA device; measures the shared driver memory pool"]
fn scratch_reuses_storage_and_releases_it_with_last_execution() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    backend.synchronize()?;
    let base = backend.memory_pool_stats()?.used;
    let first = backend.inner.dense_scratch.acquire(&backend, 1024 * 1024)?;
    backend.synchronize()?;
    let allocated = backend.memory_pool_stats()?.used;
    assert!(allocated >= base + 6 * 1024 * 1024);
    let second = backend.inner.dense_scratch.acquire(&backend, 1024 * 1024)?;
    backend.synchronize()?;
    assert_eq!(backend.memory_pool_stats()?.used, allocated);
    drop(first);
    backend.synchronize()?;
    assert_eq!(backend.memory_pool_stats()?.used, allocated);
    drop(second);
    backend.synchronize()?;
    assert_eq!(backend.memory_pool_stats()?.used, base);
    Ok(())
}

#[test]
fn shared_scratch_preserves_distinct_weights_and_graph_replay() -> Result<()> {
    let first = template()?;
    let backend = &first.backend;
    let mut second = first.clone();
    second.down = first.up.clone();
    let stream = backend.inner.stream.clone();
    let upload = |values: &[f32]| -> Result<DeviceBuffer<bf16>> {
        let mut host = backend.inner.context.allocate_pinned(values.len())?;
        host.copy_from_slice(&values.iter().copied().map(bf16::from_f32).collect::<Vec<_>>())?;
        let mut buffer = backend.inner.pool.allocate(&stream, values.len())?;
        stream.copy_to_device(&mut host, &mut buffer)?;
        Ok(buffer)
    };
    for tokens in [1, 3] {
        let values = [0.5_f32, -1.0, 2.0, -0.5, 1.5, 0.25];
        let input = upload(&values[..2 * tokens])?;
        let other = upload(&[-1.0_f32, 0.5, -0.5, 2.0, 0.25, 1.5][..2 * tokens])?;
        let allocate = || backend.inner.pool.allocate(&stream, 2 * tokens);
        let mut resources = (
            first.prepare(tokens)?,
            second.prepare(tokens)?,
            input,
            other,
            allocate()?,
            allocate()?,
        );
        // Prepare lazy library state before capture, with both plans alive.
        resources.0.execute(&resources.2, &mut resources.4)?;
        resources.1.execute(&resources.3, &mut resources.5)?;
        backend.synchronize()?;
        let mut graph = stream.capture(resources, |r| -> Result<()> {
            r.0.execute(&r.2, &mut r.4)?;
            r.1.execute(&r.3, &mut r.5)
        })?;
        for _ in 0..3 {
            graph.launch(&stream)?;
            let r = graph.resources();
            for (output, inputs, identity) in [
                (&r.4, &values[..2 * tokens], false),
                (&r.5, &[-1.0_f32, 0.5, -0.5, 2.0, 0.25, 1.5][..2 * tokens], true),
            ] {
                let mut host = backend.inner.context.allocate_pinned(2 * tokens)?;
                stream.copy_to_host(output, &mut host)?;
                for (actual, input) in
                    host.to_vec()?.as_chunks::<2>().0.iter().zip(inputs.as_chunks::<2>().0)
                {
                    let gated = |gate: f32, up: f32| {
                        bf16::from_f32(gate / (1.0 + (-gate).exp()) * up).to_f32()
                    };
                    let a = gated(2.0 * input[0], input[0]);
                    let b = gated(0.5 * input[1], input[1]);
                    let expected = if identity {
                        [a, b]
                    } else {
                        [0.25_f32.mul_add(b, a), (-0.5_f32).mul_add(a, b)]
                    };
                    for (actual, expected) in actual.iter().zip(expected) {
                        assert!(
                            (actual.to_f32() - expected).abs() < 0.04,
                            "actual={actual:?}, expected={expected}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
