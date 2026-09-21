use super::*;
use crate::CudaConfig;

#[test]
fn scratch_shares_capacity_only_on_the_same_runtime_and_releases_with_owners() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let first = backend.inner.layer_scratch.acquire(&backend, 16, 64)?;
    let second = backend.inner.layer_scratch.acquire(&backend, 32, 32)?;
    assert!(Arc::ptr_eq(&first.buffers, &second.buffers));
    let other = backend.inner.layer_scratch.acquire(&backend, 32, 64)?;
    assert!(!Arc::ptr_eq(&first.buffers, &other.buffers));
    let auxiliary = backend.auxiliary_backend();
    let separate = auxiliary.inner.layer_scratch.acquire(&auxiliary, 16, 64)?;
    assert!(!Arc::ptr_eq(&first.buffers, &separate.buffers));
    let weak = Arc::downgrade(&first.buffers);
    drop(first);
    assert!(weak.upgrade().is_some());
    drop(second);
    assert!(weak.upgrade().is_none());
    Ok(())
}

#[test]
fn shared_layer_temporaries_preserve_order_across_graph_replays() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let first = backend.inner.layer_scratch.acquire(&backend, 1, 64)?;
    let second = backend.inner.layer_scratch.acquire(&backend, 1, 64)?;
    let stream = backend.inner.stream.clone();
    let mut host = backend.inner.context.allocate_pinned::<bf16>(64)?;
    host.copy_from_slice(&[bf16::ONE; 64])?;
    let mut input = backend.inner.pool.allocate::<bf16>(&stream, 64)?;
    stream.copy_to_device(&mut host, &mut input)?;
    let outputs = [
        backend.inner.pool.allocate::<bf16>(&stream, 64)?,
        backend.inner.pool.allocate::<bf16>(&stream, 64)?,
    ];
    let kernel = crate::kernels::ElementwiseBf16::compile(&backend.inner.compiler, 64)?;
    backend.synchronize()?;
    let mut graph = stream.capture(
        ([first, second], input, outputs, kernel, stream.clone()),
        |r| -> Result<()> {
            for (index, (scratch, output)) in r.0.iter().zip(&mut r.2).enumerate() {
                let mut guard = scratch.lock()?;
                let scratch = &mut *guard;
                r.3.add(&r.4, &r.1, &r.1, &mut scratch.normalized)?;
                r.3.add(&r.4, &scratch.normalized, &r.1, &mut scratch.attention)?;
                r.3.add(&r.4, &scratch.attention, &r.1, &mut scratch.residual)?;
                r.3.add(&r.4, &scratch.residual, &r.1, &mut scratch.moe)?;
                r.3.add(
                    &r.4,
                    &scratch.moe,
                    if index == 0 {
                        &r.1
                    } else {
                        &scratch.normalized
                    },
                    output,
                )?;
                drop(guard);
            }
            Ok(())
        },
    )?;
    for _ in 0..3 {
        graph.launch(&stream)?;
        for (output, expected) in graph.resources().2.iter().zip([6.0, 7.0]) {
            stream.copy_to_host(output, &mut host)?;
            assert_eq!(host.to_vec()?, vec![bf16::from_f32(expected); 64]);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires an idle CUDA device; measures the shared driver memory pool"]
fn shared_layer_buffers_allocate_once_and_release_after_last_owner() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    backend.synchronize()?;
    let base = backend.memory_pool_stats()?.used;
    let first = backend.inner.layer_scratch.acquire(&backend, 1024, 1024)?;
    backend.synchronize()?;
    let allocated = backend.memory_pool_stats()?.used;
    assert!(allocated >= base + 8 * 1024 * 1024);
    let second = backend.inner.layer_scratch.acquire(&backend, 1024, 1024)?;
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
