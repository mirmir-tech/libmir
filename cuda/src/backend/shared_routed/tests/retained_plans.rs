use super::*;
use crate::backend::shared_routed::plan::SharedRoutedExecutionPlan;

#[test]
#[ignore = "requires an idle CUDA device; measures the shared driver memory pool"]
fn replacement_releases_retained_scratch_before_allocating_a_large_plan() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut decoder = decoder()?;
    decoder.num_experts = None;
    decoder.top_k_experts = None;
    decoder.moe_intermediate_size = None;
    decoder.shared_expert_intermediate_size = None;
    decoder.intermediate_size = 64;
    let fixture = fixture::HybridFixture::new(&decoder)?;
    let template = backend.load_shared_routed_model_template(
        &decoder,
        &fixture.catalog(),
        crate::SharedRoutedModelLoadConfig {
            cache: CacheConfig {
                block_size: 16,
                block_count: 512,
                dtype: KvCacheDType::BFloat16,
            },
            max_sequence_blocks: 512,
        },
    )?;
    backend.synchronize()?;
    let base = backend.memory_pool_stats()?.used;
    let mut plans = template.plans.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    for tokens in [1024, 1008] {
        plans.reserve(tokens);
        plans.insert(tokens, SharedRoutedExecutionPlan::new(&template, tokens)?);
    }
    backend.synchronize()?;
    assert_eq!(plans.len(), 2);
    assert!(backend.memory_pool_stats()?.used > base);

    plans.reserve(2048);
    backend.synchronize()?;
    assert_eq!(plans.len(), 0);
    assert_eq!(backend.memory_pool_stats()?.used, base);
    plans.insert(2048, SharedRoutedExecutionPlan::new(&template, 2048)?);
    plans.reserve(1);
    plans.insert(1, SharedRoutedExecutionPlan::new(&template, 1)?);
    assert!(plans.get_mut(1).is_some());
    plans.reserve(4096);
    assert_eq!(plans.len(), 1);
    assert!(plans.get_mut(1).is_some());
    assert!(plans.get_mut(2048).is_none());
    plans.insert(4096, SharedRoutedExecutionPlan::new(&template, 4096)?);
    assert_eq!(plans.len(), 2);
    Ok(())
}
