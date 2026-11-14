use std::{io::Write, sync::Arc};

use crate::engine::{
    Array, Error, KvCache, KvPageFormat, PagedArenaPool, PagedContextMode, Result, Stream,
    clear_memory_cache, memory_stats,
};

#[test]
#[ignore = "bounded GPU allocation diagnostic, no model weights"]
fn diagnoses_shared_terminal_page_copy_amplification() -> Result<()> {
    diagnose(1_025)
}

#[test]
#[ignore = "bounded GPU allocation diagnostic, no model weights"]
fn diagnoses_aligned_terminal_page_control() -> Result<()> {
    diagnose(1_024)
}

fn diagnose(tokens: usize) -> Result<()> {
    const ROWS: usize = 5;
    const CAPACITY: usize = 1_024;
    let pool = Arc::new(PagedArenaPool::default());
    let stream = Stream::new_gpu_with_config_and_pool(Arc::default(), Arc::clone(&pool))?;
    let new_cache = || {
        KvCache::new_paged_with_pool_capacity(
            256,
            16,
            KvPageFormat::Native,
            CAPACITY,
            Arc::clone(&pool),
            0,
        )
    };
    let token = Array::from_f32(&[1.0; 512], &[1, 2, 1, 256])?;
    let mut seed = new_cache()?;
    seed.reserve(CAPACITY * 16)?;
    let context = seed.update(&token, &token, &stream)?;
    stream.eval_many_with_paged_arenas(&[])?;
    stream.synchronize()?;
    stream.detach_paged_arena_graphs()?;
    drop(context);
    let sentinel = seed.snapshot_at(0)?;
    seed.reset()?;

    let initial = Array::from_f32(&vec![1.0; tokens * 512], &[1, 2, i32::try_from(tokens)?, 256])?;
    let mut caches = Vec::new();
    let mut snapshots = Vec::new();
    for _ in 0..ROWS {
        let mut cache = new_cache()?;
        cache.reserve(tokens + 256)?;
        let context = cache.update_for_attention_mode(
            &initial,
            &initial,
            &stream,
            0,
            PagedContextMode::Native,
        )?;
        stream.eval_many_with_paged_arenas(&[])?;
        stream.synchronize()?;
        stream.detach_paged_arena_graphs()?;
        cache.detach_evaluated_graphs(&stream)?;
        drop(context);
        snapshots.push(cache.snapshot_at(tokens)?);
        caches.push(cache);
    }
    drop(initial);
    clear_memory_cache()?;
    let before = memory_stats()?;
    let mut contexts = Vec::new();
    for cache in &mut caches {
        contexts.push(cache.update_for_attention_mode(
            &token,
            &token,
            &stream,
            0,
            PagedContextMode::Native,
        )?);
    }
    let roots = contexts
        .iter()
        .filter_map(|context| context.paged.as_ref())
        .map(|pages| &pages.page_dependency)
        .collect::<Vec<_>>();
    stream.eval_many_with_paged_arenas(&roots)?;
    stream.synchronize()?;
    let after = memory_stats()?;
    assert!(
        after.active <= before.active + 1024 * 1024,
        "page copies must not allocate historical arena generations: before={before:?}, after={after:?}"
    );
    let shape = contexts
        .last()
        .and_then(|context| context.paged.as_ref())
        .ok_or(Error::NullHandle("native page context"))?
        .key_pages
        .shape()?;
    writeln!(
        std::io::stderr().lock(),
        "page_copy rows={ROWS} tokens={tokens} arena_shape={shape:?} baseline={} active={} peak={} delta={}",
        before.active,
        after.active,
        after.peak,
        after.active.saturating_sub(before.active),
    )?;
    drop(roots);
    drop(contexts);
    stream.detach_paged_arena_graphs()?;
    report_memory("after_release")?;
    drop(caches);
    report_memory("after_drop_writer_plans")?;
    drop(seed);
    report_memory("after_drop_reset_seed_plan")?;
    drop(snapshots);
    drop(sentinel);
    Ok(())
}

fn report_memory(stage: &str) -> Result<()> {
    clear_memory_cache()?;
    writeln!(std::io::stderr().lock(), "page_copy {stage}={:?}", memory_stats()?)?;
    Ok(())
}
