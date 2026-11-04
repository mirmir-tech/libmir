use std::num::NonZeroUsize;

use super::*;

#[test]
#[ignore = "actual-model 512/1024/2048 prefill chunk correctness and resource diagnostic"]
fn diagnoses_larger_prefill_chunks() -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let context = 4097;
    let mut settings = diagnostics::isolated_config();
    let metal = Arc::make_mut(&mut settings);
    metal.tuning.mode = TuningMode::Disabled;
    metal.cache.prefix_cache_entries = 4;
    let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
    let prompt = (1_000..5_097).collect::<Vec<u32>>();
    let mut reference = None;
    for chunk in [512, 1024, 2048] {
        model.info.prefill_step = chunk;
        model.stream.set_prefill_chunk_limit(
            NonZeroUsize::new(chunk).ok_or(crate::engine::Error::ShapeOverflow)?,
        );
        let observation = run(&mut model, &prompt, context - 1, 64)?;
        let agrees = reference.as_ref().is_none_or(|tokens| tokens == &observation.tokens);
        assert_eq!(observation.progress.len(), (context - 1) / chunk + 2);
        writeln!(
            std::io::stderr().lock(),
            "chunks.sample: {}",
            json!({
                "chunk": chunk, "context": context, "matches_reference": agrees,
                "observation": observation,
            })
        )?;
        assert!(agrees, "larger chunk changed the greedy trajectory");
        reference.get_or_insert(observation.tokens);
    }
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok(())
}
