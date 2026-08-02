use libmir::{GenerationOverrides, Library, Result, RuntimeConfig, SamplingLogits};

use super::{join_prefill, request};

#[test]
#[ignore = "loads a real checkpoint; set MODEL"]
fn interleaves_concurrent_prefills_without_crossing_kv_sessions() -> Result<()> {
    let path = std::env::var_os("MODEL").ok_or(libmir::Error::MissingEnvironment("MODEL"))?;
    let mut config = RuntimeConfig::default();
    config.kv_cache.block_count = 256;
    config.scheduler.max_batch_requests = 2;
    config.scheduler.max_batch_tokens = 128;
    config.scheduler.decode_batch_wait_us = 50_000;
    let model =
        Library::new(config).load(path, GenerationOverrides::default(), &mut |_event| {})?;
    let prompt = model.prepare(&request(&model))?.tokens.token_ids;
    let first_prompt = prompt.iter().copied().cycle().take(512).collect::<Vec<_>>();
    let second_prompt = prompt.iter().copied().cycle().take(640).collect::<Vec<_>>();
    let first_reference_prompt = first_prompt.clone();
    let second_reference_prompt = second_prompt.clone();
    let mut first = model.session();
    let mut second = model.session();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
    let first_barrier = barrier.clone();
    let first = std::thread::spawn(move || {
        first_barrier.wait();
        first.prefill(&first_prompt, SamplingLogits::None, &mut |_event| {})
    });
    let second_barrier = barrier.clone();
    let second = std::thread::spawn(move || {
        second_barrier.wait();
        second.prefill(&second_prompt, SamplingLogits::None, &mut |_event| {})
    });
    barrier.wait();
    let first = join_prefill(first)?;
    let second = join_prefill(second)?;
    let mut first_reference = model.session();
    let mut second_reference = model.session();
    let first_reference =
        first_reference.prefill(&first_reference_prompt, SamplingLogits::None, &mut |_event| {})?;
    let second_reference = second_reference.prefill(
        &second_reference_prompt,
        SamplingLogits::None,
        &mut |_event| {},
    )?;
    assert_eq!(first.next_token, first_reference.next_token);
    assert_eq!(second.next_token, second_reference.next_token);
    assert!(
        first
            .trace
            .as_deref()
            .is_some_and(|trace| trace.contains("token-budget-round-robin"))
    );
    assert!(
        second
            .trace
            .as_deref()
            .is_some_and(|trace| trace.contains("token-budget-round-robin"))
    );
    Ok(())
}
