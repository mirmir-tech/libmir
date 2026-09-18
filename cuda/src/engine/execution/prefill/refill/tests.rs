use runtime::{
    backend::{ModelHandle, PrefillRequest, SamplingLogits},
    kv::BlockTable,
};

use super::validate;

fn request(session_id: uuid::Uuid) -> PrefillRequest {
    PrefillRequest {
        model: ModelHandle {
            id: "model".into(),
            backend: "cuda".into(),
        },
        session_id,
        prompt_tokens: vec![1; 128],
        cache_checkpoints: vec![],
        block_table: BlockTable::with_block_size(16),
        cached_tokens: 0,
        generation_tokens: None,
        sampling_logits: SamplingLogits::None,
    }
}

#[test]
fn rejects_duplicates_and_mismatched_models_before_preparation() {
    let active = uuid::Uuid::new_v4();
    let incoming = request(uuid::Uuid::new_v4());
    let model = incoming.model.clone();
    assert!(validate(&model, [active].into_iter(), std::slice::from_ref(&incoming)).is_ok());
    assert!(validate(&model, [active].into_iter(), &[request(active)]).is_err());
    assert!(validate(&model, [active].into_iter(), &[incoming.clone(), incoming.clone()]).is_err());
    let mut wrong = incoming;
    wrong.model.id = "another".into();
    assert!(validate(&model, [active].into_iter(), &[wrong]).is_err());
    assert!(validate(&model, [active].into_iter(), &[]).is_err());
}

#[test]
fn prepend_preserves_existing_progress_and_request_order() {
    use std::time::{Duration, Instant};

    use super::super::{CudaPrefillBatch, Sequence, prefix::PrefixReuse};
    fn batch(request: PrefillRequest) -> CudaPrefillBatch {
        CudaPrefillBatch {
            model_id: request.model.id.clone(),
            scheduled_tokens: request.prompt_tokens.len(),
            sequences: vec![Sequence::new(request, PrefixReuse::default(), Duration::ZERO)],
            cursor: 0,
            rounds: 0,
            token_budget: 1024,
            started: Instant::now(),
        }
    }
    let old_id = uuid::Uuid::new_v4();
    let new_id = uuid::Uuid::new_v4();
    let mut old = request(old_id);
    old.prompt_tokens.resize(8192, 1);
    let mut active = batch(old);
    active.sequences[0].consumed = 4096;
    active.sequences[0].chunks = 4;
    active.rounds = 4;
    let started = active.started;
    super::prepend(&mut active, batch(request(new_id)));
    assert_eq!(
        active.sequences.iter().map(|s| s.request.session_id).collect::<Vec<_>>(),
        [new_id, old_id]
    );
    assert_eq!(active.sequences[1].consumed, 4096);
    assert_eq!(active.sequences[1].chunks, 4);
    assert_eq!(active.rounds, 4);
    assert_eq!(active.started, started);
    assert_eq!(active.scheduled_tokens, 8192 + 128);
}
