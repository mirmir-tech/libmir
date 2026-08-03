use uuid::Uuid;

use super::{ModelHandle, PrefillRequest, SamplingLogits};
use crate::kv::BlockTable;

fn request(tokens: usize, block: usize) -> PrefillRequest {
    PrefillRequest {
        model: ModelHandle {
            id: "model".into(),
            backend: "test".into(),
        },
        session_id: Uuid::nil(),
        prompt_tokens: vec![0; tokens],
        cache_checkpoints: Vec::new(),
        block_table: BlockTable::with_block_size(block),
        cached_tokens: 0,
        sampling_logits: SamplingLogits::None,
    }
}

#[test]
fn terminal_checkpoint_retains_one_uncommitted_tail_block() {
    assert_eq!(request(4_103, 16).terminal_cache_checkpoint(), Some(4_080));
    assert_eq!(request(16, 16).terminal_cache_checkpoint(), None);
    assert_eq!(request(15, 16).terminal_cache_checkpoint(), None);
    assert_eq!(request(4_103, 0).terminal_cache_checkpoint(), None);
}
