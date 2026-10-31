use super::LoadedModel;

pub(super) fn can_finish(
    loaded: &LoadedModel,
    position: usize,
    remaining: usize,
    count: usize,
    checkpoint: Option<usize>,
) -> bool {
    loaded.stream.config().diagnostics.prefill_output == crate::config::PrefillOutput::LastChunk
        && remaining == count
        && checkpoint != Some(position + count)
        && loaded.prefill_chunk_len(position, count + 1) == count + 1
}

pub(super) fn can_defer(
    loaded: &LoadedModel,
    position: usize,
    remaining: usize,
    count: usize,
    checkpoint: Option<usize>,
) -> bool {
    loaded.stream.config().diagnostics.prefill_output == crate::config::PrefillOutput::DeferredToken
        && remaining == count
        && count <= 16
        && checkpoint != Some(position + count)
}
