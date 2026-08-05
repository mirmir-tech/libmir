use super::{super::required_prefill_pages, sequence::Sequence};
use crate::native::{error::Result, model::LoadedModel};

fn required_values(pending: bool, tokens: usize, position: usize, page_size: usize) -> usize {
    usize::from(pending) * required_prefill_pages(tokens, position, page_size)
}

pub(super) fn required(sequence: &Sequence, page_size: usize) -> usize {
    required_values(
        sequence.page_reservation_pending,
        sequence.request.prompt_tokens.len(),
        sequence.position,
        page_size,
    )
}

pub(super) fn ensure(sequence: &Sequence, loaded: &mut LoadedModel) -> Result<()> {
    let page_size = loaded.stream.config().kv_cache.block_size.max(1);
    loaded.reserve_prefill_pages(required(sequence, page_size))
}

#[cfg(test)]
mod tests {
    use super::required_values;

    #[test]
    fn rechecks_only_until_physical_reservation_is_installed() {
        assert_eq!(required_values(true, 8_194, 0, 16), 514);
        assert_eq!(required_values(false, 8_194, 0, 16), 0);
    }
}
