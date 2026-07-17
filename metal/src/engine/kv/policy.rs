use super::{PagedContextMode, Stream};

pub const NATIVE_PAGED_ATTENTION_MIN_CONTEXT: usize = 8_192;

pub const fn paged_attention_enabled() -> bool {
    true
}

pub fn paged_attention_min_context(stream: &Stream) -> usize {
    stream.config().cache.paged_attention_min_context
}

pub fn native_paged_attention_mode(
    head_dim: i32,
    query_heads: i32,
    kv_heads: i32,
    context: usize,
    force_native: bool,
) -> PagedContextMode {
    let supported = head_dim > 0
        && head_dim <= 512
        && query_heads > 0
        && kv_heads > 0
        && query_heads % kv_heads == 0
        && context > 0;
    if !supported {
        return PagedContextMode::View;
    }
    if force_native
        || automatic_native_paged_attention(head_dim, kv_heads, query_heads / kv_heads, context)
    {
        PagedContextMode::Native
    } else {
        PagedContextMode::NativeIfFragmented
    }
}

fn automatic_native_paged_attention(
    head_dim: i32,
    kv_heads: i32,
    group_factor: i32,
    context: usize,
) -> bool {
    context >= NATIVE_PAGED_ATTENTION_MIN_CONTEXT
        && head_dim <= 256
        && head_dim % 32 == 0
        && kv_heads >= 8
        && (5..=32).contains(&group_factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enables_only_benchmarked_long_context_shapes_automatically() {
        assert!(automatic_native_paged_attention(256, 8, 8, 8_192));
        assert!(!automatic_native_paged_attention(256, 8, 8, 4_096));
        assert!(!automatic_native_paged_attention(512, 8, 8, 8_192));
        assert!(!automatic_native_paged_attention(256, 8, 4, 8_192));
        assert!(!automatic_native_paged_attention(256, 2, 8, 8_192));
    }

    #[test]
    fn uses_native_attention_for_supported_fragmented_pages() {
        assert_eq!(
            native_paged_attention_mode(256, 16, 2, 128, false),
            PagedContextMode::NativeIfFragmented
        );
        assert_eq!(native_paged_attention_mode(768, 16, 2, 128, false), PagedContextMode::View);
    }
}
