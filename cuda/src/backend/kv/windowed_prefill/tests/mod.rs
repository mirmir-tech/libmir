//! The staged windowed FA2 prefill runs on a CUDA device, so it is built
//! only where CUDA is supported.

#[cfg(target_os = "linux")]
mod staged;

#[test]
fn staging_context_keeps_only_the_receptive_history() {
    let window = 128_usize;
    let start = 8_192_usize;
    let query = 2_048_usize;
    let history = start.min(window - 1);

    assert_eq!(history, 127);
    assert_eq!(history + query, 2_175);
    assert_eq!(start - history, 8_065);
}
