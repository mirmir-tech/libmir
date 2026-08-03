use runtime::backend::SamplingLogits;

use super::{bucket_sizes, build_outputs, sample_policies};

#[test]
fn prepares_common_and_maximum_decode_buckets() {
    assert_eq!(bucket_sizes(1).collect::<Vec<_>>(), [1]);
    assert_eq!(bucket_sizes(7).collect::<Vec<_>>(), [1, 2, 4, 5, 7]);
    assert_eq!(bucket_sizes(16).collect::<Vec<_>>(), [1, 2, 4, 5, 8, 10, 16]);
}

#[test]
fn preserves_mixed_host_and_device_sampling_rows() -> crate::Result<()> {
    let policies = [SamplingLogits::Full, SamplingLogits::None];
    assert_eq!(sample_policies(&policies), [SamplingLogits::None, SamplingLogits::None]);
    let outputs = build_outputs(&policies, &[7, 8], Some(&[1.0, 2.0, 3.0, 4.0]), 2)?;
    assert_eq!(outputs[0].event.token_id, None);
    assert_eq!(
        outputs[0].logits.as_ref().map(|trace| trace.values.as_slice()),
        Some(&[1.0, 2.0][..])
    );
    assert_eq!(outputs[1].event.token_id, Some(8));
    assert!(outputs[1].logits.is_none());
    Ok(())
}
