use super::ClampedRoutedExecutionPlan;
use crate::PagedPrefillBatch;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::backend::clamped_routed) enum ClampedRoutedDecodeSignature {
    Direct { context_bucket: usize },
    Split { partitions: usize },
}

impl ClampedRoutedExecutionPlan {
    pub(in crate::backend::clamped_routed) fn decode_signature(
        &self,
        batch: &PagedPrefillBatch,
    ) -> ClampedRoutedDecodeSignature {
        let partitions = self
            .batch_split_decode
            .as_ref()
            .map_or(0, |split| split.capture_partitions(batch));
        if partitions > 0 {
            ClampedRoutedDecodeSignature::Split { partitions }
        } else {
            ClampedRoutedDecodeSignature::Direct {
                context_bucket: batch.fmha_max_context_tokens(),
            }
        }
    }
}
