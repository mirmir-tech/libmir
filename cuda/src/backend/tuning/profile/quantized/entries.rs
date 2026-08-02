use std::collections::HashMap;

use super::{
    super::{QuantizedRuntimeEntry, storage::StoredQuantizedEntry},
    QuantizedProfileFormat, QuantizedProfileRequest,
};

pub(in crate::backend::tuning::profile) fn stored_entries(
    entries: &HashMap<QuantizedProfileRequest, QuantizedRuntimeEntry>,
) -> Vec<StoredQuantizedEntry> {
    let mut stored = entries
        .iter()
        .map(|(request, entry)| StoredQuantizedEntry {
            request: *request,
            execution: entry.execution,
            average_ns: entry.average_ns,
        })
        .collect::<Vec<_>>();
    stored.sort_by_key(|entry| {
        (
            entry.request.phase as u8,
            entry.request.tokens,
            entry.request.input_features,
            entry.request.output_features,
            format_key(entry.request.format),
        )
    });
    stored
}

fn format_key(format: QuantizedProfileFormat) -> (usize, usize, usize, usize) {
    match format {
        QuantizedProfileFormat::Affine { group_size, bits } => (0, group_size, bits, 0),
        QuantizedProfileFormat::MxFp8 => (1, 0, 0, 0),
        QuantizedProfileFormat::DirectFp8DynamicE4M3OutputChannel { scale_dtype, bias } => {
            (2, scale_dtype as usize, usize::from(bias), 0)
        },
        QuantizedProfileFormat::DirectFp8StaticE4M3 { weight_scale, scale_dtype, bias } => {
            (3, weight_scale as usize, scale_dtype as usize, usize::from(bias))
        },
        QuantizedProfileFormat::DirectFp8Bf16E5M2WeightOnly { bias } => {
            (4, usize::from(bias), 0, 0)
        },
    }
}
