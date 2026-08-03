use runtime::progress::{ProgressEvent, ProgressStage, ProgressUnit};

pub(super) fn metal_progress(event: metal::MetalProgressEvent) -> ProgressEvent {
    ProgressEvent {
        stage: match event.stage {
            metal::MetalProgressStage::LoadWeights => ProgressStage::LoadWeights,
            metal::MetalProgressStage::PrefillTokens => ProgressStage::PrefillTokens,
        },
        current: event.current,
        total: event.total,
        unit: match event.unit {
            metal::MetalProgressUnit::Byte => ProgressUnit::Byte,
            metal::MetalProgressUnit::Token => ProgressUnit::Token,
        },
        detail: event.detail,
    }
}
