#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryBatching {
    #[default]
    Joined,
    Rows,
    Gathered,
    Persistent,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GdnExecution {
    #[default]
    Native,
    PackedDecode,
    PackedPrefill,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyValueProjection {
    #[default]
    Separate,
    JoinedDecode,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RopeBatching {
    #[default]
    Rows,
    Offsets,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RouterPrecision {
    #[default]
    Model,
    Float32,
}

use std::path::PathBuf;

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GatedDeltaPrefill {
    #[default]
    Rows,
    Packed,
    DiagnosePacked,
    ComparePacked,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PrefixRetention {
    #[default]
    View,
    CompactCheckpoint,
}

#[derive(Debug, Clone, Default)]
pub struct MetalDiagnosticsConfig {
    #[cfg(test)]
    pub(crate) history_batching: HistoryBatching,
    #[cfg(test)]
    pub(crate) gdn_execution: GdnExecution,
    #[cfg(test)]
    pub(crate) key_value_projection: KeyValueProjection,
    #[cfg(test)]
    pub(crate) rope_batching: RopeBatching,
    #[cfg(test)]
    pub(crate) router_precision: RouterPrecision,
    #[cfg(test)]
    pub(crate) moe_decode_probe: Option<MoeDecodeProbe>,
    #[cfg(test)]
    pub(crate) moe_prefill: MoePrefill,
    #[cfg(test)]
    pub(crate) gated_delta_prefill: GatedDeltaPrefill,
    #[cfg(test)]
    pub(crate) prefill_component_window: Option<std::ops::Range<usize>>,
    #[cfg(test)]
    pub(crate) prefix_retention: PrefixRetention,
    #[cfg(test)]
    pub(crate) attention_candidate: Option<crate::engine::BatchAttentionExecution>,
    pub profile_layers: bool,
    pub profile_components: bool,
    pub profile_graph_build: bool,
    pub prefill_evaluation_layers: Option<usize>,
    pub graph_dump: Option<PathBuf>,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MoePrefill {
    #[default]
    Default,
    CompareRoutes,
    CompareProjection,
    CompareIndexed,
    MeasureIndexedNumerics,
    CompareAligned,
    CompareAlignmentBudgets,
    CompareTiles,
    CompareTileWidths,
    ProfileTiles,
    Tiles64,
    Aligned,
    ClampedSorted,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub enum MoeDecodeProbe {
    Components { step: usize },
    GateUpFusion { step: usize },
    SubmissionDrift { step: usize },
    FusionDrift { step: usize },
    SubmissionLengths { step: usize },
    ContinuousWindows { step: usize },
}
