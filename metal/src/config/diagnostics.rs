use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct MetalDiagnosticsConfig {
    pub profile_layers: bool,
    pub profile_components: bool,
    pub profile_graph_build: bool,
    pub prefill_evaluation_layers: Option<usize>,
    pub graph_dump: Option<PathBuf>,
}
