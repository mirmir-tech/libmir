use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BackendCapability {
    Prefill,
    Decode,
    Streaming,
    PrefixCache,
    Quantization(String),
    GraphCapture,
    ContinuousBatching,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub name: String,
    pub device: String,
    pub capabilities: Vec<BackendCapability>,
}
