#[derive(Debug, Clone)]
/// Point-in-time view of memory reported by the active accelerator backend.
pub struct MemorySnapshot {
    /// Total device or unified memory, when the backend can report it.
    pub total_bytes: Option<u64>,
    /// Memory currently available to the process, when known.
    pub available_bytes: Option<u64>,
    /// Bytes held by live model and inference allocations.
    pub active_bytes: u64,
    /// Reusable bytes retained by backend allocators.
    pub cached_bytes: u64,
    /// Additional immediately free memory required by the backend allocator.
    pub allocation_reserve_bytes: u64,
    /// Human-readable source of the memory figures.
    pub source: String,
    /// Whether host and accelerator share a unified memory pool.
    pub unified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Conservative memory estimate for loading a model with a given cache
/// configuration.
pub struct ModelMemoryEstimate {
    /// Bytes occupied by model weight files.
    pub weight_bytes: u64,
    /// Bytes reserved for the configured K/V cache.
    pub kv_cache_bytes: u64,
    /// Estimated temporary runtime workspace.
    pub workspace_bytes: u64,
    /// Sum of weights, cache, and workspace.
    pub required_bytes: u64,
    /// K/V storage consumed by one cached token across attention layers.
    pub kv_bytes_per_token: u64,
    /// Token capacity implied by the configured cache blocks.
    pub cache_capacity_tokens: u64,
    /// Maximum context length declared by the model.
    pub model_context_tokens: u64,
}
