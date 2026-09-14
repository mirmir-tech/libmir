#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeReservation {
    #[default]
    OnePage,
    /// Reserve up to the request's generation budget using reclaimable spare
    /// pages.
    GenerationBudget,
    #[cfg(test)]
    #[serde(skip_deserializing)]
    Tokens(std::num::NonZeroUsize),
}

impl std::fmt::Display for DecodeReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OnePage => f.write_str("one_page"),
            Self::GenerationBudget => f.write_str("generation_budget"),
            #[cfg(test)]
            Self::Tokens(tokens) => write!(f, "tokens({tokens})"),
        }
    }
}

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct MetalCacheConfig {
    pub decode_reservation: DecodeReservation,
    pub prefix_cache_entries: usize,
    pub prefix_cache_bytes: Option<usize>,
    pub prefill_step: Option<usize>,
    pub kv_reserve_tokens: usize,
    pub paged_attention_min_context: usize,
    pub force_native_paged_attention: bool,
}

impl Default for MetalCacheConfig {
    fn default() -> Self {
        Self {
            decode_reservation: DecodeReservation::default(),
            prefix_cache_entries: 10,
            prefix_cache_bytes: None,
            prefill_step: None,
            kv_reserve_tokens: 256,
            paged_attention_min_context: 128,
            force_native_paged_attention: false,
        }
    }
}
