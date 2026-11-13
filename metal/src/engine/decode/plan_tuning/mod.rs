#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum DecodePlan {
    SeparateGateUp,
    FusedGateUp,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct DecodePlanKey {
    pub model: String,
    pub weight_bytes: u64,
    pub context_bucket: usize,
    pub batch: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodePlanAction {
    Execute(DecodePlan),
    Measure,
}

impl DecodePlan {
    pub(crate) const fn fused_gate_up(self) -> bool {
        matches!(self, Self::FusedGateUp)
    }
}

pub fn context_bucket(tokens: usize) -> usize {
    tokens.max(1_024).checked_next_power_of_two().unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_contexts_use_power_of_two_buckets() {
        assert_eq!(context_bucket(128), 1_024);
        assert_eq!(context_bucket(1_024), 1_024);
        assert_eq!(context_bucket(1_025), 2_048);
        assert_eq!(context_bucket(8_192), 8_192);
    }
}
