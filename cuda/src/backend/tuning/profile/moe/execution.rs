use super::ClampedMoeExecution;

impl ClampedMoeExecution {
    pub(in crate::backend) const fn for_batch(
        self,
        tokens: usize,
        experts: usize,
        top_k: usize,
    ) -> Self {
        let assignments = tokens.saturating_mul(top_k);
        let large_threshold = experts.saturating_mul(64).saturating_mul(9).div_ceil(10);
        if assignments >= large_threshold {
            match self {
                Self::MarlinN128K128 | Self::MarlinN128K64 => Self::MarlinM64N128K64,
                Self::MarlinN64K128 => Self::MarlinM64N64K128,
                execution => execution,
            }
        } else {
            match self {
                Self::MarlinN128K64
                | Self::MarlinN64K128
                | Self::MarlinM64N256K64
                | Self::MarlinM64N128K64
                | Self::MarlinM64N64K128 => Self::MarlinN128K128,
                execution => execution,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_profile_adapts_marlin_row_tile_to_batch() {
        let large = ClampedMoeExecution::MarlinM64N128K64;
        assert_eq!(large.for_batch(1, 32, 4), ClampedMoeExecution::MarlinN128K128);
        assert_eq!(large.for_batch(2_032, 32, 4), large);

        let small = ClampedMoeExecution::MarlinN64K128;
        assert_eq!(small.for_batch(16, 32, 4), ClampedMoeExecution::MarlinN128K128);
        assert_eq!(small.for_batch(2_032, 32, 4), ClampedMoeExecution::MarlinM64N64K128);
    }
}
