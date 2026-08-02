use runtime::kv::KvCacheDType;

use super::{ClampedRoutedConfig, ClampedRoutedLayout};
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ClampedRoutedQkvLowering {
    PackedFused,
    SeparateComposed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ClampedRoutedCapabilityPlan {
    pub qkv: ClampedRoutedQkvLowering,
}

impl ClampedRoutedCapabilityPlan {
    pub(super) fn lower(
        config: ClampedRoutedConfig,
        layout: ClampedRoutedLayout,
        cache_dtype: KvCacheDType,
    ) -> Result<Self> {
        admit_attention(config, cache_dtype)?;
        admit_experts(config, layout)?;
        let qkv = match layout {
            ClampedRoutedLayout::Native | ClampedRoutedLayout::Dense => {
                ClampedRoutedQkvLowering::PackedFused
            },
            ClampedRoutedLayout::Mlx => ClampedRoutedQkvLowering::SeparateComposed,
        };
        tracing::debug!(
            target: "libmir::cuda::lowering",
            operation = "biased YaRN QKV",
            storage = layout.storage(),
            geometry = %attention_geometry(config),
            lowering = ?qkv,
            "lowered clamped-routed CUDA operation"
        );
        Ok(Self { qkv })
    }
}

fn admit_attention(config: ClampedRoutedConfig, dtype: KvCacheDType) -> Result<()> {
    let storage = format!("BF16 projections with paged {} K/V", dtype.as_str());
    if !matches!(
        dtype,
        KvCacheDType::Auto | KvCacheDType::BFloat16 | KvCacheDType::Fp8 | KvCacheDType::Fp8E4M3
    ) {
        return Err(Error::MissingCapability {
            operation: "sink softmax attention",
            storage,
            geometry: attention_geometry(config),
            requirement: "the available paged sink-attention kernel supports BF16 or FP8 E4M3 K/V",
        });
    }
    if config.head_dim.is_multiple_of(2) && config.head_dim <= 256 {
        return Ok(());
    }
    Err(Error::MissingCapability {
        operation: "sink softmax attention with YaRN QKV",
        storage,
        geometry: attention_geometry(config),
        requirement: "the available fused transform requires an even head dimension <= 256",
    })
}

fn admit_experts(config: ClampedRoutedConfig, layout: ClampedRoutedLayout) -> Result<()> {
    if layout == ClampedRoutedLayout::Dense {
        return Ok(());
    }
    if config.hidden.is_multiple_of(32) && config.intermediate.is_multiple_of(32) {
        return Ok(());
    }
    Err(Error::MissingCapability {
        operation: "routed clamped SwiGLU",
        storage: layout.storage().into(),
        geometry: format!(
            "hidden={}, intermediate={}, experts={}, top_k={}",
            config.hidden, config.intermediate, config.experts, config.top_k
        ),
        requirement: "MXFP4 expert blocks require hidden and intermediate multiples of 32",
    })
}

fn attention_geometry(config: ClampedRoutedConfig) -> String {
    format!(
        "query_heads={}, kv_heads={}, head_dim={}",
        config.query_heads, config.kv_heads, config.head_dim
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_composed_qkv_for_separate_affine_storage() -> Result<()> {
        let plan = ClampedRoutedCapabilityPlan::lower(
            config(64, 96, 16),
            ClampedRoutedLayout::Mlx,
            KvCacheDType::BFloat16,
        )?;
        assert_eq!(plan.qkv, ClampedRoutedQkvLowering::SeparateComposed);
        Ok(())
    }

    #[test]
    fn reports_operation_storage_geometry_and_requirement() -> Result<()> {
        let Err(error) = ClampedRoutedCapabilityPlan::lower(
            config(48, 96, 16),
            ClampedRoutedLayout::Native,
            KvCacheDType::BFloat16,
        ) else {
            return Err(Error::InvalidExecutionPlan("unaligned MXFP4 geometry was admitted"));
        };
        let message = error.to_string();
        assert!(message.contains("routed clamped SwiGLU"));
        assert!(message.contains("interleaved MXFP4"));
        assert!(message.contains("hidden=48"));
        assert!(message.contains("multiples of 32"));
        Ok(())
    }

    #[test]
    fn reports_attention_capability_separately() -> Result<()> {
        let Err(error) = ClampedRoutedCapabilityPlan::lower(
            config(64, 96, 258),
            ClampedRoutedLayout::Mlx,
            KvCacheDType::BFloat16,
        ) else {
            return Err(Error::InvalidExecutionPlan("unsupported attention was admitted"));
        };
        let message = error.to_string();
        assert!(message.contains("sink softmax attention"));
        assert!(message.contains("paged bfloat16 K/V"));
        assert!(message.contains("head_dim=258"));
        assert!(message.contains("even head dimension <= 256"));
        Ok(())
    }

    #[test]
    fn reports_the_requested_kv_storage_format() -> Result<()> {
        let Err(error) = ClampedRoutedCapabilityPlan::lower(
            config(64, 96, 16),
            ClampedRoutedLayout::Native,
            KvCacheDType::Int4PerTokenHead,
        ) else {
            return Err(Error::InvalidExecutionPlan("unsupported K/V storage was admitted"));
        };
        let message = error.to_string();
        assert!(message.contains("paged int4_per_token_head K/V"));
        assert!(message.contains("supports BF16 or FP8 E4M3 K/V"));
        Ok(())
    }

    fn config(hidden: usize, intermediate: usize, head_dim: usize) -> ClampedRoutedConfig {
        ClampedRoutedConfig {
            vocab: 128,
            hidden,
            intermediate,
            query_heads: 4,
            kv_heads: 2,
            head_dim,
            experts: 8,
            top_k: 2,
            epsilon: 1.0e-5,
            scale: 0.25,
            theta: 150_000.0,
            factor: 32.0,
            initial_context: 4096.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            swiglu_limit: 7.0,
        }
    }
}
