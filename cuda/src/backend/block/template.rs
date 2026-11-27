use mircuda::{DeviceBuffer, bf16};
use models::weights::HybridMoeExpertBindings;

use super::{
    BatchedDecodeMoeLayer, CudaBackend, DecodeMoeBlockConfig, DecodeMoeBlockExecutor,
    DecodeMoeBlockWeights, NvFp4ExpertBank, PrefillMoeBlockBf16, experts::ExpertWeights,
    graph::weights::CapturedBlockWeights,
};
use crate::{
    CudaTensorSet, Error, PagedKvCache, Result,
    backend::{
        attention::AttentionLayout,
        dense::{ExpertBankLayout, GgmlExpertWeights},
        linear::DenseExpertWeights,
    },
    kernels::BatchedSplitAttentionWorkspace,
};

/// Immutable model-layer resources shared by session-local CUDA executors.
///
/// Expert banks and captured weights retain existing device allocations.
/// [`Self::instantiate`] creates only mutable plans, scratch, K/V state, and a
/// graph for one session.
#[derive(Clone)]
pub struct DecodeMoeLayerTemplate {
    backend: CudaBackend,
    config: DecodeMoeBlockConfig,
    weights: CapturedBlockWeights,
    experts: ExpertWeights,
}

impl CudaBackend {
    /// Prepares immutable routed-MoE resources from uploaded checkpoint
    /// tensors.
    pub fn prepare_decode_moe_layer_template(
        &self,
        config: DecodeMoeBlockConfig,
        weights: DecodeMoeBlockWeights<'_>,
        gate: NvFp4ExpertBank,
        up: NvFp4ExpertBank,
        down: NvFp4ExpertBank,
    ) -> Result<DecodeMoeLayerTemplate> {
        validate_banks(config, &gate, &up, &down)?;
        Ok(DecodeMoeLayerTemplate {
            backend: self.clone(),
            config,
            weights: CapturedBlockWeights::prepare(self, weights)?,
            experts: ExpertWeights::NvFp4 {
                gate,
                up,
                down,
                activation_mode: models::weights::BlockActivationMode::WeightAndActivation,
            },
        })
    }

    pub(crate) fn prepare_dense_decode_moe_layer_template(
        &self,
        config: DecodeMoeBlockConfig,
        weights: DecodeMoeBlockWeights<'_>,
        experts: DenseExpertWeights,
    ) -> Result<DecodeMoeLayerTemplate> {
        Ok(DecodeMoeLayerTemplate {
            backend: self.clone(),
            config,
            weights: CapturedBlockWeights::prepare(self, weights)?,
            experts: ExpertWeights::Dense(experts),
        })
    }

    /// [`Self::prepare_dense_decode_moe_layer_template`] with GGML expert
    /// banks read in place; `None` when the banks are not GGML.
    pub(crate) fn prepare_ggml_decode_moe_layer_template(
        &self,
        config: DecodeMoeBlockConfig,
        weights: DecodeMoeBlockWeights<'_>,
        tensors: &CudaTensorSet,
        experts: &HybridMoeExpertBindings<'_>,
    ) -> Result<Option<DecodeMoeLayerTemplate>> {
        let shape = (config.experts, config.attention.hidden_size, config.expert_intermediate);
        let Some(experts) = GgmlExpertWeights::load_hybrid(tensors, experts, shape)? else {
            return Ok(None);
        };
        Ok(Some(DecodeMoeLayerTemplate {
            backend: self.clone(),
            config,
            weights: CapturedBlockWeights::prepare(self, weights)?,
            experts: ExpertWeights::Ggml(Box::new(experts)),
        }))
    }
}

impl DecodeMoeLayerTemplate {
    /// Returns the fixed layer configuration.
    #[must_use]
    pub const fn config(&self) -> DecodeMoeBlockConfig {
        self.config
    }

    /// Whether the experts' weights bind at execution, so prefill plans
    /// serve every layer of the geometry.
    #[must_use]
    pub(in crate::backend) const fn experts_bind_at_execution(&self) -> bool {
        matches!(self.experts, ExpertWeights::Dense(_) | ExpertWeights::Ggml(_))
    }

    /// What a prefill plan compiles in from this layer's attention
    /// projections and expert banks.
    pub(in crate::backend) fn layouts(&self) -> (AttentionLayout, ExpertBankLayout) {
        let experts = match &self.experts {
            ExpertWeights::Dense(banks) => ExpertBankLayout::Dense(banks.layout()),
            ExpertWeights::Ggml(_) => ExpertBankLayout::Ggml,
            ExpertWeights::NvFp4 { .. } => ExpertBankLayout::NvFp4,
        };
        (AttentionLayout::of(self.weights.borrow().attention), experts)
    }

    /// Creates private execution state for one session without copying weights.
    pub fn instantiate(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &DeviceBuffer<bf16>,
    ) -> Result<DecodeMoeBlockExecutor> {
        let cache = self
            .backend
            .prepare_paged_kv(self.config.attention.layer, self.config.attention.cache)?;
        self.instantiate_with_cache(input, output, cache)
    }

    pub(in crate::backend) fn instantiate_with_cache(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &DeviceBuffer<bf16>,
        cache: PagedKvCache,
    ) -> Result<DecodeMoeBlockExecutor> {
        let hidden = self.config.attention.hidden_size;
        if input.len() != hidden || output.len() != hidden {
            return Err(Error::InvalidDecoderKernel(
                "layer template requires hidden-sized input and output buffers",
            ));
        }
        let block = super::DecodeMoeBlockBf16::new_with_cache(
            &self.backend, self.config, &self.experts, cache,
        )?;
        tracing::debug!(
            layer = self.config.attention.layer,
            hidden,
            experts = self.config.experts,
            "instantiated session-local CUDA MoE layer"
        );
        Ok(DecodeMoeBlockExecutor::new(
            block,
            input.clone(),
            self.weights.clone(),
            output.clone(),
        ))
    }

    /// Creates a fixed-token prefill executor sharing this layer's weights.
    pub fn instantiate_prefill(&self, tokens: usize) -> Result<PrefillMoeBlockBf16> {
        PrefillMoeBlockBf16::new(&self.backend, self.config, &self.experts, tokens)
    }

    /// Creates a fixed-row grouped-MoE decode executor without copying weights.
    pub fn instantiate_batch(&self, rows: usize) -> Result<BatchedDecodeMoeLayer> {
        let cache = self
            .backend
            .prepare_paged_kv(self.config.attention.layer, self.config.attention.cache)?;
        self.instantiate_batch_with_cache(rows, cache)
    }

    pub(in crate::backend) fn instantiate_batch_with_cache(
        &self,
        rows: usize,
        cache: PagedKvCache,
    ) -> Result<BatchedDecodeMoeLayer> {
        self.instantiate_batch_with_cache_workspace(rows, cache, None)
    }

    pub(in crate::backend) fn instantiate_batch_with_cache_workspace(
        &self,
        rows: usize,
        cache: PagedKvCache,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<BatchedDecodeMoeLayer> {
        let block = super::BatchedDecodeMoeBlockBf16::new_with_cache(
            &self.backend, self.config, &self.experts, rows, cache, workspace,
        )?;
        Ok(BatchedDecodeMoeLayer::new(block, self.weights.clone()))
    }
}

fn validate_banks(
    config: DecodeMoeBlockConfig,
    gate: &NvFp4ExpertBank,
    up: &NvFp4ExpertBank,
    down: &NvFp4ExpertBank,
) -> Result<()> {
    let gate = gate.config();
    let up = up.config();
    let down = down.config();
    let valid = gate == up
        && gate.experts == config.experts
        && gate.input_features == config.attention.hidden_size
        && gate.output_features == config.expert_intermediate
        && down.experts == config.experts
        && down.input_features == config.expert_intermediate
        && down.output_features == config.attention.hidden_size;
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidNvFp4("expert banks differ from layer template configuration"))
    }
}
