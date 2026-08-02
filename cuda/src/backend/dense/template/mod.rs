use mircuda::{DeviceBuffer, bf16};

use super::{
    BatchedDecodeDenseLayer, DecodeDenseSwiGlu, DenseSwiGluConfig, DenseSwiGluWeights,
    PrefillDenseSwiGlu,
};
use crate::{CudaBackend, Error, PagedKvCache, Result, kernels::BatchedSplitAttentionWorkspace};

#[cfg(test)]
mod tests;
mod weights;

use weights::DenseWeights;
pub use weights::{
    DenseDownSource, DenseGateUpSource, DenseOutputSource, DenseQkvSource, DenseWeightSource,
};

#[derive(Clone)]
pub struct DenseSwiGluLayerTemplate {
    backend: CudaBackend,
    config: DenseSwiGluConfig,
    weights: DenseWeights,
}

impl CudaBackend {
    pub fn prepare_dense_swiglu_layer_template(
        &self,
        config: DenseSwiGluConfig,
        source: DenseWeightSource<'_>,
    ) -> Result<DenseSwiGluLayerTemplate> {
        Ok(DenseSwiGluLayerTemplate {
            backend: self.clone(),
            config,
            weights: DenseWeights::new(self, config, source)?,
        })
    }
}

impl DenseSwiGluLayerTemplate {
    #[must_use]
    pub const fn config(&self) -> DenseSwiGluConfig {
        self.config
    }

    pub(in crate::backend) fn instantiate_with_cache(
        &self,
        input: &DeviceBuffer<bf16>,
        output: &DeviceBuffer<bf16>,
        cache: PagedKvCache,
    ) -> Result<DecodeDenseSwiGlu> {
        let hidden = self.config.attention.hidden_size;
        if input.len() != hidden || output.len() != hidden {
            return Err(Error::InvalidDecoderKernel("dense template activation size mismatch"));
        }
        DecodeDenseSwiGlu::new_with_cache(&self.backend, self.config, cache, self.weights.borrow())
    }

    pub(in crate::backend) fn instantiate_prefill(
        &self,
        tokens: usize,
    ) -> Result<PrefillDenseSwiGlu> {
        PrefillDenseSwiGlu::new(&self.backend, self.config, tokens, self.weights.borrow())
    }

    pub(in crate::backend) fn instantiate_batch_with_cache_workspace(
        &self,
        rows: usize,
        cache: PagedKvCache,
        workspace: Option<BatchedSplitAttentionWorkspace>,
    ) -> Result<BatchedDecodeDenseLayer> {
        BatchedDecodeDenseLayer::new(&self.backend, self.clone(), rows, cache, workspace)
    }

    pub(in crate::backend) fn weights(&self) -> DenseSwiGluWeights<'_> {
        self.weights.borrow()
    }
}
