mod batch;
mod encoder;
mod head;
mod weights;

use models::decision::{DecisionCheckpoint, DecisionRow};

use crate::{
    CudaBackend, CudaConfig, Result,
    kernels::{DecisionAttention, DecisionDenseAttention, DecisionElementwise},
};

/// A Laya decision checkpoint resident on a CUDA device, evaluated in f32.
pub struct CudaDecisionModel {
    backend: CudaBackend,
    elementwise: DecisionElementwise,
    attention: DecisionAttention,
    dense: DecisionDenseAttention,
    plans: weights::Plans,
    encoder: encoder::Encoder,
    head: head::Head,
    positions: usize,
}

impl CudaDecisionModel {
    pub fn load(checkpoint: &DecisionCheckpoint) -> Result<Self> {
        let backend = CudaBackend::new(CudaConfig::default())?;
        let uploader = weights::Uploader {
            backend: &backend,
            plan: &checkpoint.tensors,
        };
        let encoder = encoder::Encoder::load(&checkpoint.encoder, &uploader)?;
        let head = head::Head::load(
            &uploader,
            checkpoint.agent.head_layers,
            checkpoint.head_attention_heads(),
            checkpoint.encoder.hidden_size,
        )?;
        Ok(Self {
            elementwise: DecisionElementwise::compile(&backend.inner.compiler)?,
            attention: DecisionAttention::compile(&backend.inner.compiler)?,
            dense: DecisionDenseAttention::compile(&backend.inner.compiler)?,
            plans: weights::Plans::default(),
            encoder,
            head,
            positions: checkpoint.agent.max_len,
            backend,
        })
    }

    /// Raw option logits of every row, one per marker.
    pub fn logits(&self, rows: &[DecisionRow]) -> Result<Vec<Vec<f32>>> {
        let batch = batch::DeviceBatch::new(&self.backend, rows, self.positions)?;
        let device = encoder::Device {
            backend: &self.backend,
            elementwise: &self.elementwise,
            attention: &self.attention,
            dense: &self.dense,
            plans: &self.plans,
        };
        let hidden = self.encoder.forward(&device, &batch)?;
        self.head.logits(&device, hidden, &batch)
    }
}
