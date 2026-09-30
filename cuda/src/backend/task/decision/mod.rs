mod attention;
mod batch;
mod device;
mod encoder;
mod head;
mod plans;
mod rope;
mod staging;
mod weights;

use std::sync::Mutex;

use mircuda::{DeviceBuffer, bf16};
use models::decision::{DecisionCheckpoint, DecisionRow};

use crate::{
    CudaBackend, CudaConfig, Error, Result,
    kernels::{DecisionAttention, DecisionElementwise, DecisionLayout},
};

/// Arithmetic of the encoder's matrix products. Norms, softmax, residuals
/// and the decision head stay f32 in both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DecisionPrecision {
    /// Full single precision, matching the CPU and Metal backends.
    #[default]
    F32,
    /// bf16 operands on tensor cores with f32 accumulation.
    Bf16,
}

enum Encoder {
    F32(encoder::Encoder<f32>),
    Bf16(encoder::Encoder<bf16>),
}

impl Encoder {
    fn band(&self) -> Option<usize> {
        match self {
            Self::F32(encoder) => encoder.band(),
            Self::Bf16(encoder) => encoder.band(),
        }
    }

    fn forward(
        &self,
        device: &device::Device<'_>,
        batch: &batch::DeviceBatch,
    ) -> Result<DeviceBuffer<f32>> {
        match self {
            Self::F32(encoder) => encoder.forward(device, batch),
            Self::Bf16(encoder) => encoder.forward(device, batch),
        }
    }
}

/// A Laya decision checkpoint resident on a CUDA device.
pub struct CudaDecisionModel {
    backend: CudaBackend,
    elementwise: DecisionElementwise,
    attention: DecisionAttention,
    layout: DecisionLayout,
    plans: plans::Plans,
    encoder: Encoder,
    head: head::Head,
    positions: usize,
    staging: Mutex<staging::Staging>,
}

impl CudaDecisionModel {
    pub fn load(checkpoint: &DecisionCheckpoint, precision: DecisionPrecision) -> Result<Self> {
        let backend = CudaBackend::new(CudaConfig::default())?;
        let uploader = weights::Uploader {
            backend: &backend,
            plan: &checkpoint.tensors,
        };
        let (config, positions) = (&checkpoint.encoder, checkpoint.agent.max_len);
        let encoder = match precision {
            DecisionPrecision::F32 => {
                Encoder::F32(encoder::Encoder::load(config, &uploader, positions)?)
            },
            DecisionPrecision::Bf16 => {
                Encoder::Bf16(encoder::Encoder::load(config, &uploader, positions)?)
            },
        };
        let head = head::Head::load(
            &uploader,
            checkpoint.agent.head_layers,
            checkpoint.head_attention_heads(),
            config.hidden_size,
        )?;
        Ok(Self {
            elementwise: DecisionElementwise::compile(&backend.inner.compiler)?,
            attention: DecisionAttention::compile(&backend.inner.compiler)?,
            layout: DecisionLayout::compile(&backend.inner.compiler)?,
            plans: plans::Plans::default(),
            encoder,
            head,
            positions,
            staging: Mutex::new(staging::Staging::new(&backend, 4 * positions)?),
            backend,
        })
    }

    /// Arithmetic of the encoder's matrix products.
    #[must_use]
    pub const fn precision(&self) -> DecisionPrecision {
        match self.encoder {
            Encoder::F32(_) => DecisionPrecision::F32,
            Encoder::Bf16(_) => DecisionPrecision::Bf16,
        }
    }

    /// Raw option logits of every row, one per marker.
    pub fn logits(&self, rows: &[DecisionRow]) -> Result<Vec<Vec<f32>>> {
        let mut staging = self
            .staging
            .lock()
            .map_err(|_| Error::InvalidDecoderKernel("decision staging is poisoned"))?;
        let batch = batch::DeviceBatch::new(
            (&self.backend, &mut staging),
            rows,
            self.positions,
            self.encoder.band(),
        )?;
        let device = device::Device {
            backend: &self.backend,
            elementwise: &self.elementwise,
            attention: &self.attention,
            layout: &self.layout,
            plans: &self.plans,
        };
        let hidden = self.encoder.forward(&device, &batch)?;
        let logits = self.head.logits(&device, hidden, &batch)?;
        let values = staging.download(&self.backend, &logits)?;
        drop(staging);
        let mut values = values.into_iter();
        Ok(batch
            .marker_counts
            .iter()
            .map(|&count| values.by_ref().take(count).collect())
            .collect())
    }
}
