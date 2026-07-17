mod batch;
mod config;
mod decode;
pub(in crate::backend) mod graph;
mod prefill;
mod projection;
mod scratch;
mod template;

pub(super) use batch::BatchedDecodeDenseLayer;
pub use config::{DenseDownWeight, DenseGateUpWeights, DenseSwiGluConfig, DenseSwiGluWeights};
pub use decode::DecodeDenseSwiGlu;
use decode::validate;
pub use prefill::PrefillDenseSwiGlu;
use projection::{DownProjection, GateUpBuffers, GateUpProjection};
use scratch::DenseScratch;
pub use template::{
    DenseDownSource, DenseGateUpSource, DenseOutputSource, DenseQkvSource,
    DenseSwiGluLayerTemplate, DenseWeightSource,
};

use super::{Bf16LinearPairWeights, CudaBackend, DecodeAttentionConfig, DecodeAttentionWeights};
use crate::CudaTensor;
