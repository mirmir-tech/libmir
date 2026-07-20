mod execute;
mod layer;
mod load;
mod scratch;

use models::layout::EncoderConfig;

use crate::{CudaBackend, CudaTensorSet};

pub struct CudaSequenceScoringModel {
    backend: CudaBackend,
    config: EncoderConfig,
    tensors: CudaTensorSet,
}
