mod execute;
mod layer;
mod load;
mod scratch;

use models::{layout::DecoderConfig, weights::TextTensorLayout};

use crate::{CudaBackend, CudaTensorSet};

pub struct CudaTextEmbeddingModel {
    backend: CudaBackend,
    config: DecoderConfig,
    layout: TextTensorLayout,
    tensors: CudaTensorSet,
}
