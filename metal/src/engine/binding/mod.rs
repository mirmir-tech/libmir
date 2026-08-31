use models::weights::TensorBinding;

use super::{ModelTensors, NormWeight, Result, Stream};

mod projection;

pub(super) use projection::{BoundEmbedding, BoundLinear, GraphLinear};

pub(super) fn adjusted_norm(
    tensors: &ModelTensors,
    binding: &TensorBinding,
    shift: f32,
    stream: &Stream,
) -> Result<NormWeight> {
    NormWeight::load_name_adjusted(tensors, &binding.source, shift, stream)
}
