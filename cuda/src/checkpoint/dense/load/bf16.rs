//! BF16 dense layers; a layer reading an earlier layer's keys and values
//! projects queries alone.

use super::{common_source, tensor};
use crate::{
    CudaBackend, CudaTensorSet, DenseDownSource, DenseFeedForwardSource, DenseGateUpSource,
    DenseOutputSource, DenseQkvSource, DenseSwiGluLayerTemplate, Result,
    checkpoint::dense::{DenseAttentionBindings, DenseLayerBindings},
};

pub(super) fn load(
    backend: &CudaBackend,
    block: crate::DenseSwiGluConfig,
    tensors: &CudaTensorSet,
    bindings: DenseLayerBindings<'_>,
) -> Result<DenseSwiGluLayerTemplate> {
    let feed_forward = bindings.feed_forward.gated()?;
    let query = tensor(tensors, &bindings.attention.query().source)?;
    let qkv = match bindings.attention {
        DenseAttentionBindings::Projected(attention) => Some(backend.pack_bf16_linears([
            query,
            tensor(tensors, &attention.key.source)?,
            tensor(tensors, &attention.value.source)?,
        ])?),
        DenseAttentionBindings::Shared { .. } => None,
    };
    let gate_up = backend.pack_bf16_linear_pair(
        tensor(tensors, &feed_forward.gate.source)?,
        tensor(tensors, &feed_forward.up.source)?,
    )?;
    backend.prepare_dense_swiglu_layer_template(
        block,
        common_source(
            tensors,
            bindings,
            qkv.as_ref().map_or(
                DenseQkvSource::Query(DenseOutputSource::Bf16(query)),
                DenseQkvSource::Bf16,
            ),
            DenseOutputSource::Bf16(tensor(tensors, &bindings.attention.output().source)?),
            DenseFeedForwardSource::Gated {
                gate_up: DenseGateUpSource::Bf16(&gate_up),
                down: DenseDownSource::Bf16(tensor(tensors, &feed_forward.down.source)?),
            },
        )?,
    )
}
