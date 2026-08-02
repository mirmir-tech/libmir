use std::collections::HashSet;

use models::{
    layout::DecoderConfig,
    weights::{HybridMoeLayerBindings, TensorCatalog, TensorInfo},
};

use super::{
    NvFp4MoeLayerLoadConfig,
    load::{tensor, weights},
    model::payload_bytes,
    source::common,
};
use crate::{
    CudaBackend, CudaTensorSet, DecodeMoeLayerTemplate, Error, Result, backend::DenseExpertWeights,
};

impl CudaBackend {
    pub(super) fn load_dense_moe_layer_template_tracked(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        layer: usize,
        bindings: &HybridMoeLayerBindings<'_>,
        load: NvFp4MoeLayerLoadConfig,
    ) -> Result<(DecodeMoeLayerTemplate, u64)> {
        let block = load.block(decoder, layer)?;
        let physical = dense_sources(catalog, bindings)?;
        let tensors = upload(self, &physical)?;
        let names = common(bindings)
            .into_iter()
            .map(|binding| binding.source.clone())
            .collect::<Vec<_>>();
        let dense_gate_up = self
            .pack_bf16_linear_pair(tensor(&tensors, &names[9])?, tensor(&tensors, &names[10])?)?;
        let qkv = self.pack_bf16_linears([
            tensor(&tensors, &names[1])?,
            tensor(&tensors, &names[2])?,
            tensor(&tensors, &names[3])?,
        ])?;
        let experts = DenseExpertWeights::load_hybrid(
            self,
            &tensors,
            &bindings.experts,
            block.experts,
            block.attention.hidden_size,
            block.expert_intermediate,
        )?;
        let layer_weights = weights(&tensors, &names, &qkv, &dense_gate_up)?;
        let bytes = payload_bytes(physical)?;
        let template =
            self.prepare_dense_decode_moe_layer_template(block, layer_weights, experts)?;
        Ok((template, bytes))
    }
}

fn dense_sources<'a>(
    catalog: &'a TensorCatalog,
    bindings: &HybridMoeLayerBindings<'_>,
) -> Result<Vec<&'a TensorInfo>> {
    let mut seen = HashSet::new();
    bindings
        .physical_sources()
        .into_iter()
        .filter(|name| seen.insert((*name).to_owned()))
        .map(|name| {
            catalog
                .tensors
                .iter()
                .find(|tensor| tensor.name == name)
                .ok_or_else(|| Error::MissingTensor(name.into()))
        })
        .collect()
}

fn upload(backend: &CudaBackend, tensors: &[&TensorInfo]) -> Result<CudaTensorSet> {
    let cast = backend.prepare_dense_cast()?;
    let mut upload = backend.begin_tensor_upload();
    for tensor in tensors {
        upload.enqueue_float_as_bf16(tensor, &cast)?;
    }
    upload.finish()
}
