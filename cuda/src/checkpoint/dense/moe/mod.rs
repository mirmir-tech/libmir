pub(in crate::checkpoint) mod model;
use std::collections::HashSet;

use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{HybridMoeLayerBindings, TensorBinding, TensorCatalog, TensorInfo, TensorStorage},
};

use super::super::{
    NvFp4MoeLayerLoadConfig,
    load::{tensor, weights},
    model::payload_bytes,
    source::common,
};
use crate::{
    CudaBackend, CudaTensor, CudaTensorSet, DecodeMoeLayerTemplate, Error, GgmlCheckpointWeight,
    Result, backend::DenseExpertWeights,
};

impl CudaBackend {
    pub(in crate::checkpoint) fn load_dense_moe_layer_template_tracked(
        &self,
        decoder: &DecoderConfig,
        (semantic, plan): (&SemanticModelSpec, &architecture::cuda::CudaDecoderPlan),
        catalog: &TensorCatalog,
        layer: usize,
        bindings: &HybridMoeLayerBindings<'_>,
        load: NvFp4MoeLayerLoadConfig,
    ) -> Result<(DecodeMoeLayerTemplate, u64)> {
        let block = load.block(decoder, (semantic, plan), layer)?;
        let physical = dense_sources(catalog, bindings)?;
        let raw = bindings
            .bindings()
            .into_iter()
            .flat_map(TensorBinding::full_precision_sources)
            .collect::<HashSet<_>>();
        let mut tensors = upload(self, &physical, &raw)?;
        // GGUF projections outside the expert banks run in BF16 here; the
        // banks stay GGML blocks.
        // Layers whose keys double as values list the key projection twice.
        let mut dequantized = HashSet::new();
        for binding in common(bindings) {
            let ggml = matches!(binding.storage, TensorStorage::Ggml { .. });
            if ggml && dequantized.insert(binding.source.as_str()) {
                let weight = GgmlCheckpointWeight::load_binding(&tensors, binding)?;
                let shape = binding.logical_shape.clone().unwrap_or_default();
                let values = weight.dequantize_bf16(self)?;
                tensors.replace(CudaTensor::from_bf16(binding.source.clone(), shape, values))?;
            }
        }
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
        let layer_weights = weights(&tensors, &names, &qkv, &dense_gate_up)?;
        let bytes = payload_bytes(physical)?;
        if let Some(template) = self.prepare_ggml_decode_moe_layer_template(
            block,
            layer_weights,
            &tensors,
            &bindings.experts,
        )? {
            return Ok((template, bytes));
        }
        let experts = DenseExpertWeights::load_hybrid(
            self,
            &tensors,
            &bindings.experts,
            block.experts,
            block.attention.hidden_size,
            block.expert_intermediate,
        )?;
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

/// Uploads `tensors`, those named in `raw` in their checkpoint dtype and
/// every other floating tensor as BF16.
fn upload(
    backend: &CudaBackend,
    tensors: &[&TensorInfo],
    raw: &HashSet<&str>,
) -> Result<CudaTensorSet> {
    let cast = backend.prepare_dense_cast()?;
    let mut upload = backend.begin_tensor_upload();
    for tensor in tensors {
        if raw.contains(tensor.name.as_str()) {
            upload.enqueue(tensor)?;
        } else {
            upload.enqueue_float_as_bf16(tensor, &cast)?;
        }
    }
    upload.finish()
}
