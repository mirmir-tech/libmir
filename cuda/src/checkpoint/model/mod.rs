use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorBinding, TensorCatalog, TensorInfo, TensorStorage, WeightBindingPlan},
};

use super::{DenseSwiGluLayerLoadConfig, NvFp4MoeLayerLoadConfig};
use crate::{
    CudaBackend, CudaMoeModelTemplate, CudaTensor, CudaTensorSet, Error, Result,
    backend::CheckpointProjectionWeight as ProjectionWeight,
};

#[cfg(test)]
mod tests;

impl CudaBackend {
    /// Loads a complete BF16 dense `SwiGLU` decoder without host conversion.
    #[cfg(test)]
    pub(crate) fn load_dense_swiglu_model_template_with_progress(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        load: DenseSwiGluLayerLoadConfig,
        progress: &mut dyn FnMut(u64, String),
    ) -> Result<CudaMoeModelTemplate> {
        let spec = SemanticModelSpec::discover(decoder, catalog)?;
        let bindings = WeightBindingPlan::discover(&spec, catalog)?;
        self.load_dense_swiglu_model_template_with_bindings(
            decoder, &bindings, catalog, load, progress,
        )
    }

    pub(crate) fn load_dense_swiglu_model_template_with_bindings(
        &self,
        decoder: &DecoderConfig,
        bindings: &WeightBindingPlan,
        catalog: &TensorCatalog,
        load: DenseSwiGluLayerLoadConfig,
        progress: &mut dyn FnMut(u64, String),
    ) -> Result<CudaMoeModelTemplate> {
        let source = ModelSource::discover(decoder, bindings, catalog)?;
        let tensors = source.upload(self)?;
        let mut completed = source.payload_bytes()?;
        progress(completed, "model boundary tensors".into());
        let embedding = ProjectionWeight::load_binding_prepared(self, &tensors, source.embedding)?;
        let final_norm = tensor(&tensors, source.final_norm)?.clone();
        let output = ProjectionWeight::load_binding_prepared(self, &tensors, source.output)?;
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for layer in 0..decoder.num_hidden_layers {
            let (template, bytes) = self.load_dense_swiglu_layer_tracked(
                decoder,
                catalog,
                layer,
                bindings.dense_decoder_layer(layer)?,
                load,
            )?;
            completed = completed
                .checked_add(bytes)
                .ok_or(Error::InvalidDecoderKernel("checkpoint progress byte overflow"))?;
            progress(completed, format!("layer {}/{}", layer + 1, decoder.num_hidden_layers));
            layers.push(template);
        }
        tracing::debug!(
            backend = "cuda",
            layers = layers.len(),
            hidden = decoder.hidden_size,
            vocab = decoder.vocab_size,
            tied_output = decoder.tie_word_embeddings,
            "loaded BF16 dense SwiGLU model template"
        );
        CudaMoeModelTemplate::new_dense_bound(
            self,
            decoder.clone(),
            embedding,
            final_norm,
            output,
            layers,
        )
    }

    /// Loads all routed-MoE layers and model boundary tensors without host-side
    /// tensor conversion.
    pub fn load_nvfp4_moe_model_template(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        load: NvFp4MoeLayerLoadConfig,
    ) -> Result<CudaMoeModelTemplate> {
        let mut ignored = |_completed, _detail| {};
        self.load_nvfp4_moe_model_template_with_progress(decoder, catalog, load, &mut ignored)
    }

    pub(crate) fn load_nvfp4_moe_model_template_with_progress(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        load: NvFp4MoeLayerLoadConfig,
        progress: &mut dyn FnMut(u64, String),
    ) -> Result<CudaMoeModelTemplate> {
        let spec = SemanticModelSpec::discover(decoder, catalog)?;
        let bindings = WeightBindingPlan::discover(&spec, catalog)?;
        self.load_nvfp4_moe_model_template_with_bindings(
            decoder, &bindings, catalog, load, progress,
        )
    }

    pub(crate) fn load_nvfp4_moe_model_template_with_bindings(
        &self,
        decoder: &DecoderConfig,
        bindings: &WeightBindingPlan,
        catalog: &TensorCatalog,
        load: NvFp4MoeLayerLoadConfig,
        progress: &mut dyn FnMut(u64, String),
    ) -> Result<CudaMoeModelTemplate> {
        let source = ModelSource::discover(decoder, bindings, catalog)?;
        let tensors = source.upload(self)?;
        let mut completed = source.payload_bytes()?;
        progress(completed, "model boundary tensors".into());
        let embedding = tensor(&tensors, &source.embedding.source)?.clone();
        let final_norm = tensor(&tensors, source.final_norm)?.clone();
        let output = tensor(&tensors, &source.output.source)?.clone();
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for layer in 0..decoder.num_hidden_layers {
            let layer_bindings = bindings.hybrid_moe_layer(layer)?;
            let (template, bytes) = self.load_nvfp4_moe_layer_template_tracked(
                decoder, catalog, layer, &layer_bindings, load,
            )?;
            completed = completed
                .checked_add(bytes)
                .ok_or(Error::InvalidDecoderKernel("checkpoint progress byte overflow"))?;
            progress(completed, format!("layer {}/{}", layer + 1, decoder.num_hidden_layers));
            layers.push(template);
        }
        tracing::debug!(
            backend = "cuda",
            layers = layers.len(),
            hidden = decoder.hidden_size,
            vocab = decoder.vocab_size,
            tied_output = decoder.tie_word_embeddings,
            "loaded NVFP4 routed MoE model template"
        );
        CudaMoeModelTemplate::new(self, decoder.clone(), embedding, final_norm, output, layers)
    }
}

pub(super) struct ModelSource<'a> {
    pub(super) embedding: &'a TensorBinding,
    pub(super) final_norm: &'a str,
    pub(super) output: &'a TensorBinding,
    tensors: Vec<&'a TensorInfo>,
}

impl<'a> ModelSource<'a> {
    pub(super) fn discover(
        decoder: &DecoderConfig,
        bindings: &'a WeightBindingPlan,
        catalog: &'a TensorCatalog,
    ) -> Result<Self> {
        let boundary = bindings.decoder_boundary_with_tied_output(decoder.tie_word_embeddings)?;
        let final_norm = required(catalog, &boundary.final_norm.source)?;
        let mut seen = std::collections::HashSet::new();
        let mut tensors = Vec::new();
        for name in boundary
            .embedding
            .physical_sources()
            .into_iter()
            .chain(boundary.output.physical_sources())
            .chain(std::iter::once(boundary.final_norm.source.as_str()))
        {
            if seen.insert(name) {
                tensors.push(required(catalog, name)?);
            }
        }
        Ok(Self {
            embedding: boundary.embedding,
            final_norm: &final_norm.name,
            output: boundary.output,
            tensors,
        })
    }

    pub(super) fn upload(&self, backend: &CudaBackend) -> Result<CudaTensorSet> {
        let cast = backend.prepare_dense_cast()?;
        let mut upload = backend.begin_tensor_upload();
        let bindings = [self.embedding, self.output];
        let raw = bindings
            .iter()
            .flat_map(|binding| runtime_raw_sources(binding))
            .collect::<std::collections::HashSet<_>>();
        let metadata = bindings
            .iter()
            .filter_map(|binding| packed_shape_source(binding))
            .collect::<std::collections::HashSet<_>>();
        for tensor in &self.tensors {
            if metadata.contains(tensor.name.as_str()) {
                continue;
            }
            if raw.contains(tensor.name.as_str()) {
                upload.enqueue(tensor)?;
            } else {
                upload.enqueue_as_bf16(tensor, &cast)?;
            }
        }
        upload.finish()
    }

    pub(super) fn payload_bytes(&self) -> Result<u64> {
        payload_bytes(self.tensors.iter().copied())
    }
}
fn runtime_raw_sources(binding: &TensorBinding) -> Vec<&str> {
    match &binding.storage {
        TensorStorage::AffineQuantized { .. }
        | TensorStorage::BlockQuantized { .. }
        | TensorStorage::Float8 { .. } => binding.physical_sources(),
        TensorStorage::PackedInt8 { scales, .. } | TensorStorage::PackedInt4 { scales, .. } => {
            vec![binding.source.as_str(), scales.as_str()]
        },
        _ => Vec::new(),
    }
}

fn packed_shape_source(binding: &TensorBinding) -> Option<&str> {
    match &binding.storage {
        TensorStorage::PackedInt8 { shape, .. } | TensorStorage::PackedInt4 { shape, .. } => {
            Some(shape)
        },
        _ => None,
    }
}

pub(super) fn payload_bytes<'a>(tensors: impl IntoIterator<Item = &'a TensorInfo>) -> Result<u64> {
    tensors.into_iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(u64::try_from(tensor.payload_bytes()?)?)
            .ok_or(Error::InvalidDecoderKernel("checkpoint progress byte overflow"))
    })
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}

pub(super) fn tensor<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}
