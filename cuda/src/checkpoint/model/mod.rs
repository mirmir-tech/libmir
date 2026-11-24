use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TensorInfo, TensorStorage, WeightBindingPlan},
};

use super::DenseSwiGluLayerLoadConfig;
use crate::{
    CudaBackend, CudaMoeModelTemplate, CudaTensor, CudaTensorSet, Error, Result,
    backend::CheckpointProjectionWeight as ProjectionWeight,
};

mod nvfp4_moe;
mod source;
#[cfg(test)]
mod tests;

pub(in crate::checkpoint) use source::ModelSource;

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
            decoder, &spec, &bindings, catalog, load, progress,
        )
    }

    pub(crate) fn load_dense_swiglu_model_template_with_bindings(
        &self,
        decoder: &DecoderConfig,
        semantic: &SemanticModelSpec,
        bindings: &WeightBindingPlan,
        catalog: &TensorCatalog,
        load: DenseSwiGluLayerLoadConfig,
        progress: &mut dyn FnMut(u64, String),
    ) -> Result<CudaMoeModelTemplate> {
        let source =
            ModelSource::discover_biased(semantic.decoder.tie_word_embeddings, bindings, catalog)?;
        let mut tensors = source.upload(self)?;
        let mut completed = source.payload_bytes()?;
        progress(completed, "model boundary tensors".into());
        let embedding = ProjectionWeight::load_binding_prepared(self, &tensors, source.embedding)?;
        let offset = semantic.decoder.final_norm.weight_offset;
        if offset != 0.0 {
            tensors.add_bf16(source.final_norm, offset)?;
        }
        let final_norm_biased = source.final_norm_bias.is_some();
        if final_norm_biased {
            let binding = bindings
                .decoder_boundary_with_tied_output(semantic.decoder.tie_word_embeddings)?
                .final_norm;
            crate::checkpoint::dense::append_norm_bias(self, &mut tensors, binding)?;
        }
        let final_norm = tensor(&tensors, source.final_norm)?.clone();
        let output = match &source.output.storage {
            TensorStorage::Ggml {
                companion: models::weights::GgmlCompanion::Bias(_),
                ..
            } => ProjectionWeight::Ggml(crate::GgmlCheckpointWeight::load_binding_with_bias(
                &tensors, source.output,
            )?),
            _ => ProjectionWeight::load_binding_prepared(self, &tensors, source.output)?,
        };
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for layer in 0..decoder.num_hidden_layers {
            let (template, bytes) = self.load_dense_swiglu_layer_tracked(
                decoder,
                semantic,
                catalog,
                layer,
                crate::checkpoint::dense::DenseLayerBindings::of_layer(bindings, semantic, layer)?,
                load,
            )?;
            completed = completed
                .checked_add(bytes)
                .ok_or(Error::InvalidDecoderKernel("checkpoint progress byte overflow"))?;
            progress(completed, format!("layer {}/{}", layer + 1, decoder.num_hidden_layers));
            layers.push(template);
        }
        // The final norm follows the first layer's formula.
        let layers_biased = layers.first().is_some_and(|layer| {
            layer.config().attention.stream_norm == crate::kernels::NormFormula::MeanCentredBiased
        });
        if layers_biased != final_norm_biased {
            return Err(Error::UnsupportedDecoderLayer(
                "CUDA biases the final norm exactly when it biases the layer norms".into(),
            ));
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
            embedding_multiplier(semantic)?,
            embedding,
            final_norm,
            output,
            layers,
        )
    }
}

/// The model spec's embedding multiplier, narrowed for the embedding kernel.
pub(super) fn embedding_multiplier(semantic: &SemanticModelSpec) -> Result<f32> {
    let multiplier = semantic.decoder.embedding_multiplier;
    #[allow(clippy::cast_possible_truncation)]
    let narrowed = multiplier as f32;
    if narrowed.is_finite() && narrowed > 0.0 {
        Ok(narrowed)
    } else {
        Err(Error::UnsupportedDecoderLayer(format!(
            "embedding multiplier {multiplier} is not a positive finite f32"
        )))
    }
}

pub(super) fn payload_bytes<'a>(tensors: impl IntoIterator<Item = &'a TensorInfo>) -> Result<u64> {
    tensors.into_iter().try_fold(0_u64, |total, tensor| {
        total
            .checked_add(u64::try_from(tensor.payload_bytes()?)?)
            .ok_or(Error::InvalidDecoderKernel("checkpoint progress byte overflow"))
    })
}

pub(super) fn tensor<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}
