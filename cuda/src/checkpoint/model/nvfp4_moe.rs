use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, WeightBindingPlan},
};

use super::{ModelSource, embedding_multiplier, tensor};
use crate::{CudaBackend, CudaMoeModelTemplate, Error, NvFp4MoeLayerLoadConfig, Result};

impl CudaBackend {
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
            decoder, &spec, &bindings, catalog, load, progress,
        )
    }

    pub(crate) fn load_nvfp4_moe_model_template_with_bindings(
        &self,
        decoder: &DecoderConfig,
        semantic: &SemanticModelSpec,
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
                decoder, semantic, catalog, layer, &layer_bindings, load,
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
        CudaMoeModelTemplate::new(
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
