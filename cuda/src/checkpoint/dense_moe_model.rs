use models::{
    layout::DecoderConfig,
    weights::{TensorCatalog, WeightBindingPlan},
};

use super::{
    NvFp4MoeLayerLoadConfig,
    model::{ModelSource, tensor},
};
use crate::{CudaBackend, CudaMoeModelTemplate, Error, Result};

impl CudaBackend {
    pub(crate) fn load_dense_moe_model_template_with_bindings(
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
            let (template, bytes) = self.load_dense_moe_layer_template_tracked(
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
            "loaded dense routed MoE model template"
        );
        CudaMoeModelTemplate::new(self, decoder.clone(), embedding, final_norm, output, layers)
    }
}
