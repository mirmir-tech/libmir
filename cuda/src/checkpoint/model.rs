use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{TensorCatalog, TensorInfo, WeightBindingPlan},
};

use super::{DenseSwiGluLayerLoadConfig, NvFp4MoeLayerLoadConfig};
use crate::{CudaBackend, CudaMoeModelTemplate, CudaTensor, CudaTensorSet, Error, Result};

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
        let embedding = tensor(&tensors, source.embedding)?.clone();
        let final_norm = tensor(&tensors, source.final_norm)?.clone();
        let output = tensor(&tensors, source.output)?.clone();
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
        CudaMoeModelTemplate::new_dense(
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
        let embedding = tensor(&tensors, source.embedding)?.clone();
        let final_norm = tensor(&tensors, source.final_norm)?.clone();
        let output = tensor(&tensors, source.output)?.clone();
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

struct ModelSource<'a> {
    embedding: &'a str,
    final_norm: &'a str,
    output: &'a str,
    tensors: Vec<&'a TensorInfo>,
}

impl<'a> ModelSource<'a> {
    fn discover(
        decoder: &DecoderConfig,
        bindings: &WeightBindingPlan,
        catalog: &'a TensorCatalog,
    ) -> Result<Self> {
        let boundary = bindings.decoder_boundary_with_tied_output(decoder.tie_word_embeddings)?;
        let embedding = required(catalog, &boundary.embedding.source)?;
        let final_norm = required(catalog, &boundary.final_norm.source)?;
        let output = required(catalog, &boundary.output.source)?;
        let mut tensors = vec![embedding, final_norm];
        if output.name != embedding.name {
            tensors.push(output);
        }
        Ok(Self {
            embedding: &embedding.name,
            final_norm: &final_norm.name,
            output: &output.name,
            tensors,
        })
    }

    fn upload(&self, backend: &CudaBackend) -> Result<CudaTensorSet> {
        let mut upload = backend.begin_tensor_upload();
        for tensor in &self.tensors {
            upload.enqueue(tensor)?;
        }
        upload.finish()
    }

    fn payload_bytes(&self) -> Result<u64> {
        payload_bytes(self.tensors.iter().copied())
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

fn tensor<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}
