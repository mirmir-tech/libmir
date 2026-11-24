use std::collections::HashSet;

use models::{
    execution::TaskExecutionPlan,
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{
        LayerTensorRole, TensorBinding, TensorCatalog, TensorInfo, TensorStorage, WeightBindingPlan,
    },
};

use super::{SharedRoutedModelLoadConfig, model::payload_bytes};
use crate::{
    CudaAdmitted, CudaArchitecture, CudaBackend, CudaSharedRoutedModelTemplate, Error, Result,
};

impl CudaBackend {
    pub fn load_shared_routed_model_template(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        load: SharedRoutedModelLoadConfig,
    ) -> Result<CudaSharedRoutedModelTemplate> {
        let mut ignored = |_completed, _detail| {};
        let spec = SemanticModelSpec::discover(decoder, catalog)?;
        let bindings = WeightBindingPlan::discover(&spec, catalog)?;
        let task = TaskExecutionPlan::Generation { decoder: decoder.clone() };
        let CudaArchitecture::Generation(admitted) = crate::admit_architecture(&task, Some(&spec))?
        else {
            return Err(Error::State("CUDA admitted a non-generation runtime".into()));
        };
        self.load_shared_routed_model_template_with_progress(
            decoder,
            (&admitted, &spec),
            catalog,
            &bindings,
            load,
            &mut ignored,
        )
    }

    pub(crate) fn load_shared_routed_model_template_with_progress(
        &self,
        decoder: &DecoderConfig,
        (admitted, semantic): (&CudaAdmitted, &SemanticModelSpec),
        catalog: &TensorCatalog,
        bindings: &WeightBindingPlan,
        load: SharedRoutedModelLoadConfig,
        progress: &mut dyn FnMut(u64, String),
    ) -> Result<CudaSharedRoutedModelTemplate> {
        if load.max_sequence_blocks == 0 {
            return Err(Error::UnsupportedDecoderLayer(
                "shared-routed model sequence block capacity is empty".into(),
            ));
        }
        let prediction = load.speculation.is_some()
            && decoder.state_space.is_some()
            && !semantic.decoder.prediction_layers.is_empty();
        let source = shared_routed_source(bindings, decoder, catalog, prediction)?;
        let raw = raw_sources(bindings);
        let cast = self.prepare_dense_cast()?;
        let mut upload = self.begin_tensor_upload();
        for tensor in &source {
            if raw.contains(tensor.name.as_str()) {
                upload.enqueue(tensor)?;
            } else {
                upload.enqueue_float_as_bf16(tensor, &cast)?;
            }
        }
        let tensors = upload.finish()?;
        let bytes = payload_bytes(source.iter().copied())?;
        progress(bytes, format!("uploaded {} shared-routed checkpoint tensors", source.len()));
        tracing::debug!(
            backend = "cuda",
            layers = decoder.num_hidden_layers,
            tensors = source.len(),
            bytes,
            "loaded shared-routed mixed-mixer model template"
        );
        CudaSharedRoutedModelTemplate::from_tensors(
            self,
            decoder,
            admitted,
            semantic,
            &tensors,
            catalog,
            bindings,
            load.cache,
            load.max_sequence_blocks,
            load.speculation,
        )
    }
}

/// Every tensor uploaded with the template, with the prediction module's
/// when `prediction`. Individual expert tensors are excluded: their banks are
/// built straight from the checkpoint.
fn shared_routed_source<'a>(
    bindings: &WeightBindingPlan,
    decoder: &DecoderConfig,
    catalog: &'a TensorCatalog,
    prediction: bool,
) -> Result<Vec<&'a TensorInfo>> {
    let individual = bindings
        .tensors
        .iter()
        // Quantized per-expert tensors stream from the checkpoint into their
        // banks; BF16 ones are stacked from the uploaded set.
        .filter(|binding| {
            matches!(
                binding.role.layer_tensor(),
                Some(LayerTensorRole::ExpertProjection { expert: Some(_), .. })
            ) && !matches!(binding.storage, TensorStorage::Dense { .. })
        })
        .flat_map(|binding| binding.physical_sources())
        .collect::<HashSet<_>>();
    let tied = decoder.tie_word_embeddings;
    let mut names = bindings.decoder_boundary_with_tied_output(tied)?.physical_sources();
    for layer in 0..decoder.num_hidden_layers {
        if decoder.state_space.is_some() {
            names.extend(bindings.state_space_hybrid_layer(layer)?.physical_sources());
        } else if decoder.latent.is_some() {
            names.extend(bindings.latent_decoder_layer(layer)?.physical_sources());
        } else {
            names.extend(bindings.mixed_decoder_layer(layer)?.physical_sources());
        }
    }
    if prediction {
        names.extend(bindings.prediction_layer(0)?.physical_sources());
    }
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| !individual.contains(name))
        .filter(|name| seen.insert(*name))
        .map(|name| required(catalog, name))
        .collect()
}

/// Tensors uploaded in their checkpoint dtype; every other floating tensor
/// is converted to BF16.
fn raw_sources(bindings: &WeightBindingPlan) -> HashSet<&str> {
    bindings
        .tensors
        .iter()
        .flat_map(TensorBinding::full_precision_sources)
        .collect()
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}
