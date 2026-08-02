use std::collections::HashSet;

use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{
        LayerTensorRole, LogicalTensorRole, TensorCatalog, TensorInfo, TensorStorage,
        WeightBindingPlan,
    },
};

use super::{SharedRoutedModelLoadConfig, model::payload_bytes};
use crate::{CudaBackend, CudaSharedRoutedModelTemplate, Error, Result};

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
        self.load_shared_routed_model_template_with_progress(
            decoder, &spec, catalog, &bindings, load, &mut ignored,
        )
    }

    pub(crate) fn load_shared_routed_model_template_with_progress(
        &self,
        decoder: &DecoderConfig,
        semantic: &SemanticModelSpec,
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
        let source = shared_routed_source(bindings, decoder.num_hidden_layers, catalog)?;
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
            semantic,
            &tensors,
            catalog,
            bindings,
            load.cache,
            load.max_sequence_blocks,
        )
    }
}

fn shared_routed_source<'a>(
    bindings: &WeightBindingPlan,
    layers: usize,
    catalog: &'a TensorCatalog,
) -> Result<Vec<&'a TensorInfo>> {
    let individual = bindings
        .tensors
        .iter()
        .filter(|binding| {
            matches!(
                binding.role,
                LogicalTensorRole::Layer {
                    tensor: LayerTensorRole::ExpertProjection { expert: Some(_), .. },
                    ..
                }
            )
        })
        .flat_map(|binding| binding.physical_sources())
        .collect::<HashSet<_>>();
    let mut names = bindings.decoder_boundary()?.physical_sources();
    for layer in 0..layers {
        names.extend(bindings.hybrid_decoder_layer(layer)?.physical_sources());
    }
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| !individual.contains(name))
        .filter(|name| seen.insert(*name))
        .map(|name| required(catalog, name))
        .collect()
}

fn raw_sources(bindings: &WeightBindingPlan) -> HashSet<&str> {
    bindings
        .tensors
        .iter()
        .filter(|binding| {
            matches!(
                binding.storage,
                TensorStorage::AffineQuantized { .. }
                    | TensorStorage::BlockQuantized { .. }
                    | TensorStorage::Float8 { .. }
                    | TensorStorage::PackedInt8 { .. }
                    | TensorStorage::PackedInt4 { .. }
                    | TensorStorage::Awq { .. }
                    | TensorStorage::Gptq { .. }
                    | TensorStorage::BitsAndBytes4Bit { .. }
            )
        })
        .flat_map(|binding| binding.physical_sources())
        .collect()
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
