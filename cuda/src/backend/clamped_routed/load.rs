use architecture::{cuda::CudaDecoderPlan, lowering::MixerLowering};
use models::{
    execution::DecoderExecutionContract,
    layout::DecoderConfig,
    weights::{TensorCatalog, TensorInfo},
};
use runtime::kv::CacheConfig;

use super::{
    ClampedRoutedCapabilityPlan, ClampedRoutedConfig, ClampedRoutedLayout,
    CudaClampedRoutedModelTemplate, layer::ClampedRoutedLayerTemplate, weights,
};
use crate::{CudaBackend, CudaTensorSet, Error, Result};

impl CudaBackend {
    #[allow(clippy::too_many_arguments)]
    pub fn load_clamped_routed_model_template_with_progress(
        &self,
        decoder: &DecoderConfig,
        plan: &CudaDecoderPlan,
        contract: &DecoderExecutionContract,
        catalog: &TensorCatalog,
        cache: CacheConfig,
        max_sequence_blocks: usize,
        ring_sessions: usize,
        progress: &mut dyn FnMut(u64, String),
    ) -> Result<CudaClampedRoutedModelTemplate> {
        let bindings = &contract.bindings;
        // Admission checked the spec; the loader reads the admitted plan.
        if plan.layers().len() != decoder.num_hidden_layers {
            return Err(Error::PlanLayerCount {
                plan: plan.layers().len(),
                decoder: decoder.num_hidden_layers,
            });
        }
        let config = ClampedRoutedConfig::from_admitted(plan, &contract.semantic)?;
        let layout = ClampedRoutedLayout::discover(bindings)?;
        let capabilities = ClampedRoutedCapabilityPlan::lower(config, layout, cache.dtype)?;
        if ring_sessions == 0 {
            return Err(Error::InvalidPagedKv("windowed KV session capacity is empty"));
        }
        let boundary_bindings = bindings.decoder_boundary()?;
        let boundary_names = owned_names(boundary_bindings.physical_sources());
        let boundary = upload_names(self, catalog, &boundary_names)?;
        let (embedding, final_norm, output) =
            weights::boundary(self, layout, &boundary, boundary_bindings, config)?;
        let mut completed = bytes_for_names(catalog, &boundary_names)?;
        progress(completed, "clamped-routed model boundary tensors".into());
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for (layer, lowered) in plan.layers().iter().enumerate() {
            let MixerLowering::Softmax { window, .. } = lowered.mixer else {
                return Err(Error::PlanMixer { layer });
            };
            let layer_bindings = bindings.routed_decoder_layer(layer)?;
            let names = owned_names(layer_bindings.physical_sources());
            let tensors = upload_names(self, catalog, &names)?;
            completed = completed
                .checked_add(bytes_for_names(catalog, &names)?)
                .ok_or(Error::InvalidDecoderKernel("checkpoint progress byte overflow"))?;
            layers.push(ClampedRoutedLayerTemplate::new(
                self,
                config,
                capabilities.qkv,
                weights::layer(self, layout, config, &tensors, layer_bindings)?,
                window.and_then(models::semantic::AttentionWindow::sliding),
            ));
            progress(
                completed,
                format!("clamped-routed layer {}/{}", layer + 1, decoder.num_hidden_layers),
            );
        }
        Ok(CudaClampedRoutedModelTemplate {
            backend: self.clone(),
            decoder: decoder.clone(),
            embedding,
            final_norm,
            output,
            layers,
            config,
            cache,
            max_sequence_blocks,
            ring_sessions,
        })
    }
}

fn upload_names(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    names: &[String],
) -> Result<CudaTensorSet> {
    let cast = backend.prepare_dense_cast()?;
    let mut upload = backend.begin_tensor_upload();
    for name in names {
        upload.enqueue_float_as_bf16(required_any(catalog, &[name])?, &cast)?;
    }
    upload.finish()
}

fn required_any<'a>(
    catalog: &'a TensorCatalog,
    names: &[impl AsRef<str>],
) -> Result<&'a TensorInfo> {
    names
        .iter()
        .find_map(|name| catalog.tensors.iter().find(|tensor| tensor.name == name.as_ref()))
        .ok_or_else(|| {
            Error::MissingTensor(names.iter().map(AsRef::as_ref).collect::<Vec<_>>().join(" | "))
        })
}

fn owned_names(names: Vec<&str>) -> Vec<String> {
    names.into_iter().map(str::to_owned).collect()
}

fn bytes_for_names(catalog: &TensorCatalog, names: &[String]) -> Result<u64> {
    names.iter().try_fold(0_u64, |total, name| {
        let bytes = required_any(catalog, &[name])?.payload_bytes()?;
        total
            .checked_add(u64::try_from(bytes)?)
            .ok_or(Error::InvalidDecoderKernel("checkpoint progress byte overflow"))
    })
}
