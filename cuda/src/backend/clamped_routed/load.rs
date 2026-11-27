use architecture::{cuda::CudaDecoderPlan, lowering::MixerLowering};
use models::{execution::DecoderExecutionContract, layout::DecoderConfig, weights::TensorCatalog};
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
        let (boundary, mut completed) =
            upload_names(self, catalog, &boundary_bindings.physical_sources())?;
        let (embedding, final_norm, output) =
            weights::boundary(self, layout, &boundary, boundary_bindings, config)?;
        progress(completed, "clamped-routed model boundary tensors".into());
        let mut layers = Vec::with_capacity(decoder.num_hidden_layers);
        for (layer, lowered) in plan.layers().iter().enumerate() {
            let MixerLowering::Softmax { window, .. } = lowered.mixer else {
                return Err(Error::PlanMixer { layer });
            };
            let layer_bindings = bindings.routed_decoder_layer(layer)?;
            let (tensors, bytes) = upload_names(self, catalog, &layer_bindings.physical_sources())?;
            completed = add_bytes(completed, bytes)?;
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

/// Uploads the tensors `names` as BF16, each found once in the sorted
/// catalog, and their payload bytes, for progress.
fn upload_names(
    backend: &CudaBackend,
    catalog: &TensorCatalog,
    names: &[&str],
) -> Result<(CudaTensorSet, u64)> {
    let cast = backend.prepare_dense_cast()?;
    let mut upload = backend.begin_tensor_upload();
    let mut bytes = 0;
    for &name in names {
        let tensor = catalog.get(name).ok_or_else(|| Error::MissingTensor(name.to_owned()))?;
        upload.enqueue_float_as_bf16(tensor, &cast)?;
        bytes = add_bytes(bytes, u64::try_from(tensor.payload_bytes()?)?)?;
    }
    Ok((upload.finish()?, bytes))
}

fn add_bytes(total: u64, bytes: u64) -> Result<u64> {
    total
        .checked_add(bytes)
        .ok_or(Error::InvalidDecoderKernel("checkpoint progress byte overflow"))
}
