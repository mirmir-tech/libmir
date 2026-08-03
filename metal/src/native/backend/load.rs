use std::sync::Arc;

use foundation::model::ModelManifest;
use runtime::{
    Result as RuntimeResult,
    backend::{Backend, ModelHandle},
    kv::CacheConfig,
};

use super::MetalBackend;
use crate::{
    MetalProgressEvent,
    native::{backend::worker::ModelClient, error::Result, trace},
};

impl MetalBackend {
    pub fn load_model_with_progress(
        &self,
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> RuntimeResult<ModelHandle> {
        Ok(self.load_model_inner(manifest, None, Some(progress))?)
    }

    pub fn load_model_with_progress_and_reservation(
        &self,
        manifest: &ModelManifest,
        reserved_bytes: usize,
        cache: CacheConfig,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> RuntimeResult<ModelHandle> {
        Ok(self.load_model_inner(manifest, Some((reserved_bytes, cache)), Some(progress))?)
    }

    pub(super) fn load_model_inner(
        &self,
        manifest: &ModelManifest,
        reservation: Option<(usize, CacheConfig)>,
        progress: Option<&mut dyn FnMut(MetalProgressEvent)>,
    ) -> Result<ModelHandle> {
        progress.map_or_else(
            || {
                let mut ignored = |_event: MetalProgressEvent| {};
                self.load_model_with_callback(manifest, reservation, &mut ignored)
            },
            |progress| self.load_model_with_callback(manifest, reservation, progress),
        )
    }

    fn load_model_with_callback(
        &self,
        manifest: &ModelManifest,
        reservation: Option<(usize, CacheConfig)>,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<ModelHandle> {
        let span = tracing::debug_span!(
            "mlx.native.load_model",
            model_id = %manifest.id,
            model_path = %manifest.path
        );
        let _guard = span.enter();
        let mut config = (*self.config).clone();
        if let Some((reserved_bytes, cache)) = reservation {
            config.expert_fusion_reserve_bytes = Some(reserved_bytes);
            config.kv_cache = cache;
            config.cache.kv_reserve_tokens = usize::try_from(cache.block_count)
                .unwrap_or(usize::MAX)
                .saturating_mul(cache.block_size);
        }
        let client = ModelClient::spawn(
            manifest.clone(),
            Arc::new(config),
            Arc::clone(&self.paged_arenas),
            progress,
        )?;
        let backend = self.info();
        let model_trace = client.run(move |loaded| Ok(trace::build(loaded, backend)))?;
        trace::emit(&model_trace);
        self.models.lock()?.models.insert(manifest.id.clone(), client);
        Ok(ModelHandle {
            id: manifest.id.clone(),
            backend: "mlx-native".into(),
        })
    }
}
