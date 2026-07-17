use foundation::model::ModelManifest;
use runtime::{
    Result as RuntimeResult,
    backend::{Backend, ModelHandle},
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
        Ok(self.load_model_inner(manifest, Some(progress))?)
    }

    pub(super) fn load_model_inner(
        &self,
        manifest: &ModelManifest,
        progress: Option<&mut dyn FnMut(MetalProgressEvent)>,
    ) -> Result<ModelHandle> {
        progress.map_or_else(
            || {
                let mut ignored = |_event: MetalProgressEvent| {};
                self.load_model_with_callback(manifest, &mut ignored)
            },
            |progress| self.load_model_with_callback(manifest, progress),
        )
    }

    fn load_model_with_callback(
        &self,
        manifest: &ModelManifest,
        progress: &mut dyn FnMut(MetalProgressEvent),
    ) -> Result<ModelHandle> {
        let span = tracing::debug_span!(
            "mlx.native.load_model",
            model_id = %manifest.id,
            model_path = %manifest.path
        );
        let _guard = span.enter();
        let client = ModelClient::spawn(manifest.clone(), self.config.clone(), progress)?;
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
