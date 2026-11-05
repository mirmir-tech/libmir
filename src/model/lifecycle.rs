use std::sync::Arc;

use super::{Library, Model};
use crate::{DeviceTelemetrySnapshot, Error, MemorySnapshot, Result};

impl Library {
    /// Creates an OS telemetry observer without initializing or retaining the
    /// engine.
    #[must_use]
    pub fn device_telemetry_reader(&self) -> crate::DeviceTelemetryReader {
        crate::DeviceTelemetryReader::from_config(&self.config)
    }

    /// Returns current memory information from the lazily initialized backend.
    pub fn memory_snapshot(&self) -> Result<MemorySnapshot> {
        Ok(self.engine()?.memory_snapshot()?)
    }

    /// Returns the latest utilization, temperature, and power readings for the
    /// configured accelerator when the platform exposes them.
    pub fn device_telemetry_snapshot(&self) -> Result<DeviceTelemetrySnapshot> {
        Ok(self.engine()?.device_telemetry_snapshot())
    }
}

impl Model {
    pub(crate) fn release_decode_session(&self, session_id: uuid::Uuid) {
        self.inner.coordinator.release(session_id);
    }

    #[must_use]
    /// Returns whether model clones or sessions currently prevent unloading.
    pub fn is_in_use(&self) -> bool {
        Arc::strong_count(&self.inner) > 1
    }

    /// Unloads backend resources when this is the model's sole remaining owner.
    pub fn unload(self) -> Result<()> {
        let Self { inner } = self;
        let Ok(inner) = Arc::try_unwrap(inner) else {
            return Err(Error::ModelInUse);
        };
        let engine = inner.engine.clone();
        let handle = inner.handle.clone();
        engine.unload_model(&handle)?;
        drop(inner);
        engine.clear_memory_cache()?;
        Ok(())
    }
}
