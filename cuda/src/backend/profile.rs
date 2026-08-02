use std::time::Duration;

use mircuda::{Event, ProfilerRange};

use super::CudaBackend;
use crate::Result;

pub struct DeviceTimer {
    started: Event,
    completed: Event,
}

pub struct ProfilerCapture(ProfilerRange);

impl CudaBackend {
    pub(crate) fn start_device_timer(&self) -> Result<DeviceTimer> {
        let started = self.inner.context.create_event(true)?;
        let completed = self.inner.context.create_event(true)?;
        started.record(&self.inner.stream)?;
        Ok(DeviceTimer { started, completed })
    }

    pub(crate) fn start_profiler_capture(&self) -> Result<ProfilerCapture> {
        Ok(ProfilerCapture(self.inner.context.start_profiler_range()?))
    }
}

impl DeviceTimer {
    pub(crate) fn finish(self, backend: &CudaBackend) -> Result<Duration> {
        self.completed.record(&backend.inner.stream)?;
        self.completed.synchronize()?;
        let seconds = self.started.elapsed_ms(&self.completed)? / 1_000.0;
        Ok(Duration::from_secs_f32(seconds))
    }
}

impl ProfilerCapture {
    pub(crate) fn stop(self) -> Result<()> {
        Ok(self.0.stop()?)
    }
}
