use std::time::{Duration, Instant};

use runtime::backend::{DecodeOutput, DecodeTimings};

use crate::{CudaBackend, Result, backend::DeviceTimer};

pub(super) struct DecodeProfile {
    started: Instant,
    device: DeviceTimer,
    backend_wait: Duration,
    rows: usize,
}

impl DecodeProfile {
    pub(super) fn begin(
        backend: &CudaBackend,
        backend_wait: Duration,
        rows: usize,
        enabled: bool,
    ) -> Result<Option<Self>> {
        enabled
            .then(|| {
                Ok(Self {
                    started: Instant::now(),
                    device: backend.start_device_timer()?,
                    backend_wait,
                    rows,
                })
            })
            .transpose()
    }

    pub(super) fn finish(self, backend: &CudaBackend, outputs: &mut [DecodeOutput]) -> Result<()> {
        let device_execution = self.device.finish(backend)?;
        let timings = DecodeTimings {
            scheduler_queue: Duration::ZERO,
            backend_wait: self.backend_wait,
            backend_execution: self.started.elapsed(),
            device_execution: Some(device_execution),
            batch_rows: self.rows,
        };
        for output in outputs {
            output.timings = Some(timings);
        }
        Ok(())
    }
}
