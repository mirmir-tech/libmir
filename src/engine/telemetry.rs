use super::{Engine, EngineInner};
use crate::{DeviceTelemetryReader, DeviceTelemetrySnapshot, telemetry::Source};

impl Engine {
    pub(crate) fn device_telemetry_snapshot(&self) -> DeviceTelemetrySnapshot {
        self.device_telemetry_reader().snapshot()
    }

    pub(crate) fn device_telemetry_reader(&self) -> DeviceTelemetryReader {
        DeviceTelemetryReader(match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                let device = cuda.device_info();
                Source::Nvidia {
                    ordinal: device.ordinal,
                    name: device.name.clone(),
                }
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => Source::Apple,
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => Source::Unavailable,
        })
    }
}
