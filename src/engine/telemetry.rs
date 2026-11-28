use super::{Engine, EngineInner};
#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::telemetry::Source;
use crate::{DeviceTelemetryReader, DeviceTelemetrySnapshot};

impl Engine {
    pub(crate) fn device_telemetry_snapshot(&self) -> DeviceTelemetrySnapshot {
        self.device_telemetry_reader().snapshot()
    }

    pub(crate) fn device_telemetry_reader(&self) -> DeviceTelemetryReader {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                let device = cuda.device_info();
                DeviceTelemetryReader(Source::Nvidia {
                    ordinal: device.ordinal,
                    name: device.name.clone(),
                })
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => DeviceTelemetryReader(Source::Apple),
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable(never) => match *never {},
        }
    }
}
