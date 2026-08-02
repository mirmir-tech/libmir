use super::{Engine, EngineInner};
use crate::DeviceTelemetrySnapshot;

impl Engine {
    pub(crate) fn device_telemetry_snapshot(&self) -> DeviceTelemetrySnapshot {
        match &self.inner {
            #[cfg(feature = "cuda")]
            EngineInner::Cuda(cuda) => {
                let device = cuda.device_info();
                #[cfg(target_os = "linux")]
                return crate::telemetry::nvidia_snapshot(device.ordinal, &device.name);
                #[cfg(not(target_os = "linux"))]
                DeviceTelemetrySnapshot {
                    device_name: device.name.clone(),
                    ..DeviceTelemetrySnapshot::default()
                }
            },
            #[cfg(feature = "metal")]
            EngineInner::Metal(_) => {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                return crate::telemetry::apple_snapshot();
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                DeviceTelemetrySnapshot {
                    device_name: "Metal GPU".to_owned(),
                    ..DeviceTelemetrySnapshot::default()
                }
            },
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            EngineInner::Unavailable => DeviceTelemetrySnapshot::default(),
        }
    }
}
