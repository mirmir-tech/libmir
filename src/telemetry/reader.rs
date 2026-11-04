use super::DeviceTelemetrySnapshot;

/// An OS telemetry source with no inference runtime or accelerator ownership.
#[derive(Clone, Debug, Default)]
pub struct DeviceTelemetryReader(pub(crate) Source);

#[derive(Clone, Debug, Default)]
pub enum Source {
    #[default]
    Unavailable,
    #[cfg(feature = "metal")]
    Apple,
    #[cfg(feature = "cuda")]
    Nvidia { ordinal: usize, name: String },
}

impl DeviceTelemetryReader {
    pub(crate) fn from_config(config: &crate::RuntimeConfig) -> Self {
        #[cfg(all(feature = "metal", any(not(feature = "cuda"), target_os = "macos")))]
        {
            let _ = config;
            Self(Source::Apple)
        }
        #[cfg(all(feature = "cuda", any(not(feature = "metal"), not(target_os = "macos"))))]
        {
            Self(Source::Nvidia {
                ordinal: config.cuda.device_ordinal,
                name: "NVIDIA GPU".into(),
            })
        }
        #[cfg(not(any(feature = "cuda", feature = "metal")))]
        {
            let _ = config;
            Self::default()
        }
    }

    /// Reads best-effort system metrics independently of model lifetime.
    #[must_use]
    pub fn snapshot(&self) -> DeviceTelemetrySnapshot {
        match &self.0 {
            Source::Unavailable => DeviceTelemetrySnapshot::default(),
            #[cfg(feature = "metal")]
            Source::Apple => {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                return super::apple_snapshot();
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                DeviceTelemetrySnapshot {
                    device_name: "Metal GPU".into(),
                    ..Default::default()
                }
            },
            #[cfg(feature = "cuda")]
            Source::Nvidia { ordinal, name } => {
                #[cfg(target_os = "linux")]
                return super::nvidia_snapshot(*ordinal, name);
                #[cfg(not(target_os = "linux"))]
                {
                    let _ = ordinal;
                    DeviceTelemetrySnapshot {
                        device_name: name.clone(),
                        ..Default::default()
                    }
                }
            },
        }
    }
}
