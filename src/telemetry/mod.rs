#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod apple;
#[cfg(all(feature = "cuda", target_os = "linux"))]
mod nvidia;

/// Best-effort dynamic readings for the accelerator selected by a library.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceTelemetrySnapshot {
    /// Human-readable accelerator name when available.
    pub device_name: String,
    /// Current accelerator utilization from zero to one hundred percent.
    pub utilization_percent: Option<f64>,
    /// Current accelerator temperature in degrees Celsius.
    pub temperature_celsius: Option<f64>,
    /// Current accelerator power draw in Watts.
    pub power_watts: Option<f64>,
    /// Configured accelerator power limit in Watts when reported by hardware.
    pub power_limit_watts: Option<f64>,
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
pub use apple::snapshot as apple_snapshot;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use nvidia::snapshot as nvidia_snapshot;
