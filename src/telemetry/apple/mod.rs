//! Apple silicon GPU readings. Under `metal-telemetry`, utilization,
//! temperature and power come from macmon's `IOReport` and `IOHID` sampling.
//! Those are private macOS interfaces, which the Mac App Store rejects, so
//! builds without it read utilization from public `IOKit` registry properties
//! and report no temperature or power.

#[cfg(not(feature = "metal-telemetry"))]
mod accelerator;
#[cfg(feature = "metal-telemetry")]
mod sampler;

use std::sync::OnceLock;

#[cfg(feature = "metal-telemetry")]
pub use sampler::snapshot;
use sysinfo::{CpuRefreshKind, RefreshKind, System};

use super::DeviceTelemetrySnapshot;

#[cfg(not(feature = "metal-telemetry"))]
#[must_use]
pub fn snapshot() -> DeviceTelemetrySnapshot {
    DeviceTelemetrySnapshot {
        utilization_percent: accelerator::utilization_percent(),
        ..named_snapshot()
    }
}

fn named_snapshot() -> DeviceTelemetrySnapshot {
    DeviceTelemetrySnapshot {
        device_name: chip_name().to_owned(),
        ..DeviceTelemetrySnapshot::default()
    }
}

/// The chip's marketing name, such as "Apple M3 Max", read once through sysctl.
fn chip_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        let cpus = RefreshKind::nothing().with_cpu(CpuRefreshKind::nothing());
        System::new_with_specifics(cpus)
            .cpus()
            .first()
            .map(|cpu| cpu.brand().trim().to_owned())
            .filter(|brand| !brand.is_empty())
            .unwrap_or_else(|| "Apple GPU".to_owned())
    })
}
