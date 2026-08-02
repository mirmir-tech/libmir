use std::sync::OnceLock;

use nvml_wrapper::{Nvml, enum_wrappers::device::TemperatureSensor};

use super::DeviceTelemetrySnapshot;

static NVML: OnceLock<Result<Nvml, String>> = OnceLock::new();

pub fn snapshot(ordinal: usize, fallback_name: &str) -> DeviceTelemetrySnapshot {
    let mut snapshot = DeviceTelemetrySnapshot {
        device_name: fallback_name.to_owned(),
        ..DeviceTelemetrySnapshot::default()
    };
    let Ok(nvml) = NVML.get_or_init(|| Nvml::init().map_err(|error| error.to_string())) else {
        return snapshot;
    };
    let Ok(index) = u32::try_from(ordinal) else {
        return snapshot;
    };
    let Ok(device) = nvml.device_by_index(index) else {
        return snapshot;
    };
    snapshot.device_name = device.name().unwrap_or_else(|_| fallback_name.to_owned());
    snapshot.utilization_percent =
        device.utilization_rates().ok().map(|utilization| f64::from(utilization.gpu));
    snapshot.temperature_celsius = device.temperature(TemperatureSensor::Gpu).ok().map(f64::from);
    snapshot.power_watts = device.power_usage().ok().map(milliwatts_to_watts);
    snapshot.power_limit_watts = device.power_management_limit().ok().map(milliwatts_to_watts);
    snapshot
}

fn milliwatts_to_watts(value: u32) -> f64 {
    f64::from(value) / 1_000.0
}
