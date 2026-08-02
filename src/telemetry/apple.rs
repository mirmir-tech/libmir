use std::{
    sync::{Mutex, OnceLock, mpsc},
    thread,
};

use macmon::{Metrics, Sampler, SocInfo};

use super::DeviceTelemetrySnapshot;

const SAMPLE_INTERVAL_MS: u32 = 1_000;
static SAMPLES: OnceLock<Mutex<SampleState>> = OnceLock::new();

struct SampleState {
    receiver: mpsc::Receiver<Metrics>,
    latest: Option<DeviceTelemetrySnapshot>,
}

pub fn snapshot() -> DeviceTelemetrySnapshot {
    let state = SAMPLES.get_or_init(spawn_sampler);
    let Ok(mut state) = state.lock() else {
        return named_snapshot();
    };
    if let Some(metrics) = state.receiver.try_iter().last() {
        state.latest = Some(from_metrics(&metrics));
    }
    state.latest.clone().unwrap_or_else(named_snapshot)
}

fn spawn_sampler() -> Mutex<SampleState> {
    let (sender, receiver) = mpsc::channel();
    let _worker =
        thread::Builder::new()
            .name("libmir-device-telemetry".to_owned())
            .spawn(move || {
                let Ok(mut sampler) = Sampler::new() else {
                    tracing::warn!("Apple device telemetry sampler is unavailable");
                    return;
                };
                while let Ok(metrics) = sampler.get_metrics(SAMPLE_INTERVAL_MS) {
                    if sender.send(metrics).is_err() {
                        return;
                    }
                }
                tracing::warn!("Apple device telemetry sampler stopped");
            });
    Mutex::new(SampleState { receiver, latest: None })
}

fn from_metrics(metrics: &Metrics) -> DeviceTelemetrySnapshot {
    let mut snapshot = named_snapshot();
    snapshot.utilization_percent = finite_positive_or_zero(metrics.gpu_active_ratio * 100.0);
    snapshot.temperature_celsius = finite_positive(metrics.temp.gpu_temp_avg);
    snapshot.power_watts = finite_positive_or_zero(metrics.gpu_power);
    snapshot
}

fn named_snapshot() -> DeviceTelemetrySnapshot {
    DeviceTelemetrySnapshot {
        device_name: SocInfo::new().map_or_else(|_| "Apple GPU".to_owned(), |soc| soc.chip_name),
        ..DeviceTelemetrySnapshot::default()
    }
}

fn finite_positive(value: f32) -> Option<f64> {
    value.is_finite().then_some(f64::from(value)).filter(|value| *value > 0.0)
}

fn finite_positive_or_zero(value: f32) -> Option<f64> {
    value.is_finite().then_some(f64::from(value)).filter(|value| *value >= 0.0)
}
