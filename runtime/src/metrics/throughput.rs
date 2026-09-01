use std::time::Duration;

use super::{ThroughputRate, ThroughputUnit};

impl ThroughputRate {
    #[must_use]
    pub fn new(units: usize, duration: Duration, unit: ThroughputUnit) -> Self {
        let per_second = if units == 0 || duration.is_zero() {
            None
        } else {
            units
                .to_string()
                .parse::<f64>()
                .ok()
                .map(|units| units / duration.as_secs_f64())
        };
        Self { per_second, unit }
    }
}
