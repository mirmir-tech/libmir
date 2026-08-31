use std::time::Duration;

use super::GenerationMetricsRecorder;

impl Default for GenerationMetricsRecorder {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationMetricsRecorder {
    pub fn record_recovery_attempt(&mut self) {
        self.recovery_attempts += 1;
    }

    pub fn record_recovery_token(&mut self) {
        self.recovery_tokens += 1;
    }

    pub fn record_reasoning_exit(&mut self) {
        self.reasoning_exits += 1;
    }

    pub fn record_reasoning_exit_token(&mut self) {
        self.reasoning_exit_tokens += 1;
    }
}

pub(super) fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}
