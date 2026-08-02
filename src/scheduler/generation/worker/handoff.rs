use std::{
    collections::HashSet,
    time::{Duration, Instant},
};

use super::Worker;

#[derive(Default)]
pub(super) struct PrefillHandoff {
    sessions: HashSet<uuid::Uuid>,
    started: Option<Instant>,
    expected: usize,
}

impl Worker {
    pub(super) const fn prefill_handoff_active(&self) -> bool {
        self.prefill_handoff.started.is_some()
    }

    pub(super) fn begin_prefill_handoff(&mut self, sessions: impl IntoIterator<Item = uuid::Uuid>) {
        if !self.prefill_profile.limit_deep_prefill_waves || self.prefill.is_empty() {
            return;
        }
        self.prefill_handoff.begin(sessions);
    }

    pub(super) fn collect_prefill_handoff(&mut self) {
        if !self.prefill_handoff_active() || !self.decode.is_empty() {
            return;
        }
        if self.prefill.is_empty() {
            self.finish_prefill_handoff(0, "prefill queue drained");
            return;
        }
        while self.decode.is_empty() && !self.prefill_handoff.sessions.is_empty() && !self.stopping
        {
            match self.commands.recv() {
                Ok(command) => self.admit(command),
                Err(_) => self.stopping = true,
            }
        }
        if self.decode.is_empty() && self.prefill_handoff.sessions.is_empty() {
            self.finish_prefill_handoff(0, "all continuations released");
        }
    }

    pub(super) fn resolve_prefill_handoff(&mut self, session: uuid::Uuid) {
        self.prefill_handoff.sessions.remove(&session);
    }

    pub(super) fn observe_prefill_handoff_decode(&mut self, rows: usize) {
        if rows > 0 && self.prefill_handoff_active() {
            self.finish_prefill_handoff(rows, "decode continuation admitted");
        }
    }

    fn finish_prefill_handoff(&mut self, decode_rows: usize, outcome: &'static str) {
        let elapsed = self.prefill_handoff.elapsed();
        tracing::debug!(
            expected_sessions = self.prefill_handoff.expected,
            unresolved_sessions = self.prefill_handoff.sessions.len(),
            decode_rows,
            handoff_ms = elapsed.as_secs_f64() * 1_000.0,
            outcome,
            "completed event-driven prefill handoff"
        );
        self.prefill_handoff.clear();
    }
}

impl PrefillHandoff {
    fn begin(&mut self, sessions: impl IntoIterator<Item = uuid::Uuid>) {
        self.sessions = sessions.into_iter().collect();
        self.expected = self.sessions.len();
        self.started = (!self.sessions.is_empty()).then(Instant::now);
    }

    fn elapsed(&self) -> Duration {
        self.started.map_or(Duration::ZERO, |started| started.elapsed())
    }

    fn clear(&mut self) {
        self.sessions.clear();
        self.started = None;
        self.expected = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::PrefillHandoff;

    #[test]
    fn handoff_tracks_decode_or_release_resolution() {
        let first = uuid::Uuid::new_v4();
        let second = uuid::Uuid::new_v4();
        let mut handoff = PrefillHandoff::default();
        handoff.begin([first, second]);
        assert_eq!(handoff.expected, 2);
        assert!(handoff.started.is_some());
        assert!(handoff.sessions.remove(&first));
        assert_eq!(handoff.sessions, std::iter::once(second).collect());
        handoff.clear();
        assert!(handoff.started.is_none());
        assert!(handoff.sessions.is_empty());
    }
}
