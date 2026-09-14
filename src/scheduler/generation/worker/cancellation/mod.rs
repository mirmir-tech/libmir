use super::Worker;
use crate::{Error, Result};

impl Worker {
    pub(super) fn cancel_prefills(&mut self) {
        if let Err(error) = self.cancel_queued_prefills() {
            self.fail_all(&error.to_string());
            return;
        }
        if let Some(active) = &mut self.active_prefill {
            let sessions = active
                .requests
                .iter()
                .filter(|pending| pending.cancellation.is_cancelled())
                .map(|pending| pending.request.session_id)
                .collect::<Vec<_>>();
            if !sessions.is_empty() {
                if let Err(error) =
                    self.engine.cancel_generation_prefill(&mut active.batch, &sessions)
                {
                    self.fail_active_prefill(&error.to_string());
                    return;
                }
                for pending in active
                    .requests
                    .extract_if(.., |pending| sessions.contains(&pending.request.session_id))
                {
                    pending.response.complete(Err(Error::Cancelled));
                }
                if active.requests.is_empty() {
                    self.active_prefill = None;
                }
            }
        }
        let completed = self
            .completed_prefill
            .extract_if(.., |(pending, _)| pending.cancellation.is_cancelled())
            .collect::<Vec<_>>();
        for (pending, _) in completed {
            let result = self.engine.release_session(&self.model, pending.request.session_id);
            pending.response.complete(match result {
                Ok(()) => Err(Error::Cancelled),
                Err(error) => Err(error.into()),
            });
        }
        if self.active_prefill.is_none() {
            self.publish_completed_prefill();
        }
    }

    fn cancel_queued_prefills(&mut self) -> Result<()> {
        let cohort_rows = self.prefill_cohort.as_ref().map_or(0, |cohort| cohort.remaining);
        let sessions = self
            .prefill
            .iter()
            .take(cohort_rows)
            .filter(|pending| pending.cancellation.is_cancelled())
            .map(|pending| pending.request.session_id)
            .collect::<Vec<_>>();
        if !sessions.is_empty() {
            if let Some(cohort) = &self.prefill_cohort {
                self.engine.discard_prefill_cohort_sessions(&cohort.lease, &sessions)?;
            }
            self.advance_prefill_cohort(sessions.len());
        }
        // Snapshot the selection before removing anything: cancellation can race
        // this pass, and a newly flagged cohort row still owns its lease/count.
        let cancelled = self
            .prefill
            .iter()
            .skip(cohort_rows)
            .filter(|pending| pending.cancellation.is_cancelled())
            .map(|pending| pending.request.session_id)
            .chain(sessions)
            .collect::<Vec<_>>();
        self.prefill.retain(|pending| {
            if cancelled.contains(&pending.request.session_id) {
                pending.response.complete(Err(Error::Cancelled));
                false
            } else {
                true
            }
        });
        Ok(())
    }
}

#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod tests;
