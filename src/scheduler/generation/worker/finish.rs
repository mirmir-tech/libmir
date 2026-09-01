use super::Worker;
use crate::{
    engine::EngineGenerationStepOutput,
    scheduler::generation::{
        PendingDecode,
        completion::{complete_decode, complete_decode_errors, complete_prefill_errors},
    },
};

impl Worker {
    pub(super) fn complete_step(
        &mut self,
        decode: Vec<PendingDecode>,
        output: EngineGenerationStepOutput,
    ) {
        let rows = decode.len();
        complete_decode(decode, output.decode);
        self.observe_decode(rows);
        match output.prefill {
            Ok(true) => self.finish_prefill(),
            Ok(false) => {},
            Err(error) => self.fail_active_prefill(&error.to_string()),
        }
    }

    fn finish_prefill(&mut self) {
        let Some(active) = self.active_prefill.take() else {
            return;
        };
        match self.engine.finish_generation_prefill(active.batch) {
            Ok(outputs) if outputs.len() == active.requests.len() => {
                self.completed_prefill.extend(active.requests.into_iter().zip(outputs));
                if hold_prefill_completion(
                    self.prefill_profile.interleave_prefill_decode,
                    self.prefill_cohort.is_some(),
                ) {
                    return;
                }
                let completed = std::mem::take(&mut self.completed_prefill);
                let continuations = completed
                    .iter()
                    .filter(|(pending, _)| pending.expects_decode)
                    .map(|(pending, _)| pending.request.session_id)
                    .collect::<Vec<_>>();
                self.begin_prefill_handoff(continuations);
                for (pending, mut output) in completed {
                    output.timings.get_or_insert_default().scheduler_queue =
                        pending.scheduler_queue;
                    pending.response.complete(Ok(output));
                }
            },
            Ok(_) => {
                let message = "backend returned another prefill batch size";
                complete_prefill_errors(active.requests, message);
                self.fail_completed_prefill(message);
            },
            Err(error) => {
                complete_prefill_errors(active.requests, &error.to_string());
                self.fail_completed_prefill(&error.to_string());
            },
        }
    }

    pub(super) fn fail_active_prefill(&mut self, message: &str) {
        if let Some(active) = self.active_prefill.take() {
            complete_prefill_errors(active.requests, message);
        }
        self.fail_completed_prefill(message);
    }

    pub(super) fn fail_all(&mut self, message: &str) {
        self.prefill_cohort = None;
        complete_decode_errors(self.decode.drain(..).collect(), message);
        complete_prefill_errors(self.prefill.drain(..).collect(), message);
        self.fail_active_prefill(message);
    }

    pub(super) fn fail_completed_prefill(&mut self, message: &str) {
        let pending = self.completed_prefill.drain(..).map(|(pending, _)| pending).collect();
        complete_prefill_errors(pending, message);
    }
}

const fn hold_prefill_completion(interleave: bool, cohort_has_more_waves: bool) -> bool {
    !interleave && cohort_has_more_waves
}

#[cfg(test)]
mod tests {
    use super::hold_prefill_completion;

    #[test]
    fn non_interleaved_backends_release_the_logical_cohort_together() {
        assert!(hold_prefill_completion(false, true));
        assert!(!hold_prefill_completion(false, false));
        assert!(!hold_prefill_completion(true, true));
    }
}
