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
                let continuations = active
                    .requests
                    .iter()
                    .filter(|pending| pending.expects_decode)
                    .map(|pending| pending.request.session_id)
                    .collect::<Vec<_>>();
                self.begin_prefill_handoff(continuations);
                for (pending, output) in active.requests.into_iter().zip(outputs) {
                    pending.response.complete(Ok(output));
                }
            },
            Ok(_) => complete_prefill_errors(
                active.requests,
                "backend returned another prefill batch size",
            ),
            Err(error) => complete_prefill_errors(active.requests, &error.to_string()),
        }
    }

    pub(super) fn fail_active_prefill(&mut self, message: &str) {
        if let Some(active) = self.active_prefill.take() {
            complete_prefill_errors(active.requests, message);
        }
    }

    pub(super) fn fail_all(&mut self, message: &str) {
        self.prefill_cohort = None;
        complete_decode_errors(self.decode.drain(..).collect(), message);
        complete_prefill_errors(self.prefill.drain(..).collect(), message);
        self.fail_active_prefill(message);
    }
}
