use runtime::backend::DecodeOutput;

use super::{PendingDecode, PendingPrefill};

pub(super) fn complete_decode(pending: Vec<PendingDecode>, outputs: Vec<DecodeOutput>) {
    if pending.len() != outputs.len() {
        complete_decode_errors(pending, "backend returned another decode batch size");
        return;
    }
    for (pending, mut output) in pending.into_iter().zip(outputs) {
        if let Some(timings) = output.timings.as_mut() {
            timings.scheduler_queue = pending.scheduler_queue;
        }
        pending.response.complete(Ok(output));
    }
}

pub(super) fn complete_decode_errors(pending: Vec<PendingDecode>, message: &str) {
    for pending in pending {
        pending.response.complete(Err(message.into()));
    }
}

pub(super) fn complete_prefill_errors(pending: Vec<PendingPrefill>, message: &str) {
    for pending in pending {
        pending.response.complete(Err(super::super::scheduler_error(message)));
    }
}
