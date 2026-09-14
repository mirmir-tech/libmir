use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::{DecodeInput, LoadedModel, NativeOutput, Result};

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::native) enum Stage {
    SampleSubmit,
    GraphBuild,
    RootsSubmit,
    TokenRead,
    Complete,
}

#[derive(serde::Serialize)]
struct Span {
    stage: Stage,
    end_ms: f64,
}

/// Observes existing call boundaries; never adds GPU barriers or host reads.
/// Submission calls can block, so these are wall spans, not pure CPU timings.
#[derive(serde::Serialize)]
pub(in crate::native) struct Profile {
    start_unix_ns: u128,
    #[serde(skip)]
    started: Instant,
    spans: Vec<Span>,
}

impl Profile {
    fn new() -> Result<Self> {
        let spans = Vec::with_capacity(5);
        let start_unix_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        Ok(Self {
            start_unix_ns,
            started: Instant::now(),
            spans,
        })
    }

    pub(super) fn mark(&mut self, stage: Stage) {
        self.spans.push(Span {
            stage,
            end_ms: self.started.elapsed().as_secs_f64() * 1000.0,
        });
    }
}

impl LoadedModel {
    pub(in crate::native) fn decode_batch_profiled(
        &mut self,
        inputs: &[DecodeInput],
    ) -> Result<(Vec<NativeOutput>, Profile)> {
        let mut profile = Profile::new()?;
        let outputs = self.decode_batch_inner(inputs, Some(&mut profile))?;
        Ok((outputs, profile))
    }
}
