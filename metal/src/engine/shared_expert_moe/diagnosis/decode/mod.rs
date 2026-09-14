mod drift;
mod fusion;
mod profile;

use super::{Array, Result, SharedExpertMoe, Stream, numerics::Difference};

#[derive(Clone, Copy, serde::Serialize)]
struct Case {
    layer: usize,
    step: usize,
}

impl SharedExpertMoe {
    pub(crate) fn diagnose_decode(
        &self,
        input: &Array,
        layer: usize,
        probe: crate::config::MoeDecodeProbe,
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(input.shape()?, [5, 1, 2048]);
        assert_eq!(self.config.top_k, 8);
        if matches!(
            probe,
            crate::config::MoeDecodeProbe::SubmissionDrift { .. }
                | crate::config::MoeDecodeProbe::FusionDrift { .. }
                | crate::config::MoeDecodeProbe::SubmissionLengths { .. }
                | crate::config::MoeDecodeProbe::ContinuousWindows { .. }
        ) && layer != 0
        {
            return Ok(());
        }
        stream.eval_many(&[input])?;
        stream.synchronize()?;
        match probe {
            crate::config::MoeDecodeProbe::ContinuousWindows { step } => {
                self.diagnose_continuous_windows(input, Case { layer, step }, stream)
            },
            crate::config::MoeDecodeProbe::SubmissionLengths { step } => {
                self.diagnose_submission_lengths(input, Case { layer, step }, stream)
            },
            crate::config::MoeDecodeProbe::FusionDrift { step } => {
                self.diagnose_fusion_drift(input, Case { layer, step }, stream)
            },
            crate::config::MoeDecodeProbe::SubmissionDrift { step } => {
                self.diagnose_submission_drift(input, Case { layer, step }, stream)
            },
            crate::config::MoeDecodeProbe::Components { step } => {
                let reference = self.forward(input, stream)?.to_vec_f32(stream)?;
                self.profile_decode(input, Case { layer, step }, &reference, stream)
            },
            crate::config::MoeDecodeProbe::GateUpFusion { step } => {
                self.compare_decode_fusion(input, Case { layer, step }, stream)
            },
        }
    }
}
