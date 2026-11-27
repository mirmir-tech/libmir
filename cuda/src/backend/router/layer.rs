//! A unit router's top-k with the correction bias of the one layer whose
//! weights its executor already binds.

use mircuda::{Compiler, DeviceBuffer, Stream, bf16};

use crate::{
    Error, Result,
    kernels::{RouterUnitScoring, RouterUnitSpec, RouterUnitTopK, RouterUnitWeighting},
};

#[derive(Debug)]
pub(in crate::backend) struct LayerRouterTopK {
    top_k: RouterUnitTopK,
    correction_bias: Option<DeviceBuffer<f32>>,
}

impl LayerRouterTopK {
    /// A sigmoid router takes its layer's `[experts]` correction bias; the
    /// others take none.
    pub(in crate::backend) fn compile(
        compiler: &Compiler,
        spec: RouterUnitSpec,
        scoring: RouterUnitScoring,
        correction_bias: Option<&DeviceBuffer<f32>>,
    ) -> Result<Self> {
        let sigmoid = scoring.weighting == RouterUnitWeighting::SigmoidBiased;
        if sigmoid != correction_bias.is_some() {
            return Err(Error::InvalidRouter(
                "a unit router takes a correction bias exactly when it routes by sigmoid",
            ));
        }
        Ok(Self {
            top_k: RouterUnitTopK::compile_scored(compiler, spec, scoring)?,
            correction_bias: correction_bias.cloned(),
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        stream: &Stream,
        scores: &DeviceBuffer<bf16>,
        selected: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.top_k
            .execute(stream, scores, self.correction_bias.as_ref(), selected, weights)
    }
}
