use super::{Array, Error, Result, RoutedGateUp, SharedExpertMoe, Stream};
use crate::engine::{SortedExpertInputs, kernels::expert_group::tiles::TilePlan};

impl SharedExpertMoe {
    // Shared by actual-layer replay and the test-only whole-model candidate.
    pub(super) fn tiled_mlp(
        &self,
        grouped: &SortedExpertInputs,
        plan: &TilePlan,
        stream: &Stream,
    ) -> Result<Array> {
        let tiles = plan.prepare(&grouped.indices, self.config.expert_count, stream)?;
        let RoutedGateUp::Separate { gate, up, .. } = &self.routed_gate_up else {
            return Err(Error::InvalidModel("tile probe requires separate banks".into()));
        };
        let gate =
            gate.gather_mxfp4_tiles(&grouped.input, &grouped.indices, plan, &tiles, stream)?;
        let up = up.gather_mxfp4_tiles(&grouped.input, &grouped.indices, plan, &tiles, stream)?;
        self.routed_down.gather_mxfp4_tiles(
            &gate.silu_mul(&up, stream)?,
            &grouped.indices,
            plan,
            &tiles,
            stream,
        )
    }
}
