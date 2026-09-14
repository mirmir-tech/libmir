use std::io::Write;

use super::{Array, Plan, Result, SharedExpertMoe, Stream};

impl SharedExpertMoe {
    pub(super) fn describe_aligned_layout(
        &self,
        input: &Array,
        candidates: &[Plan<'_>],
        stream: &Stream,
    ) -> Result<()> {
        let routing = self.diagnostic_routing(input, stream)?;
        for plan in std::iter::once(&Plan::Grouped).chain(candidates) {
            let prepared =
                plan.prepare(input, &routing.indices, self.config.expert_count, stream)?;
            // Diagnostic host observation only, outside all measured intervals.
            let ids = prepared.indices.to_vec_u32(stream)?;
            let runs = ids
                .chunks(16)
                .map(|tile| 1 + tile.windows(2).filter(|pair| pair[0] != pair[1]).count())
                .sum::<usize>();
            let mixed = ids.chunks(16).filter(|tile| tile.windows(2).any(|p| p[0] != p[1])).count();
            writeln!(
                std::io::stderr().lock(),
                "moe.layout: {}",
                serde_json::json!({
                    "shape": input.shape()?, "plan": plan.name(), "rows": ids.len(),
                    "tiles_16": ids.len().div_ceil(16), "mixed_tiles_16": mixed,
                    "tile_expert_runs_16": runs,
                })
            )?;
        }
        Ok(())
    }
}
