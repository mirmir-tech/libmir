mod measurement;
mod profile;

use std::io::Write;

use super::{AlignedGroup, Array, Plan, Result, RouterOutput, SharedExpertMoe, Stream};
use crate::engine::kernels::expert_group::aligned::PaddingBudget;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Pattern {
    Actual,
    HotSet,
}

#[derive(Clone, Copy, serde::Serialize)]
struct Case {
    layer: usize,
    sequence: i32,
    pattern: Pattern,
}

impl SharedExpertMoe {
    pub(crate) fn compare_alignment_budgets(
        &self,
        input: &Array,
        layer: usize,
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(input.shape()?, [5, 512, 2048]);
        let four = AlignedGroup::with_budget(PaddingBudget::Four)?;
        let eight = AlignedGroup::with_budget(PaddingBudget::Eight)?;
        let tile_plan = crate::engine::kernels::expert_group::tiles::TilePlan::new()?;
        let wide_plan = crate::engine::kernels::expert_group::tiles::TilePlan::with_columns(
            crate::engine::kernels::expert_group::tiles::ColumnTile::Width64,
        )?;
        let plans = if stream.config().diagnostics.moe_prefill
            == crate::config::MoePrefill::CompareTileWidths
        {
            vec![Plan::Grouped, Plan::GroupedTiles(&tile_plan), Plan::GroupedTiles(&wide_plan)]
        } else if matches!(
            stream.config().diagnostics.moe_prefill,
            crate::config::MoePrefill::CompareTiles | crate::config::MoePrefill::ProfileTiles
        ) {
            vec![Plan::Grouped, Plan::GroupedTiles(&tile_plan)]
        } else {
            vec![Plan::Grouped, Plan::GroupedAligned(&four), Plan::GroupedAligned(&eight)]
        };
        for sequence in [128, 512] {
            let input = input.slice(&[0, 0, 0], &[5, sequence, 2048], stream)?;
            let routing = self.diagnostic_routing(&input, stream)?;
            stream.eval_many(&[&input, &routing.indices, &routing.weights])?;
            stream.synchronize()?;
            for pattern in [Pattern::Actual, Pattern::HotSet] {
                let case = Case {
                    layer,
                    sequence: i32::try_from(sequence)?,
                    pattern,
                };
                let hot;
                let selected = match pattern {
                    Pattern::Actual => &routing,
                    Pattern::HotSet => {
                        let top_k = self.config.top_k;
                        let ids = (0..5 * sequence)
                            .flat_map(|token| {
                                (0..top_k).map(move |choice| (token + choice) % top_k)
                            })
                            .map(|id| Ok(u32::try_from(id)?))
                            .collect::<Result<Vec<_>>>()?;
                        hot = RouterOutput {
                            indices: Array::from_u32(
                                &ids,
                                &[5, i32::try_from(sequence)?, i32::try_from(top_k)?],
                            )?,
                            weights: Array::from_native(routing.weights.native().clone())?,
                        };
                        &hot
                    },
                };
                stream.eval_many(&[&selected.indices, &selected.weights])?;
                stream.synchronize()?;
                if stream.config().diagnostics.moe_prefill
                    == crate::config::MoePrefill::ProfileTiles
                {
                    self.profile_tiles(case, &input, selected, &tile_plan, stream)?;
                } else {
                    self.budget_layout(case, &input, selected, &plans, stream)?;
                    self.measure_budgets(case, &input, selected, &plans, stream)?;
                }
            }
        }
        Ok(())
    }

    fn budget_layout(
        &self,
        case: Case,
        input: &Array,
        routing: &RouterOutput,
        plans: &[Plan<'_>],
        stream: &Stream,
    ) -> Result<()> {
        let actual = routing.indices.to_vec_u32(stream)?;
        let active_experts = actual.iter().copied().collect::<std::collections::HashSet<_>>().len();
        for &plan in plans {
            let prepared =
                plan.prepare(input, &routing.indices, self.config.expert_count, stream)?;
            stream.eval_many(&[&prepared.input, &prepared.indices])?;
            stream.synchronize()?;
            let ids = prepared.indices.to_vec_u32(stream)?;
            let runs = if let Plan::GroupedTiles(_) = plan {
                let mut counts = vec![0_usize; self.config.expert_count];
                for &id in &ids {
                    counts[usize::try_from(id)?] += 1;
                }
                counts.into_iter().map(|count| count.div_ceil(16)).sum()
            } else {
                ids.chunks(16)
                    .map(|tile| 1 + tile.windows(2).filter(|p| p[0] != p[1]).count())
                    .sum::<usize>()
            };
            let input_allocation_bytes = prepared.input.native().allocation()?.map(|a| a.bytes());
            let memory = crate::engine::memory_stats()?;
            writeln!(
                std::io::stderr().lock(),
                "moe.budget_layout: {}",
                serde_json::json!({
                    "case": case, "plan": plan.name(), "rows": ids.len(), "active_experts": active_experts,
                    "tile_expert_runs_16": runs, "input_allocation_bytes": input_allocation_bytes,
                    "allocator_active_bytes": memory.active, "allocator_cached_bytes": memory.cached,
                })
            )?;
        }
        Ok(())
    }
}
