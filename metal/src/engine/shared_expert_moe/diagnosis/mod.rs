mod aligned;
mod budgets;
mod decode;
mod measurement;
mod numerics;
mod precision;
mod profile;
mod rows;

use super::{Array, Result, RoutedGateUp, SharedExpertMoe, Stream};
use crate::engine::{
    RouterOutput, SortedExpertInputs, kernels::expert_group::aligned::AlignedGroup,
};

#[derive(Clone, Copy, Debug)]
enum Plan<'a> {
    Grouped,
    SortedFused,
    SortedGraph,
    GroupedProjection(&'a RoutedGateUp),
    GroupedIndexed,
    GroupedTiles(&'a crate::engine::kernels::expert_group::tiles::TilePlan),
    GroupedAligned(&'a AlignedGroup),
}

impl SharedExpertMoe {
    pub(crate) fn diagnose_prefill(&self, input: &Array, stream: &Stream) -> Result<()> {
        assert_eq!(input.shape()?, [5, 512, 2048]);
        let aligned = (stream.config().diagnostics.moe_prefill
            == crate::config::MoePrefill::CompareAligned)
            .then(AlignedGroup::new)
            .transpose()?;
        let fusion = if stream.config().diagnostics.moe_prefill
            == crate::config::MoePrefill::CompareProjection
        {
            let RoutedGateUp::Separate { gate, up, .. } = &self.routed_gate_up else {
                return Err(crate::engine::Error::InvalidModel(
                    "expected separate MoE banks".into(),
                ));
            };
            let (projection, width) =
                gate.fuse_mxfp4_expert_projection(up, stream)?.ok_or_else(|| {
                    crate::engine::Error::InvalidModel("MoE fusion not available".into())
                })?;
            Some(RoutedGateUp::Fused { projection, width, interleaved: false })
        } else {
            None
        };
        for sequence in [128, 512] {
            let input = input.slice(&[0, 0, 0], &[5, sequence, 2048], stream)?;
            stream.eval_many(&[&input])?;
            stream.synchronize()?;
            let candidates = aligned.as_ref().map_or_else(
                || {
                    if matches!(
                        stream.config().diagnostics.moe_prefill,
                        crate::config::MoePrefill::CompareIndexed
                            | crate::config::MoePrefill::MeasureIndexedNumerics
                    ) {
                        vec![Plan::GroupedIndexed]
                    } else {
                        fusion.as_ref().map_or_else(
                            || vec![Plan::SortedFused, Plan::SortedGraph],
                            |fused| vec![Plan::GroupedProjection(fused)],
                        )
                    }
                },
                |plan| vec![Plan::GroupedAligned(plan)],
            );
            if aligned.is_some() {
                self.describe_aligned_layout(&input, &candidates, stream)?;
            }
            self.compare_plans(&input, &candidates, stream)?;
            if stream.config().diagnostics.moe_prefill == crate::config::MoePrefill::CompareRoutes {
                self.profile_prefill(&input, stream)?;
            }
        }
        Ok(())
    }

    fn diagnostic_routing(&self, input: &Array, stream: &Stream) -> Result<RouterOutput> {
        self.router.route_unit(input, i32::try_from(self.config.top_k)?, stream)
    }

    fn diagnostic_forward(&self, input: &Array, plan: Plan<'_>, stream: &Stream) -> Result<Array> {
        let routing = self.diagnostic_routing(input, stream)?;
        self.replay_routed(input, &routing, plan, stream)
    }

    fn replay_routed(
        &self,
        input: &Array,
        routing: &RouterOutput,
        plan: Plan<'_>,
        stream: &Stream,
    ) -> Result<Array> {
        let grouped = plan.prepare(input, &routing.indices, self.config.expert_count, stream)?;
        let output = if let Plan::GroupedProjection(fused) = plan {
            let (gate, up) = fused.gather(&grouped.input, &grouped.indices, true, stream)?;
            self.routed_down
                .gather(&gate.silu_mul(&up, stream)?, &grouped.indices, true, stream)?
        } else if let Plan::GroupedTiles(plan) = plan {
            self.tiled_mlp(&grouped, plan, stream)?
        } else if matches!(plan, Plan::GroupedIndexed) {
            let RoutedGateUp::Separate { gate, up, .. } = &self.routed_gate_up else {
                return Err(crate::engine::Error::InvalidModel(
                    "indexed probe requires separate banks".into(),
                ));
            };
            let gate = gate
                .gather_mxfp4_indexed(&grouped.source, &grouped.rows, &grouped.indices, stream)?;
            let up =
                up.gather_mxfp4_indexed(&grouped.source, &grouped.rows, &grouped.indices, stream)?;
            self.routed_down
                .gather(&gate.silu_mul(&up, stream)?, &grouped.indices, true, stream)?
        } else {
            self.routed_mlp(&grouped.input, &grouped.indices, true, false, stream)?
        };
        let output = plan.restore(&grouped, &output, &routing.weights, stream)?;
        output.add(&self.shared(input, stream)?, stream)
    }
}

impl Plan<'_> {
    const fn name(self) -> &'static str {
        match self {
            Self::Grouped => "grouped_fused",
            Self::SortedFused => "sorted_fused",
            Self::SortedGraph => "sorted_graph",
            Self::GroupedProjection(_) => "grouped_projection",
            Self::GroupedIndexed => "grouped_indexed",
            Self::GroupedTiles(plan) => match plan.columns() {
                crate::engine::kernels::expert_group::tiles::ColumnTile::Width32 => "grouped_tiles",
                crate::engine::kernels::expert_group::tiles::ColumnTile::Width64 => {
                    "grouped_tiles64"
                },
            },
            Self::GroupedAligned(plan) => match plan.budget() {
                crate::engine::kernels::expert_group::aligned::PaddingBudget::Four => {
                    "grouped_aligned4"
                },
                crate::engine::kernels::expert_group::aligned::PaddingBudget::Eight => {
                    "grouped_aligned"
                },
            },
        }
    }

    fn prepare(
        self,
        input: &Array,
        indices: &Array,
        experts: usize,
        stream: &Stream,
    ) -> Result<SortedExpertInputs> {
        match self {
            Self::GroupedAligned(plan) => input.align_expert_inputs(indices, experts, plan, stream),
            Self::Grouped
            | Self::GroupedProjection(_)
            | Self::GroupedIndexed
            | Self::GroupedTiles(_) => input.group_expert_inputs(indices, experts, stream),
            Self::SortedFused | Self::SortedGraph => input.sort_expert_inputs(indices, stream),
        }
    }

    fn restore(
        self,
        grouped: &SortedExpertInputs,
        output: &Array,
        weights: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        match self {
            Self::SortedGraph => grouped.restore(output, stream)?.weighted_sum(weights, -2, stream),
            Self::Grouped
            | Self::SortedFused
            | Self::GroupedProjection(_)
            | Self::GroupedAligned(_)
            | Self::GroupedTiles(_)
            | Self::GroupedIndexed => grouped.restore_weighted(output, weights, stream),
        }
    }
}
