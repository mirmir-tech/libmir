mod measurement;
use super::{
    super::{RoutedGateUp, numerics::Difference},
    Array, Case, Result, RouterOutput, SharedExpertMoe, Stream,
};
use crate::engine::{
    Error,
    binding::BoundLinear,
    kernels::expert_group::tiles::{TilePlan, WorkTiles},
};

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Stage {
    Group,
    Worklist,
    Gate,
    Up,
    Down,
}

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Variant {
    Mlx,
    Tiles,
}

impl SharedExpertMoe {
    pub(super) fn profile_tiles(
        &self,
        case: Case,
        input: &Array,
        routing: &RouterOutput,
        plan: &TilePlan,
        stream: &Stream,
    ) -> Result<()> {
        let RoutedGateUp::Separate { gate, up, .. } = &self.routed_gate_up else {
            return Err(Error::InvalidModel("tile profile requires separate banks".into()));
        };
        measurement::probe(case, Stage::Group, &[Variant::Mlx], stream, |_| {
            let grouped =
                input.group_expert_inputs(&routing.indices, self.config.expert_count, stream)?;
            Ok(vec![grouped.input, grouped.indices])
        })?;
        let grouped =
            input.group_expert_inputs(&routing.indices, self.config.expert_count, stream)?;
        stream.eval_many(&[&grouped.input, &grouped.indices])?;
        stream.synchronize()?;
        measurement::probe(case, Stage::Worklist, &[Variant::Tiles], stream, |_| {
            let tiles = plan.prepare(&grouped.indices, self.config.expert_count, stream)?;
            Ok(vec![Array::from_native(tiles.descriptors().clone())?])
        })?;
        let tiles = plan.prepare(&grouped.indices, self.config.expert_count, stream)?;
        stream.native().eval(tiles.descriptors())?;
        stream.synchronize()?;
        let projections = Projections {
            plan,
            tiles: &tiles,
            indices: &grouped.indices,
            stream,
        };
        projections.probe(case, Stage::Gate, gate, &grouped.input)?;
        projections.probe(case, Stage::Up, up, &grouped.input)?;
        let gate = gate.gather(&grouped.input, &grouped.indices, true, stream)?;
        let up = up.gather(&grouped.input, &grouped.indices, true, stream)?;
        let activated = gate.silu_mul(&up, stream)?;
        stream.eval_many(&[&activated])?;
        stream.synchronize()?;
        projections.probe(case, Stage::Down, &self.routed_down, &activated)
    }
}

struct Projections<'a> {
    plan: &'a TilePlan,
    tiles: &'a WorkTiles,
    indices: &'a Array,
    stream: &'a Stream,
}

impl Projections<'_> {
    fn probe(&self, case: Case, stage: Stage, bank: &BoundLinear, input: &Array) -> Result<()> {
        use std::io::Write;
        let forward = |variant| match variant {
            Variant::Mlx => bank.gather(input, self.indices, true, self.stream),
            Variant::Tiles => {
                bank.gather_mxfp4_tiles(input, self.indices, self.plan, self.tiles, self.stream)
            },
        };
        let reference = forward(Variant::Mlx)?.to_vec_f32(self.stream)?;
        let values = forward(Variant::Tiles)?.to_vec_f32(self.stream)?;
        let difference = Difference::between(&values, &reference);
        writeln!(
            std::io::stderr().lock(),
            "moe.tile_component_parity: {}",
            serde_json::json!({
                "case": case, "stage": stage, "difference": difference, "elements": values.len(),
            })
        )?;
        if difference.differing != 0 {
            return Ok(());
        }
        measurement::probe(case, stage, &[Variant::Mlx, Variant::Tiles], self.stream, |variant| {
            Ok(vec![forward(variant)?])
        })
    }
}
