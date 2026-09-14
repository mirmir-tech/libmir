use super::{Array, Dtype, MxFp4Linear, MxFp4LinearLayout, Result, Stream};
use crate::engine::binding::BoundLinear;

impl BoundLinear {
    pub(crate) fn gather_mxfp4_tiles(
        &self,
        input: &Array,
        rhs: &Array,
        plan: &crate::engine::kernels::expert_group::tiles::TilePlan,
        tiles: &crate::engine::kernels::expert_group::tiles::WorkTiles,
        stream: &Stream,
    ) -> Result<Array> {
        let Self::MxFp4(projection) = self else {
            return Err(super::Error::InvalidQuantization("tile probe requires MXFP4".into()));
        };
        if !matches!(projection.layout, MxFp4LinearLayout::Gathered) {
            return Err(super::Error::InvalidQuantization("tile probe requires a bank".into()));
        }
        let output = plan.project(input, &projection.weight, &projection.scales, tiles, stream)?;
        if projection.has_bias {
            let graph = stream.native().graph();
            let bias = graph.take(projection.bias.native(), rhs.native(), 0)?;
            return Array::from_native(
                graph.add(output.native(), &graph.expand_dims(&bias, &[-2])?)?,
            );
        }
        Ok(output)
    }

    pub(crate) fn gather_mxfp4_indexed(
        &self,
        input: &Array,
        lhs: &Array,
        rhs: &Array,
        stream: &Stream,
    ) -> Result<Array> {
        let Self::MxFp4(projection) = self else {
            return Err(super::Error::InvalidQuantization("indexed probe requires MXFP4".into()));
        };
        if !matches!(projection.layout, MxFp4LinearLayout::Gathered) {
            return Err(super::Error::InvalidQuantization("indexed probe requires a bank".into()));
        }
        let graph = stream.native().graph();
        let output = graph.gather_mxfp4_with_indices(
            input.native(),
            mirtal::MxFp4 {
                weight: projection.weight.native(),
                scales: projection.scales.native(),
            },
            mirtal::GatherIndices {
                lhs: Some(lhs.native()),
                rhs: Some(rhs.native()),
            },
            mirtal::GatherQmmOptions { transpose: true, sorted_indices: true },
        )?;
        let output = if projection.has_bias {
            let bias = graph.take(projection.bias.native(), rhs.native(), 0)?;
            graph.add(&output, &graph.expand_dims(&bias, &[-2])?)?
        } else {
            output
        };
        Array::from_native(output)
    }

    /// Test-only bank fusion. Concatenate output rows on device and retain
    /// gathered output biases, including a bias present on just one half.
    pub(crate) fn fuse_mxfp4_expert_projection(
        &self,
        up: &Self,
        stream: &Stream,
    ) -> Result<Option<(Self, usize)>> {
        let (Self::MxFp4(gate), Self::MxFp4(up)) = (self, up) else {
            return Ok(None);
        };
        if !matches!(gate.layout, MxFp4LinearLayout::Gathered)
            || !matches!(up.layout, MxFp4LinearLayout::Gathered)
            || gate.input_features != up.input_features
            || gate.weight.shape()?[0] != up.weight.shape()?[0]
            || gate.weight.dtype()? != Dtype::Uint32
            || up.weight.dtype()? != Dtype::Uint32
        {
            return Ok(None);
        }
        let output_features = gate
            .output_features
            .checked_add(up.output_features)
            .ok_or(super::Error::ShapeOverflow)?;
        let projection = MxFp4Linear {
            weight: Array::concatenate(&[&gate.weight, &up.weight], 1, stream)?,
            scales: Array::concatenate(&[&gate.scales, &up.scales], 1, stream)?,
            bias: Array::concatenate(&[&gate.bias, &up.bias], 1, stream)?,
            input_features: gate.input_features,
            output_features,
            has_bias: gate.has_bias || up.has_bias,
            layout: MxFp4LinearLayout::Gathered,
        };
        stream.eval_many(&[&projection.weight, &projection.scales, &projection.bias])?;
        stream.synchronize()?;
        Ok(Some((Self::MxFp4(projection), gate.output_features)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fused_mxfp4_banks_preserve_gathered_bias_and_output_halves() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let gate = bank(0x2222_2222, true, &stream)?;
        let up = bank(0x4444_4444, false, &stream)?;
        let (fused, width) = gate
            .fuse_mxfp4_expert_projection(&up, &stream)?
            .ok_or(super::super::Error::ShapeOverflow)?;
        assert_eq!(width, 64);
        let input =
            Array::from_f32(&vec![0.5; 4 * 32], &[4, 1, 32])?.astype(Dtype::Bfloat16, &stream)?;
        for (indices, sorted) in [([0, 0, 1, 1], true), ([1, 0, 1, 0], false)] {
            let indices = Array::from_u32(&indices, &[4])?;
            let output = fused.gather(&input, &indices, sorted, &stream)?;
            let (actual_gate, actual_up) =
                crate::engine::fused_gate_up::split_last(&output, width, &stream)?;
            let expected_gate = gate.gather(&input, &indices, sorted, &stream)?;
            let expected_up = up.gather(&input, &indices, sorted, &stream)?;
            assert_eq!(actual_gate.to_vec_f32(&stream)?, expected_gate.to_vec_f32(&stream)?);
            assert_eq!(actual_up.to_vec_f32(&stream)?, expected_up.to_vec_f32(&stream)?);
            assert!(actual_gate.to_vec_f32(&stream)?.iter().all(|&v| v > 0.0));
        }
        Ok(())
    }

    #[test]
    fn tile_projection_preserves_gathered_output_bias() -> Result<()> {
        let stream = Stream::new_gpu()?;
        let plan = crate::engine::kernels::expert_group::tiles::TilePlan::new()?;
        let input =
            Array::from_f32(&vec![0.5; 33 * 32], &[33, 1, 32])?.astype(Dtype::Bfloat16, &stream)?;
        let ids = [vec![0; 17], vec![1; 16]].concat();
        let indices = Array::from_u32(&ids, &[33])?;
        let tiles = plan.prepare(&indices, 2, &stream)?;
        for has_bias in [false, true] {
            let bank = bank(0x2345_abcd, has_bias, &stream)?;
            let expected = bank.gather(&input, &indices, true, &stream)?;
            let actual = bank.gather_mxfp4_tiles(&input, &indices, &plan, &tiles, &stream)?;
            assert_eq!(actual.to_vec_f32(&stream)?, expected.to_vec_f32(&stream)?);
        }
        Ok(())
    }

    fn bank(packed: u32, has_bias: bool, stream: &Stream) -> Result<BoundLinear> {
        Ok(BoundLinear::MxFp4(MxFp4Linear {
            weight: Array::from_u32(&vec![packed; 2 * 64 * 4], &[2, 64, 4])?,
            scales: Array::from_u32(&[127; 128], &[2, 64, 1])?.astype(Dtype::Uint8, stream)?,
            bias: Array::from_f32(
                &vec![
                    if has_bias {
                        1.0
                    } else {
                        0.0
                    };
                    128
                ],
                &[2, 64],
            )?
            .astype(Dtype::Bfloat16, stream)?,
            input_features: 32,
            output_features: 64,
            has_bias,
            layout: MxFp4LinearLayout::Gathered,
        }))
    }
}
