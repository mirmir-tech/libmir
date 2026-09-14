use std::{collections::BTreeSet, io::Write, num::NonZeroUsize};

use super::reference;
use crate::engine::{Array, Dtype, Result, Stream, probe::Projection};

impl Projection {
    #[expect(clippy::float_cmp, reason = "compare discrete BF16 rounding outcomes")]
    pub fn report(&self, step: usize, stream: &Stream) -> Result<()> {
        let shape = self.input.shape()?;
        let [batch, sequence, width] = <[i32; 3]>::try_from(shape.as_slice()).map_err(|_| {
            crate::engine::Error::InvalidModel("width probe needs rank three".into())
        })?;
        assert_eq!(sequence, 1);
        let k = usize::try_from(width)?;
        let n = usize::try_from(self.weight.shape()?[0])?;
        let weights = mirtal::MxFp4 {
            weight: self.weight.native(),
            scales: self.scales.native(),
        };
        let graph = stream.native().graph();
        let packed = Array::from_native(graph.mxfp4_matmul(self.input.native(), weights, true)?)?
            .to_vec_f32(stream)?;
        let plan = mirtal::MxFp4Matmul::new()?;
        let fixed = Array::from_native(plan.matmul(
            self.input.native(),
            weights,
            NonZeroUsize::new(4).ok_or(crate::engine::Error::ShapeOverflow)?,
            stream.native(),
        )?)?
        .to_vec_f32(stream)?;
        let mut scalar = Vec::new();
        let mut fixed_scalar = Vec::new();
        for row in 0..usize::try_from(batch)? {
            let input = self.input.slice(&[row, 0, 0], &[row + 1, 1, k], stream)?;
            scalar.extend(
                Array::from_native(graph.mxfp4_matmul(input.native(), weights, true)?)?
                    .to_vec_f32(stream)?,
            );
            fixed_scalar.extend(
                Array::from_native(plan.matmul(
                    input.native(),
                    weights,
                    NonZeroUsize::new(4).ok_or(crate::engine::Error::ShapeOverflow)?,
                    stream.native(),
                )?)?
                .to_vec_f32(stream)?,
            );
        }
        assert_eq!(fixed_scalar, fixed, "FP32 projection depends on row count");
        assert!(scalar.iter().chain(&packed).chain(&fixed).all(|v| v.is_finite()));
        let changed = scalar
            .iter()
            .zip(&packed)
            .enumerate()
            .filter_map(|(index, (a, b))| (a.to_bits() != b.to_bits()).then_some(index))
            .collect::<BTreeSet<_>>();
        let mut selected = changed.clone();
        for row in 0..usize::try_from(batch)? {
            selected.extend(reference::columns(n).iter().map(|column| row * n + column));
        }
        let input = self.input.to_vec_f32(stream)?;
        let weight = self.weight.to_vec_u32(stream)?;
        let scale = self.scales.astype(Dtype::Uint32, stream)?.to_vec_u32(stream)?;
        assert!(scale.iter().all(|&v| v < 255));
        let mut not_nearest = [0; 3];
        let mut differing = Vec::new();
        for &index in &selected {
            let row = index / n;
            let column = index % n;
            let sum = (0..k)
                .map(|i| {
                    f64::from(input[row * k + i])
                        * reference::decode(
                            weight[column * (k / 8) + i / 8],
                            i % 8,
                            scale[column * (k / 32) + i / 32],
                        )
                })
                .sum::<f64>();
            let nearest = reference::nearest_bf16(sum);
            for (count, value) in
                not_nearest.iter_mut().zip([scalar[index], packed[index], fixed[index]])
            {
                *count += usize::from(f64::from(value) != nearest);
            }
            if changed.contains(&index) {
                differing.push(serde_json::json!({
                    "row":row,"column":column,"scalar":scalar[index],"packed":packed[index],
                    "fp32":fixed[index],"reference":sum,"nearest_bf16":nearest,
                }));
            }
        }
        writeln!(
            std::io::stderr().lock(),
            "width.projection: {}",
            serde_json::json!({
                "step":step,"layer":self.layer,"kind":self.kind,"batch":batch,"k":k,"n":n,
                "changed":differing,"samples":selected.len(),"not_nearest_bf16":{
                    "scalar":not_nearest[0],"packed":not_nearest[1],"fp32":not_nearest[2]
                },"fp32_row_count_exact":true,
            })
        )?;
        Ok(())
    }
}
