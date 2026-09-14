mod candidate;
mod cost;
mod reference;
mod split;
mod width;

use std::io::Write;

use super::*;
use crate::engine::binding::BoundLinear;

impl BoundLinear {
    #[expect(clippy::float_cmp, reason = "count exact BF16 rounding outcomes")]
    pub(crate) fn probe_mxfp4_accuracy(
        &self,
        input: &Array,
        label: &str,
        stream: &Stream,
    ) -> Result<()> {
        let Self::MxFp4(projection) = self else {
            return Err(Error::InvalidQuantization("accuracy probe requires MXFP4".into()));
        };
        assert!(!projection.has_bias, "probe compares the matrix product before output bias");
        let shape = input.shape()?;
        let [batch, sequence, width] = <[i32; 3]>::try_from(shape.as_slice())
            .map_err(|_| Error::InvalidModel("accuracy probe needs rank three input".into()))?;
        let flat = input.reshape(&[1, batch * sequence, width], stream)?;
        let reference = reference::dot_samples(projection, &flat, stream)?;
        let mut scalar = Vec::new();
        let mut fixed_scalar = Vec::new();
        for row in 0..usize::try_from(batch)? {
            let input = input.slice(
                &[row, 0, 0],
                &[row + 1, usize::try_from(sequence)?, usize::try_from(width)?],
                stream,
            )?;
            scalar.extend(self.forward(&input, stream)?.to_vec_f32(stream)?);
            fixed_scalar.extend(projection.fixed_reduction(&input, stream)?.to_vec_f32(stream)?);
        }
        let packed = self.forward(&flat, stream)?.to_vec_f32(stream)?;
        let fixed = projection.fixed_reduction(&flat, stream)?.to_vec_f32(stream)?;
        assert_eq!(fixed_scalar, fixed, "fixed reduction must not depend on M");
        for (plan, values) in
            [("mlx.scalar", &scalar), ("mlx.packed", &packed), ("fixed.fp32", &fixed)]
        {
            let values = values
                .chunks_exact(projection.output_features)
                .flat_map(|row| {
                    reference::columns(projection.output_features)
                        .into_iter()
                        .map(|column| f64::from(row[column]))
                })
                .collect::<Vec<_>>();
            assert_eq!(values.len(), reference.len());
            assert!(values.iter().chain(&reference).all(|v| v.is_finite()));
            let error = values.iter().zip(&reference).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
            let norm = reference.iter().map(|v| v * v).sum::<f64>();
            let best =
                reference.iter().map(|&v| (reference::nearest_bf16(v) - v).powi(2)).sum::<f64>();
            writeln!(
                std::io::stderr().lock(),
                "mxfp4.accuracy: {}",
                serde_json::json!({
                    "label":label,"sequence":sequence,"batch":batch,"k":projection.input_features,"n":projection.output_features,
                    "plan":plan,"samples":values.len(),"relative_l2":(error/norm).sqrt(),
                    "max_abs":values.iter().zip(&reference).map(|(a,b)| (a-b).abs()).fold(0.0,f64::max),
                    "squared_error_over_bf16_min":error/best.max(f64::MIN_POSITIVE),
                    "not_nearest_bf16":values.iter().zip(&reference).filter(|(a,b)| **a!=reference::nearest_bf16(**b)).count()
                })
            )?;
        }
        split::compare(projection, &flat, label, sequence, batch, [&scalar, &packed], stream)?;
        candidate::compare(projection, input, label, stream)?;
        Ok(())
    }
}

impl MxFp4Linear {
    fn fixed_reduction(&self, input: &Array, stream: &Stream) -> Result<Array> {
        stream.kernels().mxfp4_linear(
            [input, &self.weight, &self.scales, &self.bias],
            self.input_features,
            self.output_features,
            stream,
        )
    }
}
