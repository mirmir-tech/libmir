use std::io::Write;

use super::*;
use crate::config::RouterPrecision;

impl BoundLinear {
    pub(in crate::engine) fn probe_router_precision(
        &self,
        input: &Array,
        top_k: i32,
        label: &str,
        stream: &Stream,
    ) -> Result<()> {
        let shape = input.shape()?;
        let (rows, sequence, width) = (
            usize::try_from(shape[0])?,
            usize::try_from(shape[1])?,
            usize::try_from(shape[2])?,
        );
        for precision in [RouterPrecision::Model, RouterPrecision::Float32] {
            let flat = input.reshape(&[1, shape[0] * shape[1], shape[2]], stream)?;
            let packed = self.probe_route(&flat, top_k, precision, stream)?;
            let mut scalar_ids = Vec::new();
            let mut scalar_weights = Vec::new();
            for row in 0..rows {
                let input = input.slice(&[row, 0, 0], &[row + 1, sequence, width], stream)?;
                let routing = self.probe_route(&input, top_k, precision, stream)?;
                scalar_ids.extend(routing.indices.to_vec_u32(stream)?);
                scalar_weights.extend(routing.weights.to_vec_f32(stream)?);
            }
            let mut packed_ids = packed.indices.to_vec_u32(stream)?;
            let packed_weights = packed.weights.to_vec_f32(stream)?;
            assert_eq!(packed_ids.len(), scalar_ids.len());
            assert!(scalar_weights.iter().chain(&packed_weights).all(|v| v.is_finite()));
            assert_eq!(packed.weights.dtype()?, input.dtype()?);
            let k = usize::try_from(top_k)?;
            for ids in scalar_ids.chunks_exact_mut(k) {
                ids.sort_unstable();
            }
            for ids in packed_ids.chunks_exact_mut(k) {
                ids.sort_unstable();
            }
            let changed = scalar_ids
                .chunks_exact(k)
                .zip(packed_ids.chunks_exact(k))
                .filter(|(a, b)| a != b)
                .count();
            writeln!(
                std::io::stderr().lock(),
                "router.precision: {}",
                serde_json::json!({
                "case": label, "precision": format!("{precision:?}"),
                "tokens": rows*sequence, "changed_expert_sets": changed })
            )?;
        }
        Ok(())
    }

    fn probe_route(
        &self,
        input: &Array,
        top_k: i32,
        precision: RouterPrecision,
        stream: &Stream,
    ) -> Result<RouterOutput> {
        if precision == RouterPrecision::Float32 {
            self.route_precise(input, top_k, stream)
        } else {
            self.forward(input, stream)?.router_top_k_unit(top_k, stream)
        }
    }
}
