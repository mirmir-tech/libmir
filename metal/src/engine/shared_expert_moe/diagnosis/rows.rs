use std::io::Write;

use super::{Array, Result, SharedExpertMoe, Stream};

impl SharedExpertMoe {
    pub(crate) fn diagnose_row_batching(&self, input: &Array, stream: &Stream) -> Result<()> {
        let shape = input.shape()?;
        let [batch, sequence, hidden] = <[i32; 3]>::try_from(shape.as_slice()).map_err(|_| {
            crate::engine::Error::InvalidModel("expected row probe rank three".into())
        })?;
        let flat = input.reshape(&[1, batch * sequence, hidden], stream)?;
        let packed = self.row_components(&flat, stream)?;
        let mut rows = Vec::new();
        for row in 0..usize::try_from(batch)? {
            let input = input.slice(
                &[row, 0, 0],
                &[row + 1, usize::try_from(sequence)?, usize::try_from(hidden)?],
                stream,
            )?;
            rows.push(self.row_components(&input, stream)?);
        }
        let ids = rows
            .iter()
            .map(|row| row[1].to_vec_u32(stream))
            .collect::<Result<Vec<_>>>()?
            .concat();
        let packed_ids = packed[1].to_vec_u32(stream)?;
        let mut changed_sets = Vec::new();
        for (token, (a, b)) in ids
            .chunks_exact(self.config.top_k)
            .zip(packed_ids.chunks_exact(self.config.top_k))
            .enumerate()
        {
            let mut a = a.to_vec();
            let mut b = b.to_vec();
            a.sort_unstable();
            b.sort_unstable();
            if a != b {
                changed_sets.push(token);
            }
        }
        let scalar_scores = rows
            .iter()
            .map(|row| row[0].to_vec_f32(stream))
            .collect::<Result<Vec<_>>>()?
            .concat();
        let packed_scores = packed[0].to_vec_f32(stream)?;
        let mut boundaries = Vec::new();
        for &token in &changed_sets {
            let range = token * self.config.expert_count..(token + 1) * self.config.expert_count;
            let mut a = scalar_scores[range.clone()].to_vec();
            let mut b = packed_scores[range].to_vec();
            a.sort_by(|a, b| b.total_cmp(a));
            b.sort_by(|a, b| b.total_cmp(a));
            boundaries.push(serde_json::json!({"token": token,
                "scalar_gap": a[self.config.top_k-1] - a[self.config.top_k],
                "packed_gap": b[self.config.top_k-1] - b[self.config.top_k]}));
        }
        writeln!(
            std::io::stderr().lock(),
            "moe.routing_sets: {}",
            serde_json::json!({
            "changed_tokens": changed_sets, "total_tokens": batch * sequence, "boundaries": boundaries })
        )?;
        for (component, output) in packed.iter().enumerate() {
            let a = rows
                .iter()
                .map(|row| row[component].to_vec_f32(stream))
                .collect::<Result<Vec<_>>>()?
                .concat();
            let b = output.to_vec_f32(stream)?;
            assert_eq!(a.len(), b.len());
            assert!(a.iter().chain(&b).all(|value| value.is_finite()));
            let differing = a.iter().zip(&b).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let max_abs = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            writeln!(
                std::io::stderr().lock(),
                "moe.rows: {}",
                serde_json::json!({
                "component": (["scores", "indices", "weights", "routed", "shared"][component]),
                "differing": differing, "max_abs":max_abs,"elements": a.len()})
            )?;
        }
        let first = input.slice(
            &[0, 0, 0],
            &[1, usize::try_from(sequence)?, usize::try_from(hidden)?],
            stream,
        )?;
        let repeated = Array::concatenate(&vec![&first; usize::try_from(batch)?], 1, stream)?;
        let control = self.router.forward(&repeated, stream)?;
        let a = control
            .slice(&[0, 0, 0], &[1, usize::try_from(sequence)?, self.config.expert_count], stream)?
            .to_vec_f32(stream)?;
        let b = packed[0]
            .slice(&[0, 0, 0], &[1, usize::try_from(sequence)?, self.config.expert_count], stream)?
            .to_vec_f32(stream)?;
        assert_eq!(a, b, "same M router result must not depend on neighboring rows");
        writeln!(std::io::stderr().lock(), "moe.same_m_neighbor_control: exact")?;
        self.precise_router_control(input, stream)?;
        Ok(())
    }

    fn precise_router_control(&self, input: &Array, stream: &Stream) -> Result<()> {
        let input = Array::from_native(
            stream.native().graph().astype(input.native(), mirtal::DType::Float32)?,
        )?;
        let shape = input.shape()?;
        let rows = usize::try_from(shape[0])?;
        let sequence = usize::try_from(shape[1])?;
        let width = usize::try_from(shape[2])?;
        let flat = input.reshape(&[1, shape[0] * shape[1], shape[2]], stream)?;
        let scores = self.router.forward(&flat, stream)?;
        let packed = scores.router_top_k_unit(i32::try_from(self.config.top_k)?, stream)?;
        let mut scalar_scores = Vec::new();
        let mut scalar_ids = Vec::new();
        for row in 0..rows {
            let input = input.slice(&[row, 0, 0], &[row + 1, sequence, width], stream)?;
            let scores = self.router.forward(&input, stream)?;
            scalar_scores.extend(scores.to_vec_f32(stream)?);
            scalar_ids.extend(
                scores
                    .router_top_k_unit(i32::try_from(self.config.top_k)?, stream)?
                    .indices
                    .to_vec_u32(stream)?,
            );
        }
        let packed_scores = scores.to_vec_f32(stream)?;
        assert!(scalar_scores.iter().chain(&packed_scores).all(|value| value.is_finite()));
        let mut packed_ids = packed.indices.to_vec_u32(stream)?;
        for ids in scalar_ids.chunks_exact_mut(self.config.top_k) {
            ids.sort_unstable();
        }
        for ids in packed_ids.chunks_exact_mut(self.config.top_k) {
            ids.sort_unstable();
        }
        let changed = scalar_ids
            .chunks_exact(self.config.top_k)
            .zip(packed_ids.chunks_exact(self.config.top_k))
            .filter(|(a, b)| a != b)
            .count();
        let max_abs = scalar_scores
            .iter()
            .zip(packed_scores)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        writeln!(
            std::io::stderr().lock(),
            "moe.fp32_router_control: {}",
            serde_json::json!({"changed_expert_sets": changed, "max_abs": max_abs})
        )?;
        Ok(())
    }

    fn row_components(&self, input: &Array, stream: &Stream) -> Result<[Array; 5]> {
        let scores = self.router.forward(input, stream)?;
        let routing = scores.router_top_k_unit(i32::try_from(self.config.top_k)?, stream)?;
        let routed = self.routed(input, &routing.indices, &routing.weights, stream)?;
        let shared = self.shared(input, stream)?;
        Ok([scores, routing.indices, routing.weights, routed, shared])
    }
}

impl SharedExpertMoe {
    pub(crate) fn probe_router_precision(
        &self,
        input: &Array,
        label: &str,
        stream: &Stream,
    ) -> Result<()> {
        self.router
            .probe_router_precision(input, i32::try_from(self.config.top_k)?, label, stream)
    }
}
