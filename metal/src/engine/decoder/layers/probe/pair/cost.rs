use std::{io::Write, time::Instant};

use super::{
    Plan::{Joined, Separate},
    *,
};

impl ProjectionPair {
    pub fn verify(&self, stream: &Stream) -> Result<()> {
        let (first, second) = self.forward(Separate, stream)?;
        let (key, value) = self.forward(Joined, stream)?;
        let mut errors = Vec::new();
        for (reference, candidate) in [(&first, &key), (&second, &value)] {
            assert_eq!(candidate.dtype()?, reference.dtype()?);
            assert_eq!(candidate.shape()?, reference.shape()?);
            let actual = candidate.to_vec_f32(stream)?;
            let expected = reference.to_vec_f32(stream)?;
            let differing =
                actual.iter().zip(&expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let max_abs =
                actual.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            errors.push((differing, max_abs));
        }
        writeln!(
            std::io::stderr().lock(),
            "kv.parity: {}",
            serde_json::json!({
                "layer": self.layer, "input": self.input.shape()?,
                "key": key.shape()?, "value": value.shape()?,
                "bitwise_equal": errors.iter().all(|(count, _)| *count == 0),
                "key_difference": errors[0], "value_difference": errors[1],
                "additional_projection_bytes": projection_bytes(&self.joined)?,
            })
        )?;
        assert!(
            errors.iter().all(|(count, _)| *count == 0),
            "K/V output mismatch at layer {}",
            self.layer
        );
        Ok(())
    }

    pub fn qualify(&self, stream: &Stream) -> Result<()> {
        Self::qualify_interleaved(std::slice::from_ref(self), stream)
    }

    pub fn qualify_interleaved(pairs: &[Self], stream: &Stream) -> Result<()> {
        assert!(!pairs.is_empty());
        let layers = pairs.iter().map(|pair| pair.layer).collect::<Vec<_>>();
        let layer = (pairs.len() == 1).then_some(pairs[0].layer);
        for pair in pairs {
            pair.verify(stream)?;
        }
        let mut samples = [Vec::<f64>::new(), Vec::<f64>::new()];
        for (run, plan) in [
            Separate, Joined, Joined, Separate, Separate, Joined, Joined, Separate, Joined,
            Separate, Separate, Joined,
        ]
        .into_iter()
        .enumerate()
        {
            let started = Instant::now();
            let mut roots = Vec::with_capacity(2048);
            for call in 0..1024 {
                roots.extend(<[Array; 2]>::from(pairs[call % pairs.len()].forward(plan, stream)?));
            }
            let build_ms = started.elapsed().as_secs_f64() * 1000.0;
            stream.eval_many(&roots.iter().collect::<Vec<_>>())?;
            stream.synchronize()?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let memory = crate::engine::memory_stats()?;
            drop(roots);
            writeln!(
                std::io::stderr().lock(),
                "kv.cost: {}",
                serde_json::json!({
                    "layer": layer, "layers": layers, "run": run, "plan": plan, "measured": run >= 4,
                    "calls": 1024, "build_ms": build_ms, "elapsed_ms": elapsed_ms,
                    "active_bytes": memory.active, "cached_bytes": memory.cached,
                })
            )?;
            if run >= 4 {
                let values = &mut samples[match plan {
                    Separate => 0,
                    Joined => 1,
                }];
                values.push(elapsed_ms);
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(0.0, f64::max);
                if max > min * 1.15 {
                    return Err(Error::BenchmarkStability {
                        case: format!("KV/layers{layers:?}"),
                        variant: format!("{plan:?}"),
                        spread_percent: (max / min - 1.0) * 100.0,
                    });
                }
            }
        }
        for pair in pairs {
            pair.verify(stream)?;
        }
        let blocks = [0, 2].map(|start| {
            1.0 - samples[1][start..start + 2].iter().sum::<f64>()
                / samples[0][start..start + 2].iter().sum::<f64>()
        });
        let means = samples.map(|values| values.iter().sum::<f64>() / 4.0);
        writeln!(
            std::io::stderr().lock(),
            "kv.gate: {}",
            serde_json::json!({
                "layer": layer, "layers": layers, "separate_ms": means[0], "joined_ms": means[1],
                "reduction": 1.0 - means[1] / means[0], "block_reductions": blocks,
                "passes": means[1] <= means[0] * 0.97 && blocks.iter().all(|&gain| gain > 0.0),
            })
        )?;
        Ok(())
    }
}

fn projection_bytes(projection: &BoundLinear) -> Result<usize> {
    if let BoundLinear::Dense(projection) = projection {
        return Ok(projection.joined_probe_layout()?.1);
    }
    let BoundLinear::MxFp4(projection) = projection else {
        return Err(Error::InvalidQuantization("projection bytes require MXFP4".into()));
    };
    let mut bytes = projection
        .weight
        .byte_len()?
        .checked_add(projection.scales.byte_len()?)
        .ok_or(Error::ShapeOverflow)?;
    if projection.has_bias {
        bytes = bytes.checked_add(projection.bias.byte_len()?).ok_or(Error::ShapeOverflow)?;
    }
    Ok(bytes)
}
