use std::{num::NonZeroUsize, time::Instant};

use mirtal::MxFp4RowTile;

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Native,
    Split8,
    Split4,
}

impl BoundLinear {
    pub(crate) fn probe_mxfp4_cost(
        &self,
        input: &Array,
        label: &str,
        stream: &Stream,
    ) -> Result<()> {
        self.probe_projection_cost(input, label, stream, Variant::Split8)
    }

    pub(crate) fn probe_mxfp4_native_cost(
        &self,
        input: &Array,
        label: &str,
        stream: &Stream,
    ) -> Result<()> {
        self.probe_projection_cost(input, label, stream, Variant::Native)
    }

    fn probe_projection_cost(
        &self,
        input: &Array,
        label: &str,
        stream: &Stream,
        baseline: Variant,
    ) -> Result<()> {
        let Self::MxFp4(projection) = self else {
            return Err(Error::InvalidQuantization("cost probe requires MXFP4".into()));
        };
        let shape = input.shape()?;
        let sequence = shape[1];
        let input = input.reshape(&[1, shape[0] * shape[1], shape[2]], stream)?;
        stream.eval_many(&[&input, &projection.weight, &projection.scales])?;
        stream.synchronize()?;
        assert_eq!(shape, [5, 8, 512], "split-budget probe uses the captured short down input");
        let baseline_parts = NonZeroUsize::new(8).ok_or(Error::ShapeOverflow)?;
        let candidate_parts = NonZeroUsize::new(4).ok_or(Error::ShapeOverflow)?;
        let candidate = Variant::Split4;
        let weights = mirtal::MxFp4 {
            weight: projection.weight.native(),
            scales: projection.scales.native(),
        };
        assert!(!projection.has_bias, "cost probe compares an unbiased projection");
        let split_eight = candidate::compiled(stream, baseline_parts, MxFp4RowTile::Rows16)?;
        let split_four = candidate::compiled(stream, candidate_parts, MxFp4RowTile::Rows16)?;
        let mut samples = Vec::new();
        let warm = [baseline, candidate, candidate, baseline];
        let measured =
            [baseline, candidate, candidate, baseline, baseline, candidate, candidate, baseline];
        for (run, variant) in warm.into_iter().chain(measured).enumerate() {
            let started = Instant::now();
            let mut outputs = Vec::with_capacity(256);
            for _ in 0..256 {
                outputs.push(match variant {
                    Variant::Native => self.forward(&input, stream)?,
                    Variant::Split8 | Variant::Split4 => {
                        let plan = if variant == Variant::Split8 {
                            &split_eight
                        } else {
                            &split_four
                        };
                        let [output] = plan.call(
                            stream.native(),
                            [input.native(), weights.weight, weights.scales],
                        )?;
                        Array::from_native(output)?
                    },
                });
            }
            let build_ms = started.elapsed().as_secs_f64() * 1000.0;
            stream.eval_many(&outputs.iter().collect::<Vec<_>>())?;
            stream.synchronize()?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let memory = crate::engine::memory_stats()?;
            drop(outputs);
            writeln!(
                std::io::stderr().lock(),
                "mxfp4.split_cost: {}",
                serde_json::json!({
                    "baseline":format!("{baseline:?}"),"label":label,"sequence":sequence,"run":run,"measured":run>=4,"variant":format!("{variant:?}"),"row_tile":16,"partitions_candidate":candidate_parts.get(),
                    "calls":256,"build_ms":build_ms,"elapsed_ms":elapsed_ms,"active_bytes":memory.active,"cache_bytes":memory.cached
                })
            )?;
            if run >= 4 {
                samples.push((variant, elapsed_ms));
                for mode in [baseline, candidate] {
                    let values = samples
                        .iter()
                        .filter(|(variant, _)| *variant == mode)
                        .map(|(_, time)| *time)
                        .collect::<Vec<_>>();
                    if values.len() >= 2 {
                        let spread = values.iter().copied().fold(0.0, f64::max)
                            / values.iter().copied().fold(f64::INFINITY, f64::min)
                            - 1.0;
                        if spread > 0.15 {
                            return Err(Error::BenchmarkStability {
                                case: format!("{label}/C5/{sequence}"),
                                variant: format!("{mode:?}"),
                                spread_percent: spread * 100.0,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
