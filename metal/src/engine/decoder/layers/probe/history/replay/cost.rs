use std::time::Instant;

use super::*;

impl Replay {
    pub fn measure_assembly(replays: &[Self], stream: &Stream) -> Result<()> {
        assert!(!replays.is_empty());
        let mut samples = [Vec::<f64>::new(), Vec::<f64>::new()];
        for (run, plan) in [
            Plan::Assemble,
            Plan::Resident,
            Plan::Resident,
            Plan::Assemble,
            Plan::Assemble,
            Plan::Resident,
            Plan::Resident,
            Plan::Assemble,
            Plan::Resident,
            Plan::Assemble,
            Plan::Assemble,
            Plan::Resident,
        ]
        .into_iter()
        .enumerate()
        {
            stream.synchronize()?;
            let started = Instant::now();
            let mut roots = Vec::with_capacity(1024);
            for call in 0..1024 {
                roots.push(replays[call % replays.len()].forward(plan, stream)?);
            }
            let build_ms = started.elapsed().as_secs_f64() * 1000.0;
            stream.eval_many(&roots.iter().collect::<Vec<_>>())?;
            stream.synchronize()?;
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let memory = crate::engine::memory_stats()?;
            drop(roots);
            writeln!(
                std::io::stderr().lock(),
                "history.cost: {}",
                serde_json::json!({
                    "run":run,"plan":plan,"measured":run>=4,"calls":1024,"build_ms":build_ms,"elapsed_ms":elapsed_ms,
                    "active_bytes":memory.active,"cached_bytes":memory.cached,
                })
            )?;
            if run >= 4 {
                let values = &mut samples[usize::from(plan == Plan::Resident)];
                values.push(elapsed_ms);
                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                let max = values.iter().copied().fold(0.0, f64::max);
                if max > min * 1.15 {
                    return Err(Error::BenchmarkStability {
                        case: "history/attention-replay".into(),
                        variant: format!("{plan:?}"),
                        spread_percent: (max / min - 1.0) * 100.0,
                    });
                }
            }
        }
        let blocks = [0, 2].map(|i| {
            1.0 - samples[1][i..i + 2].iter().sum::<f64>()
                / samples[0][i..i + 2].iter().sum::<f64>()
        });
        let means = samples.map(|v| v.iter().sum::<f64>() / 4.0);
        writeln!(
            std::io::stderr().lock(),
            "history.cost.summary: {}",
            serde_json::json!({
                "assemble_ms":means[0],"resident_ms":means[1],"reduction":1.0-means[1]/means[0],
                "block_reductions":blocks,"material_cost":means[1]<=means[0]*0.90 && blocks.iter().all(|&g|g>0.0),
                "counterfactual_only":true,
            })
        )?;
        for replay in replays {
            replay.inspect(stream)?;
        }
        Ok(())
    }
}
