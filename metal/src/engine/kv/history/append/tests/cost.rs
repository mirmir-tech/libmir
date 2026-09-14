use std::{io::Write, time::Instant};

use super::*;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Plan {
    Fresh,
    Prepared,
}

#[test]
#[ignore = "bounded host graph-construction benchmark; no GPU timing or evaluation"]
fn compares_history_append_graph_build() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let append = &stream.kernels().history_append;
    let dtype = mirtal::DType::Bfloat16;
    let (history, updates) = fixture(&stream, dtype, [3, 2, 2304, 256], 2082)?;
    stream.eval_many(&history.buffers.iter().chain(&updates).collect::<Vec<_>>())?;
    stream.synchronize()?;
    let mut samples = [Vec::new(), Vec::new()];
    for (run, plan) in [
        Plan::Fresh,
        Plan::Prepared,
        Plan::Prepared,
        Plan::Fresh,
        Plan::Fresh,
        Plan::Prepared,
        Plan::Prepared,
        Plan::Fresh,
        Plan::Prepared,
        Plan::Fresh,
        Plan::Fresh,
        Plan::Prepared,
    ]
    .into_iter()
    .enumerate()
    {
        let mut roots = Vec::with_capacity(4096);
        let started = Instant::now();
        for _ in 0..4096 {
            let outputs = match plan {
                Plan::Prepared => append.execute(&history, [&updates[0], &updates[1]], &stream)?,
                Plan::Fresh => {
                    let geometry = Geometry::new(history.shape, history.tokens)?;
                    let [keys, values] = append
                        .library
                        .export("history_append")?
                        .with_dtype_suffix(history.buffers[0].native().dtype()?)?
                        .dispatch_aliasing_array(
                            stream.native(),
                            &[
                                history.buffers[0].native(),
                                history.buffers[1].native(),
                                updates[0].native(),
                                updates[1].native(),
                            ],
                            &geometry.dispatch(),
                        )?;
                    [Array::from_native(keys)?, Array::from_native(values)?]
                },
            };
            roots.push(outputs);
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        // Deliberately discard all unevaluated graphs: the test isolates host
        // dispatch construction, not repeated writes or device execution.
        drop(roots);
        writeln!(
            std::io::stderr().lock(),
            "append.build: {}",
            serde_json::json!({
                "run":run,"plan":plan,"measured":run>=4,"calls":4096,"elapsed_ms":elapsed_ms,
            })
        )?;
        if run >= 4 {
            let values = &mut samples[usize::from(matches!(plan, Plan::Prepared))];
            values.push(elapsed_ms);
            let min = values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = values.iter().copied().fold(0.0, f64::max);
            if max > min * 1.15 {
                return Err(Error::BenchmarkStability {
                    case: "history/host-build".into(),
                    variant: format!("{plan:?}"),
                    spread_percent: (max / min - 1.0) * 100.0,
                });
            }
        }
    }
    let blocks = [0, 2].map(|i| {
        1.0 - samples[1][i..i + 2].iter().sum::<f64>() / samples[0][i..i + 2].iter().sum::<f64>()
    });
    let means = samples.map(|v| v.iter().sum::<f64>() / 4.0);
    writeln!(
        std::io::stderr().lock(),
        "append.build.gate: {}",
        serde_json::json!({
            "fresh_ms":means[0],"prepared_ms":means[1],"reduction":1.0-means[1]/means[0],"blocks":blocks,
            "passes": means[1]<=means[0]*0.97 && blocks.iter().all(|&r|r>0.0),"host_only":true,
        })
    )?;
    Ok(())
}
