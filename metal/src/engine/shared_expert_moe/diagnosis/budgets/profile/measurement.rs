use std::{io::Write, time::Instant};

use super::{Array, Case, Result, Stage, Stream, Variant};

pub(super) fn probe(
    case: Case,
    stage: Stage,
    variants: &[Variant],
    stream: &Stream,
    forward: impl Fn(Variant) -> Result<Vec<Array>>,
) -> Result<()> {
    // All dependencies are already materialized. Per-component barriers change
    // overlap; these measurements only choose where to optimize next.
    let measure = |variant| -> Result<(f64, f64)> {
        stream.synchronize()?;
        let started = Instant::now();
        let roots = forward(variant)?;
        let graph_ms = started.elapsed().as_secs_f64() * 1000.0;
        stream.eval_many(&roots.iter().collect::<Vec<_>>())?;
        stream.synchronize()?;
        Ok((graph_ms, started.elapsed().as_secs_f64() * 1000.0))
    };
    for _ in 0..3 {
        for &variant in variants {
            measure(variant)?;
        }
    }
    let mut samples = vec![Vec::new(); variants.len()];
    for block in 0..3 {
        let order = (0..variants.len()).map(|i| (block + i) % variants.len()).collect::<Vec<_>>();
        for (slot, &index) in order.iter().chain(order.iter().rev()).enumerate() {
            let (graph_ms, total_ms) = measure(variants[index])?;
            samples[index].push(total_ms);
            writeln!(
                std::io::stderr().lock(),
                "moe.tile_component: {}",
                serde_json::json!({
                    "case": case, "stage": stage, "variant": variants[index], "block": block,
                    "slot": slot, "graph_ms": graph_ms, "total_ms": total_ms,
                })
            )?;
        }
        if samples.iter().any(|values| {
            let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = values.iter().copied().fold(0.0, f64::max);
            hi > lo * 1.15
        }) {
            writeln!(
                std::io::stderr().lock(),
                "moe.tile_component_stop: {}",
                serde_json::json!({
                    "case": case, "stage": stage, "block": block,
                })
            )?;
            break;
        }
    }
    Ok(())
}
