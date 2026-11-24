use std::time::Instant;

use crate::engine::{Array, Result, Stream};

pub(super) fn emit_component(
    output: &Array,
    stream: &Stream,
    layer: usize,
    attention: &str,
    component: &str,
    started: Instant,
) -> Result<()> {
    output.async_eval(stream)?;
    stream.synchronize()?;
    tracing::debug!(
        layer,
        attention,
        component,
        milliseconds = started.elapsed().as_secs_f64() * 1_000.0,
        "MLX hybrid linear MoE component profile"
    );
    Ok(())
}
