use super::{Array, Result, Stream};

pub fn export_once(logits: &Array, stream: &Stream) -> Result<()> {
    let Some(path) = stream.take_graph_dump_path() else {
        return Ok(());
    };
    logits.export_graph_dot(path)?;
    tracing::debug!(path = %path.display(), "exported MLX decode graph");
    Ok(())
}
