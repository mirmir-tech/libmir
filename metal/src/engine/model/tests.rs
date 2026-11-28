use std::fs;

use super::*;

#[test]
fn reports_a_shard_only_after_materialization() -> Result<()> {
    let root = std::env::temp_dir().join(format!(
        "libmir-materialized-progress-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    write_safetensors(&root.join("model.safetensors"))?;
    let layout = ModelLayout::inspect(&root)?;
    let mut events = Vec::new();

    let tensors = ModelTensors::load_layout_materialized_with_progress(
        &layout,
        &Stream::new_cpu()?,
        |current, total, detail| events.push((current, total, detail)),
    )?;

    assert_eq!(events.len(), 2);
    assert_eq!(events[0].0, 0);
    assert!(events[0].2.starts_with("materializing weight shard 1/1"));
    assert_eq!(events[1].0, events[1].1);
    assert!(events[1].2.starts_with("materialized weight shard 1/1"));
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn write_safetensors(path: &Path) -> Result<()> {
    let mut header = r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_owned();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(&1.0_f32.to_le_bytes());
    fs::write(path, data)?;
    Ok(())
}
