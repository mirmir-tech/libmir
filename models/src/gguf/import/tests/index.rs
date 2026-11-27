//! A `SafeTensors` index beside an import names weights the import does
//! not hold: the loader reads the import's, as its manifest records them,
//! while an ordinary checkpoint still reads the shards its index names.

use std::fs;

use super::{import, root};
use crate::{
    gguf::tests::TestResult,
    layout::{ModelLayout, WeightContainer},
};

const INDEX: &str = "model.safetensors.index.json";

fn index(names: &[&str]) -> TestResult<Vec<u8>> {
    let map = names
        .iter()
        .enumerate()
        .map(|(tensor, name)| (format!("tensor.{tensor}"), serde_json::json!(name)))
        .collect::<serde_json::Map<_, _>>();
    Ok(serde_json::to_vec(&serde_json::json!({ "weight_map": map }))?)
}

#[test]
fn a_leftover_index_does_not_choose_an_imports_weights() -> TestResult {
    let root = root("index")?;
    let source = root.join("chosen.gguf");
    fs::write(&source, b"gguf bytes")?;
    let output = root.join("output");
    fs::create_dir_all(&output)?;
    fs::write(output.join(INDEX), index(&["other.safetensors"])?)?;
    fs::write(output.join("other.safetensors"), b"safetensors bytes")?;
    import(vec![source], 1).write(&output)?;

    let layout = ModelLayout::inspect(&output)?;
    let weights = layout.weights.iter().map(|weight| weight.path.clone()).collect::<Vec<_>>();
    assert_eq!(weights, [output.join("chosen.gguf")]);
    assert_eq!(layout.container(), Some(WeightContainer::Gguf));
    assert_eq!(layout.safetensors_index_path, None);
    // The user's files stay.
    assert!(output.join(INDEX).is_file() && output.join("other.safetensors").is_file());
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A checkpoint downloaded over an import, its `config.json` over the
/// import's, is read as that checkpoint: the manifest no longer tells the
/// directory's model.
#[test]
fn a_checkpoint_downloaded_over_an_import_is_that_checkpoint() -> TestResult {
    let root = root("index-over")?;
    let source = root.join("chosen.gguf");
    fs::write(&source, b"gguf bytes")?;
    let output = root.join("output");
    import(vec![source], 1).write(&output)?;
    fs::write(output.join("config.json"), b"{\"model_type\": \"llama\"}")?;
    fs::write(output.join(INDEX), index(&["model.safetensors"])?)?;
    fs::write(output.join("model.safetensors"), b"safetensors bytes")?;

    let layout = ModelLayout::inspect(&output)?;
    let weights = layout.weights.iter().map(|weight| weight.path.clone()).collect::<Vec<_>>();
    assert_eq!(weights, [output.join("model.safetensors")]);
    assert_eq!(layout.container(), Some(WeightContainer::SafeTensors));
    assert_eq!(layout.safetensors_index_path, Some(output.join(INDEX)));
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn an_indexed_checkpoint_reads_the_shards_its_index_names() -> TestResult {
    let root = root("indexed")?;
    let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"];
    fs::write(root.join("config.json"), b"{}")?;
    fs::write(root.join(INDEX), index(&shards)?)?;
    for shard in shards {
        fs::write(root.join(shard), b"safetensors bytes")?;
    }
    // A file the index does not name is no weight.
    fs::write(root.join("stray.safetensors"), b"safetensors bytes")?;

    let layout = ModelLayout::inspect(&root)?;
    let weights = layout.weights.iter().map(|weight| weight.path.clone()).collect::<Vec<_>>();
    assert_eq!(weights, shards.map(|shard| root.join(shard)));
    assert_eq!(layout.container(), Some(WeightContainer::SafeTensors));
    assert_eq!(layout.safetensors_index_path, Some(root.join(INDEX)));
    fs::remove_dir_all(root)?;
    Ok(())
}
