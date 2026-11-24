//! A GGUF import directory read through `WeightContainer`: split text shards,
//! the vision tower's `mmproj` and the `SafeTensors` file its import assembles.

use std::fs;

use super::{GgmlType, TestResult, Writer, vision::mmproj};
use crate::{
    layout::{WeightContainer, WeightFile},
    weights::TensorCatalog,
};

fn shard(number: u32, writer: Writer) -> Writer {
    let unsigned = |value: u32| value.to_le_bytes().to_vec();
    writer.kv("split.no", 4, unsigned(number)).kv("split.count", 4, unsigned(2))
}

fn safetensors() -> Vec<u8> {
    let header = r#"{"model.visual.patch_embed.proj.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes
}

#[test]
fn reads_each_container_of_a_gguf_import_once() -> TestResult {
    let root = std::env::temp_dir().join(format!("libmir-gguf-container-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    let first = shard(0, Writer::new().text("general.architecture", "nemotron_h")?)
        .kv("split.tensors.count", 4, 2_u32.to_le_bytes().to_vec())
        .tensor("blk.0.ssm_d", &[1, 4], GgmlType::F32)?;
    let second = shard(1, Writer::new()).tensor("blk.1.ssm_a", &[1, 4], GgmlType::F32)?;
    fs::write(root.join("model-00001-of-00002.gguf"), first.bytes()?)?;
    fs::write(root.join("model-00002-of-00002.gguf"), second.bytes()?)?;
    fs::write(root.join("mmproj-F16.gguf"), mmproj()?.bytes()?)?;
    fs::write(root.join("mmproj-patch-embedding.safetensors"), safetensors())?;
    fs::write(root.join("config.json"), "{}")?;
    let files = |container: WeightContainer| -> TestResult<Vec<WeightFile>> {
        Ok(container
            .discover(&root)?
            .into_iter()
            .map(WeightFile::open)
            .collect::<Result<_, _>>()?)
    };
    let gguf = files(WeightContainer::Gguf)?;
    let safetensors = files(WeightContainer::SafeTensors)?;
    let catalog = TensorCatalog::from_weight_files(&[gguf.clone(), safetensors.clone()].concat());
    let text = WeightContainer::Gguf.read_catalog(&gguf);
    let vision = WeightContainer::SafeTensors.read_catalog(&safetensors);
    fs::remove_dir_all(&root)?;
    assert_eq!(gguf.len(), 3);
    assert!(gguf.iter().all(|file| file.container == WeightContainer::Gguf));
    let names = text?.into_iter().map(|tensor| tensor.name).collect::<Vec<_>>();
    for name in ["backbone.layers.0.mixer.D", "backbone.layers.1.mixer.A_log"] {
        // Both shards are listed, yet the split model is read once.
        assert_eq!(names.iter().filter(|found| *found == name).count(), 1, "{name}");
    }
    assert!(names.iter().any(|name| name == "model.visual.pos_embed.weight"));
    let vision = vision?;
    assert_eq!(vision.len(), 1);
    assert_eq!(vision[0].name, "model.visual.patch_embed.proj.weight");
    let catalog = catalog?;
    assert!(catalog.get("model.visual.patch_embed.proj.weight").is_some());
    assert!(catalog.get("backbone.layers.1.mixer.A_log").is_some());
    Ok(())
}
