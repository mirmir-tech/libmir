//! A GGUF import with a separate vision projector: two GGUF groups, the
//! text model and `mmproj`, each mapped and loaded whichever sorts first.

use std::{collections::BTreeSet, fs, path::Path};

use super::{GgufShards, TensorFile};
use crate::engine::{Error, ModelTensors, Result, Stream};

const MAGIC: &[u8; 4] = b"GGUF";
const ALIGNMENT: u64 = 32;
const F32: u32 = 0;

/// A GGUF v3 file of `metadata` strings and F32 `tensors`, laid out as
/// llama.cpp writes it.
fn gguf(metadata: &[(&str, &str)], tensors: &[(&str, &[u64])]) -> Result<Vec<u8>> {
    let string = |value: &str| -> Result<Vec<u8>> {
        let mut bytes = u64::try_from(value.len())?.to_le_bytes().to_vec();
        bytes.extend_from_slice(value.as_bytes());
        Ok(bytes)
    };
    let mut out = MAGIC.to_vec();
    out.extend(3_u32.to_le_bytes());
    out.extend(u64::try_from(tensors.len())?.to_le_bytes());
    out.extend(u64::try_from(metadata.len())?.to_le_bytes());
    for (key, value) in metadata {
        out.extend(string(key)?);
        out.extend(8_u32.to_le_bytes());
        out.extend(string(value)?);
    }
    let mut offset = 0_u64;
    for (name, dims) in tensors {
        out.extend(string(name)?);
        out.extend(u32::try_from(dims.len())?.to_le_bytes());
        for dim in *dims {
            out.extend(dim.to_le_bytes());
        }
        out.extend(F32.to_le_bytes());
        out.extend(offset.to_le_bytes());
        offset = (offset + dims.iter().product::<u64>() * 4).div_ceil(ALIGNMENT) * ALIGNMENT;
    }
    let header = u64::try_from(out.len())?.div_ceil(ALIGNMENT) * ALIGNMENT;
    out.resize(usize::try_from(header + offset)?, 0);
    Ok(out)
}

/// Loads the import from `root` and reads one text and one vision tensor.
fn text_and_vision(root: &Path, text: &str, projector: &str) -> Result<(Vec<i32>, Vec<i32>)> {
    fs::create_dir_all(root)?;
    fs::write(root.join("config.json"), "{}")?;
    let model = gguf(&[("general.architecture", "llama")], &[("token_embd.weight", &[4, 2])])?;
    fs::write(root.join(text), model)?;
    let tower = gguf(
        &[("general.architecture", "clip"), ("clip.vision.projector_type", "gemma4v")],
        &[("v.blk.0.attn_out.weight", &[4, 4])],
    )?;
    fs::write(root.join(projector), tower)?;
    // Weights load on the CPU stream, as the model loaders read them.
    let tensors = ModelTensors::load(root, &Stream::new_cpu()?);
    let shapes = tensors.and_then(|tensors| {
        let text = tensors.get("model.embed_tokens.weight")?.shape()?;
        let vision = tensors
            .get("model.vision_tower.encoder.layers.0.self_attn.o_proj.linear.weight")?
            .shape()?;
        Ok((text, vision))
    });
    fs::remove_dir_all(root)?;
    shapes
}

#[test]
fn a_separate_projector_loads_beside_its_text_model_in_either_order() -> Result<()> {
    let root = std::env::temp_dir().join(format!("libmir-metal-mmproj-{}", std::process::id()));
    // `mmproj.gguf` sorts before `model.gguf`, and after `a-model.gguf`.
    for (index, text) in ["model.gguf", "a-model.gguf"].into_iter().enumerate() {
        let shapes = text_and_vision(&root.join(index.to_string()), text, "mmproj.gguf")?;
        assert_eq!(shapes, (vec![2, 4], vec![4, 4]), "text model {text}");
    }
    Ok(())
}

/// A GGUF file outside the model's catalog is refused by type, and before
/// anything is read: the path need not even exist.
#[test]
fn an_uncatalogued_gguf_file_is_refused_unread() -> Result<()> {
    let missing = std::env::temp_dir().join("libmir-metal-never-written.gguf");
    let stream = Stream::new_cpu()?;
    let refused = |result: Result<TensorFile>| matches!(result, Err(Error::UncataloguedGguf(path)) if path == missing);
    assert!(refused(TensorFile::load(&missing, &stream)));
    let mut shards = GgufShards::of(&[])?;
    let weight = models::layout::WeightFile {
        path: missing.clone(),
        bytes: 0,
        container: models::layout::WeightContainer::Gguf,
    };
    assert!(refused(shards.load_weight(&weight, &BTreeSet::new(), &stream)));
    Ok(())
}
