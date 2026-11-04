//! Writes `mir-model-spec.toml` for a checkpoint: the semantics derived from
//! its configuration and tensors, plus properties the configuration does not
//! declare, given explicitly on the command line. The written spec replaces
//! derivation when the model loads.
//!
//! Usage: `model-spec <model> [value-rms] [attention-scale=X]
//! [embedding-multiplier=X]`. `value-rms` declares an RMS norm without a
//! learned scale on the values of every Q/K-normalized attention layer;
//! `attention-scale` sets the softmax scale of every attention layer.
#![allow(clippy::print_stdout)]

use std::{env, fs, path::PathBuf};

use models::{
    layout::{DecoderConfig, ModelLayout},
    semantic::{MixerSpec, QkNormalization, SemanticModelSpec},
    weights::TensorCatalog,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let root = PathBuf::from(args.next().ok_or("missing model directory")?);
    let layout = ModelLayout::inspect(&root)?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let mut spec = SemanticModelSpec::discover(&decoder, &catalog)?;
    for arg in args {
        match arg.split_once('=') {
            None if arg == "value-rms" => declare_value_rms(&mut spec)?,
            Some(("attention-scale", value)) => {
                let scale = value.parse()?;
                for attention in attentions(&mut spec) {
                    attention.scale = scale;
                }
            },
            Some(("embedding-multiplier", value)) => {
                spec.decoder.embedding_multiplier = value.parse()?;
            },
            _ => return Err(format!("unknown declaration {arg}").into()),
        }
    }
    let path = root.join("mir-model-spec.toml");
    fs::write(&path, spec.to_toml()?)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn declare_value_rms(spec: &mut SemanticModelSpec) -> Result<(), Box<dyn std::error::Error>> {
    for attention in attentions(spec) {
        if attention.qk_normalization == QkNormalization::None {
            return Err("value-rms needs Q/K-normalized attention in every layer".into());
        }
        attention.qk_normalization = QkNormalization::QueryKeyValueRms;
    }
    Ok(())
}

fn attentions(
    spec: &mut SemanticModelSpec,
) -> impl Iterator<Item = &mut models::semantic::AttentionSpec> {
    spec.decoder.layers.iter_mut().filter_map(|layer| match &mut layer.mixer {
        MixerSpec::SoftmaxAttention(attention) => Some(attention),
        MixerSpec::LinearAttention(_) | MixerSpec::StateSpace(_) => None,
    })
}
