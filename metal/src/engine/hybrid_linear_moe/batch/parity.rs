use std::io::Write;

use super::*;
use crate::engine::{
    ModelTensors,
    decoder::{LayerContext, LoweredLayer},
    lowering,
};

#[test]
#[ignore = "real Qwen first scalar/packed prefill component difference"]
fn diagnoses_first_prefill_difference() -> std::result::Result<(), Box<dyn std::error::Error>> {
    diagnose(Probe::Model)
}

#[test]
#[ignore = "real Qwen first scalar/packed component difference with FP32 routing"]
fn diagnoses_first_precise_prefill_difference()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    diagnose(Probe::Float32)
}

#[test]
#[ignore = "real Qwen ordinary MXFP4 against independent f64 decoded-weight dot products"]
fn diagnoses_mxfp4_projection_accuracy() -> std::result::Result<(), Box<dyn std::error::Error>> {
    diagnose(Probe::MxFp4Accuracy)
}

#[test]
#[ignore = "one bounded S8 versus S4 compiled FP32 down-projection cost gate"]
fn measures_mxfp4_split_budget_cost() -> std::result::Result<(), Box<dyn std::error::Error>> {
    diagnose(Probe::MxFp4Cost)
}

#[test]
#[ignore = "one bounded native versus S4 compiled FP32 down-projection cost gate"]
fn measures_native_mxfp4_split_budget_cost() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    diagnose(Probe::MxFp4NativeCost)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Probe {
    Model,
    Float32,
    MxFp4Accuracy,
    MxFp4Cost,
    MxFp4NativeCost,
}

impl Probe {
    fn precision(self) -> crate::config::RouterPrecision {
        match self {
            Self::Model => crate::config::RouterPrecision::Model,
            Self::Float32 | Self::MxFp4Accuracy | Self::MxFp4Cost | Self::MxFp4NativeCost => {
                crate::config::RouterPrecision::Float32
            },
        }
    }

    fn sequence(self) -> u32 {
        match self {
            Self::MxFp4Accuracy | Self::MxFp4Cost | Self::MxFp4NativeCost => 128,
            Self::Model | Self::Float32 => 8,
        }
    }
}

fn diagnose(probe: Probe) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let precision = probe.precision();
    let path = std::env::var("MIRMIR_BENCH_MODEL")?;
    let layout = models::layout::ModelLayout::inspect(std::path::Path::new(&path))?;
    let decoder = models::layout::DecoderConfig::from_layout(&layout)?;
    let catalog = models::weights::TensorCatalog::from_layout(&layout)?;
    let contract =
        models::execution::DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(std::path::Path::new(&path), &load_stream)?;
    let mut config = crate::MetalConfig::default();
    config.diagnostics.router_precision = precision;
    config.tuning.mode = runtime::tuning::TuningMode::Disabled;
    config.cache.prefix_cache_entries = 0;
    let stream = Stream::new_gpu_with_config(std::sync::Arc::new(config))?;
    let model = HybridLinearMoeModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        256,
        &stream,
    )?;
    let sequence = probe.sequence();
    let tokens = (0..5u32)
        .flat_map(|row| (0..sequence).map(move |token| row + token))
        .collect::<Vec<_>>();
    let mut packed = model
        .embedding
        .lookup(&Array::from_u32(&tokens, &[5, i32::try_from(sequence)?])?, &stream)?;
    let mut scalar = split(&packed, &stream)?;
    let mut scalar_caches = (0..5).map(|_| model.new_cache(&stream)).collect::<Result<Vec<_>>>()?;
    let mut packed_caches = (0..5).map(|_| model.new_cache(&stream)).collect::<Result<Vec<_>>>()?;
    for (index, layer) in model.layers.iter().enumerate() {
        for (hidden, cache) in scalar.iter_mut().zip(&mut scalar_caches) {
            *hidden = layer.forward_mixer(hidden, cache, index, context(&stream))?;
        }
        packed = layer.mix_packed_prefill(
            &packed,
            &mut packed_caches.iter_mut().collect::<Vec<_>>(),
            &[0; 5],
            &stream,
        )?;
        if !compare(&format!("layer={index},mixer"), &scalar, &packed, &stream)? {
            if !matches!(probe, Probe::Model | Probe::Float32) {
                return Err(crate::engine::Error::InvalidModel(
                    "accuracy capture mixer differs".into(),
                )
                .into());
            }
            break;
        }
        let input = layer.post_attention_norm.apply(&packed, layer.rms_norm_eps, &stream)?;
        match probe {
            Probe::MxFp4Cost => {
                return Ok(layer.moe.probe_shared_projection_cost(&input, &stream)?);
            },
            Probe::MxFp4NativeCost => {
                return Ok(layer.moe.probe_shared_projection_native_cost(&input, &stream)?);
            },
            Probe::MxFp4Accuracy => {
                return Ok(layer.moe.probe_shared_projection_accuracy(&input, &stream)?);
            },
            Probe::Model | Probe::Float32 => {},
        }
        if precision == crate::config::RouterPrecision::Model {
            layer.moe.diagnose_row_batching(&input, &stream)?;
        }
        for hidden in &mut scalar {
            *hidden = layer.forward_feed_forward(hidden, context(&stream))?;
        }
        packed = layer.feed_forward_packed_prefill(&packed, &stream)?;
        if !compare(&format!("layer={index},ffn"), &scalar, &packed, &stream)? {
            if precision == crate::config::RouterPrecision::Float32 {
                layer.moe.diagnose_effective_rows(&input, &stream)?;
            }
            break;
        }
    }
    stream.synchronize()?;
    Ok(())
}

fn context(stream: &Stream) -> LayerContext<'_> {
    LayerContext {
        position: 0,
        causal: true,
        positions: None,
        image: None,
        stream,
    }
}

fn split(input: &Array, stream: &Stream) -> Result<Vec<Array>> {
    let shape = input.shape()?;
    let rows = usize::try_from(shape[0])?;
    let sequence = usize::try_from(shape[1])?;
    let width = usize::try_from(shape[2])?;
    (0..rows)
        .map(|row| input.slice(&[row, 0, 0], &[row + 1, sequence, width], stream))
        .collect()
}

fn compare(label: &str, scalar: &[Array], packed: &Array, stream: &Stream) -> Result<bool> {
    let a = scalar
        .iter()
        .map(|a| a.to_vec_f32(stream))
        .collect::<Result<Vec<_>>>()?
        .concat();
    let b = packed.to_vec_f32(stream)?;
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(&b).all(|value| value.is_finite()));
    let differing = a.iter().zip(&b).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let max_abs = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    writeln!(
        std::io::stderr().lock(),
        "component.parity: {}",
        serde_json::json!({"label":label,"differing":differing,"max_abs":max_abs,"elements":a.len()})
    )?;
    Ok(differing == 0)
}
