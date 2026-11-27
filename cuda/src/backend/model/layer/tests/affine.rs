//! An MLX-style mixed-precision model: some layers store `v_proj` and
//! `down_proj` at 8 bits, the rest at 4. Each layer kind gets its own shared
//! prefill plan and matches the same layer over its dequantized BF16 values.

use mircuda::bf16;

use super::{HIDDEN, TOKENS, config, session, shared_prefill, tensor};
use crate::{
    AffineQuantizedWeight, CudaBackend, CudaConfig, CudaTensor, DenseDownSource,
    DenseFeedForwardSource, DenseGateUpSource, DenseOutputSource, DenseQkvSource,
    DenseWeightSource, ProjectionFormat, Result,
    backend::{linear::ggml_tests::copy, model::layer::SessionLayer},
};

const GROUP: usize = 32;

/// A `[HIDDEN, HIDDEN]` affine weight of `bits` bits and its dequantized
/// values, which BF16 holds exactly: power-of-two scales over centred codes.
fn affine(
    backend: &CudaBackend,
    bits: usize,
    seed: usize,
) -> Result<(AffineQuantizedWeight, CudaTensor)> {
    let levels = 1_usize << bits;
    let code = |row: usize, column: usize| (row * 7 + column * 3 + seed) % levels;
    let per_word = 32 / bits;
    let words = (0..HIDDEN)
        .flat_map(|row| {
            (0..HIDDEN / per_word).map(move |word| {
                (0..per_word).fold(0_u32, |packed, slot| {
                    let value = u32::try_from(code(row, word * per_word + slot)).unwrap_or(0);
                    packed | (value << (slot * bits))
                })
            })
        })
        .collect::<Vec<_>>();
    #[allow(clippy::cast_precision_loss)]
    let (scale, middle) =
        (2.0_f32.powi(-1 - i32::try_from(bits).unwrap_or(0)), levels as f32 / 2.0);
    let groups = HIDDEN / GROUP;
    let parameter = |value: f32| copy(backend, &vec![bf16::from_f32(value); HIDDEN * groups]);
    #[allow(clippy::cast_precision_loss)]
    let values = (0..HIDDEN * HIDDEN)
        .map(|index| bf16::from_f32(scale * (code(index / HIDDEN, index % HIDDEN) as f32 - middle)))
        .collect::<Vec<_>>();
    let shape = vec![HIDDEN, groups];
    let weight = AffineQuantizedWeight {
        weight: CudaTensor::from_u32(
            "weight".into(),
            vec![HIDDEN, HIDDEN * bits / 32],
            copy(backend, &words)?,
        ),
        scales: CudaTensor::from_bf16("scales".into(), shape.clone(), parameter(scale)?),
        biases: CudaTensor::from_bf16("biases".into(), shape, parameter(-scale * middle)?),
    };
    let dense =
        CudaTensor::from_bf16("dense".into(), vec![HIDDEN, HIDDEN], copy(backend, &values)?);
    Ok((weight, dense))
}

/// Layer `index` with `v_proj` and `down_proj` of `wide` bits and every other
/// projection of 4, as affine weights and as their BF16 values.
fn layers(backend: &CudaBackend, index: usize, wide: usize) -> Result<[SessionLayer; 2]> {
    let norm = tensor(backend, "norm", &[HIDDEN], 3)?;
    let bits = [4, 4, wide, 4, 4, 4, wide];
    let weights = bits
        .iter()
        .enumerate()
        .map(|(projection, &bits)| affine(backend, bits, index * 11 + projection * 5))
        .collect::<Result<Vec<_>>>()?;
    let [q, k, v, o, gate, up, down] = [0, 1, 2, 3, 4, 5, 6].map(|index| &weights[index]);
    let source = |qkv, output, gate_up, down| DenseWeightSource {
        input_norm: Some(&norm),
        qkv,
        query_norm: None,
        key_norm: None,
        output,
        post_attention_norm: Some(&norm),
        sandwich: None,
        feed_forward: DenseFeedForwardSource::Gated { gate_up, down },
        per_layer_input: None,
    };
    let quantized = backend.prepare_dense_swiglu_layer_template(
        config(index, ProjectionFormat::Affine, false),
        source(
            DenseQkvSource::Affine([&q.0, &k.0, &v.0]),
            DenseOutputSource::Affine(&o.0),
            DenseGateUpSource::Affine { gate: &gate.0, up: &up.0 },
            DenseDownSource::Affine(&down.0),
        ),
    )?;
    let qkv = backend.pack_bf16_linears([&q.1, &k.1, &v.1])?;
    let gate_up = backend.pack_bf16_linear_pair(&gate.1, &up.1)?;
    let dense = backend.prepare_dense_swiglu_layer_template(
        config(index + 8, ProjectionFormat::Bf16, false),
        source(
            DenseQkvSource::Bf16(&qkv),
            DenseOutputSource::Bf16(&o.1),
            DenseGateUpSource::Bf16(&gate_up),
            DenseDownSource::Bf16(&down.1),
        ),
    )?;
    Ok([session(backend, quantized)?, session(backend, dense)?])
}

/// Prefills `layers` as a session does, every layer through the first plan
/// whose signature it shares, and returns each output and the plan count.
fn prefill(backend: &CudaBackend, layers: &mut [SessionLayer]) -> Result<(Vec<Vec<bf16>>, usize)> {
    let mut plans = Vec::new();
    let mut outputs = Vec::new();
    for layer in layers {
        let signature = layer.prefill_signature();
        let shared = plans
            .iter()
            .position(|plan: &super::SharedLayerPrefill| plan.supports(&signature));
        let index = if let Some(index) = shared {
            index
        } else {
            plans.push(layer.instantiate_shared_prefill(TOKENS)?);
            plans.len() - 1
        };
        outputs.push(shared_prefill(backend, &mut plans[index], layer)?);
    }
    Ok((outputs, plans.len()))
}

#[test]
fn mixed_bit_layers_take_their_own_prefill_plans() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let [narrow, narrow_dense] = layers(&backend, 0, 4)?;
    let [wide, wide_dense] = layers(&backend, 1, 8)?;
    let [narrow_again, narrow_again_dense] = layers(&backend, 2, 4)?;
    let mut quantized = [narrow, wide, narrow_again];
    let (actual, plans) = prefill(&backend, &mut quantized)?;
    // The 4-bit layers share a plan; the mixed layer has its own.
    assert_eq!(plans, 2);
    let (expected, _) = prefill(&backend, &mut [narrow_dense, wide_dense, narrow_again_dense])?;
    for (layer, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let (actual, expected) = (actual.to_f32(), expected.to_f32());
            assert!(
                (actual - expected).abs() <= expected.abs().mul_add(1.0e-2, 4.0e-3),
                "layer {layer} [{index}]: {actual} vs {expected}"
            );
        }
    }
    Ok(())
}
