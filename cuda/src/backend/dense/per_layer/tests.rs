//! The per-layer input gate over GGML blocks against the same gate over
//! their dequantized BF16 values, and the load-time checks that refuse
//! tensors its projections cannot read.

use mircuda::{DeviceBuffer, bf16, f16};
use models::gguf::GgmlType;

use super::{PerLayerGate, PerLayerInputWeights};
use crate::{
    AffineQuantizedWeight, CudaBackend, CudaConfig, CudaTensor, Error, GgmlCheckpointWeight,
    Result,
    backend::linear::{
        CheckpointProjectionWeight,
        ggml_tests::{copy, copy_blocks, input, q8_0, read},
    },
    kernels::GgmlKernelType,
};

const HIDDEN: usize = 64;
const WIDTH: usize = 32;
const EPSILON: f32 = 1.0e-6;

/// `Q8_0` blocks of a `[rows, columns]` matrix and their dequantized values.
fn blocks(rows: usize, columns: usize) -> (Vec<u8>, Vec<bf16>) {
    let bytes = q8_0(1, rows, columns, &[]);
    let values = bytes
        .as_chunks::<34>()
        .0
        .iter()
        .flat_map(|block| {
            let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
            block[2..].iter().map(move |&q| bf16::from_f32(d * f32::from(q.cast_signed())))
        })
        .collect();
    (bytes, values)
}

fn bf16_tensor(
    backend: &CudaBackend,
    name: &str,
    shape: &[usize],
    values: &[bf16],
) -> Result<CudaTensor> {
    Ok(CudaTensor::from_bf16(name.into(), shape.to_vec(), copy(backend, values)?))
}

/// The gate `[WIDTH, HIDDEN]` and projection `[HIDDEN, WIDTH]`, as GGML
/// blocks or as their BF16 values.
fn weights(backend: &CudaBackend, ggml: bool) -> Result<PerLayerInputWeights> {
    let projection = |name: &str, rows, columns| -> Result<CheckpointProjectionWeight> {
        let (bytes, values) = blocks(rows, columns);
        Ok(if ggml {
            CheckpointProjectionWeight::Ggml(GgmlCheckpointWeight::from_blocks(
                copy_blocks(backend, &bytes)?,
                GgmlKernelType::new(GgmlType::Q8_0)?,
                (1, rows, columns),
                None,
            ))
        } else {
            CheckpointProjectionWeight::Dense(bf16_tensor(
                backend,
                name,
                &[rows, columns],
                &values,
            )?)
        })
    };
    let norm = (0..HIDDEN)
        .map(|index| bf16::from_f32(f32::from(u8::try_from(index % 7).unwrap_or(0)) / 8.0 + 0.5))
        .collect::<Vec<_>>();
    Ok(PerLayerInputWeights {
        gate: projection("inp_gate", WIDTH, HIDDEN)?,
        projection: projection("proj", HIDDEN, WIDTH)?,
        output_norm: bf16_tensor(backend, "post_norm", &[HIDDEN], &norm)?,
    })
}

/// The gate's normalized output over `rows` rows.
fn gate(backend: &CudaBackend, weights: &PerLayerInputWeights, rows: usize) -> Result<Vec<bf16>> {
    weights.validate(HIDDEN, WIDTH)?;
    let mut operation = PerLayerGate::new(backend, rows, [HIDDEN, WIDTH], EPSILON, weights)?;
    let hidden = copy(backend, &input(rows * HIDDEN))?;
    let per_layer = copy(backend, &input(rows * WIDTH + 5)[5..])?;
    let allocate = || backend.pool().allocate::<bf16>(backend.stream(), rows * HIDDEN);
    let (mut projected, mut normalized): (DeviceBuffer<bf16>, _) = (allocate()?, allocate()?);
    operation.execute(&hidden, weights, &per_layer, &mut projected, &mut normalized)?;
    read(backend, &normalized)
}

/// Quantized `inp_gate` and `proj` run as GGML blocks in decode and prefill
/// (the GEMV quantizes its input to int8, hence the tolerance).
#[test]
fn gates_ggml_blocks_as_their_dequantized_values() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let ggml = weights(&backend, true)?;
    let dense = weights(&backend, false)?;
    for rows in [1, 4, 300] {
        let expected = gate(&backend, &dense, rows)?;
        let actual = gate(&backend, &ggml, rows)?;
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            let (expected, actual) = (expected.to_f32(), actual.to_f32());
            assert!(
                (expected - actual).abs() <= expected.abs().mul_add(4.0e-2, 4.0e-2),
                "{rows} rows [{index}]: {actual} vs {expected}"
            );
        }
    }
    Ok(())
}

fn refusal(weights: &PerLayerInputWeights) -> Result<Error> {
    weights.validate(HIDDEN, WIDTH).map_or_else(Ok, |()| {
        Err(Error::InvalidDecoderKernel("per-layer input weights passed validation"))
    })
}

/// Raw GGML bytes or F32 values bound as dense are refused at load, not
/// left to fail as BF16 in the first pass.
#[test]
fn refuses_dense_weights_that_are_not_bf16() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut weights = weights(&backend, false)?;
    let bytes = copy(&backend, &vec![0_u8; WIDTH * HIDDEN])?;
    weights.gate = CheckpointProjectionWeight::Dense(CudaTensor::from_u8(
        "inp_gate".into(),
        vec![WIDTH, HIDDEN],
        bytes,
    ));
    assert!(matches!(refusal(&weights)?, Error::DTypeMismatch { expected: "BF16", .. }));
    let values = copy(&backend, &vec![0.0_f32; HIDDEN * WIDTH])?;
    weights.gate = self::weights(&backend, false)?.gate;
    weights.projection = CheckpointProjectionWeight::Dense(CudaTensor::from_f32(
        "proj".into(),
        vec![HIDDEN, WIDTH],
        values,
    ));
    assert!(matches!(refusal(&weights)?, Error::DTypeMismatch { expected: "BF16", .. }));
    Ok(())
}

/// GGML blocks of the wrong geometry and a norm of the wrong width are
/// refused at load.
#[test]
fn refuses_mismatched_geometry() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let mut weights = weights(&backend, true)?;
    // The projection's blocks bound as the gate: `[HIDDEN, WIDTH]`.
    weights.gate = weights.projection.clone();
    assert!(!matches!(refusal(&weights)?, Error::DTypeMismatch { .. }));
    let mut weights = self::weights(&backend, true)?;
    let norm = vec![bf16::ONE; WIDTH];
    weights.output_norm = bf16_tensor(&backend, "post_norm", &[WIDTH], &norm)?;
    assert!(matches!(refusal(&weights)?, Error::InvalidQuantizedTensor { .. }));
    Ok(())
}

/// MLX-style affine projections of `bits` bits in groups of 32.
fn affine(backend: &CudaBackend, bits: usize) -> Result<PerLayerInputWeights> {
    let matrix = |name: &str, rows: usize, columns: usize| -> Result<CheckpointProjectionWeight> {
        let (words, groups) = (columns * bits / 32, columns / 32);
        let packed = copy(backend, &vec![0x8888_8888_u32; rows * words])?;
        let parameters = |seed| -> Result<CudaTensor> {
            let values = vec![bf16::from_f32(seed); rows * groups];
            Ok(CudaTensor::from_bf16(name.into(), vec![rows, groups], copy(backend, &values)?))
        };
        Ok(CheckpointProjectionWeight::Affine(AffineQuantizedWeight {
            weight: CudaTensor::from_u32(name.into(), vec![rows, words], packed),
            scales: parameters(0.01)?,
            biases: parameters(-0.02)?,
        }))
    };
    let mut weights = weights(backend, false)?;
    weights.gate = matrix("inp_gate", WIDTH, HIDDEN)?;
    weights.projection = matrix("proj", HIDDEN, WIDTH)?;
    Ok(weights)
}

/// Layers of one format but different bits (MLX mixed-precision recipes)
/// have different layouts, so they share no prefill plan, and a plan built
/// for one refuses the other's weights instead of reading them in its own
/// layout.
#[test]
fn refuses_affine_weights_of_another_layout() -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let (four, eight) = (affine(&backend, 4)?, affine(&backend, 8)?);
    assert_ne!(four.layouts(), eight.layouts());
    assert_eq!(four.layouts(), affine(&backend, 4)?.layouts());
    gate(&backend, &four, 1)?;
    let mut operation = PerLayerGate::new(&backend, 1, [HIDDEN, WIDTH], EPSILON, &four)?;
    let hidden = copy(&backend, &input(HIDDEN))?;
    let per_layer = copy(&backend, &input(WIDTH))?;
    let allocate = || backend.pool().allocate::<bf16>(backend.stream(), HIDDEN);
    let (mut projected, mut normalized) = (allocate()?, allocate()?);
    let refused = operation.execute(&hidden, &eight, &per_layer, &mut projected, &mut normalized);
    assert!(matches!(refused, Err(Error::InvalidExecutionPlan(_))), "{refused:?}");
    Ok(())
}
