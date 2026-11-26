//! Interleaved rotary rows and the Q/K norms that scale them.

use mircuda::{DeviceBuffer, bf16};
use models::weights::{
    DenseFeedForwardBindings, DenseLayerFeedForwardBindings, DenseSoftmaxBindings, LayerTensorRole,
    LogicalTensorRole, TensorBinding, TensorStorage,
};

use super::restore;
use crate::{
    CudaBackend, CudaConfig, CudaTensor, CudaTensorSet, Result,
    checkpoint::dense::{DenseAttentionBindings, DenseLayerBindings},
};

const HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const ROTARY: usize = 4;

fn binding(source: &str, shape: &[usize]) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::InputNorm,
        },
        source: source.into(),
        shape: shape.to_vec(),
        logical_shape: None,
        transforms: Vec::new(),
        storage: TensorStorage::Dense { dtype: "BF16".into(), bias: None },
    }
}

fn copy(backend: &CudaBackend, values: &[bf16]) -> Result<DeviceBuffer<bf16>> {
    let mut host = backend.context().allocate_pinned::<bf16>(values.len())?;
    host.copy_from_slice(values)?;
    let mut device = backend.pool().allocate::<bf16>(backend.stream(), values.len())?;
    backend.stream().copy_to_device(&mut host, &mut device)?;
    backend.stream().synchronize()?;
    Ok(device)
}

fn read(backend: &CudaBackend, values: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host = backend.context().allocate_pinned::<bf16>(values.len())?;
    backend.stream().copy_to_host(values, &mut host)?;
    Ok(host.to_vec()?)
}

/// A BF16 tensor whose value `i` is `offset + i`.
fn tensor(backend: &CudaBackend, name: &str, shape: &[usize], offset: u16) -> Result<CudaTensor> {
    let values = (0..shape.iter().product::<usize>())
        .map(|index| bf16::from_f32(f32::from(offset + u16::try_from(index).unwrap_or(0))))
        .collect::<Vec<_>>();
    Ok(CudaTensor::from_bf16(name.into(), shape.to_vec(), copy(backend, &values)?))
}

/// Each tensor's values after restoring with these norm sources, minus the
/// offsets they started from.
fn restored(norms: (&str, &str), shapes: [&[usize]; 2]) -> Result<Vec<Vec<u16>>> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let names = ["q", "k", norms.0, norms.1];
    let shapes = [&[HEADS * HEAD_DIM, 1][..], &[HEAD_DIM, 1], shapes[0], shapes[1]];
    // Distinct ranges, all integers BF16 holds exactly.
    let offsets = [0, 32, 64, 128];
    let mut tensors = Vec::new();
    for ((name, shape), offset) in names.iter().zip(shapes).zip(offsets) {
        if !tensors.iter().any(|tensor: &CudaTensor| tensor.name() == *name) {
            tensors.push(tensor(&backend, name, shape, offset)?);
        }
    }
    let mut set = CudaTensorSet::from_tensors(backend.context(), backend.stream(), tensors);
    let [q, k, q_norm, k_norm] = [0, 1, 2, 3].map(|i| binding(names[i], shapes[i]));
    let (value, output, gate) = (binding("v", &[8, 1]), binding("o", &[1, 8]), binding("g", &[1]));
    let bindings = DenseLayerBindings {
        input_norm: None,
        attention: DenseAttentionBindings::Projected(DenseSoftmaxBindings {
            query: &q,
            key: &k,
            value: &value,
            output: &output,
            query_norm: Some(&q_norm),
            key_norm: Some(&k_norm),
        }),
        post_attention_norm: None,
        sandwich: None,
        feed_forward: DenseLayerFeedForwardBindings::Gated(DenseFeedForwardBindings {
            gate: &gate,
            up: &gate,
            down: &gate,
        }),
        per_layer_input: None,
    };
    restore(&backend, &mut set, bindings, (HEAD_DIM, ROTARY))?;
    let mut values = Vec::new();
    for (name, offset) in names.iter().zip(offsets) {
        let device = set.get(name).and_then(CudaTensor::as_bf16).cloned();
        let device = device.ok_or_else(|| crate::Error::MissingTensor((*name).into()))?;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        values.push(
            read(&backend, &device)?
                .iter()
                .map(|v| (v.to_f32() as u16).wrapping_sub(offset))
                .collect(),
        );
    }
    Ok(values)
}

/// The source row of restored row `row` (`rope_rows_u8.cu`).
const fn source(row: usize) -> usize {
    let within = row % HEAD_DIM;
    if within < ROTARY {
        row - within + (within % (ROTARY / 2)) * 2 + within / (ROTARY / 2)
    } else {
        row
    }
}

/// Every norm feature follows the row its projection moved; features past
/// the rotary part keep their place.
#[test]
fn norms_follow_their_rows() -> Result<()> {
    let values = restored(("q_norm", "k_norm"), [&[HEADS, HEAD_DIM], &[HEAD_DIM]])?;
    let [q, k, q_norm, k_norm] = [&values[0], &values[1], &values[2], &values[3]];
    for row in 0..HEADS * HEAD_DIM {
        let from = u16::try_from(source(row))?;
        assert_eq!((q[row], q_norm[row]), (from, from), "query row {row}");
    }
    assert!((0..HEADS * HEAD_DIM).any(|row| source(row) != row), "the rows moved");
    for row in 0..HEAD_DIM {
        let from = u16::try_from(source(row))?;
        assert_eq!((k[row], k_norm[row]), (from, from), "key row {row}");
        if row >= ROTARY {
            assert_eq!(k_norm[row], u16::try_from(row)?, "unrotated feature {row}");
        }
    }
    Ok(())
}

/// One tensor bound as both norms moves once: not left in place, and not
/// moved twice (the 4-row permutation is its own inverse).
#[test]
fn a_shared_norm_moves_once() -> Result<()> {
    let values = restored(("qk_norm", "qk_norm"), [&[HEAD_DIM], &[HEAD_DIM]])?;
    let norm = &values[2];
    for (row, &moved) in norm.iter().enumerate() {
        assert_eq!(moved, u16::try_from(source(row))?, "feature {row}");
    }
    assert_ne!(norm[1], 1, "the shared norm moved");
    Ok(())
}
