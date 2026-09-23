use super::{super::super::*, allocate, copy, read};
use crate::{CudaConfig, CudaTensor, backend::linear::CheckpointProjectionWeight};

#[test]
fn packed_bf16_qkv_gate_preserves_prefill_and_decode() -> Result<()> {
    for key_dim in [32, 128] {
        compare(key_dim)?;
    }
    Ok(())
}

pub(super) fn dense_weights(
    backend: &CudaBackend,
    key_dim: usize,
) -> Result<(AffineGatedDeltaLayerConfig, AffineGatedDeltaLayerWeights)> {
    let config = AffineGatedDeltaLayerConfig {
        hidden_size: 128,
        key_heads: 1,
        value_heads: 2,
        key_dim,
        value_dim: 128,
        convolution_kernel_size: 4,
        group_size: 0,
        bits: 0,
        rms_norm_epsilon: 1.0e-6,
        norm_weight_shift: 0.0,
    };
    let tensor = |name: &str, shape: Vec<usize>, seed: usize| {
        let values = pattern(shape.iter().product(), seed);
        Ok::<_, crate::Error>(CudaTensor::from_bf16(name.into(), shape, copy(backend, &values)?))
    };
    let dense = |name, rows, seed| {
        tensor(name, vec![rows, config.hidden_size], seed).map(CheckpointProjectionWeight::Dense)
    };
    let weights = AffineGatedDeltaLayerWeights {
        qkv: dense("qkv", config.mixed_width()?, 1)?,
        gate: dense("gate", config.value_width()?, 3)?,
        alpha: dense("alpha", config.value_heads, 5)?,
        beta: dense("beta", config.value_heads, 7)?,
        output: CheckpointProjectionWeight::Dense(tensor(
            "output",
            vec![config.hidden_size, config.value_width()?],
            9,
        )?),
        convolution: tensor("conv", vec![config.mixed_width()?, 4, 1], 11)?,
        norm: tensor("norm", vec![128], 13)?,
        a_log: tensor("a", vec![2], 15)?,
        dt_bias: tensor("dt", vec![2], 17)?,
        identity: super::super::weights::next_layer_identity(),
    };
    Ok((config, weights))
}

fn compare(key_dim: usize) -> Result<()> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let (config, weights) = dense_weights(&backend, key_dim)?;
    let original_qkv = read(&backend, buffer(Some(&weights.qkv))?)?;
    let original_gate = read(&backend, buffer(Some(&weights.gate))?)?;
    let packed = CudaAffineGatedDeltaLayer::new(&backend, config, weights)?;
    assert!(packed.packed_qkv_gate.is_some());
    let combined = buffer(packed.packed_qkv_gate.as_ref())?;
    assert_eq!(read(&backend, combined)?, [original_qkv, original_gate].concat());
    let mut separate = packed.clone();
    separate.packed_qkv_gate = None;
    let mut actual_state = packed.prepare_state()?;
    let mut expected_state = separate.prepare_state()?;
    for tokens in [33, 5, 1, 1] {
        let input = copy(&backend, &pattern(tokens * config.hidden_size, tokens))?;
        let mut actual = allocate(&backend, tokens * config.hidden_size)?;
        let mut expected = allocate(&backend, tokens * config.hidden_size)?;
        packed.prepare(tokens)?.execute(&input, &mut actual_state, &mut actual)?;
        separate.prepare(tokens)?.execute(&input, &mut expected_state, &mut expected)?;
        let actual = read(&backend, &actual)?;
        let expected = read(&backend, &expected)?;
        assert!(expected.iter().any(|value| value.to_f32().abs() > 0.0));
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual.to_f32() - expected.to_f32()).abs() <= 0.001,
                "key_dim={key_dim} tokens={tokens}: {actual:?} != {expected:?}"
            );
        }
        assert_eq!(actual_state.offset(), expected_state.offset());
    }
    Ok(())
}

pub(super) fn pattern(elements: usize, seed: usize) -> Vec<bf16> {
    (0..elements)
        .map(|index| {
            let value = i16::try_from((index * 17 + seed) % 31).unwrap_or_default() - 15;
            bf16::from_f32(f32::from(value) / 64.0)
        })
        .collect()
}

fn buffer(weight: Option<&CheckpointProjectionWeight>) -> Result<&DeviceBuffer<bf16>> {
    weight
        .and_then(CheckpointProjectionWeight::dense_bf16)
        .and_then(CudaTensor::as_bf16)
        .ok_or(crate::Error::InvalidExecutionPlan("expected dense test weight"))
}
