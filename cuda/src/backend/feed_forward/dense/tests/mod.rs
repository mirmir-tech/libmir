mod scratch;

use models::weights::TensorInfo;

use super::*;
use crate::CudaConfig;

fn template() -> Result<DenseFeedForward> {
    let backend = CudaBackend::new(CudaConfig::default())?;
    let path = std::env::temp_dir().join(format!("dense-mixed-{}.bin", uuid::Uuid::new_v4()));
    let values = [2.0_f32, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0, 1.0, 0.25, -0.5, 1.0];
    let bytes = values
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(&path, bytes)?;
    let mut upload = backend.begin_tensor_upload();
    for (index, name) in ["gate", "up", "down"].into_iter().enumerate() {
        let offset = u64::try_from(index)? * 8;
        upload.enqueue(&TensorInfo {
            name: name.into(),
            file: path.clone(),
            dtype: "BF16".into(),
            shape: vec![2, 2],
            data_start: 0,
            data_offsets: [offset, offset + 8],
        })?;
    }
    let tensors = upload.finish()?;
    std::fs::remove_file(path)?;
    let weight = |name: &str| {
        tensors
            .get(name)
            .cloned()
            .map(CheckpointProjectionWeight::Dense)
            .ok_or_else(|| Error::MissingTensor(name.into()))
    };
    let decoder = DecoderConfig::from_value(
        &serde_json::json!({"hidden_size":2, "intermediate_size":2, "num_hidden_layers":1, "num_attention_heads":1, "num_key_value_heads":1, "vocab_size":2, "hidden_act":"silu"}),
    )?;
    Ok(DenseFeedForward {
        backend,
        hidden: 2,
        intermediate: 2,
        activation: GatedActivation::try_from(&decoder)?,
        gate: weight("gate")?,
        up: weight("up")?,
        down: weight("down")?,
    })
}

#[test]
fn dense_swiglu_matches_nonzero_reference_for_decode_and_prefill() -> Result<()> {
    let template = template()?;
    let backend = &template.backend;
    for tokens in [1, 3] {
        let values = [0.5_f32, -1.0, 2.0, -0.5, 1.5, 0.25];
        let values = &values[..2 * tokens];
        let mut host = backend.inner.context.allocate_pinned::<bf16>(values.len())?;
        host.copy_from_slice(&values.iter().copied().map(bf16::from_f32).collect::<Vec<_>>())?;
        let mut input = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
        backend.inner.stream.copy_to_device(&mut host, &mut input)?;
        let mut output = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
        template.prepare(tokens)?.execute(&input, &mut output)?;
        backend.inner.stream.copy_to_host(&output, &mut host)?;
        for (actual, input) in
            host.to_vec()?.as_chunks::<2>().0.iter().zip(values.as_chunks::<2>().0)
        {
            let gated =
                |gate: f32, up: f32| bf16::from_f32(gate / (1.0 + (-gate).exp()) * up).to_f32();
            let first = gated(2.0 * input[0], input[0]);
            let second = gated(0.5 * input[1], input[1]);
            let expected = [0.25_f32.mul_add(second, first), (-0.5_f32).mul_add(first, second)];
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual.to_f32() - expected).abs() < 0.04,
                    "actual={actual:?}, expected={expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn dense_residual_shifted_norm_matches_nonzero_reference() -> Result<()> {
    use crate::{backend::feed_forward::FeedForwardExecution, kernels::ShiftedRmsNorm};
    let template = template()?;
    let backend = &template.backend;
    let round = |value: f32| bf16::from_f32(value).to_f32();
    let inputs = [0.5_f32, -1.0, 2.0, -0.5, 1.5, 0.25];
    let updates = [0.25_f32, 0.125, -0.5, 0.25, 0.75, -0.5];
    let upload = |values: &[f32]| -> Result<DeviceBuffer<bf16>> {
        let mut host = backend.inner.context.allocate_pinned(values.len())?;
        host.copy_from_slice(&values.iter().copied().map(bf16::from_f32).collect::<Vec<_>>())?;
        let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
        backend.inner.stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    };
    let weight = upload(&[0.25, -0.5])?;
    for tokens in [1, 3] {
        let elements = 2 * tokens;
        let input = upload(&inputs[..elements])?;
        let update = upload(&updates[..elements])?;
        let allocate = || backend.inner.pool.allocate(&backend.inner.stream, elements);
        let mut residual = allocate()?;
        let mut normalized = allocate()?;
        let mut ff_output = allocate()?;
        let mut output = allocate()?;
        let norm = ShiftedRmsNorm::compile(&backend.inner.compiler, tokens, 2, 1e-6, 1.0)?;
        let add = ElementwiseBf16::compile(&backend.inner.compiler, elements)?;
        let mut execution = FeedForwardExecution::Dense(Box::new(template.prepare(tokens)?));
        execution.execute_residual_norm(
            &norm, &input, &update, &weight, &mut residual, &mut normalized, &mut ff_output,
            &mut output, &add,
        )?;
        let mut host = backend.inner.context.allocate_pinned(elements)?;
        backend.inner.stream.copy_to_host(&output, &mut host)?;
        let actual = host.to_vec()?;
        for row in 0..tokens {
            let offset = 2 * row;
            let a = round(inputs[offset] + updates[offset]);
            let b = round(inputs[offset + 1] + updates[offset + 1]);
            let inverse_rms = (f32::midpoint(a * a, b * b) + 1e-6).sqrt().recip();
            let x = round(a * inverse_rms * 1.25);
            let y = round(b * inverse_rms * 0.5);
            let silu = |value: f32| value / (1.0 + (-value).exp());
            let gate = round(silu(round(2.0 * x)) * x);
            let up = round(silu(round(0.5 * y)) * y);
            let expected = [
                round(a + round(0.25_f32.mul_add(up, gate))),
                round(b + round((-0.5_f32).mul_add(gate, up))),
            ];
            for (actual, expected) in actual[offset..offset + 2].iter().zip(expected) {
                assert!(
                    (actual.to_f32() - expected).abs() < 0.04,
                    "actual={actual:?}, expected={expected}"
                );
            }
        }
    }
    Ok(())
}
