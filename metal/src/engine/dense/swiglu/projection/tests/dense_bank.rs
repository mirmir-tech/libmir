use std::fs;

use models::weights::{BindingTransform, ExpertProjectionRole, LayerTensorRole};

use super::*;

/// Per-expert BF16/F32 matrices, one stored transposed, stack into one
/// bank whose gather gives each token its own expert's product and bias.
#[test]
fn stacks_individual_dense_experts_into_a_gathered_bank() -> Result<()> {
    let root = fixture("gathered")?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let bindings = [expert(0, true, false), expert(1, true, true)];
    let refs = bindings.iter().collect::<Vec<_>>();
    let bank = BoundLinear::load_individual_bank(&tensors, &refs, &stream)?;
    let input = Array::from_f32(&[1.0, 1.0, 1.0, 1.0, 2.0, 3.0], &[2, 1, 3])?;
    let indices = Array::from_u32(&[1, 0], &[2])?;
    let output = bank.gather(&input, &indices, false, &stream)?.to_vec_f32(&stream)?;
    // Expert 1 of [1, 1, 1], then expert 0 of [1, 2, 3], each with its bias.
    assert_eq!(output, [30.0, 46.0, 24.0, 52.0]);
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A bank whose experts only partly carry a bias has no one product.
#[test]
fn refuses_a_dense_bank_partly_biased() -> Result<()> {
    let root = fixture("partly-biased")?;
    let tensors = ModelTensors::load(&root, &Stream::new_cpu()?)?;
    let stream = Stream::new_gpu()?;
    let bindings = [expert(0, true, false), expert(1, false, true)];
    let refs = bindings.iter().collect::<Vec<_>>();
    let refused = BoundLinear::load_individual_bank(&tensors, &refs, &stream);
    assert!(refused.is_err_and(|error| error.to_string().contains("bias")));
    drop(tensors);
    fs::remove_dir_all(root)?;
    Ok(())
}

fn expert(expert: usize, biased: bool, transposed: bool) -> TensorBinding {
    TensorBinding {
        role: LogicalTensorRole::Layer {
            index: 0,
            tensor: LayerTensorRole::ExpertProjection {
                expert: Some(expert),
                projection: ExpertProjectionRole::Up,
            },
        },
        source: format!("weight{expert}"),
        shape: if transposed {
            vec![3, 2]
        } else {
            vec![2, 3]
        },
        logical_shape: Some(vec![2, 3]),
        transforms: if transposed {
            vec![BindingTransform::Transpose]
        } else {
            Vec::new()
        },
        storage: TensorStorage::Dense {
            dtype: models::weights::DenseDType::F32,
            bias: biased.then(|| format!("bias{expert}")),
        },
    }
}

/// Expert 0's `[2, 3]` weight as stored; expert 1's stored `[3, 2]`.
fn fixture(name: &str) -> Result<std::path::PathBuf> {
    let root =
        std::env::temp_dir().join(format!("libmir-metal-dense-bank-{name}-{}", std::process::id()));
    fs::create_dir_all(&root)?;
    fs::write(root.join("config.json"), "{}")?;
    let tensors: [(&str, &[usize], &[f32]); 4] = [
        ("weight0", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ("weight1", &[3, 2], &[-1.0, 2.0, 0.0, 2.0, 1.0, 2.0]),
        ("bias0", &[2], &[10.0, 20.0]),
        ("bias1", &[2], &[30.0, 40.0]),
    ];
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = payload.len();
        payload.extend(values.iter().flat_map(|value| value.to_le_bytes()));
        header.insert(
            name.into(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header)?;
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut data = u64::try_from(header.len())?.to_le_bytes().to_vec();
    data.extend(header);
    data.extend(payload);
    fs::write(root.join("model.safetensors"), data)?;
    Ok(root)
}
