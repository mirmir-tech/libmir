//! mlx-community's GPT-OSS in MXFP4 as the Hub serves it: the
//! configuration declares the experts' `mxfp4` mode, and their scales are
//! the U8 exponents of MXFP4 blocks. The remote contract binds them as a
//! downloaded checkpoint's are, not as grouped-affine scales.

use super::*;

#[test]
fn a_hub_mlx_mxfp4_checkpoint_binds_its_experts_as_mxfp4() -> Result<()> {
    let mut config = config();
    config["quantization"] = json!({ "group_size": 32, "bits": 4, "mode": "mxfp4" });
    let mut tensors = catalog("BF16", "BF16", "model.safetensors").tensors;
    tensors.retain(|tensor| !tensor.name.contains(".mlp.experts."));
    for projection in ["gate", "up", "down"] {
        let prefix = format!("model.layers.0.mlp.experts.{projection}_proj");
        for (name, shape, dtype) in [
            (format!("{prefix}.weight"), vec![2, 32, 4], "U32"),
            (format!("{prefix}.scales"), vec![2, 32, 1], "U8"),
            (format!("{prefix}.bias"), vec![2, 32], "BF16"),
        ] {
            let mut tensor = tensor_with_dtype(&name, shape, dtype);
            tensor.file = "model.safetensors".into();
            tensors.push(tensor);
        }
    }
    let contract = RemoteModelContract::inspect_generation(&config, &TensorCatalog::new(tensors))?;
    let metal = contract.admission(BackendTarget::Metal);
    // MXFP4 runs for selected compositions, GPT-OSS's among them: partial,
    // as for OpenAI's checkpoint, not refused.
    assert_eq!(metal.status, AdmissionStatus::Partial, "{metal:?}");
    assert!(metal.checks.iter().all(|check| check.status() != AdmissionStatus::Unsupported));
    assert!(
        metal
            .checks
            .iter()
            .any(|check| check.kind() == AdmissionCheckKind::Encoding(EncodingKind::MxFp4))
    );
    Ok(())
}
