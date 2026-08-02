use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};

use super::run;
use crate::{
    CudaDenseVectorPolicy, CudaDenseVendorPolicy, CudaDenseWeightPolicy, CudaOutputHeadPolicy,
    DenseRole, ProjectionFormat,
};

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn checkpoint_dense_weight_meets_promotion_gate() -> Result<(), Box<dyn std::error::Error>> {
    let Some(policy) = candidate_policy()? else {
        return Ok(());
    };
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(std::path::Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let prompts = super::prompts(&layout)?;
    let baseline = run(
        &decoder,
        &catalog,
        &prompts,
        CudaOutputHeadPolicy::Bf16,
        CudaDenseVectorPolicy::Disabled,
        CudaDenseVendorPolicy::Disabled,
        CudaDenseWeightPolicy::Bf16,
        ProjectionFormat::Bf16,
    )?;
    let candidate = run(
        &decoder,
        &catalog,
        &prompts,
        CudaOutputHeadPolicy::Bf16,
        CudaDenseVectorPolicy::Disabled,
        CudaDenseVendorPolicy::Disabled,
        policy,
        ProjectionFormat::Bf16,
    )?;
    assert_sequences_equal(&baseline.sequences, &candidate.sequences, policy);
    eprintln!(
        "dense weight gate {policy:?}: candidate={:.2} baseline={:.2} tok/s across {} tokens",
        candidate.tokens as f64 / candidate.elapsed.as_secs_f64(),
        baseline.tokens as f64 / baseline.elapsed.as_secs_f64(),
        candidate.tokens,
    );
    Ok(())
}

fn assert_sequences_equal(
    baseline: &[Vec<u32>],
    candidate: &[Vec<u32>],
    policy: CudaDenseWeightPolicy,
) {
    assert_eq!(candidate.len(), baseline.len(), "{policy:?} changed prompt count");
    for (prompt, (expected, actual)) in baseline.iter().zip(candidate).enumerate() {
        assert_eq!(actual.len(), expected.len(), "{policy:?} changed token count");
        if let Some((token, (&expected, &actual))) =
            expected.iter().zip(actual).enumerate().find(|(_, (left, right))| left != right)
        {
            assert_eq!(actual, expected, "{policy:?} changed prompt {prompt} at token {token}");
        }
    }
}

fn candidate_policy() -> Result<Option<CudaDenseWeightPolicy>, Box<dyn std::error::Error>> {
    let Some(value) = std::env::var_os("LIBMIR_CUDA_GATE_DENSE_WEIGHT") else {
        return Ok(None);
    };
    match value.to_str() {
        Some("block-fp8-gate-up") => {
            Ok(Some(CudaDenseWeightPolicy::BlockFp8Role(DenseRole::DenseGateUp)))
        },
        Some("fp8-int4-gate-up") => {
            Ok(Some(CudaDenseWeightPolicy::Fp8Int4Role(DenseRole::DenseGateUp)))
        },
        _ => Err("invalid LIBMIR_CUDA_GATE_DENSE_WEIGHT".into()),
    }
}
