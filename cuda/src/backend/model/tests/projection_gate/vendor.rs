use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};

use super::run;
use crate::{
    CudaDenseVectorPolicy, CudaDenseVendorPolicy, CudaDenseWeightPolicy, CudaOutputHeadPolicy,
    ProjectionFormat,
};

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn checkpoint_vendor_dense_meets_promotion_gate() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("LIBMIR_CUDA_GATE_VENDOR_DENSE").is_none() {
        return Ok(());
    }
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
        CudaDenseVendorPolicy::Tuned,
        CudaDenseWeightPolicy::Bf16,
        ProjectionFormat::Bf16,
    )?;
    assert_sequences_equal(&baseline.sequences, &candidate.sequences);
    eprintln!(
        "vendor dense gate: candidate={:.2} baseline={:.2} tok/s across {} tokens",
        candidate.tokens as f64 / candidate.elapsed.as_secs_f64(),
        baseline.tokens as f64 / baseline.elapsed.as_secs_f64(),
        candidate.tokens,
    );
    Ok(())
}

fn assert_sequences_equal(baseline: &[Vec<u32>], candidate: &[Vec<u32>]) {
    assert_eq!(candidate.len(), baseline.len(), "vendor gate changed prompt count");
    for (prompt, (expected, actual)) in baseline.iter().zip(candidate).enumerate() {
        assert_eq!(actual.len(), expected.len(), "vendor gate changed token count");
        if let Some((token, (&expected, &actual))) =
            expected.iter().zip(actual).enumerate().find(|(_, (left, right))| left != right)
        {
            assert_eq!(actual, expected, "vendor dense changed prompt {prompt} at token {token}");
        }
    }
}
