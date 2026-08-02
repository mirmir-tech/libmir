use std::path::Path;

use mircuda::bf16;
use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

mod metrics;

use metrics::{Metrics, RANK, ratio};

use super::{block_table, load_template_with_format, prompts, read_selected};
use crate::{
    CudaBackend, CudaConfig, CudaDenseVendorPolicy, CudaDenseWeightPolicy, CudaKernelAdmission,
    CudaMoeModelTemplate, CudaNumericalPolicy, CudaOutputHeadPolicy, CudaPlanningPolicy, DenseRole,
    ProjectionFormat, Result,
};

const STEPS: usize = 64;

struct Trace {
    tokens: Vec<u32>,
    logits: Vec<Vec<bf16>>,
}

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn checkpoint_throughput_quality_report() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(mode) = std::env::var_os("LIBMIR_CUDA_GATE_THROUGHPUT_QUALITY") else {
        return Ok(());
    };
    let root = std::env::var_os("LIBMIR_CUDA_QUALITY_MODEL")
        .or_else(|| std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL"));
    let Some(root) = root else {
        return Ok(());
    };
    let mode = mode.to_str().ok_or("throughput quality mode is not UTF-8")?;
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let prompts = prompts(&layout)?;
    let reference = trace(&decoder, &catalog, &prompts, stable())?;
    let metrics = compare(&decoder, &catalog, &prompts, &reference, candidate(mode)?)?;
    assert!(metrics.squared_error.is_finite());
    assert!(metrics.kl_divergence.is_finite());
    eprintln!(
        "throughput quality {mode}: steps={} top1={:.3}% top{RANK}_overlap={:.3}% \
         nrmse={:.6} max_abs={:.6} mean_kl={:.6}",
        metrics.steps,
        ratio(metrics.top1, metrics.steps),
        ratio(metrics.topk_overlap, metrics.steps * RANK),
        (metrics.squared_error / metrics.squared_reference.max(f64::EPSILON)).sqrt(),
        metrics.maximum_error,
        metrics.kl_divergence / metrics.steps as f64,
    );
    metrics.validate(mode);
    Ok(())
}

fn trace(
    decoder: &DecoderConfig,
    catalog: &TensorCatalog,
    prompts: &[Vec<u32>],
    planning: CudaPlanningPolicy,
) -> Result<Vec<Trace>> {
    let backend = CudaBackend::new(CudaConfig { planning, ..CudaConfig::default() })?;
    let template = load_template_with_format(&backend, decoder, catalog, ProjectionFormat::Bf16)?;
    prompts.iter().map(|prompt| trace_prompt(&backend, &template, prompt)).collect()
}

fn trace_prompt(
    backend: &CudaBackend,
    template: &CudaMoeModelTemplate,
    prompt: &[u32],
) -> Result<Trace> {
    let mut session = template.instantiate()?;
    let mut table = block_table(prompt.len())?;
    session.prefill_from(Uuid::nil(), prompt, 0, &table)?;
    let mut tokens = Vec::with_capacity(STEPS);
    let mut logits = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        logits.push(super::super::read(backend, session.logits())?);
        let token = read_selected(backend, session.sample(SamplingLogits::None)?)?;
        tokens.push(token);
        if step + 1 < STEPS {
            table.set_token_len(prompt.len() + step + 1);
            session.decode(Uuid::nil(), token, &table)?;
        }
    }
    Ok(Trace { tokens, logits })
}

fn compare(
    decoder: &DecoderConfig,
    catalog: &TensorCatalog,
    prompts: &[Vec<u32>],
    reference: &[Trace],
    planning: CudaPlanningPolicy,
) -> Result<Metrics> {
    let backend = CudaBackend::new(CudaConfig { planning, ..CudaConfig::default() })?;
    let template = load_template_with_format(&backend, decoder, catalog, ProjectionFormat::Bf16)?;
    let mut metrics = Metrics::default();
    for (prompt, reference) in prompts.iter().zip(reference) {
        compare_prompt(&backend, &template, prompt, reference, &mut metrics)?;
    }
    Ok(metrics)
}

fn compare_prompt(
    backend: &CudaBackend,
    template: &CudaMoeModelTemplate,
    prompt: &[u32],
    reference: &Trace,
    metrics: &mut Metrics,
) -> Result<()> {
    let mut session = template.instantiate()?;
    let mut table = block_table(prompt.len())?;
    session.prefill_from(Uuid::nil(), prompt, 0, &table)?;
    for (step, (token, expected)) in reference.tokens.iter().zip(&reference.logits).enumerate() {
        let actual = super::super::read(backend, session.logits())?;
        metrics.observe(expected, &actual);
        if step + 1 < STEPS {
            table.set_token_len(prompt.len() + step + 1);
            session.decode(Uuid::nil(), *token, &table)?;
        }
    }
    Ok(())
}

fn stable() -> CudaPlanningPolicy {
    CudaPlanningPolicy {
        output_head: CudaOutputHeadPolicy::Bf16,
        ..CudaPlanningPolicy::default()
    }
}

fn candidate(mode: &str) -> std::result::Result<CudaPlanningPolicy, &'static str> {
    let dense_weights = match mode {
        "block-fp8-gate-up" | "throughput" => {
            CudaDenseWeightPolicy::BlockFp8Role(DenseRole::DenseGateUp)
        },
        "fp8-int4-gate-up" => CudaDenseWeightPolicy::Fp8Int4Role(DenseRole::DenseGateUp),
        _ => return Err("invalid throughput quality mode"),
    };
    Ok(CudaPlanningPolicy {
        numerical: CudaNumericalPolicy::Throughput,
        admission: CudaKernelAdmission::Experimental,
        dense_vendor: if mode == "throughput" {
            CudaDenseVendorPolicy::Tuned
        } else {
            CudaDenseVendorPolicy::Disabled
        },
        dense_weights,
        output_head: CudaOutputHeadPolicy::Bf16,
        ..CudaPlanningPolicy::default()
    })
}
