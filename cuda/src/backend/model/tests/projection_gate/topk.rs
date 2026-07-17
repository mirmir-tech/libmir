use std::path::Path;

use mircuda::{DeviceBuffer, PinnedBuffer, bf16};
use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use runtime::backend::SamplingLogits;
use uuid::Uuid;

use crate::{
    CudaBackend, CudaConfig, CudaDenseVectorPolicy, CudaKernelAdmission, CudaMoeModelTemplate,
    CudaNumericalPolicy, CudaOutputHeadPolicy, CudaPlanningPolicy, DenseRole, Result,
};

const STEPS: usize = 64;
const TOP_K: usize = 64;

#[derive(Clone, Copy, Debug)]
struct Candidate {
    token: u32,
    logit: f32,
}

#[test]
#[allow(clippy::print_stderr)]
fn checkpoint_refined_output_preserves_top_k_frontier()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("LIBMIR_CUDA_GATE_OUTPUT_TOPK").is_none() {
        return Ok(());
    }
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let prompts = super::prompts(&layout)?;
    let baseline = run(
        &decoder,
        &catalog,
        &prompts,
        CudaOutputHeadPolicy::Bf16,
        CudaDenseVectorPolicy::Disabled,
    )?;
    for (label, output_head, dense_vectors) in [
        (
            "refined",
            CudaOutputHeadPolicy::Fp8BlockRefined,
            CudaDenseVectorPolicy::Disabled,
        ),
        (
            "refined+attention-output",
            CudaOutputHeadPolicy::Fp8BlockRefined,
            CudaDenseVectorPolicy::Role(DenseRole::AttentionOutput),
        ),
    ] {
        let candidate = run(&decoder, &catalog, &prompts, output_head, dense_vectors)?;
        let maximum_error = compare(&baseline, &candidate, label);
        eprintln!("top-{TOP_K} gate {label}: maximum logit error {maximum_error:.6}");
    }
    Ok(())
}

fn run(
    decoder: &DecoderConfig,
    catalog: &TensorCatalog,
    prompts: &[Vec<u32>],
    output_head: CudaOutputHeadPolicy,
    dense_vectors: CudaDenseVectorPolicy,
) -> Result<Vec<Vec<Vec<Candidate>>>> {
    let experimental = output_head != CudaOutputHeadPolicy::Bf16
        || dense_vectors != CudaDenseVectorPolicy::Disabled;
    let backend = CudaBackend::new(CudaConfig {
        planning: CudaPlanningPolicy {
            numerical: if experimental {
                CudaNumericalPolicy::Throughput
            } else {
                CudaNumericalPolicy::Validated
            },
            admission: if experimental {
                CudaKernelAdmission::Experimental
            } else {
                CudaKernelAdmission::Stable
            },
            output_head,
            dense_vectors,
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })?;
    let template = super::load_template(&backend, decoder, catalog)?;
    prompts
        .iter()
        .map(|prompt| sequence(&backend, &template, prompt, decoder.vocab_size))
        .collect()
}

fn sequence(
    backend: &CudaBackend,
    template: &CudaMoeModelTemplate,
    prompt: &[u32],
    vocab: usize,
) -> Result<Vec<Vec<Candidate>>> {
    let mut session = template.instantiate()?;
    let mut table = super::block_table(prompt.len())?;
    session.prefill_from(Uuid::nil(), prompt, 0, &table)?;
    let mut indices = (0..vocab).collect::<Vec<_>>();
    let mut frontiers = Vec::with_capacity(STEPS);
    for index in 0..STEPS {
        let logits = read_logits(backend, session.logits())?;
        frontiers.push(frontier(&logits, &mut indices)?);
        let selected = session.sample(SamplingLogits::None)?;
        let _token = super::read_selected(backend, selected)?;
        if index + 1 < STEPS {
            table.set_token_len(prompt.len() + index + 1);
            session.decode_sampled(Uuid::nil(), &table)?;
        }
    }
    Ok(frontiers)
}

fn frontier(logits: &[bf16], indices: &mut [usize]) -> Result<Vec<Candidate>> {
    let compare = |left: &usize, right: &usize| {
        logits[*right]
            .to_f32()
            .total_cmp(&logits[*left].to_f32())
            .then_with(|| left.cmp(right))
    };
    let _ = indices.select_nth_unstable_by(TOP_K, compare);
    indices[..TOP_K].sort_unstable_by(compare);
    indices[..TOP_K]
        .iter()
        .map(|token| {
            Ok(Candidate {
                token: u32::try_from(*token)?,
                logit: logits[*token].to_f32(),
            })
        })
        .collect()
}

fn compare(reference: &[Vec<Vec<Candidate>>], actual: &[Vec<Vec<Candidate>>], label: &str) -> f32 {
    assert_eq!(actual.len(), reference.len(), "{label} prompt count");
    let mut maximum = 0.0_f32;
    for (actual_prompt, reference_prompt) in actual.iter().zip(reference) {
        assert_eq!(actual_prompt.len(), reference_prompt.len(), "{label} step count");
        for (actual_step, reference_step) in actual_prompt.iter().zip(reference_prompt) {
            let actual_ids =
                actual_step.iter().map(|candidate| candidate.token).collect::<Vec<_>>();
            let reference_ids =
                reference_step.iter().map(|candidate| candidate.token).collect::<Vec<_>>();
            assert_eq!(actual_ids, reference_ids, "{label} changed ordered top-{TOP_K}");
            for (actual, reference) in actual_step.iter().zip(reference_step) {
                maximum = maximum.max((actual.logit - reference.logit).abs());
            }
        }
    }
    assert!(maximum <= 0.015_625, "{label} top-{TOP_K} logit error {maximum}");
    maximum
}

fn read_logits(backend: &CudaBackend, source: &DeviceBuffer<bf16>) -> Result<Vec<bf16>> {
    let mut host: PinnedBuffer<bf16> = backend.inner.context.allocate_pinned(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
