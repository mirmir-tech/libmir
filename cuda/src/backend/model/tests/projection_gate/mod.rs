use std::{
    path::Path,
    time::{Duration, Instant},
};

use foundation::protocol::{ChatCompletionRequest, ChatMessage};
use mircuda::PinnedBuffer;
use models::{
    chat::ChatTemplate,
    layout::{DecoderConfig, ModelLayout},
    semantic::SemanticModelSpec,
    tokenizer::TextTokenizer,
    weights::TensorCatalog,
};
use runtime::{
    backend::SamplingLogits,
    kv::{BlockId, BlockTable, CacheConfig},
};
use uuid::Uuid;

use crate::{
    CudaBackend, CudaConfig, CudaDenseVectorPolicy, CudaKernelAdmission, CudaMoeModelTemplate,
    CudaNumericalPolicy, CudaOutputHeadPolicy, CudaPlanningPolicy, DenseRole,
    DenseSwiGluLayerLoadConfig, NvFp4MoeLayerLoadConfig, ProjectionFormat, Result,
};

mod topk;

const GENERATED: usize = 128;
const BLOCKS: usize = 64;

#[test]
#[allow(clippy::cast_precision_loss, clippy::print_stderr)]
fn checkpoint_output_projections_preserve_broad_greedy_sequences()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("LIBMIR_CUDA_GATE_OUTPUT_HEAD").is_none() {
        return Ok(());
    }
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let prompts = prompts(&layout)?;
    let baseline = run(
        &decoder,
        &catalog,
        &prompts,
        CudaOutputHeadPolicy::Bf16,
        CudaDenseVectorPolicy::Disabled,
    )?;
    for policy in [CudaOutputHeadPolicy::Fp8Residual, CudaOutputHeadPolicy::Fp8BlockRefined] {
        let candidate = run(&decoder, &catalog, &prompts, policy, CudaDenseVectorPolicy::Disabled)?;
        assert_eq!(candidate.sequences, baseline.sequences, "{policy:?} changed greedy output");
        eprintln!(
            "output gate {policy:?}: {:.2} tok/s across {} tokens",
            candidate.tokens as f64 / candidate.elapsed.as_secs_f64(),
            candidate.tokens,
        );
    }
    let combined = run(
        &decoder,
        &catalog,
        &prompts,
        CudaOutputHeadPolicy::Fp8BlockRefined,
        CudaDenseVectorPolicy::Role(DenseRole::AttentionOutput),
    )?;
    assert_eq!(
        combined.sequences, baseline.sequences,
        "combined projection plan changed greedy output"
    );
    eprintln!(
        "output gate refined+attention-output: {:.2} tok/s across {} tokens",
        combined.tokens as f64 / combined.elapsed.as_secs_f64(),
        combined.tokens,
    );
    eprintln!(
        "output gate Bf16: {:.2} tok/s across {} tokens",
        baseline.tokens as f64 / baseline.elapsed.as_secs_f64(),
        baseline.tokens,
    );
    Ok(())
}

struct Report {
    sequences: Vec<Vec<u32>>,
    tokens: usize,
    elapsed: Duration,
}

fn run(
    decoder: &DecoderConfig,
    catalog: &TensorCatalog,
    prompts: &[Vec<u32>],
    output_head: CudaOutputHeadPolicy,
    dense_vectors: CudaDenseVectorPolicy,
) -> Result<Report> {
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
    let template = load_template(&backend, decoder, catalog)?;
    let mut elapsed = Duration::ZERO;
    let mut sequences = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let mut session = template.instantiate()?;
        let mut table = block_table(prompt.len())?;
        session.prefill_from(Uuid::nil(), prompt, 0, &table)?;
        let started = Instant::now();
        let mut generated = Vec::with_capacity(GENERATED);
        for index in 0..GENERATED {
            let selected = session.sample(SamplingLogits::None)?;
            generated.push(read_selected(&backend, selected)?);
            if index + 1 < GENERATED {
                table.set_token_len(prompt.len() + index + 1);
                session.decode_sampled(Uuid::nil(), &table)?;
            }
        }
        elapsed += started.elapsed();
        sequences.push(generated);
    }
    Ok(Report {
        tokens: GENERATED * prompts.len(),
        sequences,
        elapsed,
    })
}

pub(super) fn load_template(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    catalog: &TensorCatalog,
) -> Result<CudaMoeModelTemplate> {
    let cache = CacheConfig::new(u32::try_from(BLOCKS)?);
    let semantic = SemanticModelSpec::discover(decoder, catalog)?;
    let plan = crate::engine::lowering::CudaDecoderPlan::lower(&semantic);
    if plan.all_dense_and_routed() {
        backend.load_nvfp4_moe_model_template(
            decoder,
            catalog,
            NvFp4MoeLayerLoadConfig { cache, max_sequence_blocks: BLOCKS },
        )
    } else if plan.all_dense() {
        let mut ignored = |_completed, _detail| {};
        backend.load_dense_swiglu_model_template_with_progress(
            decoder,
            catalog,
            DenseSwiGluLayerLoadConfig {
                cache,
                max_sequence_blocks: BLOCKS,
                qkv_normalization: plan.graph_normalization()?,
                projection_format: ProjectionFormat::NvFp4,
            },
            &mut ignored,
        )
    } else {
        Err(crate::Error::MissingCapability {
            operation: "projection-gate graph decoder",
            storage: "NVFP4 bindings".into(),
            geometry: format!("layers={}", plan.layers().len()),
            requirement: "the test admits dense or dense-plus-routed semantic layers",
        })
    }
}

pub(super) fn prompts(layout: &ModelLayout) -> Result<Vec<Vec<u32>>> {
    let template = ChatTemplate::from_layout(layout)?;
    let tokenizer = TextTokenizer::from_layout(layout)?;
    [
        "Hello. Briefly introduce yourself and explain what you can help with.",
        "Explain mixture-of-experts routing, load balancing, and inference trade-offs.",
        "Write a safe Rust function that parses a port number and explain its error handling.",
    ]
    .into_iter()
    .map(|content| {
        let prompt = template.render(&request(content))?;
        Ok(tokenizer
            .encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?
            .token_ids)
    })
    .collect()
}

fn request(content: &str) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "projection-gate".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: content.into(),
            reasoning_content: None,
        }],
        stream: false,
        max_tokens: Some(GENERATED),
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: Some(0),
    }
}

fn block_table(prompt_tokens: usize) -> Result<BlockTable> {
    let mut table = BlockTable::with_block_size(16);
    for block in 0..BLOCKS {
        table.push(BlockId(u32::try_from(block)?));
    }
    table.set_token_len(prompt_tokens);
    Ok(table)
}

fn read_selected(backend: &CudaBackend, selected: &mircuda::DeviceBuffer<u32>) -> Result<u32> {
    let mut host: PinnedBuffer<u32> = backend.inner.context.allocate_pinned(1)?;
    backend.inner.stream.copy_to_host(selected, &mut host)?;
    Ok(host.to_vec()?[0])
}
