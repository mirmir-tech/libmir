use std::path::Path;

use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use runtime::{
    backend::SamplingLogits,
    kv::{BlockId, BlockTable, CacheConfig},
};
use uuid::Uuid;

use crate::{
    CudaBackend, CudaConfig, CudaDenseVectorPolicy, CudaDenseWeightPolicy, CudaKernelAdmission,
    CudaModelSessionConfig, CudaMoeBatchPolicy, CudaMoeFusionPolicy, CudaMoeModelTemplate,
    CudaNumericalPolicy, CudaOutputHeadPolicy, CudaPlanningPolicy, NvFp4MoeLayerLoadConfig, Result,
};

mod assertions;
mod batch;
mod long_prefill;
mod long_profile;
mod policy;
mod profile;
mod projection_gate;
mod reference;
use assertions::assert_logits_close;

#[test]
fn checkpoint_model_decodes() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let fused = std::env::var_os("LIBMIR_CUDA_PROFILE_FUSED_MOE").is_some();
    let hybrid = std::env::var_os("LIBMIR_CUDA_PROFILE_HYBRID_MOE").is_some();
    let output_policy = output_policy();
    let fp8_output = output_policy != CudaOutputHeadPolicy::Bf16;
    let dense_role = dense_role()?;
    let dense_weights = policy::dense_weight()?;
    let dense_vectors =
        std::env::var_os("LIBMIR_CUDA_PROFILE_DENSE_VECTORS").is_some() || dense_role.is_some();
    let dense_policy = dense_role.map_or(
        if dense_vectors {
            CudaDenseVectorPolicy::Tuned
        } else {
            CudaDenseVectorPolicy::Disabled
        },
        CudaDenseVectorPolicy::Role,
    );
    let planning = if fused
        || hybrid
        || dense_vectors
        || fp8_output
        || dense_weights != CudaDenseWeightPolicy::Bf16
    {
        CudaPlanningPolicy {
            numerical: CudaNumericalPolicy::Throughput,
            admission: CudaKernelAdmission::Experimental,
            dense_vectors: dense_policy,
            dense_weights,
            moe_fusion: if fused {
                CudaMoeFusionPolicy::Tuned
            } else {
                CudaMoeFusionPolicy::Disabled
            },
            moe_batch: if hybrid {
                CudaMoeBatchPolicy::W4A4Hybrid
            } else {
                CudaMoeBatchPolicy::Auto
            },
            output_head: output_policy,
            ..CudaPlanningPolicy::default()
        }
    } else {
        CudaPlanningPolicy::default()
    };
    let backend = CudaBackend::new(CudaConfig { planning, ..CudaConfig::default() })?;
    let template = template(&backend, &decoder, &catalog)?;
    if std::env::var_os("LIBMIR_CUDA_REFERENCE_GREEDY").is_some() {
        reference::print_greedy_sequence(&backend, &template, dense_vectors, output_policy)?;
    }
    let mut session = template.instantiate()?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(1));
    table.set_token_len(1);
    let logits = session.decode(Uuid::nil(), 2, &table)?;
    let first = read(&backend, logits)?;
    valid(&first, decoder.vocab_size);
    table.set_token_len(2);
    let logits = session.decode(Uuid::nil(), 3, &table)?;
    let second = read(&backend, logits)?;
    valid(&second, decoder.vocab_size);
    assert!(first.iter().zip(&second).any(|(left, right)| left != right));
    session.sample(SamplingLogits::None)?;
    table.set_token_len(3);
    let logits = session.decode_sampled(Uuid::nil(), &table)?;
    let third = read(&backend, logits)?;
    valid(&third, decoder.vocab_size);
    assert!(second.iter().zip(&third).any(|(left, right)| left != right));
    drop(session);

    let prompt = (2_u32..18).collect::<Vec<_>>();
    let mut sequential = template.instantiate()?;
    let mut expected = Vec::new();
    for (index, token) in prompt.iter().copied().enumerate() {
        table.set_token_len(index + 1);
        expected = read(&backend, sequential.decode(Uuid::nil(), token, &table)?)?;
    }
    let profiling = std::env::var_os("LIBMIR_CUDA_PROFILE_PREFILL").is_some();
    let prefill_chunk_tokens = if profiling {
        prompt.len()
    } else {
        8
    };
    let mut prefill =
        template.instantiate_with_config(CudaModelSessionConfig { prefill_chunk_tokens })?;
    table.set_token_len(prompt.len());
    let logits = prefill.prefill_from(Uuid::nil(), &prompt, 0, &table)?;
    let prefetched = read(&backend, logits)?;
    if profiling {
        profile::run(&backend, &mut sequential, &mut prefill, &prompt, &mut table)?;
    }
    let maximum_rmse = [0.1, 0.35][usize::from(dense_vectors)];
    assert_logits_close(&prefetched, &expected, maximum_rmse);
    Ok(())
}

#[test]
fn checkpoint_model_admits_experimental_decode_plans()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some(root) = std::env::var_os("LIBMIR_CUDA_NVFP4_MODEL") else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let stable_backend = CudaBackend::new(CudaConfig::default())?;
    let stable_template = template(&stable_backend, &decoder, &catalog)?;
    let mut stable = stable_template.instantiate()?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(1));
    table.set_token_len(1);
    let expected = read(&stable_backend, stable.decode(Uuid::nil(), 2, &table)?)?;
    drop(stable);
    drop(stable_template);
    let backend = CudaBackend::new(CudaConfig {
        planning: CudaPlanningPolicy {
            numerical: CudaNumericalPolicy::Throughput,
            admission: CudaKernelAdmission::Experimental,
            output_head: experimental_output_policy(),
            ..CudaPlanningPolicy::default()
        },
        ..CudaConfig::default()
    })?;
    let template = template(&backend, &decoder, &catalog)?;
    let mut session = template.instantiate()?;
    let logits = session.decode(Uuid::nil(), 2, &table)?;
    let actual = read(&backend, logits)?;
    valid(&actual, decoder.vocab_size);
    assert_logits_close(&actual, &expected, 0.15);
    Ok(())
}

pub(super) fn template(
    backend: &CudaBackend,
    decoder: &DecoderConfig,
    catalog: &TensorCatalog,
) -> Result<CudaMoeModelTemplate> {
    backend.load_nvfp4_moe_model_template(
        decoder,
        catalog,
        NvFp4MoeLayerLoadConfig {
            cache: CacheConfig {
                block_size: 16,
                block_count: 2,
                dtype: policy::cache_dtype(),
            },
            max_sequence_blocks: 2,
        },
    )
}

fn dense_role() -> std::result::Result<Option<crate::DenseRole>, Box<dyn std::error::Error>> {
    let Some(role) = std::env::var_os("LIBMIR_CUDA_PROFILE_DENSE_ROLE") else {
        return Ok(None);
    };
    match role.to_str() {
        Some("qkv") => Ok(Some(crate::DenseRole::AttentionQkv)),
        Some("attention-output") => Ok(Some(crate::DenseRole::AttentionOutput)),
        Some("gate-up") => Ok(Some(crate::DenseRole::DenseGateUp)),
        _ => Err("invalid LIBMIR_CUDA_PROFILE_DENSE_ROLE".into()),
    }
}

fn output_policy() -> CudaOutputHeadPolicy {
    if std::env::var_os("LIBMIR_CUDA_PROFILE_FP8_REFINED").is_some() {
        CudaOutputHeadPolicy::Fp8BlockRefined
    } else if std::env::var_os("LIBMIR_CUDA_PROFILE_FP8_BLOCK_VECTOR").is_some() {
        CudaOutputHeadPolicy::Fp8BlockVectorized
    } else if std::env::var_os("LIBMIR_CUDA_PROFILE_FP8_RESIDUAL").is_some() {
        CudaOutputHeadPolicy::Fp8Residual
    } else if std::env::var_os("LIBMIR_CUDA_PROFILE_FP8_OUTPUT").is_some() {
        CudaOutputHeadPolicy::Fp8Vectorized
    } else {
        CudaOutputHeadPolicy::Bf16
    }
}

fn experimental_output_policy() -> CudaOutputHeadPolicy {
    match output_policy() {
        CudaOutputHeadPolicy::Auto | CudaOutputHeadPolicy::Bf16 => {
            CudaOutputHeadPolicy::Fp8Residual
        },
        policy => policy,
    }
}

fn valid(values: &[mircuda::bf16], vocab: usize) {
    assert_eq!(values.len(), vocab);
    assert!(values.iter().all(|value| value.to_f32().is_finite()));
    assert!(values.iter().any(|value| value.to_f32() != 0.0));
}

pub(super) fn read(
    backend: &CudaBackend,
    source: &mircuda::DeviceBuffer<mircuda::bf16>,
) -> Result<Vec<mircuda::bf16>> {
    let mut host = backend.inner.context.allocate_pinned::<mircuda::bf16>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}
