//! Shared prefill plans across layers whose weights have different layouts.

use std::num::NonZeroU16;

use mircuda::{DeviceBuffer, bf16};
use runtime::kv::{BlockId, BlockTable, CacheConfig, KvCacheDType, KvStorageSpec};

use super::{DecoderLayerTemplate, SessionLayer, SharedLayerPrefill};
use crate::{
    CudaBackend, CudaTensor, DenseSwiGluConfig, DenseSwiGluLayerTemplate, GatedActivation,
    ProjectionFormat, Result,
    backend::linear::ggml_tests::{copy, input, read},
    kernels::QkvNormalization,
};

mod affine;
mod per_layer;

const HIDDEN: usize = 64;
const TOKENS: usize = 4;

fn tensor(backend: &CudaBackend, name: &str, shape: &[usize], seed: usize) -> Result<CudaTensor> {
    let values = input(shape.iter().product::<usize>() + seed);
    Ok(CudaTensor::from_bf16(
        name.into(),
        shape.to_vec(),
        copy(backend, &values[seed..])?,
    ))
}

/// A session layer of `template`, with a per-layer input when it gates one.
fn session(backend: &CudaBackend, template: DenseSwiGluLayerTemplate) -> Result<SessionLayer> {
    let config = template.config();
    let activations = backend.pool().allocate_zeroed::<bf16>(backend.stream(), HIDDEN)?;
    let cache = backend.prepare_paged_kv(config.attention.layer, config.attention.cache)?;
    let per_layer = config
        .per_layer_input
        .map(|_| backend.pool().allocate_zeroed::<bf16>(backend.stream(), HIDDEN))
        .transpose()?;
    DecoderLayerTemplate::Dense(Box::new(template))
        .instantiate(&activations, &activations, cache, per_layer)
}

/// Runs `layer` through a shared prefill `plan` and reads its output.
fn shared_prefill(
    backend: &CudaBackend,
    plan: &mut SharedLayerPrefill,
    layer: &mut SessionLayer,
) -> Result<Vec<bf16>> {
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(0));
    table.set_token_len(TOKENS);
    let storage = config(0, ProjectionFormat::Bf16, false).attention.cache;
    let mut batch = backend.prepare_paged_prefill_batch(storage, 1, 1, TOKENS)?;
    batch.prepare(&[&table], &[0], &[TOKENS])?;
    let hidden = copy(backend, &input(TOKENS * HIDDEN))?;
    let per_layer = copy(backend, &input(TOKENS * HIDDEN + 3)[3..])?;
    let per_layer = matches!(layer, SessionLayer::Dense(dense) if dense.per_layer.is_some())
        .then_some(&per_layer);
    let mut output: DeviceBuffer<bf16> =
        backend.pool().allocate_zeroed(backend.stream(), TOKENS * HIDDEN)?;
    layer.execute_shared_prefill_batch(plan.borrow(), &hidden, &mut output, &batch, per_layer)?;
    read(backend, &output)
}

/// A one-head layer `layer` of `HIDDEN` features in `format`, with
/// per-layer inputs `HIDDEN` wide when `per_layer` says so.
fn config(layer: usize, format: ProjectionFormat, per_layer: bool) -> DenseSwiGluConfig {
    DenseSwiGluConfig {
        attention: crate::DecodeAttentionConfig {
            layer,
            hidden_size: HIDDEN,
            query_heads: 1,
            rotary: None,
            rms_norm_epsilon: 1.0e-6,
            stream_norm: crate::kernels::NormFormula::Rms,
            attention_scale: 64.0_f32.sqrt().recip(),
            projection_format: format,
            qkv_normalization: QkvNormalization::NONE,
            window: None,
            score_softcap: None,
            query_temperature: None,
            max_sequence_blocks: 1,
            cache: KvStorageSpec::new(
                CacheConfig {
                    block_size: 16,
                    block_count: 1,
                    dtype: KvCacheDType::BFloat16,
                },
                1,
                HIDDEN,
            ),
            key_value_source: None,
        },
        intermediate_size: HIDDEN,
        activation: GatedActivation::GeluTanh,
        feed_forward_format: format,
        experts: None,
        feed_forward_input: crate::FeedForwardInput::AttentionResidual,
        per_layer_input: per_layer.then(|| NonZeroU16::new(64)).flatten(),
    }
}
