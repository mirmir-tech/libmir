//! Prepared clamped-routed layer execution.

use mircuda::Stream;

use super::{
    ClampedRoutedConfig, ClampedRoutedQkvLowering,
    projection::{ClampedRoutedLinear, ClampedRoutedQkv},
    scratch::ClampedRoutedScratch,
    weights::ClampedRoutedLayerWeights,
};
use crate::{
    CudaBackend, DenseRole, Result, RmsNormBf16,
    kernels::{
        ClampedRoutedAttention, ClampedRoutedKernels, ClampedRoutedSpec, ElementwiseBf16,
        RouterUnitSpec, RouterUnitTopK,
    },
};

mod execution;

#[derive(Clone)]
pub(super) struct ClampedRoutedLayerTemplate {
    backend: CudaBackend,
    config: ClampedRoutedConfig,
    qkv_lowering: ClampedRoutedQkvLowering,
    weights: ClampedRoutedLayerWeights,
    window: Option<usize>,
}

pub(super) struct ClampedRoutedLayerExecution {
    config: ClampedRoutedConfig,
    tokens: usize,
    input_norm: RmsNormBf16,
    post_norm: RmsNormBf16,
    qkv: ClampedRoutedQkv,
    output: ClampedRoutedLinear,
    router: ClampedRoutedLinear,
    top_k: RouterUnitTopK,
    kernels: ClampedRoutedKernels,
    attention: ClampedRoutedAttention,
    add: ElementwiseBf16,
    scratch: ClampedRoutedScratch,
    stream: Stream,
    window: Option<usize>,
}

impl ClampedRoutedLayerTemplate {
    pub(super) fn new(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        qkv_lowering: ClampedRoutedQkvLowering,
        weights: ClampedRoutedLayerWeights,
        window: Option<usize>,
    ) -> Self {
        Self {
            backend: backend.clone(),
            config,
            qkv_lowering,
            weights,
            window,
        }
    }

    pub(super) fn prepare(
        &self,
        tokens: usize,
        storage: runtime::kv::KvStorageSpec,
    ) -> Result<ClampedRoutedLayerExecution> {
        ClampedRoutedLayerExecution::new(self, tokens, storage)
    }

    pub(super) const fn weights(&self) -> &ClampedRoutedLayerWeights {
        &self.weights
    }
}

impl ClampedRoutedLayerExecution {
    fn new(
        template: &ClampedRoutedLayerTemplate,
        tokens: usize,
        storage: runtime::kv::KvStorageSpec,
    ) -> Result<Self> {
        let backend = &template.backend;
        let config = template.config;
        Ok(Self {
            config,
            tokens,
            input_norm: RmsNormBf16::new(backend, tokens, config.hidden, config.epsilon)?,
            post_norm: RmsNormBf16::new(backend, tokens, config.hidden, config.epsilon)?,
            qkv: ClampedRoutedQkv::new(
                backend,
                config,
                tokens,
                template.qkv_lowering,
                &template.weights.qkv,
            )?,
            output: ClampedRoutedLinear::new(
                backend,
                tokens,
                config.query_heads * config.head_dim,
                config.hidden,
                DenseRole::AttentionOutput,
                &template.weights.output,
            )?,
            router: ClampedRoutedLinear::new(
                backend,
                tokens,
                config.hidden,
                config.experts,
                DenseRole::Router,
                &template.weights.router,
            )?,
            top_k: RouterUnitTopK::compile(
                &backend.inner.compiler,
                RouterUnitSpec {
                    tokens,
                    experts: config.experts,
                    top_k: config.top_k,
                },
            )?,
            kernels: ClampedRoutedKernels::compile(&backend.inner.compiler, spec(config, tokens))?,
            attention: ClampedRoutedAttention::compile(
                &backend.inner.compiler,
                storage.cache.block_size,
                config.query_heads,
                config.kv_heads,
                config.head_dim,
                storage.cache.dtype,
            )?,
            add: ElementwiseBf16::compile(&backend.inner.compiler, tokens * config.hidden)?,
            scratch: ClampedRoutedScratch::new(backend, config, tokens)?,
            stream: backend.inner.stream.clone(),
            window: template.window,
        })
    }
}

fn spec(config: ClampedRoutedConfig, tokens: usize) -> ClampedRoutedSpec {
    ClampedRoutedSpec {
        tokens,
        hidden: config.hidden,
        intermediate: config.intermediate,
        query_heads: config.query_heads,
        kv_heads: config.kv_heads,
        head_dim: config.head_dim,
        top_k: config.top_k,
        theta: config.theta,
        factor: config.factor,
        initial_context: config.initial_context,
        beta_fast: config.beta_fast,
        beta_slow: config.beta_slow,
        swiglu_limit: config.swiglu_limit,
    }
}
