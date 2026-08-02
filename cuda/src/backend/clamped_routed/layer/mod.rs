//! Prepared clamped-routed layer execution.

use mircuda::{DeviceBuffer, Stream};

use super::{
    ClampedRoutedConfig, ClampedRoutedQkvLowering,
    experts::AutoClampedExperts,
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

mod attention;
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
    experts: Option<AutoClampedExperts>,
    attention: ClampedRoutedAttention,
    add: ElementwiseBf16,
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
        phase: crate::ExecutionPhase,
    ) -> Result<ClampedRoutedLayerExecution> {
        ClampedRoutedLayerExecution::new(self, tokens, storage, phase)
    }

    pub(super) const fn weights(&self) -> &ClampedRoutedLayerWeights {
        &self.weights
    }

    pub(super) const fn window(&self) -> Option<usize> {
        self.window
    }
}

impl ClampedRoutedLayerExecution {
    fn new(
        template: &ClampedRoutedLayerTemplate,
        tokens: usize,
        storage: runtime::kv::KvStorageSpec,
        phase: crate::ExecutionPhase,
    ) -> Result<Self> {
        let backend = &template.backend;
        let config = template.config;
        let kernels = ClampedRoutedKernels::compile(&backend.inner.compiler, spec(config, tokens))?;
        let experts = AutoClampedExperts::new(
            backend,
            config,
            tokens,
            phase,
            &template.weights.experts,
            kernels.clone(),
        );
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
            kernels,
            experts,
            attention: ClampedRoutedAttention::compile(
                backend,
                storage.cache.block_size,
                config.query_heads,
                config.kv_heads,
                config.head_dim,
                storage.cache.dtype,
                template.window,
            )?,
            add: ElementwiseBf16::compile(&backend.inner.compiler, tokens * config.hidden)?,
            stream: backend.inner.stream.clone(),
            window: template.window,
        })
    }

    pub(super) fn prepare_rope(
        &self,
        positions: &DeviceBuffer<u32>,
        scratch: &mut ClampedRoutedScratch,
    ) -> Result<()> {
        self.kernels.prepare_rope(
            &self.stream,
            positions,
            &scratch.rope_inverse,
            &mut scratch.rope_sines,
            &mut scratch.rope_cosines,
        )
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
