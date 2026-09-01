use std::sync::{Arc, Mutex};

use mircuda::{Context, MarlinNvFp4MoeSpec, MarlinNvFp4ThreadConfig, Stream};

use super::super::{
    ClampedRoutedConfig,
    weights::{MarlinMxFp4Banks, NativeExpertWeights},
};
use crate::{
    CudaBackend, Error, Result,
    backend::{
        linear::{MarlinNvFp4Scratch, MarlinNvFp4ScratchConfig, MarlinRouteBlock},
        tuning::ClampedMoeExecution,
    },
    kernels::{ClampedRoutedMarlinEpilogue, ClampedRoutedMarlinGeometry},
};

mod execute;

pub(super) struct MarlinMxFp4Candidate {
    banks: Arc<MarlinMxFp4Banks>,
    scratch: Arc<Mutex<MarlinNvFp4Scratch>>,
    epilogue: ClampedRoutedMarlinEpilogue,
    config: ClampedRoutedConfig,
    context: Context,
    stream: Stream,
    tokens: usize,
    gate_thread_config: MarlinNvFp4ThreadConfig,
    down_thread_config: MarlinNvFp4ThreadConfig,
}

impl MarlinMxFp4Candidate {
    pub(super) fn new(
        backend: &CudaBackend,
        config: ClampedRoutedConfig,
        tokens: usize,
        weights: &NativeExpertWeights,
        execution: ClampedMoeExecution,
    ) -> Result<Self> {
        let gate_thread_config = thread_config(execution)?;
        let banks = weights.marlin(backend, config.experts, config.hidden, config.intermediate)?;
        let gate_input = banks.gate_up.input_features;
        let gate_output = banks.gate_up.output_features;
        let down_input = banks.down.input_features;
        let down_output = banks.down.output_features;
        let down_thread_config =
            compatible_thread_config(down_output, down_input, gate_thread_config.moe_block_size())?;
        let assignments = product(tokens, config.top_k)?;
        MarlinNvFp4MoeSpec::new(
            config.experts,
            tokens,
            config.top_k,
            gate_output,
            gate_input,
            gate_thread_config,
        )?;
        MarlinNvFp4MoeSpec::new(
            config.experts,
            assignments,
            1,
            down_output,
            down_input,
            down_thread_config,
        )?;
        Ok(Self {
            banks,
            scratch: backend.marlin_nvfp4_scratch(MarlinNvFp4ScratchConfig {
                tokens,
                top_k: config.top_k,
                experts: config.experts,
                route_block: MarlinRouteBlock::from(gate_thread_config),
                hidden: config.hidden,
                intermediate: config.intermediate,
                padded_hidden: down_output,
                padded_intermediate: down_input,
            })?,
            epilogue: ClampedRoutedMarlinEpilogue::compile(
                backend.compiler(),
                ClampedRoutedMarlinGeometry {
                    tokens,
                    top_k: config.top_k,
                    intermediate: config.intermediate,
                    hidden: config.hidden,
                    padded_hidden: down_output,
                    padded_intermediate: down_input,
                    limit: config.swiglu_limit,
                },
            )?,
            config,
            context: backend.context().clone(),
            stream: backend.stream().clone(),
            tokens,
            gate_thread_config,
            down_thread_config,
        })
    }
}

fn thread_config(execution: ClampedMoeExecution) -> Result<MarlinNvFp4ThreadConfig> {
    match execution {
        ClampedMoeExecution::MarlinN128K128 => Ok(MarlinNvFp4ThreadConfig::N128K128),
        ClampedMoeExecution::MarlinN128K64 => Ok(MarlinNvFp4ThreadConfig::N128K64),
        ClampedMoeExecution::MarlinN64K128 => Ok(MarlinNvFp4ThreadConfig::N64K128),
        ClampedMoeExecution::MarlinM64N256K64 => Ok(MarlinNvFp4ThreadConfig::M64N256K64),
        ClampedMoeExecution::MarlinM64N128K64 => Ok(MarlinNvFp4ThreadConfig::M64N128K64),
        ClampedMoeExecution::MarlinM64N64K128 => Ok(MarlinNvFp4ThreadConfig::M64N64K128),
        ClampedMoeExecution::FusedReduce | ClampedMoeExecution::RouteParallel => {
            Err(Error::InvalidExecutionPlan("portable clamped execution is not Marlin"))
        },
    }
}

fn compatible_thread_config(
    n: usize,
    k: usize,
    block_size: usize,
) -> Result<MarlinNvFp4ThreadConfig> {
    if block_size == 64 && n.is_multiple_of(256) && k.is_multiple_of(64) {
        Ok(MarlinNvFp4ThreadConfig::M64N256K64)
    } else if block_size == 64 && n.is_multiple_of(128) && k.is_multiple_of(64) {
        Ok(MarlinNvFp4ThreadConfig::M64N128K64)
    } else if block_size == 64 && n.is_multiple_of(64) && k.is_multiple_of(128) {
        Ok(MarlinNvFp4ThreadConfig::M64N64K128)
    } else if block_size == 8 && n.is_multiple_of(128) && k.is_multiple_of(128) {
        Ok(MarlinNvFp4ThreadConfig::N128K128)
    } else if block_size == 8 && n.is_multiple_of(128) && k.is_multiple_of(64) {
        Ok(MarlinNvFp4ThreadConfig::N128K64)
    } else if block_size == 8 && n.is_multiple_of(64) && k.is_multiple_of(128) {
        Ok(MarlinNvFp4ThreadConfig::N64K128)
    } else {
        Err(Error::InvalidExecutionPlan("MXFP4 Marlin geometry is not tile aligned"))
    }
}

fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or(Error::InvalidDecoderKernel("MXFP4 Marlin size overflow"))
}
