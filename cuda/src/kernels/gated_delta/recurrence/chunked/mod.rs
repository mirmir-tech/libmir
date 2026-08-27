use mircuda::{
    CompileOptions, Compiler, LaunchConfig, TypedKernel, cuda_kernel_file, cuda_ptx_file,
};

use super::GatedDeltaSpec;
use crate::{Error, Result};

mod launch;
mod scratch;
mod symbols;

pub use scratch::GatedDeltaChunkedScratch;
use symbols::{Cumsum, H, Kkt, LogParameters, O, Solve, Uw};

pub(super) const CHUNK: usize = 64;
const CAPABILITY: (i32, i32) = (12, 1);

#[derive(Clone, Debug)]
pub struct GatedDeltaChunked {
    parameters: TypedKernel<LogParameters>,
    cumsum: TypedKernel<Cumsum>,
    kkt: TypedKernel<Kkt>,
    solve: TypedKernel<Solve>,
    uw: TypedKernel<Uw>,
    h: TypedKernel<H>,
    o: TypedKernel<O>,
    spec: GatedDeltaSpec,
}

impl GatedDeltaChunked {
    #[must_use]
    pub const fn supports(compute_capability: (i32, i32), spec: GatedDeltaSpec) -> bool {
        compute_capability.0 == CAPABILITY.0
            && compute_capability.1 == CAPABILITY.1
            && spec.tokens > 1
            && spec.key_heads == 16
            && spec.value_heads == 32
            && spec.key_dim == 128
            && spec.value_dim == 128
    }

    pub fn compile(compiler: &Compiler, spec: GatedDeltaSpec) -> Result<Self> {
        validate(spec)?;
        let native = compiler.compile(
            cuda_kernel_file!("../../../../../kernels/gated_delta_bf16.cu"),
            &CompileOptions::default(),
        )?;
        let cumsum = compiler.load_ptx(cuda_ptx_file!(
            CAPABILITY,
            "../../../../../kernels/gated_delta/chunked/sm121/cumsum.ptx"
        ))?;
        let kkt = compiler.load_ptx(cuda_ptx_file!(
            CAPABILITY,
            "../../../../../kernels/gated_delta/chunked/sm121/kkt.ptx"
        ))?;
        let solve = compiler.load_ptx(cuda_ptx_file!(
            CAPABILITY,
            "../../../../../kernels/gated_delta/chunked/sm121/solve64.ptx"
        ))?;
        let uw = compiler.load_ptx(cuda_ptx_file!(
            CAPABILITY,
            "../../../../../kernels/gated_delta/chunked/sm121/uw.ptx"
        ))?;
        let h = compiler
            .load_ptx(cuda_ptx_file!(
                CAPABILITY,
                "../../../../../kernels/gated_delta/chunked/sm121/h.ptx"
            ))?
            .kernel()?;
        h.set_max_dynamic_shared_memory_bytes(90_632)?;
        Ok(Self {
            parameters: native.kernel()?,
            cumsum: cumsum.kernel()?,
            kkt: kkt.kernel()?,
            solve: solve.kernel()?,
            uw: uw.kernel()?,
            h,
            o: compiler
                .load_ptx(cuda_ptx_file!(
                    CAPABILITY,
                    "../../../../../kernels/gated_delta/chunked/sm121/o.ptx"
                ))?
                .kernel()?,
            spec,
        })
    }

    fn config(grid: (usize, usize, usize), warps: u32, shared: u32) -> Result<LaunchConfig> {
        Ok(LaunchConfig {
            grid: (u32::try_from(grid.0)?, u32::try_from(grid.1)?, u32::try_from(grid.2)?),
            block: (warps * 32, 1, 1),
            shared_memory_bytes: shared,
        })
    }
}

pub(super) fn validate(spec: GatedDeltaSpec) -> Result<()> {
    if !GatedDeltaChunked::supports(CAPABILITY, spec) {
        return Err(Error::InvalidDecoderKernel("unsupported chunked Gated Delta geometry"));
    }
    Ok(())
}
