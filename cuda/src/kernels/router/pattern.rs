use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, cuda_export,
    cuda_kernel_file,
};

use super::super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(RoutePatternKernel = "libmir_cuda_router_route_pattern"(
    selected: &mut DeviceBuffer<u32>, tokens: u32, experts: u32,
    top_k: u32, pattern: u32,
));

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RoutePattern {
    Balanced = 0,
    HotSet = 1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RoutePatternSpec {
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
}

#[derive(Clone, Debug)]
pub struct RoutePatternGenerator {
    kernel: TypedKernel<RoutePatternKernel>,
    spec: RoutePatternSpec,
}

impl RoutePatternGenerator {
    pub fn compile(compiler: &Compiler, spec: RoutePatternSpec) -> Result<Self> {
        if spec.tokens == 0 || spec.experts == 0 || spec.top_k == 0 || spec.top_k > spec.experts {
            return Err(Error::InvalidRouter("invalid route pattern geometry"));
        }
        let source = cuda_kernel_file!("../../../kernels/router_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        pattern: RoutePattern,
        selected: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        let assignments = product(self.spec.tokens, self.spec.top_k)?;
        require("route pattern output", assignments, selected.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(assignments.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                selected,
                narrow(self.spec.tokens)?,
                narrow(self.spec.experts)?,
                narrow(self.spec.top_k)?,
                pattern as u32,
            ),
        )?)
    }
}
