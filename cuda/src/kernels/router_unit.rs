use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, product, require};
use crate::{Error, Result};

cuda_export!(
    RouterUnitKernel = "libmir_cuda_router_topk_unit_bf16"(
        scores: &DeviceBuffer<bf16>, selected: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<bf16>, experts: u32, top_k: u32,
        tokens: u32,
    )
);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RouterUnitSpec {
    pub tokens: usize,
    pub experts: usize,
    pub top_k: usize,
}

#[derive(Clone, Debug)]
pub struct RouterUnitTopK {
    kernel: TypedKernel<RouterUnitKernel>,
    spec: RouterUnitSpec,
}

impl RouterUnitTopK {
    pub fn compile(compiler: &Compiler, spec: RouterUnitSpec) -> Result<Self> {
        if spec.tokens == 0
            || spec.experts == 0
            || spec.experts > 256
            || spec.top_k == 0
            || spec.top_k > spec.experts
        {
            return Err(Error::InvalidRouter("invalid unit router geometry"));
        }
        let source = cuda_kernel_file!("../../kernels/router_unit_bf16.cu");
        let module = compiler.compile(source, &CompileOptions::default())?;
        Ok(Self { kernel: module.kernel()?, spec })
    }

    pub fn execute(
        &self,
        stream: &Stream,
        scores: &DeviceBuffer<bf16>,
        selected: &mut DeviceBuffer<u32>,
        weights: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        require(
            "unit router scores",
            product(self.spec.tokens, self.spec.experts)?,
            scores.len(),
        )?;
        let selections = product(self.spec.tokens, self.spec.top_k)?;
        require("unit router selected", selections, selected.len())?;
        require("unit router weights", selections, weights.len())?;
        Ok(self.kernel.launch(
            stream,
            LaunchConfig {
                grid: (narrow(self.spec.tokens)?, 1, 1),
                block: (32, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                scores,
                selected,
                weights,
                narrow(self.spec.experts)?,
                narrow(self.spec.top_k)?,
                narrow(self.spec.tokens)?,
            ),
        )?)
    }
}
