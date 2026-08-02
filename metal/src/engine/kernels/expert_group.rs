use super::{Kernels, template};
use crate::engine::{Error, Result};

mirtal::metal_kernel! {
    fn expert_group {
        name: "mirmir_expert_group",
        templates: [ROUTES: int = 64, EXPERTS: int = 8],
        inputs: [indices: u32],
        outputs: [order: u32, inverse: u32],
        source: file "kernels/expert_group.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct ExpertGroupKernel {
    kernel: mirtal::MetalKernel<1, 2>,
}

impl ExpertGroupKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: expert_group()? })
    }

    pub(super) fn forward(
        &self,
        stream: &mirtal::Stream,
        indices: &mirtal::Array,
        experts: usize,
    ) -> Result<[mirtal::Array; 2]> {
        let routes = indices.len();
        if indices.dtype()? != mirtal::DType::Uint32 || experts == 0 || experts > 1024 {
            return Err(Error::InvalidModel("expert grouping configuration is unsupported".into()));
        }
        let output = mirtal::OutputSpec::new(mirtal::Shape::new([routes])?, mirtal::DType::Uint32);
        let threads = experts.max(routes.min(256));
        Ok(self.kernel.dispatch(
            stream,
            [indices],
            &[output.clone(), output],
            &mirtal::Dispatch::new([threads, 1, 1], [threads, 1, 1])
                .templates([template("ROUTES", routes)?, template("EXPERTS", experts)?]),
        )?)
    }
}

impl Kernels {
    pub(crate) fn expert_group(
        &self,
        stream: &mirtal::Stream,
        indices: &mirtal::Array,
        experts: usize,
    ) -> Result<[mirtal::Array; 2]> {
        self.expert_group.forward(stream, indices, experts)
    }
}
