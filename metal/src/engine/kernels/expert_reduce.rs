use super::{Kernels, template};
use crate::engine::{Error, Result};

mirtal::metal_kernel! {
    fn expert_reduce {
        name: "mirmir_expert_restore_reduce",
        templates: [T: dtype = bf16, HIDDEN: int = 2880, TOP_K: int = 4],
        inputs: [sorted: T, inverse: u32, weights: T],
        outputs: [output: T],
        source: file "kernels/expert_reduce.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(super) struct ExpertReduceKernel {
    kernel: mirtal::MetalKernel<3, 1>,
}

impl ExpertReduceKernel {
    pub(super) fn new() -> Result<Self> {
        Ok(Self { kernel: expert_reduce()? })
    }

    pub(super) fn forward(
        &self,
        stream: &mirtal::Stream,
        [sorted, inverse, weights]: [&mirtal::Array; 3],
    ) -> Result<mirtal::Array> {
        let sorted_shape = sorted.shape()?;
        let weights_shape = weights.shape()?;
        let sorted_shape = sorted_shape.dimensions();
        let weights_shape = weights_shape.dimensions();
        if sorted_shape.len() != 3
            || sorted_shape[1] != 1
            || weights_shape.len() != 3
            || inverse.len() != weights.len()
            || sorted_shape[0] != weights.len()
        {
            return Err(Error::InvalidModel("sorted expert reduction shapes do not align".into()));
        }
        let hidden = sorted_shape[2];
        let top_k = weights_shape[2];
        let elements = weights_shape[0]
            .checked_mul(weights_shape[1])
            .and_then(|tokens| tokens.checked_mul(hidden))
            .ok_or(Error::ShapeOverflow)?;
        let output = mirtal::OutputSpec::new(
            mirtal::Shape::new([weights_shape[0], weights_shape[1], hidden])?,
            sorted.dtype()?,
        );
        let [output] = self.kernel.dispatch(
            stream,
            [sorted, inverse, weights],
            &[output],
            &mirtal::Dispatch::new([elements, 1, 1], [elements.min(256), 1, 1]).templates([
                mirtal::TemplateArg::dtype("T", sorted.dtype()?),
                template("HIDDEN", hidden)?,
                template("TOP_K", top_k)?,
            ]),
        )?;
        Ok(output)
    }
}

impl Kernels {
    pub(crate) fn expert_restore_reduce(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 3],
    ) -> Result<mirtal::Array> {
        self.expert_reduce.forward(stream, inputs)
    }
}
