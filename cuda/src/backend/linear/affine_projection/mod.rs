use mircuda::{DeviceBuffer, bf16};

use self::candidate::Candidate;
use super::{AffineQuantizedConfig, AffineQuantizedWeight, CudaBackend, ProjectionLayout};
use crate::{Error, Result, backend::tuning::QuantizedProfileRequest};

mod candidate;
mod tuning;

#[derive(Debug)]
pub(in crate::backend) struct AffineProjection {
    operation: Candidate,
    /// The planned weight's packed words and groups per row (its bits and
    /// group size); a late-bound weight must have the same.
    layout: ProjectionLayout,
}

impl AffineProjection {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::backend) fn new(
        backend: &CudaBackend,
        tokens: usize,
        input: usize,
        output: usize,
        group_size: usize,
        bits: usize,
        weights: &AffineQuantizedWeight,
    ) -> Result<Self> {
        weights.validate(1, input, output, group_size, bits)?;
        let config = AffineQuantizedConfig::new(input, output, group_size, bits);
        let request = QuantizedProfileRequest::affine(tokens, input, output, group_size, bits);
        let operation = tuning::prepare(backend, request, tokens, config, weights)?;
        Ok(Self {
            operation,
            layout: ProjectionLayout::affine(weights),
        })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weights: &AffineQuantizedWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        // The kernels check the weight's shapes against their own geometry.
        if ProjectionLayout::affine(weights) != self.layout {
            return Err(Error::InvalidExecutionPlan(
                "affine plan and late-bound weight bits or groups differ",
            ));
        }
        self.operation.execute(input, weights, output)
    }
}
