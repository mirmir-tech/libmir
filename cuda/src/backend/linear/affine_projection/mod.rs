use mircuda::{DeviceBuffer, bf16};

use self::candidate::Candidate;
use super::{AffineQuantizedConfig, AffineQuantizedWeight, CudaBackend};
use crate::{Result, backend::tuning::QuantizedProfileRequest};

mod candidate;
mod tuning;

#[derive(Debug)]
pub(in crate::backend) struct AffineProjection {
    operation: Candidate,
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
        Ok(Self { operation })
    }

    pub(in crate::backend) fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weights: &AffineQuantizedWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.operation.execute(input, weights, output)
    }
}
