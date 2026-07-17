use mircuda::{
    BlockScaledFp4Plan, BlockScaledFp4Spec, BlockScaledFp4VectorPlan, BlockScaledFp4VectorSpec,
    Context, DeviceBuffer, Stream, bf16,
};

use crate::Result;

#[derive(Debug)]
pub(super) enum NativeNvFp4Plan {
    Matrix(BlockScaledFp4Plan),
    Vector(BlockScaledFp4VectorPlan),
}

impl NativeNvFp4Plan {
    pub(super) fn new(
        context: &Context,
        stream: &Stream,
        tokens: usize,
        output_features: usize,
        input_features: usize,
    ) -> Result<Self> {
        if use_vector(tokens, output_features, input_features) {
            Ok(Self::Vector(BlockScaledFp4VectorPlan::new(
                context,
                stream,
                BlockScaledFp4VectorSpec::new(output_features, input_features)?,
            )?))
        } else {
            Ok(Self::Matrix(BlockScaledFp4Plan::new(
                context,
                stream,
                BlockScaledFp4Spec::new(tokens, output_features, input_features)?,
            )?))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        stream: &Stream,
        input: &DeviceBuffer<u8>,
        input_scales: &DeviceBuffer<u8>,
        weight: &DeviceBuffer<u8>,
        weight_scales: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
        alpha: f32,
    ) -> Result<()> {
        match self {
            Self::Matrix(plan) => {
                Ok(plan
                    .execute(stream, input, input_scales, weight, weight_scales, output, alpha)?)
            },
            Self::Vector(plan) => {
                Ok(plan
                    .execute(stream, input, input_scales, weight, weight_scales, output, alpha)?)
            },
        }
    }
}

fn use_vector(tokens: usize, output_features: usize, input_features: usize) -> bool {
    const VALIDATED: [(usize, usize); 3] = [(1_024, 4_096), (4_096, 4_096), (12_288, 4_096)];
    tokens == 1 && VALIDATED.contains(&(output_features, input_features))
}

#[cfg(test)]
mod tests {
    use super::use_vector;

    #[test]
    fn admits_only_validated_single_token_geometries() {
        for output in [1_024, 4_096, 12_288] {
            assert!(use_vector(1, output, 4_096));
            assert!(!use_vector(2, output, 4_096));
        }
        assert!(!use_vector(1, 4_096, 12_288));
        assert!(!use_vector(1, 2_816, 4_096));
    }
}
