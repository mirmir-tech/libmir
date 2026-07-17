use mircuda::{DeviceBuffer, bf16};
use runtime::backend::SamplingLogits;

use super::{CudaOutputHead, Projection};
use crate::Result;

impl CudaOutputHead {
    /// Enqueues output logits without allocation or synchronization.
    pub fn execute(
        &mut self,
        source: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        sampling: SamplingLogits,
    ) -> Result<()> {
        match &mut self.projection {
            Projection::Bf16 { operation, weight } => operation.execute(source, weight, output),
            Projection::Fp8 {
                operation,
                kernels,
                input,
                input_scales,
                weight,
                weight_scales,
                row_scales,
            } => {
                kernels.quantize_input(&self.stream, source, input, input_scales)?;
                operation
                    .execute(&self.stream, input, input_scales, weight, weight_scales, output)?;
                kernels.rescale_output(&self.stream, output, row_scales)
            },
            Projection::Fp8Vectorized { kernels, weight, row_scales } => {
                kernels.project_vectorized(&self.stream, source, weight, row_scales, output)
            },
            Projection::Fp8Residual {
                kernels,
                weight,
                row_scales,
                residual,
                residual_scales,
            } => kernels.project_residual(
                &self.stream, source, weight, row_scales, residual, residual_scales, output,
            ),
            Projection::Fp8BlockVectorized { kernels, weight, scales } => {
                kernels.project(&self.stream, source, weight, scales, output)
            },
            Projection::Fp8BlockRefined {
                kernels,
                refinement,
                exact_weight,
                weight,
                scales,
                first,
                second,
            } => {
                kernels.project(&self.stream, source, weight, scales, output)?;
                refinement.execute(&self.stream, source, exact_weight, output, first, second)
            },
            Projection::AutoRefined {
                exact_operation,
                exact_tensor,
                kernels,
                refinement,
                exact_weight,
                weight,
                scales,
                first,
                second,
            } => {
                if supports_refinement(sampling) {
                    kernels.project(&self.stream, source, weight, scales, output)?;
                    refinement.execute(&self.stream, source, exact_weight, output, first, second)
                } else {
                    exact_operation.execute(source, exact_tensor, output)
                }
            },
        }
    }
}

const fn supports_refinement(sampling: SamplingLogits) -> bool {
    match sampling {
        SamplingLogits::None => true,
        SamplingLogits::TopK { k, .. } | SamplingLogits::SampleTopK { k, .. } => {
            k <= crate::kernels::MAX_TOP_K
        },
        SamplingLogits::Sample { top_k, .. } => top_k > 0 && top_k <= crate::kernels::MAX_TOP_K,
        SamplingLogits::Full => false,
    }
}

#[cfg(test)]
mod tests {
    use runtime::backend::SamplingLogits;

    use super::supports_refinement;

    #[test]
    fn refinement_requires_bounded_candidate_set() {
        assert!(supports_refinement(SamplingLogits::None));
        assert!(supports_refinement(SamplingLogits::TopK { k: 64, vocab_size: 262_144 }));
        assert!(!supports_refinement(SamplingLogits::TopK { k: 65, vocab_size: 262_144 }));
        assert!(!supports_refinement(SamplingLogits::Full));
        assert!(!supports_refinement(SamplingLogits::Sample {
            vocab_size: 262_144,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            draw: 0.5,
        }));
    }
}
