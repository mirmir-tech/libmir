use super::geometry::product;
use crate::{Error, Result};

mod embedding;
mod gathered;
mod portable;
pub use embedding::{MxFp4Embedding, MxFp4EmbeddingOperands, MxFp4EmbeddingSpec};
pub use gathered::{MxFp4GatheredLinear, MxFp4GatheredOperands};
pub use portable::MxFp4Linear;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Fixed BF16-by-OCP-MXFP4 projection geometry.
pub struct MxFp4Spec {
    pub tokens: usize,
    pub input_features: usize,
    pub output_features: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
/// Fixed gathered BF16-by-OCP-MXFP4 matrix-bank geometry.
pub struct MxFp4GatheredSpec {
    pub input_rows: usize,
    pub selections_per_input: usize,
    pub assignments: usize,
    pub matrices: usize,
    pub input_features: usize,
    pub output_features: usize,
}

impl MxFp4GatheredSpec {
    pub fn new(
        assignments: usize,
        matrices: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        Self::new_routed(assignments, 1, matrices, input_features, output_features)
    }

    pub fn new_routed(
        input_rows: usize,
        selections_per_input: usize,
        matrices: usize,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let assignments = product(input_rows, selections_per_input)?;
        let projection = MxFp4Spec::new(1, input_features, output_features)?;
        if matrices == 0 || selections_per_input == 0 {
            return Err(Error::InvalidDecoderKernel("invalid gathered MXFP4 matrix count"));
        }
        let spec = Self {
            input_rows,
            selections_per_input,
            assignments,
            matrices,
            input_features,
            output_features,
        };
        let _ = product(matrices, projection.weight_elements()?)?;
        let _ = product(matrices, projection.scale_elements()?)?;
        let _ = product(matrices, output_features)?;
        Ok(spec)
    }

    pub fn projection(self) -> Result<MxFp4Spec> {
        MxFp4Spec::new(self.input_rows, self.input_features, self.output_features)
    }

    pub fn output_elements(self) -> Result<usize> {
        product(self.assignments, self.output_features)
    }

    pub fn weight_elements(self) -> Result<usize> {
        product(self.matrices, self.projection()?.weight_elements()?)
    }

    pub fn scale_elements(self) -> Result<usize> {
        product(self.matrices, self.projection()?.scale_elements()?)
    }

    pub fn bias_elements(self) -> Result<usize> {
        product(self.matrices, self.output_features)
    }
}

impl MxFp4Spec {
    pub fn new(tokens: usize, input_features: usize, output_features: usize) -> Result<Self> {
        if tokens == 0
            || output_features == 0
            || input_features == 0
            || !input_features.is_multiple_of(32)
        {
            return Err(Error::InvalidDecoderKernel("invalid MXFP4 linear geometry"));
        }
        let spec = Self { tokens, input_features, output_features };
        let _ = spec.input_elements()?;
        let _ = spec.weight_elements()?;
        let _ = spec.scale_elements()?;
        let _ = spec.output_elements()?;
        Ok(spec)
    }

    pub fn input_elements(self) -> Result<usize> {
        product(self.tokens, self.input_features)
    }

    pub fn weight_elements(self) -> Result<usize> {
        product(self.output_features, self.input_features / 2)
    }

    pub fn scale_elements(self) -> Result<usize> {
        product(self.output_features, self.input_features / 32)
    }

    pub fn output_elements(self) -> Result<usize> {
        product(self.tokens, self.output_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_complete_mxfp4_blocks() -> Result<()> {
        assert!(MxFp4Spec::new(2, 64, 3).is_ok());
        assert!(MxFp4Spec::new(2, 48, 3).is_err());
        assert!(MxFp4GatheredSpec::new(2, 4, 64, 3).is_ok());
        assert!(MxFp4GatheredSpec::new(2, 0, 64, 3).is_err());
        assert_eq!(MxFp4GatheredSpec::new_routed(2, 3, 4, 64, 3)?.assignments, 6);
        Ok(())
    }
}
