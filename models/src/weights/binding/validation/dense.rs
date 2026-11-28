use super::invalid;
use crate::{
    error::Result,
    layout::WeightContainer,
    weights::{DenseDType, TensorCatalog},
};

/// A dense projection's bias is in the weight's dtype, or in F32 in a GGUF
/// file, which keeps every bias in F32 beside F16 and BF16 weights. Every
/// GGUF loader converts its float tensors to the activation dtype (CUDA and
/// Metal to BF16), so the bias reaches the product in the weight's dtype;
/// `SafeTensors` biases load as stored, so they must match the weight.
pub(super) fn bias_dtype(catalog: &TensorCatalog, name: &str, weight: &DenseDType) -> Result<()> {
    let tensor = catalog.get(name).ok_or_else(|| invalid(name, "companion tensor is missing"))?;
    let gguf = tensor.container == WeightContainer::Gguf;
    if tensor.dtype == *weight || (gguf && tensor.dtype == DenseDType::F32) {
        Ok(())
    } else if gguf {
        Err(invalid(
            name,
            &format!("expected dtype {weight} or F32, found {}", tensor.dtype),
        ))
    } else {
        Err(invalid(name, &format!("expected dtype {weight}, found {}", tensor.dtype)))
    }
}
