pub use architecture::cuda::CudaDecoderPlan;

use crate::{Error, Result, kernels::QkvNormalization as KernelNormalization};

#[cfg(test)]
mod tests;

pub fn graph_normalization(plan: &CudaDecoderPlan) -> Result<KernelNormalization> {
    let normalization = match architecture::cuda::graph_normalization(plan) {
        Ok(normalization) => normalization,
        Err(error) => return Err(Error::UnsupportedDecoderLayer(error.to_string())),
    };
    Ok(match normalization {
        architecture::cuda::CudaQkvNormalization::None => KernelNormalization::NONE,
        architecture::cuda::CudaQkvNormalization::All => KernelNormalization::ALL,
        architecture::cuda::CudaQkvNormalization::QueryKey => KernelNormalization::QUERY_KEY,
    })
}
