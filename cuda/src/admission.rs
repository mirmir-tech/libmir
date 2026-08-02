pub use architecture::cuda::{CudaArchitecture, CudaDecoderRuntime};
use models::{execution::TaskExecutionPlan, semantic::SemanticModelSpec};

use crate::{Error, Result};

pub fn admit_architecture(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> Result<CudaArchitecture> {
    match architecture::cuda::admit(task, semantic) {
        Ok(architecture) => Ok(architecture),
        Err(error) => Err(Error::UnsupportedDecoderLayer(error.to_string())),
    }
}
