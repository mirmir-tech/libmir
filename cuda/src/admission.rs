pub use architecture::cuda::{CudaAdmitted, CudaArchitecture, CudaDecoderRuntime};
use models::{execution::TaskExecutionPlan, semantic::SemanticModelSpec};

use crate::Result;

pub fn admit_architecture(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> Result<CudaArchitecture> {
    Ok(architecture::cuda::admit(task, semantic)?)
}
