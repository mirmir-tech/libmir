pub use architecture::metal::MetalArchitecture;
use models::{execution::TaskExecutionPlan, semantic::SemanticModelSpec};

use crate::engine;

pub fn admit_architecture(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> engine::Result<MetalArchitecture> {
    match architecture::metal::admit(task, semantic) {
        Ok(architecture) => Ok(architecture),
        Err(error) => Err(engine::Error::InvalidModel(error.to_string())),
    }
}
