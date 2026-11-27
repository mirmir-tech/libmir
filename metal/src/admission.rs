pub use architecture::metal::MetalArchitecture;
use models::{execution::TaskExecutionPlan, semantic::SemanticModelSpec};

use crate::engine;

pub fn admit_architecture(
    task: &TaskExecutionPlan,
    semantic: Option<&SemanticModelSpec>,
) -> engine::Result<MetalArchitecture> {
    Ok(architecture::metal::admit(task, semantic)?)
}
