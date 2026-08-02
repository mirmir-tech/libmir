mod capability;
mod contract;
mod task;

pub use capability::{ArchitectureCapability, ArchitectureRequirements};
pub use contract::DecoderExecutionContract;
pub use task::{EmbeddingTask, ModelTask, PoolingMode, SequenceScoringTask, TaskExecutionPlan};

#[cfg(test)]
#[allow(clippy::self_named_module_files)]
mod tests;
