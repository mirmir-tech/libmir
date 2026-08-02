mod embedding;
mod output;
#[cfg(all(test, target_os = "linux"))]
mod tests;

pub(in crate::backend::model) use embedding::{ModelEmbedding, ModelEmbeddingTemplate};
pub(in crate::backend::model) use output::{
    ModelBatchOutputHead, ModelOutputHead, ModelOutputHeadTemplate,
};
