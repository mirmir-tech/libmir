//! Typed decisions of Laya checkpoints: a `ModernBERT` encoder read by a small
//! transformer head that scores the options of a question in one pass.
//!
//! This module holds everything that does not depend on a backend: the
//! checkpoint layout, question rendering, token rows, tensor bindings, and
//! answer decoding. Backends only turn [`DecisionRow`]s into option logits.

mod agent;
mod answer;
mod checkpoint;
mod question;
mod schema;
mod sequence;
mod tensors;

pub use agent::AgentConfig;
pub use answer::{Answer, Verdict};
pub use checkpoint::DecisionCheckpoint;
pub use question::{ChoiceOption, Question, QuestionKind, Verdicts};
pub use schema::{DecisionSchema, SchemaField};
pub use sequence::{DecisionRow, DecisionState, DecisionTokenizer, StateEnd};
pub use tensors::{DecisionTensor, DecisionTensorPlan, EncoderTensor, HeadTensor, ScorerTensor};
