use foundation::conversation::Conversation;
use models::generation::GenerationOverrides;

#[derive(Debug, Clone, Default)]
/// Backend-neutral input for one model generation.
pub struct GenerationRequest {
    /// Ordered conversation turns and callable tools rendered by the model
    /// template.
    pub conversation: Conversation,
    /// Request-scoped sampling values applied over the loaded model defaults.
    pub options: GenerationOverrides,
    /// Optional deterministic sampler seed.
    pub seed: Option<u64>,
}
