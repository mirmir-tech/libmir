use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct Session {
    pub id: Uuid,
    pub model: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
}

impl Session {
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            model: model.into(),
            prompt_tokens: 0,
            generated_tokens: 0,
        }
    }
}
