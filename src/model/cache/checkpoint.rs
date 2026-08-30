use foundation::conversation::Conversation;

use super::super::ModelDescriptor;
use crate::Result;

impl ModelDescriptor {
    pub(in crate::model) fn cache_checkpoints(
        &self,
        conversation: &Conversation,
        full_tokens: &[u32],
    ) -> Result<Vec<usize>> {
        let mut checkpoints = Vec::new();
        for message_count in 1..conversation.messages.len() {
            let mut prefix = conversation.clone();
            prefix.messages.truncate(message_count);
            let Ok(prompt) = self.template.render(&prefix) else {
                continue;
            };
            let tokens = self
                .tokenizer
                .encode_with_special_tokens(&prompt.text, prompt.add_special_tokens)?;
            let common = full_tokens
                .iter()
                .zip(&tokens.token_ids)
                .take_while(|(left, right)| left == right)
                .count();
            if common > 0 && common < full_tokens.len() {
                checkpoints.push(common);
            }
        }
        checkpoints.sort_unstable();
        checkpoints.dedup();
        Ok(checkpoints)
    }
}
