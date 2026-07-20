use super::TokenizedPrompt;

pub(super) fn tokenized(encoding: &tokenizers::Encoding, bytes: usize) -> TokenizedPrompt {
    TokenizedPrompt {
        token_ids: encoding.get_ids().to_vec(),
        type_ids: encoding.get_type_ids().to_vec(),
        attention_mask: encoding.get_attention_mask().to_vec(),
        bytes,
    }
}
