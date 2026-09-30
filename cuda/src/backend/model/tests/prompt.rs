use models::{layout::ModelLayout, tokenizer::TextTokenizer};
use runtime::kv::{BlockId, BlockTable};

/// A rendered Gemma 4 chat turn with a known one-token answer. Special-token
/// prompts have untrained embeddings that amplify harmless kernel-order
/// differences, so prefill/decode parity is judged on real text.
const QUESTION: &str = "<bos><|turn>user\nWhat is the capital of France? Answer with one \
                        word.<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
const ANSWER: &str = "Paris";

pub(super) struct Question {
    pub(super) prompt: Vec<u32>,
    tokenizer: TextTokenizer,
}

impl Question {
    pub(super) fn load(
        layout: &ModelLayout,
        table: &mut BlockTable,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = TextTokenizer::from_layout(layout)?;
        let prompt = tokenizer.encode_with_special_tokens(QUESTION, false)?.token_ids;
        // The session table starts with one 16-token block of the two the
        // test template allocates.
        let capacity = table.block_size().ok_or("block table has no block size")?;
        if prompt.len() > capacity * table.blocks().len() {
            table.push(BlockId(0));
        }
        Ok(Self { prompt, tokenizer })
    }

    pub(super) fn assert_answer(
        &self,
        logits: &[mircuda::bf16],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let token = super::assertions::maximum(logits).ok_or("empty logits")?;
        let token = u32::try_from(token)?;
        assert_eq!(self.tokenizer.decode(&[token])?.trim(), ANSWER);
        Ok(())
    }
}

/// Two decode rows of real text, each within one 16-token block. Very short
/// contexts leave a flat distribution in which ulp-level differences between
/// equally valid batch and scalar kernels compound over 30 layers into a
/// different greedy token, so rows carry enough context to be well-conditioned.
pub(super) fn rows(layout: &ModelLayout) -> Result<[Vec<u32>; 2], Box<dyn std::error::Error>> {
    let tokenizer = TextTokenizer::from_layout(layout)?;
    let row = |text: &str| -> Result<Vec<u32>, Box<dyn std::error::Error>> {
        let ids = tokenizer.encode_with_special_tokens(text, false)?.token_ids;
        if ids.len() < 2 || ids.len() > 16 {
            return Err("batch rows need 2..=16 tokens".into());
        }
        Ok(ids)
    };
    Ok([
        row("<bos>The quick brown fox jumps over the lazy")?,
        row("<bos>The capital city of France is")?,
    ])
}
