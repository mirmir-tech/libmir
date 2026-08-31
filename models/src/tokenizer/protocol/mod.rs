use std::collections::BTreeMap;

use tokenizers::Tokenizer;

#[derive(Debug, Clone, Default)]
pub struct OutputMarkerIds {
    pub reasoning: Vec<u32>,
    pub content: Vec<u32>,
    pub turn_start: Vec<u32>,
    pub channel: Vec<u32>,
    pub channel_body: Vec<u32>,
    pub channel_end: Vec<u32>,
    pub tool_calls: Vec<u32>,
}

pub(super) struct ProtocolTokenIds {
    pub(super) stop: Vec<u32>,
    pub(super) output: OutputMarkerIds,
}

impl ProtocolTokenIds {
    pub(super) fn resolve(
        tokenizer: &Tokenizer,
        added: &BTreeMap<String, u32>,
        configured_stop: &[u32],
        eos: &[u32],
    ) -> Self {
        Self {
            stop: stop_ids(tokenizer, configured_stop, eos),
            output: OutputMarkerIds {
                reasoning: ids(
                    tokenizer,
                    added,
                    &["<think>", "<|think|>", "<|analysis|>", "<|reasoning|>"],
                ),
                content: ids(tokenizer, added, &["</think>", "<|final|>", "<|content|>"]),
                turn_start: ids(tokenizer, added, &["<|start|>"]),
                channel: ids(tokenizer, added, &["<|channel>", "<|channel|>"]),
                channel_body: ids(tokenizer, added, &["<|message|>"]),
                channel_end: ids(tokenizer, added, &["<channel|>", "<|end|>", "<|return|>"]),
                tool_calls: ids(tokenizer, added, &["[TOOL_CALLS]"]),
            },
        }
    }
}

fn ids(tokenizer: &Tokenizer, added: &BTreeMap<String, u32>, tokens: &[&str]) -> Vec<u32> {
    tokens
        .iter()
        .filter_map(|token| added.get(*token).copied().or_else(|| tokenizer.token_to_id(token)))
        .collect()
}

fn stop_ids(tokenizer: &Tokenizer, configured: &[u32], eos: &[u32]) -> Vec<u32> {
    let mut ids = configured.to_vec();
    extend_unique(&mut ids, eos.iter().copied());
    extend_unique(
        &mut ids,
        [
            "<eos>",
            "</s>",
            "<|endoftext|>",
            "<|im_end|>",
            "<end_of_turn>",
            "<turn|>",
            "<|eot_id|>",
            "<|tool_response>",
        ]
        .into_iter()
        .filter_map(|token| tokenizer.token_to_id(token)),
    );
    ids
}

fn extend_unique(ids: &mut Vec<u32>, values: impl IntoIterator<Item = u32>) {
    for id in values {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
}

#[cfg(test)]
mod tests;
