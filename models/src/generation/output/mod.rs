use crate::tokenizer::TextTokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationChannel {
    Content,
    Reasoning,
    ToolCalls,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationToken {
    pub id: u32,
    pub text: String,
    pub channel: GenerationChannel,
}

pub struct OutputNormalizer {
    markers: Markers,
    state: State,
}

enum State {
    Content,
    Reasoning,
    ToolCalls,
    ChannelName(String),
}

#[derive(Default)]
struct Markers {
    reasoning: Vec<u32>,
    content: Vec<u32>,
    channel: Vec<u32>,
    channel_body: Vec<u32>,
    channel_end: Vec<u32>,
    tool_calls: Vec<u32>,
}

impl OutputNormalizer {
    #[must_use]
    pub fn new(tokenizer: &TextTokenizer, prompt: &str) -> Self {
        Self {
            markers: Markers::new(tokenizer),
            state: if prompt_requests_reasoning(prompt) {
                State::Reasoning
            } else {
                State::Content
            },
        }
    }

    #[must_use]
    pub fn push(&mut self, id: u32, text: String) -> Option<GenerationToken> {
        if self.markers.reasoning.contains(&id) {
            self.state = State::Reasoning;
            return None;
        }
        if self.markers.content.contains(&id) || self.markers.channel_end.contains(&id) {
            self.state = State::Content;
            return None;
        }
        if self.markers.tool_calls.contains(&id) {
            self.state = State::ToolCalls;
            return None;
        }
        if self.markers.channel.contains(&id) {
            self.state = State::ChannelName(String::new());
            return None;
        }
        if self.markers.channel_body.contains(&id) {
            self.finish_channel_name();
            return None;
        }
        self.push_text(id, text)
    }

    fn push_text(&mut self, id: u32, text: String) -> Option<GenerationToken> {
        let State::ChannelName(name) = &mut self.state else {
            return nonempty(id, text, channel(&self.state));
        };
        name.push_str(&text);
        let newline = name.find('\n')?;
        let remainder = name[newline + 1..].to_owned();
        let channel = named_channel(&name[..newline]);
        self.state = state(channel);
        nonempty(id, remainder, channel)
    }

    fn finish_channel_name(&mut self) {
        let State::ChannelName(name) = &self.state else {
            return;
        };
        self.state = state(named_channel(name));
    }
}

impl Markers {
    fn new(tokenizer: &TextTokenizer) -> Self {
        Self {
            reasoning: token_ids(
                tokenizer,
                &["<think>", "<|think|>", "<|analysis|>", "<|reasoning|>"],
            ),
            content: token_ids(tokenizer, &["</think>", "<|final|>", "<|content|>"]),
            channel: token_ids(tokenizer, &["<|channel>", "<|channel|>"]),
            channel_body: token_ids(tokenizer, &["<|message|>"]),
            channel_end: token_ids(tokenizer, &["<channel|>", "<|end|>", "<|return|>"]),
            tool_calls: token_ids(tokenizer, &["[TOOL_CALLS]"]),
        }
    }
}

fn token_ids(tokenizer: &TextTokenizer, tokens: &[&str]) -> Vec<u32> {
    tokens
        .iter()
        .filter_map(|token| tokenizer.added_token_id(token).or_else(|| tokenizer.token_id(token)))
        .collect()
}

fn prompt_requests_reasoning(prompt: &str) -> bool {
    unmatched(prompt, "<think>", "</think>")
        || unmatched(prompt, "<|analysis|>", "<|final|>")
        || unmatched(prompt, "<|channel>thought", "<channel|>")
        || unmatched(prompt, "<|channel|>analysis", "<|end|>")
}

fn unmatched(text: &str, start: &str, end: &str) -> bool {
    text.rfind(start)
        .is_some_and(|start| text.rfind(end).is_none_or(|end| start > end))
}

const fn channel(state: &State) -> GenerationChannel {
    match state {
        State::Reasoning => GenerationChannel::Reasoning,
        State::ToolCalls => GenerationChannel::ToolCalls,
        State::Content | State::ChannelName(_) => GenerationChannel::Content,
    }
}

fn named_channel(name: &str) -> GenerationChannel {
    match name.trim().to_ascii_lowercase().as_str() {
        "analysis" | "reasoning" | "thought" => GenerationChannel::Reasoning,
        "tool" | "tool_calls" => GenerationChannel::ToolCalls,
        _ => GenerationChannel::Content,
    }
}

const fn state(channel: GenerationChannel) -> State {
    match channel {
        GenerationChannel::Content => State::Content,
        GenerationChannel::Reasoning => State::Reasoning,
        GenerationChannel::ToolCalls => State::ToolCalls,
    }
}

fn nonempty(id: u32, text: String, channel: GenerationChannel) -> Option<GenerationToken> {
    (!text.is_empty()).then_some(GenerationToken { id, text, channel })
}

#[cfg(test)]
mod tests;
