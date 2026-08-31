use crate::tokenizer::{TextTokenizer, protocol::OutputMarkerIds};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationChannel {
    Content,
    Reasoning,
    ToolCalls,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationToken {
    pub preceding_ids: Vec<u32>,
    pub id: u32,
    pub text: String,
    pub channel: GenerationChannel,
}

pub struct OutputNormalizer {
    markers: OutputMarkerIds,
    state: State,
    pending_ids: Vec<u32>,
}

enum State {
    Content,
    Reasoning,
    ToolCalls,
    RoleName,
    ChannelName(String),
}

impl OutputNormalizer {
    #[must_use]
    pub fn new(tokenizer: &TextTokenizer, prompt: &str) -> Self {
        Self {
            markers: tokenizer.output_markers().clone(),
            state: if prompt_requests_reasoning(prompt) {
                State::Reasoning
            } else {
                State::Content
            },
            pending_ids: Vec::new(),
        }
    }

    #[must_use]
    pub fn push(&mut self, id: u32, text: String) -> Option<GenerationToken> {
        self.pending_ids.push(id);
        if self.markers.turn_start.contains(&id) {
            self.state = State::RoleName;
            return None;
        }
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
        self.push_text(text)
    }

    fn push_text(&mut self, text: String) -> Option<GenerationToken> {
        if matches!(&self.state, State::RoleName) {
            return None;
        }
        let State::ChannelName(name) = &mut self.state else {
            return self.nonempty(text, channel(&self.state));
        };
        name.push_str(&text);
        let newline = name.find('\n')?;
        let remainder = name[newline + 1..].to_owned();
        let channel = named_channel(&name[..newline]);
        self.state = state(channel);
        self.nonempty(remainder, channel)
    }

    fn finish_channel_name(&mut self) {
        let State::ChannelName(name) = &self.state else {
            return;
        };
        self.state = state(named_channel(name));
    }

    fn nonempty(&mut self, text: String, channel: GenerationChannel) -> Option<GenerationToken> {
        if text.is_empty() {
            return None;
        }
        let id = self.pending_ids.pop()?;
        Some(GenerationToken {
            preceding_ids: std::mem::take(&mut self.pending_ids),
            id,
            text,
            channel,
        })
    }
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
        State::Content | State::RoleName | State::ChannelName(_) => GenerationChannel::Content,
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

#[cfg(test)]
mod tests;

#[cfg(test)]
type Markers = OutputMarkerIds;
