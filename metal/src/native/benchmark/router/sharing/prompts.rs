use models::tokenizer::TextTokenizer;

use super::Result;

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Suite {
    Diverse,
    Related,
}

impl Suite {
    pub fn questions(self) -> [&'static str; 5] {
        match self {
            Self::Diverse => [
                "Explain to a beginner how a hash table stores and retrieves values. Include an example, collisions, and a practical trade-off. Write about 250 words.",
                "Wyjaśnij, jak rośliny wykorzystują światło podczas fotosyntezy. Omów rolę wody, dwutlenku węgla i chlorofilu. Napisz około 250 słów dla ucznia szkoły średniej.",
                "Write a short story of about 250 words about a lighthouse keeper who discovers a message in a bottle. Include dialogue and a clear ending.",
                "Compare bicycle commuting and taking the bus in a medium-sized city. Discuss convenience, weather, cost, and planning. Write about 250 words.",
                "Describe how to make a vegetable soup from carrots, potatoes, onions, and lentils. Explain the sequence of steps and why each matters. Write about 250 words.",
            ],
            Self::Related => [
                "Explain how to implement an LRU cache in Rust. Discuss the data structures, lookup, eviction, and ownership. Write about 250 words with a small example.",
                "Explain how to implement an LRU cache in Python. Discuss the data structures, lookup, eviction, and complexity. Write about 250 words with a small example.",
                "Explain how to implement an LRU cache in JavaScript. Discuss the data structures, lookup, eviction, and complexity. Write about 250 words with a small example.",
                "Explain how to implement an LRU cache in Go. Discuss the data structures, lookup, eviction, and concurrency. Write about 250 words with a small example.",
                "Explain how to implement an LRU cache in Java. Discuss the data structures, lookup, eviction, and concurrency. Write about 250 words with a small example.",
            ],
        }
    }

    pub fn encode(self, tokenizer: &TextTokenizer) -> Result<Vec<Vec<u32>>> {
        self.questions().iter().map(|question| {
            Ok(tokenizer.encode_with_special_tokens(&format!(
                "<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ), false)?.token_ids)
        }).collect()
    }
}
