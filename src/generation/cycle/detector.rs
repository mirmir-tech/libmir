use std::collections::HashMap;

const DEFAULT_MIN_TOKENS: usize = 192;
const MIN_PERIOD: usize = 4;
const MAX_PERIOD: usize = 64;
const CONSECUTIVE_REPEATS: usize = 3;
const PHRASE_LEN: usize = 12;
const PHRASE_REPEATS: u8 = 4;

#[derive(Debug, Clone, Copy)]
pub(super) struct CycleDetection {
    pub(super) span: usize,
    pub(super) kind: CycleKind,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum CycleKind {
    Consecutive,
    RecurringPhrase,
}

pub(super) struct CycleDetector {
    phrases: HashMap<[u32; PHRASE_LEN], u8>,
    seeded: bool,
    min_tokens: usize,
    consecutive_repeats: usize,
    phrase_repeats: u8,
}

impl Default for CycleDetector {
    fn default() -> Self {
        Self::new(DEFAULT_MIN_TOKENS, CONSECUTIVE_REPEATS, PHRASE_REPEATS)
    }
}

impl CycleDetector {
    pub(super) fn reasoning_exit(min_tokens: usize) -> Self {
        Self::new(min_tokens, 2, 3)
    }

    fn new(min_tokens: usize, consecutive_repeats: usize, phrase_repeats: u8) -> Self {
        Self {
            phrases: HashMap::new(),
            seeded: false,
            min_tokens: min_tokens.max(PHRASE_LEN),
            consecutive_repeats,
            phrase_repeats,
        }
    }

    pub(super) fn observe(&mut self, tokens: &[u32]) -> Option<CycleDetection> {
        if tokens.len() < self.min_tokens {
            return None;
        }
        if let Some(span) = consecutive_cycle(tokens, self.consecutive_repeats) {
            return Some(CycleDetection { span, kind: CycleKind::Consecutive });
        }
        if self.seeded {
            increment(&mut self.phrases, &tokens[tokens.len() - PHRASE_LEN..], self.phrase_repeats);
        } else {
            for phrase in tokens.windows(PHRASE_LEN) {
                increment(&mut self.phrases, phrase, self.phrase_repeats);
            }
            self.seeded = true;
        }
        let suffix = phrase(&tokens[tokens.len() - PHRASE_LEN..]);
        (self.phrases.get(&suffix).copied() == Some(self.phrase_repeats)).then_some(
            CycleDetection {
                span: PHRASE_LEN,
                kind: CycleKind::RecurringPhrase,
            },
        )
    }
}

fn consecutive_cycle(tokens: &[u32], repeats: usize) -> Option<usize> {
    let max_period = MAX_PERIOD.min(tokens.len() / repeats);
    (MIN_PERIOD..=max_period).find(|&period| {
        let suffix = &tokens[tokens.len() - period..];
        (2..=repeats).all(|repeat| {
            let end = tokens.len() - period * (repeat - 1);
            &tokens[end - period..end] == suffix
        })
    })
}

fn increment(counts: &mut HashMap<[u32; PHRASE_LEN], u8>, tokens: &[u32], repeats: u8) {
    let count = counts.entry(phrase(tokens)).or_default();
    *count = count.saturating_add(1).min(repeats);
}

fn phrase(tokens: &[u32]) -> [u32; PHRASE_LEN] {
    let mut phrase = [0; PHRASE_LEN];
    phrase.copy_from_slice(tokens);
    phrase
}
