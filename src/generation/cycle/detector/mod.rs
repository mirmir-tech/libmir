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
    min_tokens: usize,
    policy: DetectionPolicy,
}

enum DetectionPolicy {
    Recovery {
        phrases: HashMap<[u32; PHRASE_LEN], u8>,
        seeded: bool,
    },
    ReasoningExit,
}

impl Default for CycleDetector {
    fn default() -> Self {
        Self {
            min_tokens: DEFAULT_MIN_TOKENS,
            policy: DetectionPolicy::Recovery { phrases: HashMap::new(), seeded: false },
        }
    }
}

impl CycleDetector {
    pub(super) fn reasoning_exit(min_tokens: usize) -> Self {
        Self {
            min_tokens: min_tokens.max(PHRASE_LEN),
            policy: DetectionPolicy::ReasoningExit,
        }
    }

    pub(super) fn observe(&mut self, tokens: &[u32]) -> Option<CycleDetection> {
        if tokens.len() < self.min_tokens {
            return None;
        }
        if let Some(span) = consecutive_cycle(tokens, CONSECUTIVE_REPEATS) {
            return Some(CycleDetection { span, kind: CycleKind::Consecutive });
        }
        // Repeating source evidence during a comparison is not a reasoning
        // loop. Only consecutive cycles justify forcing a channel delimiter;
        // keep scattered-phrase recovery for the non-exit policy.
        let DetectionPolicy::Recovery { phrases, seeded } = &mut self.policy else {
            return None;
        };
        if *seeded {
            increment(phrases, &tokens[tokens.len() - PHRASE_LEN..], PHRASE_REPEATS);
        } else {
            for phrase in tokens.windows(PHRASE_LEN) {
                increment(phrases, phrase, PHRASE_REPEATS);
            }
            *seeded = true;
        }
        let suffix = phrase(&tokens[tokens.len() - PHRASE_LEN..]);
        (phrases.get(&suffix).copied() == Some(PHRASE_REPEATS)).then_some(CycleDetection {
            span: PHRASE_LEN,
            kind: CycleKind::RecurringPhrase,
        })
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

#[cfg(test)]
mod tests;
