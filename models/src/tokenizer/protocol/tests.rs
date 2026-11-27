use std::collections::BTreeMap;

use tokenizers::{Tokenizer, models::wordlevel::WordLevel};

use super::ProtocolTokenIds;

#[test]
fn resolves_protocol_ids_once_without_duplicates() {
    let tokenizer = Tokenizer::new(WordLevel::default());
    let added = BTreeMap::from([
        ("<|channel|>".to_owned(), 2),
        ("<|message|>".to_owned(), 3),
        ("<|start|>".to_owned(), 4),
    ]);

    let protocol = ProtocolTokenIds::resolve(&tokenizer, &added, &[7, 1], &[8, 7]);

    assert_eq!(protocol.stop, [7, 1, 8]);
    assert_eq!(protocol.output.channel, [2]);
    assert_eq!(protocol.output.channel_body, [3]);
    assert_eq!(protocol.output.turn_start, [4]);
}

fn word_level(words: &[&str]) -> Result<Tokenizer, Box<dyn std::error::Error + Send + Sync>> {
    let vocab = words.iter().zip(0_u32..).map(|(word, id)| ((*word).to_owned(), id));
    let model = WordLevel::builder().vocab(vocab.collect()).unk_token(words[0].into()).build()?;
    Ok(Tokenizer::new(model))
}

/// Phi-3's `<|end|>` stops generation; Harmony's only ends a message.
#[test]
fn end_stops_unless_the_vocabulary_is_harmony()
-> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let phi = word_level(&["<unk>", "<|end|>"])?;
    let harmony = word_level(&["<unk>", "<|end|>", "<|return|>"])?;

    let stop = |tokenizer: &Tokenizer| {
        ProtocolTokenIds::resolve(tokenizer, &BTreeMap::new(), &[], &[]).stop
    };
    assert_eq!(stop(&phi), [1]);
    assert!(stop(&harmony).is_empty());
    Ok(())
}
