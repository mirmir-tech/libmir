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
