use super::*;

#[test]
fn discovers_every_supported_tokenizer_layout() -> Result<()> {
    let json = TokenizerAssets::discover(&BTreeMap::from([(TOKENIZER_JSON.into(), 10)]))?;
    assert_eq!(json.kind, TokenizerKind::TokenizerJson);

    let sentencepiece = TokenizerAssets::discover(&BTreeMap::from([(TOKENIZER_MODEL.into(), 20)]))?;
    assert_eq!(sentencepiece.kind, TokenizerKind::SentencePieceModel);

    let bpe = TokenizerAssets::discover(&BTreeMap::from([
        (VOCAB.into(), 30),
        (MERGES.into(), 40),
        ("tokenizer_config.json".into(), 5),
    ]))?;
    assert_eq!(bpe.kind, TokenizerKind::BpeVocab);
    assert_eq!(bpe.total_bytes, 75);
    Ok(())
}

#[test]
fn rejects_vocab_without_merges() -> Result<()> {
    let error = TokenizerAssets::discover(&BTreeMap::from([(VOCAB.into(), 30)]))
        .err()
        .ok_or_else(|| ModelsError::InvalidConfig("incomplete BPE tokenizer accepted".into()))?;
    assert!(error.to_string().contains(MERGES));
    Ok(())
}
