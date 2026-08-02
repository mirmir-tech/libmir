use tokenizers::{
    Tokenizer,
    models::bpe::{BPE, Vocab},
};

use super::*;

#[test]
fn validates_complete_content_and_required_ids() -> Result<()> {
    let tokenizer = tokenizer()?;
    let report = inspect(&tokenizer, &BTreeMap::new(), &[2], Some(0), 3, &[1])?;

    assert_eq!(report.vocabulary_entries, 3);
    assert_eq!(report.max_token_id, 2);
    assert_eq!(report.required_token_ids, vec![0, 1, 2]);
    Ok(())
}

#[test]
fn rejects_ids_outside_the_embedding_table() -> Result<()> {
    let result = inspect(&tokenizer()?, &BTreeMap::new(), &[], None, 2, &[]);
    assert!(result.is_err());
    Ok(())
}

#[test]
fn rejects_a_required_id_absent_from_content() -> Result<()> {
    let result = inspect(&tokenizer()?, &BTreeMap::new(), &[], None, 4, &[3]);
    assert!(result.is_err());
    Ok(())
}

fn tokenizer() -> Result<Tokenizer> {
    let vocab = Vocab::from_iter([
        ("<pad>".to_owned(), 0),
        ("hello".to_owned(), 1),
        ("</s>".to_owned(), 2),
    ]);
    let model = BPE::builder().vocab_and_merges(vocab, Vec::new()).build()?;
    Ok(Tokenizer::new(model))
}
