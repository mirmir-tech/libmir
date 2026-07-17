use std::{fs, path::Path};

use tokenizers::{
    AddedToken, Tokenizer,
    decoders::{
        DecoderWrapper, byte_fallback::ByteFallback, metaspace::Metaspace, sequence::Sequence,
    },
    models::{
        bpe::{BPE, Merges, Vocab},
        unigram::Unigram,
    },
    pre_tokenizers::metaspace::PrependScheme,
};

use crate::{
    error::{ModelsError, Result},
    tokenizer::sentencepiece::proto::parse,
};

mod proto;

#[derive(Debug, Clone)]
pub struct SentencePieceModel {
    pub pieces: Vec<SpPiece>,
    pub model_type: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct SpPiece {
    pub piece: String,
    pub score: f32,
    pub kind: PieceKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

pub fn tokenizer_from_file(path: impl AsRef<Path>) -> Result<Tokenizer> {
    tokenizer_from_bytes(&fs::read(path)?)
}

fn tokenizer_from_bytes(bytes: &[u8]) -> Result<Tokenizer> {
    let model = parse(bytes)?;
    let mut tokenizer = match model.model_type {
        Some(1) => Tokenizer::new(unigram(&model)?),
        Some(2) | None => Tokenizer::new(bpe(&model)?),
        Some(other) => {
            return Err(ModelsError::InvalidConfig(format!(
                "unsupported SentencePiece model type {other}"
            )));
        },
    };
    tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
    tokenizer.with_decoder(Some(decoder()));
    let special = special_tokens(&model);
    let _added = tokenizer.add_special_tokens(&special);
    Ok(tokenizer)
}

impl PieceKind {
    pub const fn from_u64(value: u64) -> Self {
        match value {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

fn unigram(model: &SentencePieceModel) -> Result<Unigram> {
    let vocab = model
        .pieces
        .iter()
        .map(|piece| (piece.piece.clone(), f64::from(piece.score)))
        .collect();
    let unk_id = model.pieces.iter().position(|piece| piece.kind == PieceKind::Unknown);
    Ok(Unigram::from(vocab, unk_id, has_byte_fallback(model))?)
}

fn bpe(model: &SentencePieceModel) -> Result<BPE> {
    let mut vocab = Vocab::default();
    for (index, piece) in model.pieces.iter().enumerate() {
        let _old = vocab.insert(piece.piece.clone(), u32::try_from(index)?);
    }
    let mut builder = BPE::builder()
        .vocab_and_merges(vocab, merges(model))
        .byte_fallback(has_byte_fallback(model))
        .fuse_unk(true);
    if let Some(unknown) = model.pieces.iter().find(|piece| piece.kind == PieceKind::Unknown) {
        builder = builder.unk_token(unknown.piece.clone());
    }
    Ok(builder.build()?)
}

fn merges(model: &SentencePieceModel) -> Merges {
    let mut merges = Vec::new();
    let ranks = ranks(model);
    for piece in model.pieces.iter().filter(|piece| piece.kind == PieceKind::Normal) {
        if let Some((left, right)) = best_split(&piece.piece, &ranks) {
            merges.push((left, right));
        }
    }
    merges
}

fn ranks(model: &SentencePieceModel) -> std::collections::HashMap<&str, usize> {
    model
        .pieces
        .iter()
        .enumerate()
        .map(|(index, piece)| (piece.piece.as_str(), index))
        .collect()
}

fn best_split(
    piece: &str,
    ranks: &std::collections::HashMap<&str, usize>,
) -> Option<(String, String)> {
    let mut best = None;
    for split in char_boundaries(piece) {
        let left = &piece[..split];
        let right = &piece[split..];
        let score = ranks
            .get(left)
            .zip(ranks.get(right))
            .map(|(left, right)| ((*left).max(*right), left.saturating_add(*right)));
        if let Some(score) = score
            && best.as_ref().is_none_or(|(best_score, _, _)| score < *best_score)
        {
            best = Some((score, left.to_owned(), right.to_owned()));
        }
    }
    best.map(|(_, left, right)| (left, right))
}

fn char_boundaries(piece: &str) -> impl Iterator<Item = usize> + '_ {
    piece.char_indices().map(|(index, _)| index).filter(|index| *index > 0)
}

fn decoder() -> Sequence {
    Sequence::new(vec![
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Metaspace(Metaspace::new('▁', PrependScheme::Always, true)),
    ])
}

fn special_tokens(model: &SentencePieceModel) -> Vec<AddedToken> {
    model
        .pieces
        .iter()
        .filter(|piece| matches!(piece.kind, PieceKind::Unknown | PieceKind::Control))
        .map(|piece| AddedToken::from(piece.piece.clone(), true))
        .collect()
}

fn has_byte_fallback(model: &SentencePieceModel) -> bool {
    model.pieces.iter().any(|piece| piece.kind == PieceKind::Byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_bpe_merges_from_sentencepiece_order() {
        let model = SentencePieceModel {
            model_type: Some(2),
            pieces: vec![
                piece("<unk>", 0.0, PieceKind::Unknown),
                piece("ab", 0.0, PieceKind::Normal),
                piece("a", -10.0, PieceKind::Normal),
                piece("b", -11.0, PieceKind::Normal),
            ],
        };

        assert_eq!(merges(&model), vec![("a".into(), "b".into())]);
    }

    fn piece(piece: &str, score: f32, kind: PieceKind) -> SpPiece {
        SpPiece { piece: piece.into(), score, kind }
    }
}
