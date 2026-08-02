#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EncoderTensorRole {
    WordEmbedding,
    TokenTypeEmbedding,
    PositionEmbedding,
    EmbeddingNorm,
    Pooler,
    Classifier,
    Layer {
        index: usize,
        tensor: EncoderLayerTensorRole,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EncoderLayerTensorRole {
    Query,
    Key,
    Value,
    Qkv,
    AttentionOutput,
    AttentionNorm,
    MlpUpGate,
    MlpDown,
    MlpNorm,
}
