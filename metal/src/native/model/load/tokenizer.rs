use models::{
    layout::ModelLayout,
    tokenizer::{TextTokenizer, TokenizerInfo},
};

pub(super) fn tokenizer_info(layout: &ModelLayout) -> (Option<TokenizerInfo>, Option<String>) {
    TextTokenizer::from_layout(layout).map_or_else(
        |error| (None, Some(error.to_string())),
        |tokenizer| (Some(tokenizer.info()), None),
    )
}
