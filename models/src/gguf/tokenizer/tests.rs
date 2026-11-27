use serde_json::json;
use tokenizers::Tokenizer;

use super::prepend_bos;

#[test]
fn an_automatic_bos_leads_every_encoding_after_the_byte_level_step() -> crate::error::Result<()> {
    let byte_level = json!({ "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false });
    let post = prepend_bos(&byte_level, &(0, "<BOS>".to_owned()));
    let tokenizer = json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [{ "id": 0, "content": "<BOS>", "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true }],
        "normalizer": null, "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": post, "decoder": null,
        "model": { "type": "WordLevel", "vocab": { "<BOS>": 0, "hi": 1 }, "unk_token": "hi" },
    });
    let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&tokenizer)?)
        .map_err(|error| crate::ModelsError::InvalidGguf(error.to_string()))?;
    let encode = |special| {
        tokenizer
            .encode("hi", special)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| crate::ModelsError::InvalidGguf(error.to_string()))
    };
    assert_eq!(encode(true)?, [0, 1]);
    assert_eq!(encode(false)?, [1]);
    Ok(())
}
