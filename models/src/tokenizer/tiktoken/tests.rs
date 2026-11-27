use std::path::PathBuf;

use super::*;

/// A four-byte alphabet with two merges and a 256-token special block.
fn fixture() -> Result<(PathBuf, Tokenizer)> {
    let directory = std::env::temp_dir().join(format!("libmir-tiktoken-{}", std::process::id()));
    fs::create_dir_all(&directory)?;
    // a, b, c, " ", "ab", "abc"
    let ranks = "YQ== 0\nYg== 1\nYw== 2\nIA== 3\nYWI= 4\nYWJj 5\n";
    fs::write(directory.join("tiktoken.model"), ranks)?;
    let config = serde_json::json!({
        "tokenizer_class": "TikTokenTokenizer",
        "added_tokens_decoder": {"7": {"content": "[EOS]", "special": true}},
    });
    fs::write(directory.join("tokenizer_config.json"), config.to_string())?;
    let tokenizer = tokenizer_from_files(
        &directory.join("tiktoken.model"),
        Some(&directory.join("tokenizer_config.json")),
    )?;
    Ok((directory, tokenizer))
}

#[test]
fn rebuilds_ranks_as_byte_level_merges() -> Result<()> {
    let (directory, tokenizer) = fixture()?;
    let ids =
        |text: &str| -> Result<Vec<u32>> { Ok(tokenizer.encode(text, false)?.get_ids().to_vec()) };
    // A whole pre-token in the ranks stays one token; others merge by rank.
    assert_eq!(ids("abc")?, [5]);
    assert_eq!(ids("abab")?, [4, 4]);
    assert_eq!(ids("cab ab")?, [2, 4, 3, 4]);
    assert_eq!(ids("ab[EOS]")?, [4, 7]);
    assert_eq!(tokenizer.token_to_id("<|reserved_token_6|>"), Some(6));
    assert_eq!(tokenizer.decode(&[2, 4, 3, 4], false)?, "cab ab");
    fs::remove_dir_all(directory)?;
    Ok(())
}

#[test]
fn decodes_padded_base64() -> Result<()> {
    assert_eq!(base64("YWJj")?, b"abc");
    assert_eq!(base64("YWI=")?, b"ab");
    assert_eq!(base64("IA==")?, b" ");
    assert!(base64("YW").is_err());
    // Padding is two `=` at most, and only at the end.
    for malformed in ["====", "a===", "YQ==YWJj", "=AAA"] {
        assert!(base64(malformed).is_err(), "{malformed}");
    }
    Ok(())
}
