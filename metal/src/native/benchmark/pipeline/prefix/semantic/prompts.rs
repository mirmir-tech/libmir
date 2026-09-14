use models::tokenizer::TextTokenizer;

use super::Result;

pub(super) const ORDERS: [[usize; 6]; 3] =
    [[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [2, 0, 4, 1, 5, 3]];
pub(super) const VALUES: [&str; 6] = ["violet", "copper", "silver", "amber", "teal", "coral"];
pub(super) const LIMIT: usize = 128;
pub(super) const REPETITIONS: usize = 4;
pub(super) const CANCEL_STEP: usize = 16;
pub(super) const REFILL_STEP: usize = 32;

pub(super) fn build(
    tokenizer: &TextTokenizer,
    context: usize,
) -> Result<(Vec<u32>, Vec<Vec<u32>>)> {
    let encode = |s: &str| -> Result<Vec<u32>> {
        Ok(tokenizer.encode_with_special_tokens(s, false)?.token_ids)
    };
    let mut prefix: Vec<u32> = encode(
        "<|im_start|>user\nRead the registry below and use its values to answer the request.\nEntry A: violet. Entry B: copper.\n",
    )?;
    let middle: Vec<u32> = encode("\nEntry C: silver. Entry D: amber.\n")?;
    let end: Vec<u32> = encode("\nEntry E: teal. Entry F: coral.\nEnd of registry.\n")?;
    let filler: Vec<u32> =
        encode("Background note: the reading room opens each morning and closes every evening. ")?;
    assert!(!filler.is_empty());
    let first = context / 2 - prefix.len();
    prefix.extend(filler.iter().copied().cycle().take(first));
    prefix.extend(middle);
    let remaining = context - prefix.len() - end.len();
    prefix.extend(filler.iter().copied().cycle().take(remaining));
    prefix.extend(end);
    assert_eq!(prefix.len(), context);
    let prompts = ORDERS.iter().map(|order| {
        let keys = order.iter().map(|&index| ["A", "B", "C", "D", "E", "F"][index]).collect::<Vec<_>>().join(", ");
        let ending: Vec<u32> = encode(&format!(
            "Return the registry values for entries {keys}, in exactly that order. Repeat that sequence exactly four times in one JSON array of twenty-four strings. Return only the array, without explanation or markdown.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ))?;
        Ok(prefix.iter().copied().chain(ending).collect())
    }).collect::<Result<Vec<_>>>()?;
    Ok((prefix, prompts))
}

pub(super) fn correct(case: usize, text: &str) -> bool {
    let Ok(actual) = serde_json::from_str::<Vec<String>>(text.trim()) else {
        return false;
    };
    actual == expected(case)
}

pub(super) fn expected(case: usize) -> Vec<&'static str> {
    ORDERS[case]
        .iter()
        .cycle()
        .take(6 * REPETITIONS)
        .map(|&index| VALUES[index])
        .collect()
}
