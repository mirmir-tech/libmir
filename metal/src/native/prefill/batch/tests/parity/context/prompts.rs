use super::*;

pub(super) const ANSWERS: [&str; 5] = ["37", "58", "23", "14", "91"];

pub(super) fn build(tokenizer: &TextTokenizer, context: usize) -> Result<Vec<Vec<u32>>> {
    let cases = [
        (
            "The access code for station ORION is 37.",
            "What is the access code for station ORION? Answer only the number.",
        ),
        (
            "Kod szafki o nazwie BRZOZA to 58.",
            "Jaki jest kod szafki BRZOZA? Odpowiedz wyłącznie liczbą.",
        ),
        (
            "The expedition named AMBER has 23 members.",
            "How many members does expedition AMBER have? Answer only the number.",
        ),
        (
            "The Python program is: values = [2, 5, 7]; result = sum(values)",
            "What is the value of result in the Python program above? Answer only the number.",
        ),
        (
            "W magazynie DELTA zapisano liczbę 91 jako numer dostawy.",
            "Jaki numer dostawy zapisano w magazynie DELTA? Odpowiedz wyłącznie liczbą.",
        ),
    ];
    let encode = |text: &str| -> Result<Vec<u32>> {
        Ok(tokenizer.encode_with_special_tokens(text, false)?.token_ids)
    };
    let filler = encode(
        "Unrelated field notes: the forest has tall trees. Water flows toward the valley. Researchers describe local weather and observe birds. These notes contain no access codes or delivery numbers.\n",
    )?;
    cases.iter().enumerate().map(|(row, (fact, question))| {
        let header = encode("<|im_start|>user\nRead the record and answer the final question using its relevant fact.\n")?;
        let fact = encode(&format!("\nRelevant record: {fact}\n"))?;
        let ending = encode(&format!("\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"))?;
        let padding = context.checked_sub(header.len() + fact.len() + ending.len())
            .ok_or_else(|| Error::Benchmark("context too short for retrieval case".into()))?;
        let before = padding * (row % 3) / 2;
        Ok(header.into_iter()
            .chain(filler.iter().copied().cycle().take(before))
            .chain(fact)
            .chain(filler.iter().copied().cycle().take(padding - before))
            .chain(ending).collect())
    }).collect()
}
