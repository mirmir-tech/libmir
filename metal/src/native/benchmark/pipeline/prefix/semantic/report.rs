use super::*;

#[derive(Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Mode {
    ColdScalar,
    MixedPrefix,
}

pub(super) fn check(
    references: &[Answer],
    answers: &[Answer],
    tokenizer: &TextTokenizer,
) -> Result<()> {
    let mut wrong = 0;
    let mut completed = 0;
    let mut exact = 0;
    for (mode, results) in [(Mode::ColdScalar, references), (Mode::MixedPrefix, answers)] {
        for answer in results {
            let text = tokenizer.decode(&answer.tokens)?;
            let reference = &references[answer.case];
            let mismatch =
                answer.tokens.iter().zip(&reference.tokens).position(|(a, b)| a != b).or_else(
                    || {
                        (answer.finish != Finish::Cancelled
                            && answer.tokens.len() != reference.tokens.len())
                        .then_some(answer.tokens.len().min(reference.tokens.len()))
                    },
                );
            let correct = answer.finish == Finish::Eos && prompts::correct(answer.case, &text);
            if answer.finish != Finish::Cancelled {
                wrong += usize::from(!correct);
                if mode == Mode::MixedPrefix {
                    completed += 1;
                    exact += usize::from(mismatch.is_none() && answer.finish == reference.finish);
                }
            }
            writeln!(
                std::io::stderr().lock(),
                "prefix.quality.answer: {}",
                serde_json::json!({
                    "mode":mode,"case":answer.case,"finish":answer.finish,"text":text,"tokens":answer.tokens,
                    "correct":correct,"first_token_mismatch":mismatch,
                    "expected":prompts::expected(answer.case)
                })
            )?;
        }
    }
    writeln!(
        std::io::stderr().lock(),
        "prefix.quality.result: {}",
        serde_json::json!({
            "semantic_wrong":wrong,"mixed_completed":completed,"mixed_exact":exact
        })
    )?;
    assert!(
        references
            .iter()
            .chain(answers)
            .filter(|a| a.finish != Finish::Cancelled)
            .all(|a| a.tokens.len() >= 64),
        "semantic probe did not reach the earlier divergence region"
    );
    assert_eq!(completed, references.len());
    assert_eq!(wrong, 0, "semantic errors preserved in per-answer records");
    Ok(())
}
