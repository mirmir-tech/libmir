use models::{layout::ModelLayout, tokenizer::TextTokenizer};

use super::*;
use crate::native::benchmark::semantic;

pub(super) fn check(model: &mut LoadedModel) -> Result<()> {
    let config = BenchmarkConfig::from_env()?;
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let prompts=[
        "Explain how a hash table handles collisions. Give an example. Write about 150 words.",
        "Wyjaśnij uczniowi, jak działa fotosynteza. Napisz około 150 słów.",
        "Write a short story about a lighthouse keeper finding a message in a bottle. Write about 150 words.",
        "Compare commuting by bicycle and bus. Discuss cost and weather. Write about 150 words.",
        "Describe how to cook lentil soup. Explain the order of the steps. Write about 150 words.",
    ].map(|question| tokenizer.encode_with_special_tokens(&format!(
        "<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ),false).map(|p|p.token_ids)).into_iter().collect::<std::result::Result<Vec<_>,_>>()?;
    let stop = tokenizer.stop_token_ids();
    assert!(!stop.is_empty());
    for width in [1, 3, 5] {
        let mut reference = None;
        for plan in [Native, PackedDecode, PackedPrefill] {
            model.clear_prefix_cache();
            model.stream.set_gdn_execution(if plan == PackedPrefill {
                plan
            } else {
                Native
            });
            let before = gdn_experiment_calls();
            let mut active = semantic::start(model, &prompts[..width])?;
            model.stream.set_gdn_execution(if plan == PackedDecode {
                plan
            } else {
                Native
            });
            let mut tokens = vec![Vec::new(); width];
            let mut complete = vec![false; width];
            for _ in 0..64 {
                let mut pending = Vec::new();
                for (row, input) in active {
                    if stop.contains(&input.token) {
                        model.release_session(input.session)?;
                        complete[row] = true;
                    } else {
                        tokens[row].push(input.token);
                        pending.push((row, input));
                    }
                }
                let inputs = pending.iter().map(|(_, input)| *input).collect::<Vec<_>>();
                if inputs.is_empty() {
                    active = pending;
                    break;
                }
                let outputs = if inputs.len() == 1 {
                    let i = inputs[0];
                    vec![model.decode(i.session, i.token, i.sampling)?]
                } else {
                    model.decode_batch(&inputs)?
                };
                for ((_, input), output) in pending.iter_mut().zip(outputs) {
                    input.token = greedy_token(&output)?;
                }
                active = pending;
            }
            model.stream.synchronize()?;
            for (_, input) in active {
                model.release_session(input.session)?;
            }
            let calls = gdn_experiment_calls() - before;
            model.stream.set_gdn_execution(Native);
            let matches = reference
                .as_ref()
                .is_none_or(|prior| prior == &(tokens.clone(), complete.clone()));
            let answers = tokens
                .iter()
                .map(|ids| tokenizer.decode(ids))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            writeln!(
                std::io::stderr().lock(),
                "gdn.quality: {}",
                json!({
                    "width":width,"plan":plan,"calls":calls,"matches_reference":matches,
                    "tokens":tokens,"complete":complete,"answers":answers,
                })
            )?;
            assert!(matches, "natural-prompt GDN output changed");
            if plan == Native {
                assert_eq!(calls, 0);
            } else {
                assert!(calls > 0);
            }
            if reference.is_none() {
                reference = Some((tokens, complete));
            }
        }
    }
    Ok(())
}
