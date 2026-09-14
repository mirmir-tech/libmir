mod cost;
mod logits;
mod sharing;

use std::{io::Write, sync::Arc, time::Instant};

use models::{layout::ModelLayout, tokenizer::TextTokenizer};

use crate::{
    config::RouterPrecision,
    native::{error::Result, model::LoadedModel},
};

#[test]
#[ignore = "real model semantic smoke and elapsed observations for both router precisions"]
fn compares_router_precision_answers() -> Result<()> {
    let config = super::BenchmarkConfig::from_env()?;
    let tokenizer = TextTokenizer::from_layout(&ModelLayout::inspect(&config.model)?)?;
    let prompts = prompts(&tokenizer)?;
    let mut failures = 0;
    for precision in [RouterPrecision::Model, RouterPrecision::Float32] {
        let mut settings = super::diagnostics::isolated_config();
        let metal = Arc::make_mut(&mut settings);
        metal.tuning.mode = runtime::tuning::TuningMode::Disabled;
        metal.diagnostics.router_precision = precision;
        metal.set_max_batch_requests(5);
        let mut model = LoadedModel::load_with_config(&config.manifest(), settings, &mut |_| {})?;
        for width in [1, 5] {
            let started = Instant::now();
            let mut answers = Vec::new();
            for batch in prompts.chunks(width) {
                for tokens in
                    super::semantic::generate(&mut model, batch, &tokenizer.stop_token_ids())?
                {
                    answers.push(tokenizer.decode(&tokens)?);
                }
            }
            let wrong = answers
                .iter()
                .zip(&CASES)
                .enumerate()
                .filter(|(_, (answer, (_, expected)))| answer.trim() != *expected)
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            failures += wrong.len();
            writeln!(
                std::io::stderr().lock(),
                "router.answers: {}",
                serde_json::json!({
                "precision":format!("{precision:?}"), "batch":width, "answers":answers,"wrong":wrong,
                "elapsed_ms_observational": started.elapsed().as_secs_f64()*1000.0 })
            )?;
            assert!(model.sessions.is_empty());
        }
        drop(model);
        crate::engine::clear_memory_cache()?;
    }
    assert_eq!(
        failures, 0,
        "router candidate/baseline failed semantic smoke; inspect per-mode records"
    );
    Ok(())
}

const CASES: [(&str, &str); 10] = [
    ("What is 17 + 25? Answer only the number.", "42"),
    ("What is 9 multiplied by 7? Answer only the number.", "63"),
    (
        "A shelf has 12 books. Three are borrowed. How many remain? Answer only the number.",
        "9",
    ),
    ("Which number is largest: 8, 19, 3, 12? Answer only the number.", "19"),
    ("What is 144 divided by 12? Answer only the number.", "12"),
    ("Ile wynosi 18 plus 24? Odpowiedz wyłącznie liczbą.", "42"),
    ("Ile wynosi 8 razy 6? Odpowiedz wyłącznie liczbą.", "48"),
    ("What does Python print for len([10, 20, 30, 40])? Answer only the number.", "4"),
    ("What does Python print for sum([2, 3, 5])? Answer only the number.", "10"),
    ("How many sides does a triangle have? Answer only the number.", "3"),
];

fn prompts(tokenizer: &TextTokenizer) -> Result<Vec<Vec<u32>>> {
    CASES.iter().map(|(question,_)| {
        Ok(tokenizer.encode_with_special_tokens(&format!("<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"), false)?.token_ids)
    }).collect()
}
