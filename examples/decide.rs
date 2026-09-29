#![allow(clippy::print_stdout)]

//! Measures Laya decisions on a workshop conversation.
//!
//! ```text
//! cargo run --release --example decide --features cpu,metal -- <checkpoint> [cpu|metal] [turns]
//! ```

use std::{env, time::Instant};

use libmir::decision::{
    ChoiceOption, DecisionBackend, DecisionModel, DecisionState, Question, Verdicts,
};
use serde_json::json;

const RUNS: usize = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let root = arguments.next().ok_or("usage: decide <checkpoint> [cpu|metal] [turns]")?;
    let backend = match arguments.next().as_deref() {
        None | Some("cpu") => DecisionBackend::Cpu,
        #[cfg(feature = "metal")]
        Some("metal") => DecisionBackend::Metal,
        Some(other) => return Err(format!("backend {other} is not enabled").into()),
    };
    let turns: usize = arguments.next().map_or(Ok(4), |turns| turns.parse())?;

    let started = Instant::now();
    let model = DecisionModel::load(&root, backend)?;
    println!("backend {backend:?}, loaded in {:.2?}", started.elapsed());

    let state = DecisionState::conversation(&conversation(turns))?;
    let questions = questions()?;
    let rows = model.rows(&state, &questions)?;
    let tokens: usize = rows.iter().map(|row| row.tokens.len()).sum();
    let dropped = rows.first().map_or(0, |row| row.state_tokens_dropped);
    println!("{} questions, {tokens} tokens, {dropped} state tokens dropped", rows.len());

    let mut timings = Vec::with_capacity(RUNS);
    let mut answers = Vec::new();
    for _ in 0..RUNS {
        let started = Instant::now();
        answers = model.decide(&state, &questions)?;
        timings.push(started.elapsed());
    }
    timings.sort_unstable();
    println!(
        "median {:.2?}, fastest {:.2?}, slowest {:.2?}",
        timings[RUNS / 2],
        timings[0],
        timings[RUNS - 1]
    );
    for (question, answer) in questions.iter().zip(&answers) {
        println!(
            "{:<55} {:?} (confidence {:.3})",
            question.instruction(),
            answer.verdict,
            answer.confidence
        );
    }
    Ok(())
}

fn conversation(turns: usize) -> Vec<serde_json::Value> {
    let lines = [
        ("Anna", "Mamy już decyzję co do kolejki zdarzeń w collab?"),
        (
            "agent Anny",
            "Sprawdziłem outbox: BIGSERIAL daje porządek, ale nie ma kursora per uczestnik.",
        ),
        ("Piotr", "Poczekajcie chwilę, zaraz wrzucę szkic kontraktu gRPC."),
        ("Anna", "Agencie Piotra, czy daemon umie już wznawiać sesję po kursorze?"),
    ];
    (0..turns)
        .map(|index| {
            let (speaker, text) = lines[(index + lines.len() - turns % lines.len()) % lines.len()];
            json!({"speaker": speaker, "text": text})
        })
        .collect()
}

fn questions() -> Result<Vec<Question>, Box<dyn std::error::Error>> {
    Ok(vec![
        Question::choice(
            "Kto jest adresatem ostatniej wypowiedzi?",
            ["anna", "piotr", "agent_anny", "agent_piotra", "wszyscy"]
                .into_iter()
                .map(|label| ChoiceOption::new(label, None))
                .collect(),
        )?,
        Question::yes_no("Czy ostatnia wypowiedź zadaje otwarte pytanie?", Verdicts::default())?,
        Question::yes_no("Czy ktoś prosi, żeby poczekać?", Verdicts::default())?,
        Question::score(
            "Jak pilna jest ostatnia wypowiedź?",
            vec!["wcale".into(), "trochę".into(), "pilne".into(), "blokuje pracę".into()],
        )?,
    ])
}
