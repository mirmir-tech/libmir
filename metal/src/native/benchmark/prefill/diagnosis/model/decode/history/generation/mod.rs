use super::*;
use crate::{
    config::DecodeReservation,
    engine::probe::history::pages::{Capture, View},
};
mod cost;
mod resident;
mod resources;

#[test]
#[ignore = "Qwen reserved decode tail allocation/numerical gate; set MIRMIR_BENCH_MODEL"]
fn inspects_qwen_generation_reservation() -> Result<()> {
    inspect(2049, 30)
}

#[test]
#[ignore = "GPT-OSS reserved decode tail allocation/numerical gate; set MIRMIR_BENCH_MODEL"]
fn inspects_clamped_generation_reservation() -> Result<()> {
    inspect(129, 36)
}

fn inspect(context: usize, expected: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    let mut reference = None;
    for plan in [DecodeReservation::OnePage, DecodeReservation::GenerationBudget] {
        model.stream.set_decode_reservation(plan);
        let mut views = Vec::new();
        let result = run_budget_with_decode(
            &mut model,
            MoePrefill::Default,
            context,
            3,
            std::num::NonZeroUsize::new(32),
            |model, inputs| capture(model, inputs, &mut views),
        );
        model.stream.set_decode_reservation(DecodeReservation::OnePage);
        let observation = result?;
        let matches = reference.as_ref().is_none_or(|r: &super::super::super::Observation| {
            r.tokens == observation.tokens && r.schedule == observation.schedule
        });
        let aliases = views
            .iter()
            .map(View::aliases_arena)
            .collect::<crate::engine::Result<Vec<_>>>()?;
        writeln!(
            std::io::stderr().lock(),
            "generation.capture: {}",
            json!({
                "plan":plan,"context":context,"matches_reference":matches,"aliases":aliases,"observation":observation,
            })
        )?;
        assert!(matches);
        assert_eq!(views.len(), expected);
        assert!(aliases.iter().all(|&alias| alias == (plan != DecodeReservation::OnePage)));
        for view in &views {
            view.inspect(&model.stream)?;
        }
        if reference.is_none() {
            reference = Some(observation);
        }
    }
    Ok(())
}

fn capture(
    model: &mut LoadedModel,
    inputs: &mut [DecodeInput],
    views: &mut Vec<View>,
) -> Result<Observation> {
    let started = Instant::now();
    let mut tokens = vec![Vec::new(); inputs.len()];
    for step in 0..32 {
        for (row, input) in inputs.iter().enumerate() {
            tokens[row].push(input.token);
        }
        let capture = (step == 31).then(Capture::begin).transpose()?;
        let outputs = model.decode_batch(inputs)?;
        if let Some(capture) = capture {
            *views = capture.finish()?;
        }
        for (input, output) in inputs.iter_mut().zip(outputs) {
            input.token = greedy_token(&output)?;
        }
    }
    model
        .stream
        .eval_many(&views.iter().flat_map(View::roots).collect::<Vec<_>>())?;
    model.stream.synchronize()?;
    Ok(Observation {
        tokens,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}
