use super::*;
use crate::engine::probe::history::pages::{Capture, View};

#[test]
#[ignore = "actual Qwen page-view allocation diagnosis; set MIRMIR_BENCH_MODEL"]
fn inspects_qwen_page_views() -> Result<()> {
    inspect_pages(2049, 30)
}

#[test]
#[ignore = "actual GPT-OSS full-layer page-view diagnosis; set MIRMIR_BENCH_MODEL"]
fn inspects_clamped_page_views() -> Result<()> {
    inspect_pages(129, 36)
}

fn inspect_pages(context: usize, expected: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    let reference = run_width_with_decode(&mut model, MoePrefill::Default, context, 3, plain)?;
    let mut views = Vec::new();
    let observed =
        run_width_with_decode(&mut model, MoePrefill::Default, context, 3, |model, inputs| {
            let started = Instant::now();
            let mut tokens = vec![Vec::new(); inputs.len()];
            for step in 0..32 {
                for (row, input) in inputs.iter().enumerate() {
                    tokens[row].push(input.token);
                }
                let capture = (step == 31).then(Capture::begin).transpose()?;
                let outputs = model.decode_batch(inputs)?;
                if let Some(capture) = capture {
                    views = capture.finish()?;
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
        })?;
    assert_eq!(reference.tokens, observed.tokens);
    assert_eq!(reference.schedule, observed.schedule);
    writeln!(
        std::io::stderr().lock(),
        "history.pages.capture: {}",
        json!({
            "context":context,"records":views.len(),"matches_reference":true,"observed":observed,
        })
    )?;
    assert_eq!(views.len(), expected);
    for view in &views {
        view.inspect(&model.stream)?;
    }
    Ok(())
}
