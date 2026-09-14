use super::*;
use crate::config::MoeDecodeProbe;

#[test]
#[ignore = "actual unsorted MoE decomposition at two decode steps and three layers"]
fn diagnoses_unsorted_moe() -> Result<()> {
    let mut model = tiles::load_model()?;
    let observation = run_with_decode(&mut model, MoePrefill::Default, 2049, |model, inputs| {
        measure(model, inputs, |step| {
            [8, 24].contains(&step).then_some(MoeDecodeProbe::Components { step })
        })
    })?;
    writeln!(std::io::stderr().lock(), "decode.moe: {}", json!({"observation": observation}))?;
    Ok(())
}

#[test]
#[ignore = "one unsorted MXFP4 gate/up fusion comparison on actual/hot decode routing"]
fn compares_unsorted_gate_up() -> Result<()> {
    let mut model = tiles::load_model()?;
    let observation = run_with_decode(&mut model, MoePrefill::Default, 2049, |model, inputs| {
        measure(model, inputs, |step| {
            [8, 24].contains(&step).then_some(MoeDecodeProbe::GateUpFusion { step })
        })
    })?;
    writeln!(
        std::io::stderr().lock(),
        "decode.fusion: {}",
        json!({"observation": observation})
    )?;
    Ok(())
}

#[test]
#[ignore = "one native MoE submission/logging drift trace; attach on ready marker"]
fn diagnoses_submission_drift() -> Result<()> {
    let mut model = tiles::load_model()?;
    let observation = run_with_decode(&mut model, MoePrefill::Default, 2049, |model, inputs| {
        measure(model, inputs, |step| {
            (step == 8).then_some(MoeDecodeProbe::SubmissionDrift { step })
        })
    })?;
    writeln!(
        std::io::stderr().lock(),
        "decode.drift: {}",
        json!({"observation": observation})
    )?;
    Ok(())
}

#[test]
#[ignore = "one fixed bank-residency and variant-switching drift diagnosis"]
fn diagnoses_fusion_drift() -> Result<()> {
    let mut model = tiles::load_model()?;
    let observation = run_with_decode(&mut model, MoePrefill::Default, 2049, |model, inputs| {
        measure(model, inputs, |step| {
            (step == 8).then_some(MoeDecodeProbe::FusionDrift { step })
        })
    })?;
    writeln!(
        std::io::stderr().lock(),
        "decode.fusion_drift: {}",
        json!({"observation": observation})
    )?;
    Ok(())
}

#[test]
#[ignore = "one fixed baseline sample-length diagnosis, without Instruments"]
fn diagnoses_submission_lengths() -> Result<()> {
    let mut model = tiles::load_model()?;
    let observation = run_with_decode(&mut model, MoePrefill::Default, 2049, |model, inputs| {
        measure(model, inputs, |step| {
            (step == 8).then_some(MoeDecodeProbe::SubmissionLengths { step })
        })
    })?;
    writeln!(
        std::io::stderr().lock(),
        "decode.lengths: {}",
        json!({"observation": observation})
    )?;
    Ok(())
}

#[test]
#[ignore = "one fixed continuous-window baseline stability gate"]
fn diagnoses_continuous_windows() -> Result<()> {
    let mut model = tiles::load_model()?;
    let observation = run_with_decode(&mut model, MoePrefill::Default, 2049, |model, inputs| {
        measure(model, inputs, |step| {
            (step == 8).then_some(MoeDecodeProbe::ContinuousWindows { step })
        })
    })?;
    writeln!(
        std::io::stderr().lock(),
        "decode.windows: {}",
        json!({"observation": observation})
    )?;
    Ok(())
}

fn measure(
    model: &mut LoadedModel,
    inputs: &mut [DecodeInput],
    probe: impl Fn(usize) -> Option<MoeDecodeProbe>,
) -> Result<Observation> {
    let started = Instant::now();
    let mut tokens = vec![Vec::with_capacity(32); inputs.len()];
    for step in 0..32 {
        for (row, input) in inputs.iter().enumerate() {
            tokens[row].push(input.token);
        }
        let probe = probe(step);
        if matches!(probe, Some(MoeDecodeProbe::SubmissionDrift { .. }))
            || (matches!(probe, Some(MoeDecodeProbe::FusionDrift { .. }))
                && std::env::var_os("MIRMIR_BENCH_TRACE_DIR").is_some())
        {
            model.stream.synchronize()?;
            capture::wait_for_trace()?;
        }
        model.stream.set_moe_decode_probe(probe);
        let result = model.decode_batch(inputs);
        model.stream.set_moe_decode_probe(None);
        let outputs = result?;
        for (input, output) in inputs.iter_mut().zip(outputs) {
            input.token = greedy_token(&output)?;
        }
    }
    model.stream.synchronize()?;
    Ok(Observation {
        tokens,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    })
}
