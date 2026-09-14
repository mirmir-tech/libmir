mod full;

use std::io::BufRead;

use super::*;
use crate::config::DecodeReservation;

#[derive(Debug, thiserror::Error)]
enum DriverError {
    #[error(transparent)]
    Native(#[from] crate::native::error::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("sustained benchmark exceeded its limit")]
    Limit,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Warmup,
    Validation,
    Conditioning,
    Measured,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Variant {
    Baseline,
    Candidate,
}

impl Variant {
    fn policy(&self) -> DecodeReservation {
        match self {
            Self::Baseline => DecodeReservation::OnePage,
            Self::Candidate => DecodeReservation::GenerationBudget,
        }
    }
}

#[derive(Clone, Copy, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Workload {
    Sample32,
    Complete256,
}

#[derive(serde::Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Command {
    Run { phase: Phase, variant: Variant },
    Stop,
}

fn epoch() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[test]
#[ignore = "bounded hardware-gated sustained Qwen qualification driver; external controller required"]
fn qualifies_sustained_qwen_reservation() -> std::result::Result<(), DriverError> {
    run_driver(Workload::Sample32)
}

fn run_driver(workload: Workload) -> std::result::Result<(), DriverError> {
    let generation_budget = match workload {
        Workload::Sample32 => 32,
        Workload::Complete256 => 256,
    };
    let decode = match workload {
        Workload::Sample32 => decode::plain,
        Workload::Complete256 => full::decode_complete,
    };
    let mut model = tiles::load_model()?;
    let mut reference: Option<Observation> = None;
    let started = Instant::now();
    writeln!(std::io::stderr().lock(), "settled.ready: {}", json!({"epoch_ms":epoch()}))?;
    for (run, line) in std::io::stdin().lock().lines().enumerate() {
        let command: Command = serde_json::from_str(&line?)?;
        let Command::Run { phase, variant } = command else {
            return Ok(());
        };
        if run >= 40 || started.elapsed().as_secs() >= 300 {
            return Err(DriverError::Limit);
        }
        model.stream.set_decode_reservation(variant.policy());
        let start_epoch_ms = epoch();
        let observation = run_budget_with_decode(
            &mut model,
            MoePrefill::Default,
            2049,
            3,
            std::num::NonZeroUsize::new(generation_budget),
            decode,
        )?;
        let end_epoch_ms = epoch();
        let matches_reference = reference.as_ref().is_none_or(|prior| {
            prior.tokens == observation.tokens && prior.schedule == observation.schedule
        });
        writeln!(
            std::io::stderr().lock(),
            "settled.sample: {}",
            json!({
                "run":run,"phase":phase,"variant":variant,"start_epoch_ms":start_epoch_ms,
                "workload":workload,"generation_budget":generation_budget,
                "end_epoch_ms":end_epoch_ms,"matches_reference":matches_reference,"observation":observation
            })
        )?;
        assert!(matches_reference, "sustained benchmark changed tokens or schedule");
        if reference.is_none() {
            reference = Some(observation);
        }
    }
    Ok(())
}
