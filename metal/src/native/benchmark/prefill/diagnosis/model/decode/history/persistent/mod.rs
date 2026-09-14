use super::*;
use crate::config::HistoryBatching::{Joined, Persistent};

#[test]
#[ignore = "actual Qwen persistent history numerical gate"]
fn preserves_qwen_persistent_history() -> Result<()> {
    numerical(2049, 10)
}

#[test]
#[ignore = "actual GPT-OSS persistent history fallback gate"]
fn preserves_clamped_persistent_history() -> Result<()> {
    numerical(129, 0)
}

fn numerical(context: usize, layers: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    for positions in [candidate::Positions::Uniform, candidate::Positions::Ragged] {
        candidate::run_modes(&mut model, context, layers, positions, &[Joined, Persistent], 3)?;
    }
    Ok(())
}

mod reclamation;

mod budget;
