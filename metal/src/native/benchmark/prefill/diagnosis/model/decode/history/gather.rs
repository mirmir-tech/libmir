use super::{
    candidate::{Positions, run_modes},
    *,
};
use crate::config::HistoryBatching::{Gathered, Joined};

#[test]
#[ignore = "Qwen page-to-batch gather numerical gate; set MIRMIR_BENCH_MODEL"]
fn preserves_qwen_page_gather() -> Result<()> {
    numerical(2049, 10)
}

#[test]
#[ignore = "GPT-OSS inactive page-to-batch gather scope; set MIRMIR_BENCH_MODEL"]
fn preserves_clamped_page_gather() -> Result<()> {
    numerical(129, 0)
}

fn numerical(context: usize, layers: usize) -> Result<()> {
    let mut model = tiles::load_model()?;
    for positions in [Positions::Uniform, Positions::Ragged] {
        run_modes(&mut model, context, layers, positions, &[Joined, Gathered], 3)?;
    }
    Ok(())
}

#[test]
#[ignore = "Qwen C3/2049 page gather whole-decode ABBA; set MIRMIR_BENCH_MODEL"]
fn compares_qwen_page_gather() -> Result<()> {
    let mut model = tiles::load_model()?;
    run_modes(
        &mut model,
        2049,
        10,
        Positions::Uniform,
        &[Joined, Gathered, Joined, Gathered, Gathered, Joined, Gathered, Joined, Joined, Gathered],
        3,
    )
}
