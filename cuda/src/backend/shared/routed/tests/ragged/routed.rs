use super::*;

#[test]
fn ragged_routed_experts_preserve_independent_sessions() -> Result<()> {
    run_routed(vec![vec![17, 1, 33], vec![51], vec![1, 50], vec![1, 1, 1], vec![7, 1, 2]], None)
}

#[test]
fn padded_routed_experts_preserve_checkpoint_tails_and_mixed_decode() -> Result<()> {
    run_routed(
        vec![vec![112; 5], vec![16; 5], vec![1; 5], vec![1, 40, 1, 1, 1], vec![1; 5]],
        Some(640),
    )
}

fn run_routed(rounds: Vec<Vec<usize>>, capacity: Option<usize>) -> Result<()> {
    let mut decoder = decoder()?;
    decoder.head_dim = 64;
    decoder.full_attention_partial_rotary_factor = Some(0.09375);
    let fixture = fixture::HybridFixture::nonzero_routed(&decoder)?;
    let rows = rounds.iter().map(Vec::len).max().unwrap_or(0);
    run_fixture(&decoder, &fixture, rounds, rows, capacity, false)
}
