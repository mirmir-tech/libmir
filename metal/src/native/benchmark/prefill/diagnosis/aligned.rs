use super::*;

#[test]
#[ignore = "actual first/middle/last MoE alignment budgets and hot routing; set MIRMIR_BENCH_MODEL"]
fn compares_alignment_budgets() -> Result<()> {
    diagnose(Schedule::MoeBudgets, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

#[test]
#[ignore = "warmed actual MoE with tile-aligned expert boundaries; set MIRMIR_BENCH_MODEL"]
fn compares_warm_moe_aligned() -> Result<()> {
    diagnose(Schedule::MoeAligned, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

#[test]
#[ignore = "compact GPU expert tiles on actual and hot routing; set MIRMIR_BENCH_MODEL"]
fn compares_expert_tiles() -> Result<()> {
    diagnose(Schedule::MoeTiles, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

#[test]
#[ignore = "barrier-separated compact tile components; set MIRMIR_BENCH_MODEL"]
fn profiles_expert_tiles() -> Result<()> {
    diagnose(Schedule::MoeTileProfile, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}

#[test]
#[ignore = "compact expert column tiles 32/64; set MIRMIR_BENCH_MODEL"]
fn compares_expert_tile_widths() -> Result<()> {
    diagnose(Schedule::MoeTileWidths, PrefixRetention::View, None, GatedDeltaPrefill::Rows)
}
