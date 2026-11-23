//! Expanded prefill attention against the same dense reference.

use std::ops::Range;

use mircuda::{DeviceBuffer, bf16};
use models::layout::RotaryPairing;

use super::{
    super::{
        LatentAttention, LatentAttentionInputs, LatentAttentionLayer, LatentAttentionSpec,
        LatentAttentionWorkspace, LatentCacheView, LatentExpandedMetadata, LatentExpandedRow,
        LatentExpandedRows, LatentExpansion, LatentExpansionBuffers, LatentUpLayout,
    },
    Scenario,
    fixture::Runtime,
    run, spec,
};
use crate::Result;

/// One layer's expansion plans and scratch for a test scenario.
pub(super) struct Expansion {
    plans: LatentExpansion,
    buffers: LatentExpansionBuffers,
}

impl Expansion {
    pub(super) fn new(
        runtime: &Runtime,
        spec: LatentAttentionSpec,
        layout: LatentUpLayout,
    ) -> Result<Self> {
        let plans = LatentExpansion::new(&runtime.context, &runtime.stream, spec, layout, 512)?
            .ok_or(crate::Error::InvalidPagedKv("test geometry has no expansion"))?;
        let buffers = plans.buffers(&runtime.pool, &runtime.stream, (spec.max_queries, 4))?;
        Ok(Self { plans, buffers })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute(
        &mut self,
        attention: &LatentAttention,
        runtime: &Runtime,
        layer: &LatentAttentionLayer<'_>,
        cache: LatentCacheView<'_>,
        inputs: LatentAttentionInputs<'_>,
        launch: &[(usize, Range<usize>)],
        workspace: &mut LatentAttentionWorkspace,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        // Sequence `s` of a packed launch uses table row `s`.
        assert!(launch.iter().enumerate().all(|(row, (sequence, _))| row == *sequence));
        let rows = launch
            .iter()
            .map(|(_, positions)| LatentExpandedRow {
                start: positions.start,
                tokens: positions.len(),
            })
            .collect::<Vec<_>>();
        let mut host = [vec![0_u32], Vec::new(), vec![0_u32]];
        for row in &rows {
            host[0].push(host[0][host[0].len() - 1] + u32::try_from(row.tokens)?);
            host[1].push(u32::try_from(row.context())?);
            host[2].push(host[2][host[2].len() - 1] + u32::try_from(row.context())?);
        }
        let [query_starts, contexts, context_starts] =
            [runtime.copy(&host[0])?, runtime.copy(&host[1])?, runtime.copy(&host[2])?];
        let sequences = match rows.as_slice() {
            [row] => LatentExpandedRows::One(*row),
            rows => LatentExpandedRows::Packed {
                rows,
                metadata: LatentExpandedMetadata {
                    query_starts: &query_starts,
                    contexts: &contexts,
                    context_starts: &context_starts,
                },
            },
        };
        attention.execute_expanded(
            &runtime.stream,
            &mut self.plans,
            layer,
            cache,
            inputs,
            sequences,
            (workspace, &mut self.buffers),
            output,
        )
    }
}

/// A prompt prefilled in two chunks over the cached rows before them (the
/// first ends one row past a whole query tile of the fused attention), then
/// one token; both up-projection layouts, head counts that are not
/// multiples of sixteen, and query temperature.
#[test]
fn expanded_prefill_matches_expanded_attention() -> Result<()> {
    for (heads, layout, temperature) in [
        (32, LatentUpLayout::Split, None),
        (10, LatentUpLayout::Fused, None),
        (
            16,
            LatentUpLayout::Split,
            Some(models::semantic::QueryTemperature { beta: 0.5, window: 16, offset: 0 }),
        ),
    ] {
        // Mistral Small 4's widths: one 128-wide fused attention head.
        let geometry = LatentAttentionSpec {
            latent_rank: 256,
            nope_dim: 64,
            ..spec(heads, RotaryPairing::Interleaved, 257, 1)
        };
        run(&Scenario {
            spec: geometry,
            lengths: vec![300],
            launches: vec![vec![(0, 0..257)], vec![(0, 257..299)], vec![(0, 299..300)]],
            rope_scale: 0.9,
            temperature,
            layout,
            expanded: true,
        })?;
    }
    Ok(())
}

/// Packed sequences prefilled expanded together, at different depths, with
/// a lone query row beside longer ones; the 192-wide `DeepSeek` keys pad to the
/// 256-wide attention.
#[test]
fn packed_expanded_prefill_matches_expanded_attention() -> Result<()> {
    for (geometry, layout) in [
        (
            LatentAttentionSpec {
                latent_rank: 256,
                nope_dim: 64,
                ..spec(10, RotaryPairing::Interleaved, 290, 1)
            },
            LatentUpLayout::Fused,
        ),
        (
            LatentAttentionSpec {
                latent_rank: 256,
                nope_dim: 128,
                ..spec(16, RotaryPairing::Interleaved, 290, 1)
            },
            LatentUpLayout::Split,
        ),
    ] {
        run(&Scenario {
            spec: geometry,
            lengths: vec![300, 90, 40],
            launches: vec![
                vec![(0, 0..200), (1, 0..60)],
                vec![(0, 200..290), (1, 60..89), (2, 0..40)],
                vec![(0, 290..300), (1, 89..90)],
            ],
            rope_scale: 0.9,
            temperature: None,
            layout,
            expanded: true,
        })?;
    }
    Ok(())
}
