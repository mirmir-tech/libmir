use super::LoadedModel;

#[derive(Clone, Copy)]
pub(super) enum Schedule {
    Adaptive,
    Probe,
    Paired,
    MoePaired,
    MoeProjection,
    MoeIndexed,
    MoeIndexedNumerics,
    MoeAligned,
    MoeBudgets,
    MoeTiles,
    MoeTileWidths,
    MoeTileProfile,
    Uniform128,
}

impl Schedule {
    pub(super) const fn moe_diagnostic(self) -> crate::config::MoePrefill {
        use crate::config::MoePrefill;
        match self {
            Self::MoeProjection => MoePrefill::CompareProjection,
            Self::MoeIndexed => MoePrefill::CompareIndexed,
            Self::MoeIndexedNumerics => MoePrefill::MeasureIndexedNumerics,
            Self::MoeAligned => MoePrefill::CompareAligned,
            Self::MoeTileProfile => MoePrefill::ProfileTiles,
            Self::MoeTileWidths => MoePrefill::CompareTileWidths,
            Self::MoeTiles => MoePrefill::CompareTiles,
            Self::MoeBudgets => MoePrefill::CompareAlignmentBudgets,
            Self::MoePaired => MoePrefill::CompareRoutes,
            _ => MoePrefill::Default,
        }
    }

    pub(super) fn configure(self, model: &mut LoadedModel) -> (&'static [usize], usize) {
        match self {
            Self::Adaptive => (&[2048, 8192, 8192], 2048),
            Self::Probe => {
                model.info.prefill_step = 128;
                (&[129], 640)
            },
            Self::Paired
            | Self::MoePaired
            | Self::MoeProjection
            | Self::MoeIndexed
            | Self::MoeIndexedNumerics
            | Self::MoeAligned
            | Self::MoeTiles
            | Self::MoeTileWidths
            | Self::MoeTileProfile
            | Self::MoeBudgets => {
                model.info.prefill_step = 512;
                (&[1025], 2560)
            },
            Self::Uniform128 => {
                model.info.prefill_step = 128;
                (&[2048, 8192], 640)
            },
        }
    }
}
