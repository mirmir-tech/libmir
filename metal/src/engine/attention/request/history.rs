use super::*;
use crate::engine::probe::history;
impl AttentionRequest<'_> {
    pub(super) fn capture_history(&self) -> Result<()> {
        match &self.source {
            Source::Paged(p) => history::row(
                self.query,
                &p.key_pages,
                &p.value_pages,
                history::Mask::None,
                history::Bias::None,
                history::Reader::NativePaged,
            ),
            Source::View { keys, values, mask, bias } => history::row(
                self.query,
                keys,
                values,
                match mask {
                    Mask::None => history::Mask::None,
                    Mask::Causal => history::Mask::Causal,
                    Mask::Explicit(_) => history::Mask::Explicit,
                },
                match bias {
                    AttentionBias::None => history::Bias::None,
                    AttentionBias::Sinks(_) => history::Bias::Sinks,
                },
                history::Reader::RowView,
            ),
        }
    }
}
