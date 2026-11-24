use super::super::parse::invalid;
use crate::error::Result;

/// Kind of one physical checkpoint block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridBlockKind {
    /// Mamba-2 selective state-space mixer.
    Mamba,
    /// Softmax attention mixer.
    Attention,
    /// Routed-expert feed-forward.
    Moe,
    /// Dense ungated feed-forward (Nemotron-H's `relu²` MLP).
    Mlp,
}

/// One logical decoder layer composed from adjacent physical blocks.
///
/// A feed-forward block always directly follows the mixer it is paired with,
/// so its physical index is `block + 1`. Pairings without a checkpoint
/// fixture, such as attention without a feed-forward block, are not
/// representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayer {
    /// A Mamba-2 block followed by another mixer.
    Mamba { block: usize },
    /// A Mamba-2 block followed by its routed-expert block.
    MambaWithMoe { block: usize },
    /// A Mamba-2 block followed by its dense MLP block.
    MambaWithMlp { block: usize },
    /// An attention block followed by its routed-expert block.
    AttentionWithMoe { block: usize },
    /// An attention block followed by its dense MLP block.
    AttentionWithMlp { block: usize },
}

/// The feed-forward block of a hybrid layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridFeedForward {
    Moe,
    Mlp,
}

/// Position of a physical block inside its logical layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayerBlock {
    /// The block is the layer's mixer.
    Mixer,
    /// The block is the layer's feed-forward.
    FeedForward,
}

/// Validated physical block sequence and its logical pairing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridBlockPattern {
    physical: Vec<HybridBlockKind>,
    layers: Vec<HybridLayer>,
}

impl HybridLayer {
    /// Physical index of the mixer block.
    #[must_use]
    pub const fn mixer_block(self) -> usize {
        match self {
            Self::Mamba { block }
            | Self::MambaWithMoe { block }
            | Self::MambaWithMlp { block }
            | Self::AttentionWithMoe { block }
            | Self::AttentionWithMlp { block } => block,
        }
    }

    /// Physical index of the feed-forward block, when the layer has one.
    #[must_use]
    pub const fn feed_forward_block(self) -> Option<usize> {
        match self.feed_forward() {
            Some(_) => Some(self.mixer_block() + 1),
            None => None,
        }
    }

    /// The layer's feed-forward block, when it has one.
    #[must_use]
    pub const fn feed_forward(self) -> Option<HybridFeedForward> {
        match self {
            Self::Mamba { .. } => None,
            Self::MambaWithMoe { .. } | Self::AttentionWithMoe { .. } => {
                Some(HybridFeedForward::Moe)
            },
            Self::MambaWithMlp { .. } | Self::AttentionWithMlp { .. } => {
                Some(HybridFeedForward::Mlp)
            },
        }
    }

    /// Whether the layer's mixer is a state-space block.
    #[must_use]
    pub const fn is_state_space(self) -> bool {
        matches!(self, Self::Mamba { .. } | Self::MambaWithMoe { .. } | Self::MambaWithMlp { .. })
    }
}

impl HybridBlockPattern {
    /// Pairs each mixer block with the routed block that directly follows it.
    pub fn pair(physical: Vec<HybridBlockKind>) -> Result<Self> {
        let mut layers = Vec::new();
        let mut block = 0;
        while let Some(kind) = physical.get(block).copied() {
            let next = physical.get(block + 1).copied();
            let layer = match (kind, next) {
                (HybridBlockKind::Mamba, Some(HybridBlockKind::Moe)) => {
                    HybridLayer::MambaWithMoe { block }
                },
                (HybridBlockKind::Mamba, Some(HybridBlockKind::Mlp)) => {
                    HybridLayer::MambaWithMlp { block }
                },
                (HybridBlockKind::Mamba, _) => HybridLayer::Mamba { block },
                (HybridBlockKind::Attention, Some(HybridBlockKind::Moe)) => {
                    HybridLayer::AttentionWithMoe { block }
                },
                (HybridBlockKind::Attention, Some(HybridBlockKind::Mlp)) => {
                    HybridLayer::AttentionWithMlp { block }
                },
                (HybridBlockKind::Attention, _) => {
                    return Err(invalid(format!(
                        "attention block {block} is not followed by a feed-forward block"
                    )));
                },
                (HybridBlockKind::Moe | HybridBlockKind::Mlp, _) => {
                    return Err(invalid(format!(
                        "feed-forward block {block} does not follow a mixer block"
                    )));
                },
            };
            block += if layer.feed_forward_block().is_some() {
                2
            } else {
                1
            };
            layers.push(layer);
        }
        if layers.is_empty() {
            return Err(invalid("hybrid block pattern is empty"));
        }
        Ok(Self { physical, layers })
    }

    /// Physical block kinds in checkpoint order.
    #[must_use]
    pub fn physical(&self) -> &[HybridBlockKind] {
        &self.physical
    }

    /// Logical decoder layers in order.
    #[must_use]
    pub fn layers(&self) -> &[HybridLayer] {
        &self.layers
    }

    /// Maps a physical block to its logical layer and position.
    #[must_use]
    pub fn locate(&self, block: usize) -> Option<(usize, HybridLayerBlock)> {
        self.layers.iter().enumerate().find_map(|(index, layer)| {
            if layer.mixer_block() == block {
                Some((index, HybridLayerBlock::Mixer))
            } else if layer.feed_forward_block() == Some(block) {
                Some((index, HybridLayerBlock::FeedForward))
            } else {
                None
            }
        })
    }
}
