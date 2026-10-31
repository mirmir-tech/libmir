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
}

/// One logical decoder layer composed from adjacent physical blocks.
///
/// A routed block always directly follows the mixer it is paired with, so its
/// physical index is `block + 1`. Pairings without a checkpoint fixture, such
/// as attention without a routed block, are not representable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridLayer {
    /// A Mamba-2 block followed by a block that is not routed.
    Mamba { block: usize },
    /// A Mamba-2 block followed by its routed-expert block.
    MambaWithMoe { block: usize },
    /// An attention block followed by its routed-expert block.
    AttentionWithMoe { block: usize },
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
            | Self::AttentionWithMoe { block } => block,
        }
    }

    /// Physical index of the routed block, when the layer has one.
    #[must_use]
    pub const fn feed_forward_block(self) -> Option<usize> {
        match self {
            Self::Mamba { .. } => None,
            Self::MambaWithMoe { block } | Self::AttentionWithMoe { block } => Some(block + 1),
        }
    }

    /// Whether the layer's mixer is a state-space block.
    #[must_use]
    pub const fn is_state_space(self) -> bool {
        matches!(self, Self::Mamba { .. } | Self::MambaWithMoe { .. })
    }
}

impl HybridBlockPattern {
    /// Pairs each mixer block with the routed block that directly follows it.
    pub fn pair(physical: Vec<HybridBlockKind>) -> Result<Self> {
        let mut layers = Vec::new();
        let mut block = 0;
        while let Some(kind) = physical.get(block).copied() {
            let followed_by_moe = physical.get(block + 1) == Some(&HybridBlockKind::Moe);
            let layer = match (kind, followed_by_moe) {
                (HybridBlockKind::Mamba, true) => HybridLayer::MambaWithMoe { block },
                (HybridBlockKind::Mamba, false) => HybridLayer::Mamba { block },
                (HybridBlockKind::Attention, true) => HybridLayer::AttentionWithMoe { block },
                (HybridBlockKind::Attention, false) => {
                    return Err(invalid(format!(
                        "attention block {block} is not followed by a routed-expert block"
                    )));
                },
                (HybridBlockKind::Moe, _) => {
                    return Err(invalid(format!(
                        "routed-expert block {block} does not follow a mixer block"
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
