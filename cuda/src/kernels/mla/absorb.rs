use mircuda::{
    Context, CublasGemmOffsets, CublasGemmOperand, CublasGemmPlan, CublasGemmSpec, DeviceBuffer,
    Stream, bf16,
};

use super::{LatentAttentionSpec, LatentAttentionWorkspace};
use crate::{Error, Result, kernels::geometry::product};

/// How a checkpoint stores the per-head key and value up-projections.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LatentUpLayout {
    /// `kv_b_proj`: `[heads · (nope + value), latent]`.
    Fused,
    /// GGUF `attn_k_b` `[heads, latent, nope]` and `attn_v_b`
    /// `[heads, value, latent]`.
    Split,
}

/// BF16 up-projection weights in their checkpoint layout.
#[derive(Clone, Copy, Debug)]
pub enum LatentUpWeights<'a> {
    Fused(&'a DeviceBuffer<bf16>),
    Split {
        key: &'a DeviceBuffer<bf16>,
        value: &'a DeviceBuffer<bf16>,
    },
}

impl<'a> LatentUpWeights<'a> {
    #[must_use]
    pub const fn layout(&self) -> LatentUpLayout {
        match self {
            Self::Fused(_) => LatentUpLayout::Fused,
            Self::Split { .. } => LatentUpLayout::Split,
        }
    }

    pub(super) const fn key(&self) -> &'a DeviceBuffer<bf16> {
        match self {
            Self::Fused(weights) => weights,
            Self::Split { key, .. } => key,
        }
    }

    pub(super) const fn value(&self) -> &'a DeviceBuffer<bf16> {
        match self {
            Self::Fused(weights) => weights,
            Self::Split { value, .. } => value,
        }
    }
}

/// The two per-head projections around absorbed attention.
///
/// Both read the up-projection weights in place: queries `q_nope · W_UK`
/// into the latent space, latent outputs `· W_UVᵀ` back to every head's
/// value width.
#[derive(Debug)]
pub struct LatentAbsorption {
    query: CublasGemmPlan<bf16, bf16>,
    output: CublasGemmPlan<bf16, bf16>,
    spec: LatentAttentionSpec,
    layout: LatentUpLayout,
    tokens: usize,
}

impl LatentAbsorption {
    /// Plans both projections for exactly `tokens` query tokens over weights
    /// in `layout`.
    pub fn new(
        context: &Context,
        stream: &Stream,
        spec: LatentAttentionSpec,
        layout: LatentUpLayout,
        tokens: usize,
    ) -> Result<Self> {
        let (rank, nope, value) = (spec.latent_rank, spec.nope_dim, spec.value_dim);
        // Per head, W_UK is `[nope, latent]` and W_UV `[value, latent]`; the
        // fused layout interleaves them per head, the split one stores W_UKᵀ.
        let (key, value_weights) = match layout {
            LatentUpLayout::Fused => {
                let head = product(nope + value, rank)?;
                (operand(rank, head, false), operand(rank, head, true))
            },
            LatentUpLayout::Split => (
                operand(nope, product(rank, nope)?, true),
                operand(rank, product(value, rank)?, true),
            ),
        };
        let query = CublasGemmSpec::new(
            (tokens, rank, nope, spec.heads),
            operand(product(spec.heads, nope + spec.rope_dim)?, nope + spec.rope_dim, false),
            key,
            operand(product(spec.heads, spec.row_width())?, spec.row_width(), false),
        )?;
        let output = CublasGemmSpec::new(
            (tokens, value, rank, spec.heads),
            operand(product(spec.heads, rank)?, rank, false),
            value_weights,
            operand(product(spec.heads, value)?, value, false),
        )?;
        Ok(Self {
            query: CublasGemmPlan::new(context, stream, query)?,
            output: CublasGemmPlan::new(context, stream, output)?,
            spec,
            layout,
            tokens,
        })
    }

    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Elements the key and value weights of this layout hold.
    pub(super) fn weight_lengths(&self) -> Result<(usize, usize)> {
        let spec = self.spec;
        match self.layout {
            LatentUpLayout::Fused => {
                let fused = product(
                    product(spec.heads, spec.nope_dim + spec.value_dim)?,
                    spec.latent_rank,
                )?;
                Ok((fused, fused))
            },
            LatentUpLayout::Split => Ok((
                product(product(spec.heads, spec.latent_rank)?, spec.nope_dim)?,
                product(product(spec.heads, spec.value_dim)?, spec.latent_rank)?,
            )),
        }
    }

    pub(super) fn check(&self, weights: &LatentUpWeights<'_>) -> Result<()> {
        if weights.layout() == self.layout {
            Ok(())
        } else {
            Err(Error::InvalidPagedKv("latent up-projection layout differs from its plan"))
        }
    }

    /// Writes the absorbed half of every attention-query row from
    /// `[tokens, heads, nope + rope]` queries.
    pub(super) fn absorb_query(
        &mut self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        weights: &LatentUpWeights<'_>,
        workspace: &mut LatentAttentionWorkspace,
    ) -> Result<()> {
        Ok(self.query.execute(
            stream,
            (query, weights.key(), &mut workspace.query),
            CublasGemmOffsets::default(),
            (1.0, 0.0),
        )?)
    }

    /// Expands the merged latent output into `[tokens, heads · value]`.
    pub(super) fn expand_output(
        &mut self,
        stream: &Stream,
        weights: &LatentUpWeights<'_>,
        workspace: &LatentAttentionWorkspace,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let right = match self.layout {
            LatentUpLayout::Fused => product(self.spec.nope_dim, self.spec.latent_rank)?,
            LatentUpLayout::Split => 0,
        };
        let offsets = CublasGemmOffsets { right, ..CublasGemmOffsets::default() };
        Ok(self.output.execute(
            stream,
            (&workspace.latent_output, weights.value(), output),
            offsets,
            (1.0, 0.0),
        )?)
    }
}

const fn operand(leading: usize, stride: usize, transposed: bool) -> CublasGemmOperand {
    CublasGemmOperand { leading, stride, transposed }
}
