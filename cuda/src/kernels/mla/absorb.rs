use mircuda::{
    Context, CublasGemmOffsets, CublasGemmOperand, CublasGemmPlan, CublasGemmSpec, DeviceBuffer,
    Stream, bf16,
};

use super::{LatentAttentionSpec, LatentAttentionWorkspace};
use crate::{Result, kernels::geometry::product};

/// The two per-head projections around absorbed attention.
///
/// Both read the checkpoint's `kv_b_proj` (`[heads · (nope + value), latent]`)
/// in place: queries `q_nope · W_UK` into the latent space, latent outputs
/// `· W_UVᵀ` back to every head's value width.
#[derive(Debug)]
pub struct LatentAbsorption {
    query: CublasGemmPlan<bf16, bf16>,
    output: CublasGemmPlan<bf16, bf16>,
    spec: LatentAttentionSpec,
    tokens: usize,
}

impl LatentAbsorption {
    /// Plans both projections for exactly `tokens` query tokens.
    pub fn new(
        context: &Context,
        stream: &Stream,
        spec: LatentAttentionSpec,
        tokens: usize,
    ) -> Result<Self> {
        let projected = spec.nope_dim + spec.value_dim;
        let head_weights = product(projected, spec.latent_rank)?;
        let query = CublasGemmSpec::new(
            (tokens, spec.latent_rank, spec.nope_dim, spec.heads),
            operand(
                product(spec.heads, spec.nope_dim + spec.rope_dim)?,
                spec.nope_dim + spec.rope_dim,
                false,
            ),
            operand(spec.latent_rank, head_weights, false),
            operand(product(spec.heads, spec.row_width())?, spec.row_width(), false),
        )?;
        let output = CublasGemmSpec::new(
            (tokens, spec.value_dim, spec.latent_rank, spec.heads),
            operand(product(spec.heads, spec.latent_rank)?, spec.latent_rank, false),
            operand(spec.latent_rank, head_weights, true),
            operand(product(spec.heads, spec.value_dim)?, spec.value_dim, false),
        )?;
        Ok(Self {
            query: CublasGemmPlan::new(context, stream, query)?,
            output: CublasGemmPlan::new(context, stream, output)?,
            spec,
            tokens,
        })
    }

    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Writes the absorbed half of every attention-query row from
    /// `[tokens, heads, nope + rope]` queries.
    pub(super) fn absorb_query(
        &mut self,
        stream: &Stream,
        query: &DeviceBuffer<bf16>,
        key_value_up: &DeviceBuffer<bf16>,
        workspace: &mut LatentAttentionWorkspace,
    ) -> Result<()> {
        Ok(self.query.execute(
            stream,
            (query, key_value_up, &mut workspace.query),
            CublasGemmOffsets::default(),
            (1.0, 0.0),
        )?)
    }

    /// Expands the merged latent output into `[tokens, heads · value]`.
    pub(super) fn expand_output(
        &mut self,
        stream: &Stream,
        key_value_up: &DeviceBuffer<bf16>,
        workspace: &LatentAttentionWorkspace,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        let offsets = CublasGemmOffsets {
            right: product(self.spec.nope_dim, self.spec.latent_rank)?,
            ..CublasGemmOffsets::default()
        };
        Ok(self.output.execute(
            stream,
            (&workspace.latent_output, key_value_up, output),
            offsets,
            (1.0, 0.0),
        )?)
    }
}

const fn operand(leading: usize, stride: usize, transposed: bool) -> CublasGemmOperand {
    CublasGemmOperand { leading, stride, transposed }
}
