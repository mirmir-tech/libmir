use super::{super::Kernels, Error, Result, template};

impl Kernels {
    pub(crate) const fn aligned_group(&self) -> &AlignedGroup {
        &self.expert_group_aligned
    }
}

mirtal::metal_kernel! {
    fn aligned_group {
        name: "mirmir_aligned_expert_group",
        templates: [ROUTES: int = 64, EXPERTS: int = 8, CAPACITY: int = 128, TILE: int = 16],
        inputs: [indices: u32],
        outputs: [order: u32, inverse: u32, grouped_indices: u32],
        source: file "kernels/expert_group/aligned.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

/// Bounded experiment matching MLX's non-NAX 16-row `GatherQMM` tile.
/// Unused capacity still costs GEMM work. The explicit budget bounds extra
/// rows; the original experiment defaults to eight rows per expert.
#[derive(Debug)]
pub struct AlignedGroup {
    kernel: mirtal::MetalKernel<1, 3>,
    budget: PaddingBudget,
}

#[derive(Clone, Copy, Debug)]
pub enum PaddingBudget {
    Four,
    Eight,
}

impl PaddingBudget {
    pub(crate) const fn rows(self) -> usize {
        match self {
            Self::Four => 4,
            Self::Eight => 8,
        }
    }
}

impl AlignedGroup {
    pub(crate) fn new() -> Result<Self> {
        Self::with_budget(PaddingBudget::Eight)
    }

    pub(crate) fn with_budget(budget: PaddingBudget) -> Result<Self> {
        Ok(Self { kernel: aligned_group()?, budget })
    }

    pub(crate) const fn budget(&self) -> PaddingBudget {
        self.budget
    }

    pub(crate) fn forward(
        &self,
        stream: &mirtal::Stream,
        indices: &mirtal::Array,
        experts: usize,
    ) -> Result<[mirtal::Array; 3]> {
        let routes = indices.len();
        if indices.dtype()? != mirtal::DType::Uint32
            || routes == 0
            || !(1..=1024).contains(&experts)
        {
            return Err(Error::InvalidModel(
                "aligned grouping configuration is unsupported".into(),
            ));
        }
        let capacity = routes
            .checked_add(experts * self.budget.rows())
            .and_then(|n| n.checked_add(15))
            .map(|n| n / 16 * 16)
            .ok_or(Error::ShapeOverflow)?;
        let output = |n| -> Result<_> {
            Ok(mirtal::OutputSpec::new(mirtal::Shape::new([n])?, mirtal::DType::Uint32))
        };
        let threads = experts.max(routes.min(256));
        Ok(self.kernel.dispatch(
            stream,
            [indices],
            &[output(capacity)?, output(routes)?, output(capacity)?],
            &mirtal::Dispatch::new([threads, 1, 1], [threads, 1, 1]).templates([
                template("ROUTES", routes)?,
                template("EXPERTS", experts)?,
                template("CAPACITY", capacity)?,
                template("TILE", 16)?,
            ]),
        )?)
    }
}

#[cfg(test)]
mod tests;
