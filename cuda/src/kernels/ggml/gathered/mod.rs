//! Routed-expert projections over stacked GGML expert banks
//! (`kernels/ggml/gathered.cu`): one GEMV per (token, expert) selection, and
//! the routing-weighted sum of each token's selections.

use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_files,
};

use super::{
    super::geometry::{narrow, product, require},
    GgmlActivations, GgmlBankRows, GgmlKernelType,
    linear::QuantizeQ8,
};
use crate::{Error, Result};

mod kernels;

use kernels::GatheredKernel;

cuda_export!(CombineBf16 = "libmir_cuda_ggml_combine_bf16"(
    partial: &DeviceBuffer<bf16>, routing: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<bf16>,
    tokens: u32, top_k: u32, hidden: u32,
));

/// The geometry of one stacked bank: `experts` matrices of `rows` by
/// `columns`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GgmlExpertSpec {
    pub kind: GgmlKernelType,
    pub experts: usize,
    pub rows: usize,
    pub columns: usize,
}

impl GgmlExpertSpec {
    pub fn bank_bytes(self) -> Result<usize> {
        product(self.experts, self.kind.bytes(self.rows, self.columns)?)
    }
}

/// The selections of one routed projection: `selected` holds one expert per
/// selection, and selection `s` reads the activations of row `s / group`.
pub struct GgmlSelections<'a> {
    pub selected: &'a DeviceBuffer<u32>,
    pub count: usize,
    pub group: usize,
}

#[derive(Clone, Debug)]
pub struct GgmlGatheredLinear {
    quantize: TypedKernel<QuantizeQ8>,
    kernel: GatheredKernel,
    combine: TypedKernel<CombineBf16>,
    spec: GgmlExpertSpec,
}

impl GgmlGatheredLinear {
    pub fn compile(compiler: &Compiler, spec: GgmlExpertSpec) -> Result<Self> {
        let block = usize::try_from(spec.kind.ggml_type().block_elements())?;
        if spec.experts == 0 || spec.rows == 0 || !spec.columns.is_multiple_of(block) {
            return Err(Error::InvalidDecoderKernel("invalid GGML expert bank geometry"));
        }
        let module = compiler.compile(
            cuda_kernel_files!(
                "ggml_gathered.cu";
                "../../../../kernels/ggml/blocks.cuh",
                "../../../../kernels/ggml/legacy.cuh",
                "../../../../kernels/ggml/k_quants.cuh",
                "../../../../kernels/ggml/iq4.cuh",
                "../../../../kernels/ggml/grids/iq2.cuh",
                "../../../../kernels/ggml/grids/iq13.cuh",
                "../../../../kernels/ggml/iq.cuh",
                "../../../../kernels/ggml/dot.cuh",
                "../../../../kernels/ggml/staged.cu",
                "../../../../kernels/ggml/vector.cu",
                "../../../../kernels/ggml/gathered.cu",
            ),
            &CompileOptions::default(),
        )?;
        Ok(Self {
            quantize: module.kernel()?,
            kernel: GatheredKernel::load(&module, spec.kind)?,
            combine: module.kernel()?,
            spec,
        })
    }

    #[must_use]
    pub const fn spec(&self) -> GgmlExpertSpec {
        self.spec
    }

    /// Quantizes `rows` input rows of `columns` values into `activations`.
    pub fn quantize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        activations: &mut GgmlActivations,
        rows: usize,
    ) -> Result<()> {
        let subs = product(rows, self.spec.columns / 32)?;
        require("GGML expert input", product(rows, self.spec.columns)?, input.len())?;
        activations.require(subs)?;
        let (quantized, scales) = activations.buffers_mut();
        self.quantize.launch(
            stream,
            LaunchConfig {
                grid: (narrow(subs.div_ceil(128))?, 1, 1),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, quantized, scales, narrow(rows)?, narrow(self.spec.columns)?),
        )?;
        Ok(())
    }

    /// Writes `rows` outputs per selection, selection-major.
    pub fn multiply(
        &self,
        stream: &Stream,
        activations: &GgmlActivations,
        (weight, bank): (&DeviceBuffer<u8>, GgmlBankRows),
        selections: &GgmlSelections<'_>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if selections.group == 0 {
            return Err(Error::InvalidDecoderKernel("GGML expert selection group is zero"));
        }
        let inputs = selections.count.div_ceil(selections.group);
        activations.require(product(inputs, self.spec.columns / 32)?)?;
        bank.check(self.spec.rows)?;
        let bank_bytes =
            product(self.spec.experts, self.spec.kind.bytes(bank.expert_rows, self.spec.columns)?)?;
        require("GGML expert bank", bank_bytes, weight.len())?;
        require("GGML expert selections", selections.count, selections.selected.len())?;
        require("GGML expert output", product(selections.count, self.spec.rows)?, output.len())?;
        let (quantized, scales) = activations.buffers();
        self.kernel.launch(
            stream,
            LaunchConfig {
                // Four warps of one row each (`ggml_gemv<Block, 1>`).
                grid: (narrow(self.spec.rows.div_ceil(4))?, narrow(selections.count)?, 1),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                quantized,
                scales,
                weight,
                selections.selected,
                output,
                narrow(selections.count)?,
                narrow(selections.group)?,
                narrow(self.spec.rows)?,
                narrow(self.spec.columns)?,
                narrow(bank.expert_rows)?,
                narrow(bank.first_row)?,
            ),
        )
    }

    /// Sums each token's `top_k` selection outputs of `hidden` values,
    /// weighted by `routing`.
    pub fn combine(
        &self,
        stream: &Stream,
        partial: &DeviceBuffer<bf16>,
        routing: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<bf16>,
        top_k: usize,
    ) -> Result<()> {
        let hidden = self.spec.rows;
        let selections = routing.len();
        if top_k == 0 || !selections.is_multiple_of(top_k) {
            return Err(Error::InvalidDecoderKernel("GGML expert routing size mismatch"));
        }
        let tokens = selections / top_k;
        require("GGML expert partial", product(selections, hidden)?, partial.len())?;
        require("GGML expert combined", product(tokens, hidden)?, output.len())?;
        let elements = product(tokens, hidden)?;
        self.combine.launch(
            stream,
            LaunchConfig {
                grid: (narrow(elements.div_ceil(256))?, 1, 1),
                block: (256, 1, 1),
                shared_memory_bytes: 0,
            },
            (partial, routing, output, narrow(tokens)?, narrow(top_k)?, narrow(hidden)?),
        )?;
        Ok(())
    }
}
