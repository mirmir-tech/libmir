use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, MemoryPool, Stream, TypedKernel, bf16,
    cuda_export, cuda_kernel_files,
};

use super::{
    super::geometry::{narrow, product, require},
    GgmlSpec,
    gemv::{GemvArguments, GemvKernel, WIDTHS},
};
use crate::Result;

cuda_export!(pub(super) QuantizeQ8 = "libmir_cuda_ggml_quantize_q8"(
    input: &DeviceBuffer<bf16>, quantized: &mut DeviceBuffer<u32>, scales: &mut DeviceBuffer<f32>,
    tokens: u32, columns: u32,
));

/// Decode-batch activations quantized to int8 per 32-value sub-block: eight
/// words of values, then a scale and the exact sums of both halves (and one
/// padding float) per sub-block.
#[derive(Debug)]
pub struct GgmlActivations {
    quantized: DeviceBuffer<u32>,
    scales: DeviceBuffer<f32>,
}

impl GgmlActivations {
    pub fn new(pool: &MemoryPool, stream: &Stream, tokens: usize, columns: usize) -> Result<Self> {
        let subs = product(tokens, columns / 32)?;
        Ok(Self {
            quantized: pool.allocate(stream, product(subs, 8)?)?,
            scales: pool.allocate(stream, product(subs, 4)?)?,
        })
    }

    /// Fails unless the buffers hold at least `subs` sub-blocks.
    pub(super) fn require(&self, subs: usize) -> Result<()> {
        if self.quantized.len() < product(subs, 8)? || self.scales.len() < product(subs, 4)? {
            return Err(crate::Error::InvalidDecoderKernel("GGML activations are too short"));
        }
        Ok(())
    }

    pub(super) const fn buffers(&self) -> (&DeviceBuffer<u32>, &DeviceBuffer<f32>) {
        (&self.quantized, &self.scales)
    }

    pub(super) const fn buffers_mut(&mut self) -> (&mut DeviceBuffer<u32>, &mut DeviceBuffer<f32>) {
        (&mut self.quantized, &mut self.scales)
    }
}

/// A BF16 projection over GGML blocks for decode batches: activations are
/// quantized once, then one warp per output row reads the row once per pass
/// of up to sixteen tokens and takes int8 dot products.
#[derive(Clone, Debug)]
pub struct GgmlLinear {
    quantize: TypedKernel<QuantizeQ8>,
    kernel: GemvKernel,
    width: usize,
    spec: GgmlSpec,
}

impl GgmlLinear {
    pub fn compile(compiler: &Compiler, spec: GgmlSpec) -> Result<Self> {
        let module = compiler.compile(
            cuda_kernel_files!(
                "ggml_vector.cu";
                "../../../kernels/ggml/blocks.cuh",
                "../../../kernels/ggml/legacy.cuh",
                "../../../kernels/ggml/k_quants.cuh",
                "../../../kernels/ggml/iq4.cuh",
                "../../../kernels/ggml/grids/iq2.cuh",
                "../../../kernels/ggml/grids/iq13.cuh",
                "../../../kernels/ggml/iq.cuh",
                "../../../kernels/ggml/dot.cuh",
                "../../../kernels/ggml/staged.cu",
                "../../../kernels/ggml/vector.cu",
            ),
            &CompileOptions::default(),
        )?;
        let width = WIDTHS
            .into_iter()
            .find(|&width| width >= spec.tokens)
            .unwrap_or(WIDTHS[WIDTHS.len() - 1]);
        let kernel = GemvKernel::load(&module, spec.kind, width)?;
        Ok(Self {
            quantize: module.kernel()?,
            kernel,
            width,
            spec,
        })
    }

    #[must_use]
    pub const fn spec(&self) -> GgmlSpec {
        self.spec
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        activations: &mut GgmlActivations,
        weight: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.quantize(stream, input, activations)?;
        self.multiply(stream, activations, weight, output)
    }

    /// Quantizes the batch input into `activations`; projections that share
    /// an input share one quantization.
    pub fn quantize(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        activations: &mut GgmlActivations,
    ) -> Result<()> {
        let subs = self.require_activations(activations)?;
        require("GGML input", self.spec.input_elements()?, input.len())?;
        self.quantize.launch(
            stream,
            LaunchConfig {
                grid: (narrow(subs.div_ceil(128))?, 1, 1),
                block: (128, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                input,
                &mut activations.quantized,
                &mut activations.scales,
                narrow(self.spec.tokens)?,
                narrow(self.spec.input_features)?,
            ),
        )?;
        Ok(())
    }

    /// The projection of activations quantized by [`Self::quantize`].
    pub fn multiply(
        &self,
        stream: &Stream,
        activations: &GgmlActivations,
        weight: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.require_activations(activations)?;
        require("GGML weight", self.spec.weight_bytes()?, weight.len())?;
        require("GGML output", self.spec.output_elements()?, output.len())?;
        let config = LaunchConfig {
            grid: (
                // Four warps of one row, or of two rows for multi-token passes
                // (`rows_per_warp` in vector.cu).
                narrow(self.spec.output_features.div_ceil(if self.width >= 2 {
                    8
                } else {
                    4
                }))?,
                narrow(self.spec.tokens.div_ceil(self.width))?,
                1,
            ),
            block: (128, 1, 1),
            shared_memory_bytes: 0,
        };
        self.kernel.launch(
            stream,
            config,
            GemvArguments {
                quantized: &activations.quantized,
                scales: &activations.scales,
                weight,
                output,
                tokens: narrow(self.spec.tokens)?,
                rows: narrow(self.spec.output_features)?,
                columns: narrow(self.spec.input_features)?,
            },
        )
    }

    fn require_activations(&self, activations: &GgmlActivations) -> Result<usize> {
        let subs = product(self.spec.tokens, self.spec.input_features / 32)?;
        require("GGML activations", product(subs, 8)?, activations.quantized.len())?;
        require("GGML activation scales", product(subs, 4)?, activations.scales.len())?;
        Ok(subs)
    }
}
