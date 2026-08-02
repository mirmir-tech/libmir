use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::Fp8OutputSpec;
use crate::{
    Error, Result,
    kernels::geometry::{narrow, product, require},
};

const TOP_K: usize = 128;
const THREADS: u32 = 256;
const ITEMS_PER_THREAD: usize = 8;
const CHUNK: usize = THREADS as usize * ITEMS_PER_THREAD;

impl Fp8OutputSpec {
    pub fn new_refinement(input_features: usize, output_features: usize) -> Result<Self> {
        if input_features == 0
            || output_features == 0
            || !input_features.is_multiple_of(64)
            || !output_features.is_multiple_of(128)
        {
            return Err(Error::InvalidDecoderKernel("invalid FP8 refinement geometry"));
        }
        let _ = product(input_features, output_features)?;
        Ok(Self { input_features, output_features })
    }
}

cuda_export!(CandidateKernel = "libmir_cuda_output_refine_candidates"(
    logits: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u64>, vocab: u32,
));
cuda_export!(MergeKernel = "libmir_cuda_output_refine_merge"(
    input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>, count: u32,
));
cuda_export!(RefineKernel = "libmir_cuda_output_refine_bf16"(
    input: &DeviceBuffer<bf16>, weight: &DeviceBuffer<bf16>,
    candidates: &DeviceBuffer<u64>, output: &mut DeviceBuffer<bf16>, columns: u32,
));

#[derive(Clone, Debug)]
pub struct Fp8RefinementKernels {
    candidates: TypedKernel<CandidateKernel>,
    merge: TypedKernel<MergeKernel>,
    refine: TypedKernel<RefineKernel>,
    spec: Fp8OutputSpec,
}

impl Fp8RefinementKernels {
    pub fn compile(compiler: &Compiler, spec: Fp8OutputSpec) -> Result<Self> {
        let source = cuda_kernel_file!("../../../kernels/output_fp8_refine.cu");
        let module = compiler.compile(
            source,
            &CompileOptions {
                extra_options: vec![
                    "--std=c++17".into(),
                    format!("--define-macro=LIBMIR_OUTPUT_REFINE_TOP_K={TOP_K}"),
                ],
                ..CompileOptions::default()
            },
        )?;
        Ok(Self {
            candidates: module.kernel()?,
            merge: module.kernel()?,
            refine: module.kernel()?,
            spec,
        })
    }

    pub fn workspace_elements(vocab: usize) -> Result<usize> {
        blocks(vocab)?
            .checked_mul(TOP_K)
            .ok_or(Error::InvalidDecoderKernel("FP8 refinement workspace overflow"))
    }

    #[must_use]
    pub const fn candidate_count() -> usize {
        TOP_K
    }

    pub fn execute(
        &self,
        stream: &Stream,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        logits: &mut DeviceBuffer<bf16>,
        first: &mut DeviceBuffer<u64>,
        second: &mut DeviceBuffer<u64>,
    ) -> Result<()> {
        self.validate(input, weight, logits, first, second)?;
        let initial_blocks = blocks(self.spec.output_features)?;
        self.candidates.launch(
            stream,
            launch(initial_blocks)?,
            (logits, &mut *first, narrow(self.spec.output_features)?),
        )?;
        let mut count = initial_blocks * TOP_K;
        let mut in_first = true;
        while count > TOP_K {
            let next_blocks = blocks(count)?;
            if in_first {
                self.merge.launch(
                    stream,
                    launch(next_blocks)?,
                    (&*first, &mut *second, narrow(count)?),
                )?;
            } else {
                self.merge.launch(
                    stream,
                    launch(next_blocks)?,
                    (&*second, &mut *first, narrow(count)?),
                )?;
            }
            count = next_blocks * TOP_K;
            in_first = !in_first;
        }
        let candidates = if in_first {
            &*first
        } else {
            &*second
        };
        Ok(self.refine.launch(
            stream,
            LaunchConfig {
                grid: (narrow(TOP_K.div_ceil(8))?, 1, 1),
                block: (THREADS, 1, 1),
                shared_memory_bytes: 0,
            },
            (input, weight, candidates, logits, narrow(self.spec.input_features)?),
        )?)
    }

    fn validate(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &DeviceBuffer<bf16>,
        logits: &DeviceBuffer<bf16>,
        first: &DeviceBuffer<u64>,
        second: &DeviceBuffer<u64>,
    ) -> Result<()> {
        let workspace = Self::workspace_elements(self.spec.output_features)?;
        require("refinement input", self.spec.input_features, input.len())?;
        require("refinement BF16 weight", self.spec.weight_elements()?, weight.len())?;
        require("refinement logits", self.spec.output_features, logits.len())?;
        require("refinement first workspace", workspace, first.len())?;
        require("refinement second workspace", workspace, second.len())
    }
}

fn blocks(elements: usize) -> Result<usize> {
    elements
        .checked_add(CHUNK - 1)
        .map(|padded| padded / CHUNK)
        .ok_or(Error::InvalidDecoderKernel("FP8 refinement grid overflow"))
}

fn launch(block_count: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(block_count)?, 1, 1),
        block: (THREADS, 1, 1),
        shared_memory_bytes: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::Fp8OutputSpec;

    #[test]
    fn refinement_admits_a_half_block_tail_without_relaxing_other_output_paths() {
        assert!(Fp8OutputSpec::new_refinement(2_880, 201_088).is_ok());
        assert!(Fp8OutputSpec::new(2_880, 201_088).is_err());
    }
}
