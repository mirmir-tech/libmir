use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, LaunchConfig, Stream, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};

use super::geometry::{narrow, require};
use crate::{Error, Result};

pub const MAX_TOP_K: usize = 64;
const THREADS: u32 = 256;
const ITEMS_PER_THREAD: usize = 8;
const CHUNK: usize = THREADS as usize * ITEMS_PER_THREAD;

cuda_export!(CandidatesKernel = "libmir_cuda_sampling_candidates_bf16"(
    logits: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u64>,
    denominator: &mut DeviceBuffer<f32>, vocab: u32, logits_stride: u32, top_k: u32,
    row: u32, workspace_stride: u32,
));
cuda_export!(MergeKernel = "libmir_cuda_sampling_merge"(
    input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>,
    denominator: &mut DeviceBuffer<f32>, count: u32, top_k: u32,
    row: u32, workspace_stride: u32,
));
cuda_export!(MassKernel = "libmir_cuda_sampling_mass_bf16"(
    logits: &DeviceBuffer<bf16>, candidates: &DeviceBuffer<u64>,
    denominator: &mut DeviceBuffer<f32>, vocab: u32, logits_stride: u32, row: u32,
    workspace_stride: u32,
));
cuda_export!(FinalizeKernel = "libmir_cuda_sampling_finalize_bf16"(
    logits: &DeviceBuffer<bf16>, candidates: &DeviceBuffer<u64>,
    denominator: &DeviceBuffer<f32>, output: &mut DeviceBuffer<u32>,
    top_k: u32, top_p: f32, temperature: f32, draw: f32,
    vocab: u32, logits_stride: u32, row: u32, workspace_stride: u32,
));

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingSpec {
    pub vocab: usize,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub draw: f32,
}

pub struct SamplingWorkspace<'a> {
    pub first: &'a mut DeviceBuffer<u64>,
    pub second: &'a mut DeviceBuffer<u64>,
    pub denominator: &'a mut DeviceBuffer<f32>,
}

#[derive(Clone, Debug)]
pub struct Sampling {
    candidates: TypedKernel<CandidatesKernel>,
    merge: TypedKernel<MergeKernel>,
    mass: TypedKernel<MassKernel>,
    finalize: TypedKernel<FinalizeKernel>,
    vocab: usize,
}

impl Sampling {
    pub fn compile(compiler: &Compiler, vocab: usize) -> Result<Self> {
        if vocab == 0 {
            return Err(Error::InvalidSampling("sampling vocabulary is empty".into()));
        }
        let source = cuda_kernel_file!("../../kernels/sampling_bf16.cu");
        let options = CompileOptions {
            extra_options: vec!["--std=c++17".into()],
            ..CompileOptions::default()
        };
        let module = compiler.compile(source, &options)?;
        Ok(Self {
            candidates: module.kernel()?,
            merge: module.kernel()?,
            mass: module.kernel()?,
            finalize: module.kernel()?,
            vocab,
        })
    }

    pub fn workspace_elements(vocab: usize) -> Result<usize> {
        blocks(vocab)?
            .checked_mul(MAX_TOP_K)
            .ok_or_else(|| Error::InvalidSampling("sampling workspace overflow".into()))
    }

    pub fn execute(
        &self,
        stream: &Stream,
        logits: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<u32>,
        workspace: SamplingWorkspace<'_>,
        spec: SamplingSpec,
    ) -> Result<()> {
        self.execute_row(stream, logits, output, workspace, spec, 0)
    }

    /// Enqueues one independently configured row from a packed logits matrix.
    pub fn execute_row(
        &self,
        stream: &Stream,
        logits: &DeviceBuffer<bf16>,
        output: &mut DeviceBuffer<u32>,
        workspace: SamplingWorkspace<'_>,
        spec: SamplingSpec,
        row: usize,
    ) -> Result<()> {
        validate(spec)?;
        if spec.vocab > self.vocab {
            return Err(Error::InvalidSampling("sampling vocabulary exceeds logits".into()));
        }
        let capacity = Self::workspace_elements(self.vocab)?;
        let rows = row
            .checked_add(1)
            .ok_or_else(|| Error::InvalidSampling("sampling row overflow".into()))?;
        require("sampling logits", rows * self.vocab, logits.len())?;
        require("sampled token", rows, output.len())?;
        require("sampling first workspace", rows * capacity, workspace.first.len())?;
        require("sampling second workspace", rows * capacity, workspace.second.len())?;
        require("sampling denominator", rows, workspace.denominator.len())?;
        let SamplingWorkspace { first, second, denominator } = workspace;
        let top_k = narrow(spec.top_k)?;
        let row = narrow(row)?;
        let stride = narrow(capacity)?;
        let initial_blocks = blocks(spec.vocab)?;
        self.candidates.launch(
            stream,
            launch(initial_blocks)?,
            (
                logits,
                &mut *first,
                &mut *denominator,
                narrow(spec.vocab)?,
                narrow(self.vocab)?,
                top_k,
                row,
                stride,
            ),
        )?;
        let mut count = initial_blocks
            .checked_mul(spec.top_k)
            .ok_or_else(|| Error::InvalidSampling("sampling candidate overflow".into()))?;
        let mut in_first = true;
        while count > spec.top_k {
            let next_blocks = blocks(count)?;
            if in_first {
                self.merge.launch(
                    stream,
                    launch(next_blocks)?,
                    (&*first, &mut *second, &mut *denominator, narrow(count)?, top_k, row, stride),
                )?;
            } else {
                self.merge.launch(
                    stream,
                    launch(next_blocks)?,
                    (&*second, &mut *first, &mut *denominator, narrow(count)?, top_k, row, stride),
                )?;
            }
            count = next_blocks
                .checked_mul(spec.top_k)
                .ok_or_else(|| Error::InvalidSampling("sampling merge overflow".into()))?;
            in_first = !in_first;
        }
        let candidates = if in_first {
            &*first
        } else {
            &*second
        };
        if spec.top_k > 1 {
            let mass_blocks = blocks(spec.vocab)?.min(1_024);
            self.mass.launch(
                stream,
                launch(mass_blocks)?,
                (
                    logits,
                    candidates,
                    &mut *denominator,
                    narrow(spec.vocab)?,
                    narrow(self.vocab)?,
                    row,
                    stride,
                ),
            )?;
        }
        Ok(self.finalize.launch(
            stream,
            LaunchConfig {
                grid: (1, 1, 1),
                block: (1, 1, 1),
                shared_memory_bytes: 0,
            },
            (
                logits,
                candidates,
                &*denominator,
                output,
                top_k,
                spec.top_p,
                spec.temperature,
                spec.draw,
                narrow(spec.vocab)?,
                narrow(self.vocab)?,
                row,
                stride,
            ),
        )?)
    }
}

fn blocks(elements: usize) -> Result<usize> {
    elements
        .checked_add(CHUNK - 1)
        .map(|padded| padded / CHUNK)
        .ok_or_else(|| Error::InvalidSampling("sampling grid overflow".into()))
}

fn launch(block_count: usize) -> Result<LaunchConfig> {
    Ok(LaunchConfig {
        grid: (narrow(block_count)?, 1, 1),
        block: (THREADS, 1, 1),
        shared_memory_bytes: 0,
    })
}

fn validate(spec: SamplingSpec) -> Result<()> {
    if spec.vocab == 0
        || spec.top_k == 0
        || spec.top_k > spec.vocab
        || spec.top_k > MAX_TOP_K
        || !spec.top_p.is_finite()
        || spec.top_p <= 0.0
        || spec.top_p > 1.0
        || !spec.temperature.is_finite()
        || spec.temperature <= 0.0
        || !spec.draw.is_finite()
        || !(0.0..1.0).contains(&spec.draw)
    {
        Err(Error::InvalidSampling("invalid bounded CUDA sampling policy".into()))
    } else {
        Ok(())
    }
}
