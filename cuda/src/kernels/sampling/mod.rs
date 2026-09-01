use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, Stream, TypedKernel, bf16, cuda_kernel_file,
};

use super::geometry::{narrow, require};
use crate::{Error, Result};

mod bindings;
mod bounded;
mod full;
mod validation;
use bindings::{
    CandidatesKernel, FinalizeKernel, FullFinalizeKernel, FullMassKernel, MassKernel, MergeKernel,
};
use bounded::BoundedSampling;
use full::FullSampling;
use validation::{blocks, launch, validate};

pub const MAX_TOP_K: usize = 64;
const THREADS: u32 = 256;
const ITEMS_PER_THREAD: usize = 8;
const CHUNK: usize = THREADS as usize * ITEMS_PER_THREAD;

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
    pub block_mass: &'a mut DeviceBuffer<f32>,
}

#[derive(Clone, Debug)]
pub struct Sampling {
    candidates: TypedKernel<CandidatesKernel>,
    merge: TypedKernel<MergeKernel>,
    mass: TypedKernel<MassKernel>,
    finalize: TypedKernel<FinalizeKernel>,
    full_mass: TypedKernel<FullMassKernel>,
    full_finalize: TypedKernel<FullFinalizeKernel>,
    vocab: usize,
}

impl Sampling {
    pub fn compile(compiler: &Compiler, vocab: usize) -> Result<Self> {
        if vocab == 0 {
            return Err(Error::InvalidSampling("sampling vocabulary is empty".into()));
        }
        let source = cuda_kernel_file!("../../../kernels/sampling_bf16.cu");
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
            full_mass: module.kernel()?,
            full_finalize: module.kernel()?,
            vocab,
        })
    }

    pub fn workspace_elements(vocab: usize) -> Result<usize> {
        blocks(vocab)?
            .checked_mul(MAX_TOP_K)
            .ok_or_else(|| Error::InvalidSampling("sampling workspace overflow".into()))
    }

    pub fn block_mass_elements(vocab: usize) -> Result<usize> {
        blocks(vocab)
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
        let SamplingWorkspace { first, second, denominator, block_mass } = workspace;
        let candidate_k = spec.top_k.max(1);
        let top_k = narrow(candidate_k)?;
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
            .checked_mul(candidate_k)
            .ok_or_else(|| Error::InvalidSampling("sampling candidate overflow".into()))?;
        let mut in_first = true;
        while count > candidate_k {
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
                .checked_mul(candidate_k)
                .ok_or_else(|| Error::InvalidSampling("sampling merge overflow".into()))?;
            in_first = !in_first;
        }
        let candidates = if in_first {
            &*first
        } else {
            &*second
        };
        if spec.top_k == 0 {
            return self.execute_full(FullSampling {
                stream,
                logits,
                candidates,
                block_mass,
                output,
                spec,
                row,
                stride,
                block_count: initial_blocks,
            });
        }
        self.execute_bounded(BoundedSampling {
            stream,
            logits,
            candidates,
            denominator,
            output,
            spec,
            row,
            stride,
            top_k,
        })
    }
}
