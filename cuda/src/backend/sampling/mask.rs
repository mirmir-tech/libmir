use mircuda::{
    CompileOptions, DeviceBuffer, LaunchConfig, PinnedBuffer, TypedKernel, bf16, cuda_export,
    cuda_kernel_file,
};
use runtime::backend::SamplingLogits;

use super::CudaBackend;
use crate::{Error, Result};

cuda_export!(MaskKernel = "libmir_sampling_mask"(
    input: &DeviceBuffer<bf16>, mask: &DeviceBuffer<u32>, output: &mut DeviceBuffer<bf16>,
    vocab: u32, rows: u32, words: u32,
));

/// Per-sampler scratch; never overwrites logits retained in the prefix cache.
#[derive(Debug)]
pub(super) struct Workspace {
    kernel: TypedKernel<MaskKernel>,
    host: PinnedBuffer<u32>,
    mask: DeviceBuffer<u32>,
    logits: DeviceBuffer<bf16>,
}

pub(super) fn apply<'a>(
    backend: &CudaBackend,
    workspace: &'a mut Option<Workspace>,
    logits: &'a DeviceBuffer<bf16>,
    policies: &[SamplingLogits],
    vocab: usize,
) -> Result<&'a DeviceBuffer<bf16>> {
    if !policies.iter().any(|policy| matches!(policy, SamplingLogits::Masked { .. })) {
        return Ok(logits);
    }
    let rows = policies.len();
    let words = vocab.div_ceil(32);
    let count = rows
        .checked_mul(vocab)
        .ok_or_else(|| Error::InvalidSampling("mask size overflow".into()))?;
    if count > logits.len() || count > u32::MAX as usize {
        return Err(Error::InvalidSampling("mask exceeds logits shape".into()));
    }
    let mut bits = vec![u32::MAX; rows * words];
    for (row, policy) in policies.iter().enumerate() {
        if let SamplingLogits::Masked { mask, .. } = policy {
            if mask.vocab() > vocab {
                return Err(Error::InvalidSampling("mask vocabulary exceeds logits".into()));
            }
            let target = &mut bits[row * words..(row + 1) * words];
            target.fill(0);
            target[..mask.words().len()].copy_from_slice(mask.words());
        }
    }
    if workspace.as_ref().is_none_or(|work| work.logits.len() != count) {
        let module = backend.inner.compiler.compile(
            cuda_kernel_file!("../../../kernels/sampling_mask.cu"),
            &CompileOptions::default(),
        )?;
        *workspace = Some(Workspace {
            kernel: module.kernel()?,
            host: backend.inner.context.allocate_pinned(bits.len())?,
            mask: backend.inner.pool.allocate(&backend.inner.stream, bits.len())?,
            logits: backend.inner.pool.allocate(&backend.inner.stream, count)?,
        });
    }
    let work = workspace
        .as_mut()
        .ok_or_else(|| Error::InvalidSampling("mask scratch missing".into()))?;
    work.host.copy_from_slice(&bits)?;
    backend.inner.stream.copy_to_device(&mut work.host, &mut work.mask)?;
    work.kernel.launch(
        &backend.inner.stream,
        LaunchConfig {
            grid: (u32::try_from(count.div_ceil(256))?, 1, 1),
            block: (256, 1, 1),
            shared_memory_bytes: 0,
        },
        (
            logits,
            &work.mask,
            &mut work.logits,
            u32::try_from(vocab)?,
            u32::try_from(rows)?,
            u32::try_from(words)?,
        ),
    )?;
    Ok(&work.logits)
}
