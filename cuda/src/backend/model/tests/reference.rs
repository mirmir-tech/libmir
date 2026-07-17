use runtime::{
    backend::SamplingLogits,
    kv::{BlockId, BlockTable},
};
use uuid::Uuid;

use super::super::CudaMoeModelTemplate;
use crate::{CudaBackend, CudaOutputHeadPolicy, Result};

const PROMPT: std::ops::Range<u32> = 2..18;
const GENERATED_TOKENS: usize = 16;

#[allow(clippy::print_stderr)]
pub(super) fn print_greedy_sequence(
    backend: &CudaBackend,
    template: &CudaMoeModelTemplate,
    dense_vectors: bool,
    output_policy: CudaOutputHeadPolicy,
) -> Result<()> {
    let prompt = PROMPT.collect::<Vec<_>>();
    let mut session = template.instantiate()?;
    let mut table = BlockTable::with_block_size(16);
    table.push(BlockId(0));
    table.push(BlockId(1));
    table.set_token_len(prompt.len());
    session.prefill_from(Uuid::nil(), &prompt, 0, &table)?;

    let mut generated = Vec::with_capacity(GENERATED_TOKENS);
    for index in 0..GENERATED_TOKENS {
        let selected = session.sample(SamplingLogits::None)?;
        generated.push(read_selected(backend, selected)?);
        if index + 1 < GENERATED_TOKENS {
            table.set_token_len(prompt.len() + index + 1);
            session.decode_sampled(Uuid::nil(), &table)?;
        }
    }
    eprintln!(
        "reference greedy (dense_vectors={dense_vectors}, output_policy={output_policy:?}): {generated:?}"
    );
    Ok(())
}

fn read_selected(backend: &CudaBackend, selected: &mircuda::DeviceBuffer<u32>) -> Result<u32> {
    let mut host = backend.inner.context.allocate_pinned::<u32>(1)?;
    backend.inner.stream.copy_to_host(selected, &mut host)?;
    Ok(host.to_vec()?[0])
}
