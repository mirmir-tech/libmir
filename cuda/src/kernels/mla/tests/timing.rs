//! Decode latency of one latent attention layer over a long context, for the
//! split choice and the bandwidth headroom of the kernel.

use mircuda::bf16;
use models::layout::RotaryPairing;

use super::{
    super::{
        LatentAbsorption, LatentAttention, LatentAttentionInputs, LatentAttentionLayer,
        LatentAttentionSpec, LatentCacheView,
    },
    fixture::{Runtime, Values},
};
use crate::Result;

#[test]
#[ignore = "measures latent attention decode latency on an idle CUDA device"]
#[allow(clippy::print_stderr)]
fn measures_latent_decode_latency() -> Result<()> {
    let runtime = Runtime::new()?;
    let context = 4096_usize;
    for heads in [16, 64, 128] {
        for splits in [1, 8, 32, 64] {
            let micros = decode(&runtime, heads, context, splits)?;
            let bytes = f32::from(u16::try_from(context * 1152 / 1024)?) * 1024.0;
            eprintln!(
                "heads {heads:3} context {context} splits {splits:2}: {micros:8.1} us, cache read {:6.1} GB/s",
                bytes / micros / 1_000.0,
            );
        }
    }
    Ok(())
}

fn decode(runtime: &Runtime, heads: usize, context: usize, splits: usize) -> Result<f32> {
    let spec = LatentAttentionSpec {
        heads,
        latent_rank: 512,
        rope_dim: 64,
        nope_dim: 128,
        value_dim: 128,
        rotary_pairing: RotaryPairing::Interleaved,
        block_size: 16,
        max_queries: 1,
        splits,
    };
    let blocks = context.div_ceil(spec.block_size);
    let mut values = Values::new(7);
    let attention = LatentAttention::compile(&runtime.compiler, spec)?;
    let mut workspace = attention.workspace(&runtime.pool, &runtime.stream)?;
    let mut pages = runtime.pool.allocate_zeroed(&runtime.stream, spec.page_bytes(blocks)?)?;
    let table = (0..blocks).map(u32::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
    let block_tables = runtime.copy(&table)?;
    let latent_norm = runtime.copy_f32(&vec![1.0; spec.latent_rank])?;
    let key_value_up = runtime.copy_f32(&values.take(heads * 256 * spec.latent_rank, 0.044))?;
    let inverse_frequencies = runtime.copy(&vec![0.01_f32; spec.rope_dim / 2])?;
    let layer = LatentAttentionLayer {
        latent_norm: &latent_norm,
        key_value_up: &key_value_up,
        inverse_frequencies: &inverse_frequencies,
        epsilon: 1e-6,
        rope_scale: 1.0,
        softmax_scale: 0.07,
    };
    let query = runtime.copy_f32(&values.take(heads * 192, 1.0))?;
    let latent = runtime.copy_f32(&values.take(spec.row_width(), 1.0))?;
    let last = u32::try_from(context - 1)?;
    let positions = runtime.copy(&[last])?;
    let query_sequences = runtime.copy(&[0_u32])?;
    let slot_mapping = runtime.copy(&[last])?;
    let mut absorption = LatentAbsorption::new(&runtime.context, &runtime.stream, spec, 1)?;
    let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, heads * spec.value_dim)?;
    runtime.measure(64, || {
        attention.execute(
            &runtime.stream,
            &mut absorption,
            &layer,
            LatentCacheView {
                pages: &mut pages,
                block_tables: &block_tables,
                table_stride: blocks,
            },
            LatentAttentionInputs {
                query: &query,
                latent: &latent,
                positions: &positions,
                slot_mapping: &slot_mapping,
                query_sequences: &query_sequences,
            },
            &mut workspace,
            &mut output,
        )
    })
}
