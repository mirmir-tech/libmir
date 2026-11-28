//! A repeated prompt resumes from its snapshot without recomputing a token,
//! as the server sees it: the runtime shares the prompt's whole blocks and
//! hands the partial last one's tokens to a new block, while another request
//! has meanwhile overwritten the block the first request ended in.

use runtime::kv::BlockId;

use super::{
    fixture::{Family, load_family},
    *,
};

#[test]
fn a_repeated_prompt_resumes_from_its_snapshot_and_decodes_alike() -> Result<()> {
    for family in [Family::Hybrid, Family::Clamped] {
        let (mut model, directory) = load_family(family, 8)?;
        model.info.prefill_step = 16;
        for length in [16_u32, 17, 31, 32, 33] {
            model.clear_prefix_cache();
            let prompt = (0..length).map(|index| (index * 7 + 3) % 64).collect::<Vec<_>>();
            let whole = prompt.len() / 16;
            let first = [0, 1, 2, 3].map(BlockId);
            let (cached, reference) = prefill_and_decode(&mut model, &prompt, &first, 0)?;
            assert_eq!(cached, 0);

            // Another prompt overwrites the block the first request ended in,
            // which the runtime does not cache, and the blocks the repeat gets.
            let fresh = [5, 6, 7].map(BlockId);
            let other = (0..33).map(|index| (index * 5 + 11) % 64).collect::<Vec<u32>>();
            let _other =
                prefill_and_decode(&mut model, &other, &[&[first[whole]], &fresh[..]].concat(), 0)?;

            let repeat = [&first[..whole], &fresh[..], &[BlockId(8)]].concat();
            let (cached, repeated) =
                prefill_and_decode(&mut model, &prompt, &repeat[..4], whole * 16)?;
            assert_eq!(cached, prompt.len(), "{family:?} prompt of {length}");
            assert_eq!(repeated.len(), reference.len());
            for (repeated, reference) in repeated.iter().zip(&reference) {
                assert!(
                    (repeated - reference).abs() <= reference.abs().mul_add(1e-4, 1e-4),
                    "{family:?} prompt of {length}: {repeated} != {reference}"
                );
            }
        }
        drop(model);
        std::fs::remove_dir_all(directory)?;
    }
    Ok(())
}

/// Prefills `prompt` on `blocks`, the runtime caching its first
/// `cached_tokens`, and decodes greedily: the tokens the prefill resumed
/// from and the logits of each step.
fn prefill_and_decode(
    model: &mut LoadedModel,
    prompt: &[u32],
    blocks: &[BlockId],
    cached_tokens: usize,
) -> Result<(usize, Vec<f32>)> {
    let mut block_table = BlockTable::with_block_size(16);
    for &block in blocks {
        block_table.push(block);
    }
    let request = PrefillRequest {
        model: ModelHandle {
            id: model.info.manifest.id.clone(),
            backend: "metal".into(),
        },
        session_id: Uuid::new_v4(),
        prompt_tokens: prompt.to_vec(),
        cache_checkpoints: Vec::new(),
        block_table,
        cached_tokens,
        generation_tokens: None,
        sampling_logits: SamplingLogits::Full,
    };
    let session = request.session_id;
    let (batch, _) =
        MetalPrefillBatch::prepare(model, vec![(request, SamplingLogits::Full)], None)?;
    while !batch.execute_step(model, 64)?.complete {}
    let row = batch.finish()?.pop().ok_or(Error::NoPrefixLogits)?;
    let mut values = Vec::new();
    let mut output = row.native.output;
    for _ in 0..12 {
        let NativeOutput::Logits(logits) = output else {
            return Err(Error::InvalidPrefillBatch("expected logits".into()));
        };
        values.extend(logits.to_vec_f32(model.stream())?);
        let token = logits.argmax_u32(model.stream())?;
        output = model.decode_with_blocks(session, token, blocks, SamplingLogits::Full)?;
    }
    model.release_session(session)?;
    Ok((row.native.prefix_cache_tokens, values))
}
