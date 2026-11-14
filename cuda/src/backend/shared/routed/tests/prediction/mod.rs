//! Teacher-forced drafting on a real state-space hybrid checkpoint: the
//! prediction module's argmax at row `i` against the decoder's argmax at row
//! `i + 1`, over reference greedy continuations
//! (`benchmarks/tools/nemotron/greedy_oracle.py`). Runs when
//! `LIBMIR_CUDA_PREDICTION_MODEL` and `LIBMIR_CUDA_PREDICTION_REFERENCE` are
//! set; `--nocapture` prints the acceptance.
#![allow(clippy::print_stdout, reason = "the gate reports its measurements")]

use std::path::Path;

use models::{
    layout::{DecoderConfig, ModelLayout},
    weights::TensorCatalog,
};
use runtime::speculation::Speculation;

use super::*;
use crate::DraftRow;

mod batched;
mod speculative;

const CHUNK: usize = 256;
const BLOCK: usize = 16;
const DRAFT_TOKENS: usize = 3;

#[test]
fn prediction_module_drafts_the_decoder_greedy_token()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let (Some(root), Some(reference)) = (
        std::env::var_os("LIBMIR_CUDA_PREDICTION_MODEL"),
        std::env::var_os("LIBMIR_CUDA_PREDICTION_REFERENCE"),
    ) else {
        return Ok(());
    };
    let layout = ModelLayout::inspect(Path::new(&root))?;
    let decoder = DecoderConfig::from_layout(&layout)?;
    let catalog = TensorCatalog::from_layout(&layout)?;
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(reference)?)?;
    let backend = CudaBackend::new(CudaConfig::default())?;
    let cache = CacheConfig {
        block_size: BLOCK,
        block_count: 64,
        dtype: KvCacheDType::BFloat16,
    };
    let template = backend.load_shared_routed_model_template(
        &decoder,
        &catalog,
        crate::SharedRoutedModelLoadConfig {
            cache,
            max_sequence_blocks: 64,
            speculation: Some(Speculation {
                draft_tokens: runtime::speculation::DraftTokens::MIN,
            }),
        },
    )?;
    let caches = template.allocate_shared_kv()?;
    let (mut accepted, mut drafted, mut agreed) = (0_usize, 0_usize, 0_usize);
    let records = reference["records"].as_array().ok_or("reference has no records")?;
    for record in records {
        let ids = |key: &str| -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
            record[key]
                .as_array()
                .ok_or("record has no token ids")?
                .iter()
                .map(|id| Ok(u32::try_from(id.as_u64().ok_or("token id is not a number")?)?))
                .collect()
        };
        let prompt = ids("prompt_ids")?.len();
        let sequence = [ids("prompt_ids")?, ids("generated_ids")?].concat();
        let (decoder_argmax, draft_argmax) = forced(&template, &caches, &sequence)?;
        // Row `i` drafts token `i + 2`; the decoder predicts it at row `i + 1`.
        for row in prompt - 1..sequence.len() - 2 {
            drafted += 1;
            accepted += usize::from(draft_argmax[row] == decoder_argmax[row + 1]);
            agreed += usize::from(decoder_argmax[row] == sequence[row + 1]);
        }
    }
    let rate = |count: usize| real(count) / real(drafted);
    println!(
        "drafted={drafted} accepted={accepted} acceptance={:.4} decoder_reference_agreement={:.4}",
        rate(accepted),
        rate(agreed)
    );
    assert!(rate(accepted) >= 0.85, "draft acceptance {:.4} is below 0.85", rate(accepted));
    vision_session_does_not_draft(&template, &caches)?;
    compare(&template, &caches, records)
}

/// A vision prompt leaves the prediction module's K/V unwritten, so its
/// session decodes without drafting.
fn vision_session_does_not_draft(
    template: &CudaSharedRoutedModelTemplate,
    caches: &[Option<PagedKvCache>],
) -> Result<()> {
    let mut session = template.instantiate_with_caches(caches)?;
    assert!(session.predicts());
    let mut table = BlockTable::with_block_size(BLOCK);
    table.push(BlockId(0));
    table.set_token_len(3);
    let image = copy(&template.backend, &vec![bf16::ZERO; template.decoder.hidden_size])?;
    let positions = [0_u32, 1, 2, 0, 1, 2, 0, 1, 2];
    session.prefill_vision(Uuid::new_v4(), &[1, 2, 3], &positions, &table, (1, 2), &image, 0)?;
    assert!(!session.predicts());
    assert_eq!(session.pending_drafts(), 0);
    Ok(())
}

/// Greedy speculative decoding against plain decoding of the same prompts.
fn compare(
    template: &CudaSharedRoutedModelTemplate,
    caches: &[Option<PagedKvCache>],
    records: &[serde_json::Value],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let count = 128;
    let mut widest = 0.0_f32;
    let (mut plains, mut prompts) = (Vec::new(), Vec::new());
    for record in records {
        let prompt = record["prompt_ids"]
            .as_array()
            .ok_or("record has no prompt")?
            .iter()
            .map(|id| Ok(u32::try_from(id.as_u64().ok_or("token id is not a number")?)?))
            .collect::<std::result::Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        let plain = speculative::plain(template, caches, &prompt, count)?;
        let fast = speculative::speculative(template, caches, &prompt, count, DRAFT_TOKENS)?;
        let divergence = plain.tokens.iter().zip(&fast.tokens).position(|(a, b)| a != b);
        let margin = divergence.map(|index| plain.margins[index]);
        println!(
            "prompt_tokens={} plain_tok_s={:.2} speculative_tok_s={:.2} \
             mean_accepted_length={:.3} first_divergence={divergence:?} margin={margin:?}",
            prompt.len(),
            real(count) / plain.seconds,
            real(count) / fast.seconds,
            real(count) / real(fast.steps),
        );
        widest = widest.max(margin.unwrap_or(0.0));
        plains.push(plain);
        prompts.push(prompt);
    }
    // The first pass builds every shape; the second is timed.
    batched::batched(template, caches, &prompts[..3], count)?;
    let batched = batched::batched(template, caches, &prompts[..3], count)?;
    let generated = real(3 * count);
    println!(
        "batched rows=3 aggregate_tok_s={:.2} tokens_per_row_step={:.3}",
        generated / batched.seconds,
        real(count) / real(batched.steps)
    );
    for (row, (tokens, plain)) in batched.tokens.iter().zip(&plains).enumerate() {
        let divergence = plain.tokens.iter().zip(tokens).position(|(a, b)| a != b);
        let margin = divergence.map(|index| plain.margins[index]);
        println!("batched row={row} first_divergence={divergence:?} margin={margin:?}");
        widest = widest.max(margin.unwrap_or(0.0));
    }
    // A near-tie: the verification pass and the one-token decode pass take
    // different kernels, so the top two may swap within a nat.
    assert!(widest <= 1.0, "divergence at a clear argmax, margin {widest}");
    Ok(())
}

/// Decoder and draft argmax at every row of `sequence[..len - 1]`, each row
/// given the following token.
fn forced(
    template: &CudaSharedRoutedModelTemplate,
    caches: &[Option<PagedKvCache>],
    sequence: &[u32],
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut session = template.instantiate_with_caches(caches)?;
    let id = Uuid::new_v4();
    let rows = sequence.len() - 1;
    let mut table = BlockTable::with_block_size(BLOCK);
    for block in 0..rows.div_ceil(BLOCK) {
        table.push(BlockId(u32::try_from(block).map_err(|_| Error::InvalidPagedKv("block"))?));
    }
    let (mut decoder, mut draft) = (Vec::new(), Vec::new());
    for start in (0..rows).step_by(CHUNK) {
        let end = (start + CHUNK).min(rows);
        table.set_token_len(end);
        session.prefill_drafting(id, &sequence[start..end], &sequence[start + 1..=end], &table)?;
        for row in 0..end - start {
            decoder.push(argmax(&read(
                template,
                session.row_logits(end - start, DraftRow::Decoder(row))?,
            )?));
            draft.push(argmax(&read(
                template,
                session.row_logits(end - start, DraftRow::Prediction(row))?,
            )?));
        }
    }
    Ok((decoder, draft))
}

fn read(
    template: &CudaSharedRoutedModelTemplate,
    source: &mircuda::DeviceBuffer<bf16>,
) -> Result<Vec<bf16>> {
    let backend = &template.backend;
    let mut host = backend.inner.context.allocate_pinned::<bf16>(source.len())?;
    backend.inner.stream.copy_to_host(source, &mut host)?;
    Ok(host.to_vec()?)
}

fn argmax(values: &[bf16]) -> u32 {
    let (index, _) =
        values.iter().enumerate().fold((0, f32::NEG_INFINITY), |best, (index, value)| {
            if value.to_f32() > best.1 {
                (index, value.to_f32())
            } else {
                best
            }
        });
    u32::try_from(index).unwrap_or(u32::MAX)
}

fn real(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}
