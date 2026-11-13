use super::*;
use crate::engine::{ModelTensors, lowering};

#[test]
#[ignore = "real Qwen router precision across natural inputs, layers and initialized contexts"]
fn compares_router_precision_across_layers() -> std::result::Result<(), Box<dyn std::error::Error>>
{
    let path = std::env::var("MIRMIR_BENCH_MODEL")?;
    let root = std::path::Path::new(&path);
    let layout = models::layout::ModelLayout::inspect(root)?;
    let tokenizer = models::tokenizer::TextTokenizer::from_layout(&layout)?;
    let decoder = models::layout::DecoderConfig::from_layout(&layout)?;
    let catalog = models::weights::TensorCatalog::from_layout(&layout)?;
    let contract =
        models::execution::DecoderExecutionContract::discover(&layout, &decoder, &catalog)?;
    let lowering = lowering::plan(&contract.semantic)?;
    let load_stream = Stream::new_cpu()?;
    let tensors = ModelTensors::load(root, &load_stream)?;
    let mut config = crate::MetalConfig::default();
    config.tuning.mode = runtime::tuning::TuningMode::Disabled;
    config.cache.prefix_cache_entries = 0;
    let stream = Stream::new_gpu_with_config(std::sync::Arc::new(config))?;
    let model = HybridLinearMoeModel::load(
        &tensors,
        &decoder,
        &contract.bindings,
        lowering.layers(),
        256,
        &stream,
    )?;
    let texts = [
        "Wyjaśnij, jak działa pamięć podręczna procesora i dlaczego lokalność danych przyspiesza program. ",
        "A library has twelve shelves. Each shelf contains twenty books. Explain how to calculate the total number of books. ",
        "Write a Rust function that finds the largest value in a list of integers. Explain how an empty list should be handled. ",
        "Describe the water cycle, including evaporation, condensation and rainfall. Explain the role of sunlight in this process. ",
        "Napisz krótką opowieść o podróżniku, który odwiedził małą górską miejscowość i spotkał tam starego przyjaciela. ",
    ];
    let prompts = texts
        .iter()
        .map(|text| {
            Ok(tokenizer
                .encode_with_special_tokens(text, false)?
                .token_ids
                .into_iter()
                .cycle()
                .take(256)
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;
    for chunk in [8, 128] {
        let mut caches = (0..5).map(|_| model.new_cache(&stream)).collect::<Result<Vec<_>>>()?;
        for cache in &mut caches {
            cache.reserve(256)?;
        }
        for position in (0..=128).step_by(chunk) {
            let tokens = prompts
                .iter()
                .flat_map(|tokens| tokens[position..position + chunk].iter().copied())
                .collect::<Vec<_>>();
            let mut hidden = model
                .embedding
                .lookup(&Array::from_u32(&tokens, &[5, i32::try_from(chunk)?])?, &stream)?;
            for (index, layer) in model.layers.iter().enumerate() {
                hidden = layer.mix_packed_prefill(
                    &hidden,
                    &mut caches.iter_mut().collect::<Vec<_>>(),
                    &[i32::try_from(position)?; 5],
                    &stream,
                )?;
                if [0, model.layers.len() / 2, model.layers.len() - 1].contains(&index)
                    && [0, 128].contains(&position)
                {
                    let input =
                        layer.post_attention_norm.apply(&hidden, layer.rms_norm_eps, &stream)?;
                    layer.moe.probe_router_precision(
                        &input,
                        &format!("layer={index},chunk={chunk},offset={position}"),
                        &stream,
                    )?;
                }
                // Captures follow the baseline trajectory, independently of probe precision.
                hidden = layer.feed_forward_packed_prefill(&hidden, &stream)?;
            }
            let mut roots = vec![&hidden];
            for cache in &caches {
                cache.extend_graph_roots(&mut roots);
            }
            stream.eval_many_with_paged_arenas(&roots)?;
            stream.synchronize()?;
            stream.detach_paged_arena_graphs()?;
            for cache in &caches {
                cache.detach_evaluated_graphs(&stream)?;
            }
        }
        drop(caches);
        assert_eq!(stream.paged_arenas().resident_arenas()?, 0);
    }
    Ok(())
}
