# libmir

Libmir is a native Rust library for embedding local language-model inference in
applications. It turns Hugging Face-format checkpoints into backend-neutral
models and sessions backed by Apple Metal or NVIDIA CUDA.

Use libmir when an application needs direct ownership of model loading,
generation, scheduling, cache policy, and telemetry without running a separate
inference service or adding a Python runtime.

## What it provides

- checkpoint, tokenizer, chat-template, and architecture discovery;
- text, vision, embedding, and reranking tasks;
- streamed generation, sampling, cancellation, and reasoning channels;
- independent sessions and concurrent request scheduling;
- paged K/V storage, prefix caching, sliding windows, and quantized caches;
- explicit memory estimation and runtime telemetry;
- Metal execution through `mirtal`;
- CUDA execution through `mircuda`.

Model admission is derived from checkpoint configuration, tensor layout,
quantization, and backend capabilities rather than a model-name allowlist.

## Add it to a project

For Apple Metal:

```toml
[dependencies]
libmir = { version = "0.3.1", features = ["metal"] }
```

For NVIDIA CUDA:

```toml
[dependencies]
libmir = { version = "0.3.1", default-features = false, features = ["cuda"] }
```

## Generate from a local checkpoint

The public API covers the complete lifecycle: create a runtime, load a model,
submit a chat request, and consume generated tokens as they arrive.

```rust,no_run
use libmir::{
    Conversation, GenerationOverrides, GenerationRequest, Library, Message,
    RuntimeConfig,
};

fn main() -> libmir::Result<()> {
    let checkpoint = std::env::args_os()
        .nth(1)
        .expect("pass a local model directory");

    let library = Library::new(RuntimeConfig::default());
    let model = library.load(
        checkpoint,
        GenerationOverrides::default(),
        &mut |_| {},
    )?;

    let request = GenerationRequest {
        conversation: Conversation {
            messages: vec![Message {
                role: "user".into(),
                content: "Explain paged K/V caching in one paragraph.".into(),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: Vec::new(),
            tool_choice: Default::default(),
        },
        options: GenerationOverrides {
            max_tokens: Some(256),
            temperature: Some(0.2),
            top_p: Some(0.9),
            ..GenerationOverrides::default()
        },
        seed: None,
        ..GenerationRequest::default()
    };

    model.generate(
        &request,
        &mut |_| {},
        &mut |token| print!("{}", token.text),
    )?;
    Ok(())
}
```

Set `GenerationRequest::reasoning` to `ReasoningMode::Enabled` or
`ReasoningMode::Disabled` to select a text request’s thinking template switch.
The default preserves existing model rendering. Explicit modes require a
Jinja `enable_thinking` input or a supported built-in Qwen/Gemma template;
unsupported templates and image requests reject explicit modes.
`GenerationOverrides::max_tokens` counts all generated tokens, including
reasoning and final content. There is no separate reasoning-token budget.

Run it with a local Hugging Face-format model directory:

```sh
cargo run --release -- /path/to/model
```

Libmir does not select models from environment variables or command-line
arguments. The embedding application owns configuration and uses the public
`Library`, `Model`, and `Session` APIs to build the desired lifecycle.

## Typical uses

- add local generation to a native Rust application;
- build an inference server with application-specific transport and policy;
- run many independent conversations over one loaded model;
- expose embeddings or reranking without a hosted dependency;
- inspect model compatibility and memory requirements before loading.

## Performance

Libmir owns `MiRMiR`'s inference performance work across Metal and CUDA. We are
actively improving throughput and latency with the goal of catching up to and
then outperforming established inference engines, including
[vLLM](https://github.com/vllm-project/vllm) and
[MLX-LM](https://github.com/ml-explore/mlx-lm). See the
[benchmark index](benchmarks/index.md) for the current comparisons and test
methodology.

## Links

- [Guide](https://docs.mirmir.tech/libmir/index.html)
- [Getting started](https://docs.mirmir.tech/libmir/getting-started.html)
- [crates.io](https://crates.io/crates/libmir)
- [Rust API documentation](https://docs.rs/libmir)
- [Source and issues](https://github.com/mirmir-tech/libmir)

Licensed under Apache-2.0.

### Cached prefill admission

`RuntimeConfig::scheduler.cached_prefill_policy` defaults to
`CachedPrefillPolicy::BackendDefault`. The opt-in `InterleaveOneBlock` admits one
cached continuation at the queue head while a backend normally defers prefill
until resident decode finishes. It requires a free request slot, estimated work
within one KV block and remaining token/memory capacity. Actual prefill work is
also capped at one block per step if a backend snapshot has disappeared.
Long queue heads and existing cohorts retain their normal ordering.

This is a latency tradeoff, not a throughput optimization: the Qwen diagnostic
reduced refill first-token latency from about 1 s to 47 ms while increasing
survivor decode time by about 22%. The backend default remains unchanged.
CUDA's normal interleaving is unaffected; GPT-OSS has not yet been measured
with this opt-in policy.

### CUDA prefill completion and short admission

Mixed-attention runners whose rows can join combined ragged steps, dense or
routed, use completion-first prefill, retaining device
state between chunks and publishing completed rows before the batch ends.
Automatic terminal checkpoints below 128 tokens do not force an extra forward;
explicitly requested checkpoints remain honored. Idle all-short queues cap the
configured collection quiet window at 3 ms; a newly arriving long prompt widens
that window using the original arrival timestamps.

`RuntimeConfig::scheduler.prefill_refill_policy` defaults to
`PrefillRefillPolicy::Closed`. Explicit `ShortPrompt` lets queue-head prompts of
at most 128 tokens join a running long prefill at chunk boundaries. It requires
such a completion-first CUDA runner and rejects `CompleteCohort`. Available
request slots, resident KV pages and half the step token budget bound admission.
The policy trades long-request first-token latency for shorter waiting by new
short requests. Other CUDA runners retain round-robin prefill, and Metal retains
its existing admission and completion policy.
