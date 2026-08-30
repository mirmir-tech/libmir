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
    };

    model.generate(
        &request,
        &mut |_| {},
        &mut |token| print!("{}", token.text),
    )?;
    Ok(())
}
```

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
