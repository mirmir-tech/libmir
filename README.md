# libmir

Libmir is an embeddable native Rust library for discovering, loading, and
executing local language models. It is the inference layer used by Mirmir, but
it does not contain a CLI, HTTP server, WebSocket transport, or UI.

The runtime path has no `Python`, `PyTorch`, or `Transformers` dependency.
Accelerator integrations are explicit Cargo features: `metal` uses the
independent `mirtal` crate and `cuda` uses the independent `mircuda` crate.

## Status

Libmir is under active development and its public API is pre-1.0. Apple Metal
supports the admitted text decoder families. CUDA is connected to the same
`Library`, `Model`, `Session`, and `Engine` facade and executes end-to-end text
generation for the admitted `ModelOpt` NVFP4 routed-MoE decoder. CUDA admission is
intentionally narrower while independent checkpoint parity coverage is expanded.

Implemented decoder capabilities include dense `SwiGLU`, routed `MoE`, hybrid
linear-attention `MoE`, affine quantization, GQA/MQA, `RoPE` variants, Gated Delta,
device sampling, paged K/V, sliding windows, prefix snapshots, and
copy-on-write cache pages. Admission is based on discovered configuration and
tensor layout rather than a model-name allowlist.

## Architecture

```text
consumer
   |
   v
libmir facade: Library -> Model -> Session / generate
   |
   +-- models: checkpoint, tokenizer, template, decoder discovery
   +-- runtime: sessions, sampling, scheduler contracts, K/V policy, metrics
   +-- metal: model execution and device cache composition -> mirtal
   +-- cuda: native CUDA backend -> mircuda
```

Applications should depend on the `libmir` package and avoid direct dependencies
on its internal workspace crates.

## Backend features

Libmir enables no accelerator by default. This keeps model discovery, protocol,
scheduler, sampling, and K/V policy usable without linking a platform runtime.

```toml
[dependencies]
libmir = { version = "0.2.0", features = ["metal"] }
```

Select CUDA instead with `default-features = false, features = ["cuda"]`. Both
features may be enabled by tooling that intentionally validates both public
boundaries. The high-level facade is backend-neutral; platform features only
select which native implementation is compiled and linked.

## Requirements

The current Metal backend requires:

- Apple Silicon and macOS;
- full Xcode selected through `xcode-select`, with Clang C++20 and the macOS SDK;
- the separately downloaded Xcode Metal toolchain component;
- native MLX installed through Homebrew or selected with `MLX_PREFIX`;
- rustup and `nightly-2026-07-13` with Clippy and rustfmt;
- published `mirtal` or `mircuda` crates when building an accelerator backend.

The CUDA feature instead requires Linux, an NVIDIA driver exposing `libcuda.so`,
CUDA Toolkit 13.x with NVRTC, `nvcc`, and CCCL/CUB headers, pinned CUTLASS 4.4.2 headers, and a
supported NVIDIA GPU. It does not require Xcode, MLX, Python, `PyTorch`, cuBLAS,
or cuBLASLt. Set `CUDA_HOME`, `PATH`, and `LD_LIBRARY_PATH` when the toolkit is
not already discoverable. CUTLASS defaults to
`~/.cache/mircuda/cutlass-v4.4.2` and can be selected with
`MIRCUDA_CUTLASS_DIR`:

```sh
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
make check-cuda
```

Prepare and verify the native toolchain before building libmir:

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcodebuild -downloadComponent MetalToolchain
xcrun --sdk macosx --find clang++
xcrun --sdk macosx --find metal
brew install mlx
```

Mirtal requires the native MLX header, dynamic library, and `mlx.metallib`. The
standalone Xcode command-line tools alone are insufficient because checked Metal
kernels are compiled as part of the Rust build. `Python`, `PyTorch`, `Transformers`,
model checkpoints, `.env`, and `HF_TOKEN` are not compilation requirements.

## Dependency

From crates.io:

```toml
[dependencies]
libmir = { version = "0.2.0", features = ["metal"] }
```

## Quick Start

Set `MODEL` to a local Hugging Face-format checkpoint and run the complete
generation lifecycle with either the `metal` or `cuda` feature enabled:

```rust,ignore
use std::io::{self, Write};

use libmir::{
    CancellationToken, ChatCompletionRequest, ChatMessage, Error,
    GenerationOverrides, Library, RuntimeConfig,
};

fn main() -> libmir::Result<()> {
    let path = std::env::var_os("MODEL").ok_or(Error::MissingEnvironment("MODEL"))?;
    let library = Library::new(RuntimeConfig::default());
    let model = library.load(path, GenerationOverrides::default(), &mut |_| {})?;
    let request = ChatCompletionRequest {
        model: model.handle().id.clone(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Hello".into(),
            reasoning_content: None,
        }],
        stream: true,
        max_tokens: Some(256),
        temperature: Some(0.0),
        top_p: Some(1.0),
        top_k: Some(0),
        repetition_penalty: Some(1.0),
        seed: None,
    };
    let cancellation = CancellationToken::new();
    let output = model.generate_cancellable(&request, &mut |_| {}, &mut |token| {
        print!("{}", token.text);
        drop(io::stdout().flush());
    }, &cancellation)?;
    println!("\nfinish_reason={}", output.finish_reason);
    Ok(())
}
```

`Model::generate` renders the checkpoint chat template, tokenizes, creates an
independent session, performs prefill, selects tokens using the requested
sampling policy, decodes incrementally, and stops at a checkpoint stop token or
the resolved token limit. Streamed `GenerationToken` values carry a semantic
`Content` or `Reasoning` channel. Model-native think markers and named channel
protocols are removed from the text contract; `GenerationOutput` exposes final
`text` and `reasoning` separately. `CancellationToken` is cooperative and is
checked before prefill, after prefill, and between decode steps;
`Model::generate` remains the convenience API for an uncancelled request.

## Configuration

`RuntimeConfig` is an explicit data object. Libmir provides `Default` but does
not inspect process arguments, environment variables, dotenv files, or other
ambient configuration sources. An embedding application owns that policy and
mutates the public configuration fields before constructing `Library`:

```rust,ignore
let mut config = libmir::RuntimeConfig::default();
config.kv_cache.block_size = 32;
config.kv_cache.block_count = 2_048;
config.metal.cache.prefix_cache_entries = 32;
let library = libmir::Library::new(config);
# drop(library);
```

The Mirmir application maps CLI and environment values into this structure in
its own `config` crate. Other consumers can use their configuration framework
without inheriting Mirmir variable names or dotenv behavior.

Backend-specific fields exist only with their feature. For example,
`RuntimeConfig::metal` is available with `metal` and absent from CUDA-only and
backend-neutral builds. Likewise, `RuntimeConfig::cuda` contains the explicit
device ordinal, memory-pool release threshold, NVRTC include paths, and optional
persistent PTX cache directory when the `cuda` feature is enabled. Libmir never
chooses that directory from ambient environment state.

The native CUDA adapter can also be initialized directly. Construction is
fallible because it creates real device resources:

```rust,ignore
let config = libmir::CudaConfig::default();
let backend = libmir::CudaBackend::new(config)?;
let device = backend.device_info();
println!("CUDA device: {}", device.name);
# Ok::<(), libmir::CudaError>(())
```

`CudaConfig::planning` controls numerical and stability admission explicitly.
Backend construction derives a typed `CudaHardwareProfile` from the selected
device, including whether host and device use a unified or discrete memory
architecture, and creates one immutable `CudaExecutionPlanner`. Consumers can
inspect the planner or request plans directly; model loading uses the same path
when it prepares operations:

```rust,ignore
use libmir::{
    CudaConfig, CudaKernelAdmission, CudaNumericalPolicy, CudaPlanningPolicy,
    DensePlanRequest, DenseRole, ExecutionPhase,
};

let mut config = CudaConfig::default();
config.planning = CudaPlanningPolicy {
    numerical: CudaNumericalPolicy::Validated,
    admission: CudaKernelAdmission::Stable,
    dense_vectors: libmir::CudaDenseVectorPolicy::Disabled,
    moe_fusion: libmir::CudaMoeFusionPolicy::Disabled,
};
let backend = libmir::CudaBackend::new(config)?;
let plan = backend.execution_planner().plan_dense(DensePlanRequest {
    phase: ExecutionPhase::Decode,
    role: DenseRole::OutputHead,
    tokens: 1,
    input_features: 2_816,
    output_features: 262_144,
})?;
println!("selected {:?} from {:?}", plan.execution(), plan.source());
# Ok::<(), libmir::CudaError>(())
```

Selection keys contain hardware, phase, semantic operation role, quantization,
and tensor geometry. They never contain checkpoint family names. Every decision
emits a structured `libmir::cuda::planning` trace event, and selection happens
while preparing a model rather than in the token hot path.

The validated, stable decode plan uses split indexed routed-MoE execution. On
SM12, an explicit `Throughput` plus `Experimental` policy admits a partially
fused activation and FC2-quantization candidate. Checkpoint A/B tests require
exact output equality, and the planner will not select that candidate by
default until tuning data demonstrates a benefit for the current hardware and
memory architecture.

CUDA checkpoint loading uses `TensorUploadBatch`. Tensor ranges discovered by
`models` are read directly into pinned host allocations and enqueued to the
device without an intermediate decoded-value vector. `finish()` performs one
batch synchronization and returns a `CudaTensorSet`. BF16 dense projections use
cached `Bf16Linear` plans with checkpoint weight layout `[out, in]`. The one-row
output head uses an explicit `Bf16VectorLinear`; internal attention and
feed-forward projections retain Tensor Core GEMM until each role passes an
independent numerical gate.
`AffineQuantizedBf16Linear` consumes packed `U32` weights plus BF16
`scales`/`biases` directly. It supports 2D matrices and 3D expert banks, selecting
an expert by index without copying or dequantizing the complete weight matrix.
`AffineQuantizedBf16Qmm` provides the corresponding prefill path. Short chunks
use a scalar tiled kernel, while larger chunks use BF16 WMMA with FP32
accumulation and reuse each dequantized weight tile across 64 tokens. Neither
path materializes a full dequantized matrix.
`SelectedAffinePairBf16Linear` consumes device-resident router indices and
executes compatible gate/up expert banks in one launch.
`SelectedAffineGatedBf16Linear` additionally applies the activation selected
from checkpoint metadata through `GatedActivation`, preserving BF16 projection
rounding. `SelectedAffineReduceBf16Linear` performs all selected down
projections and their router-weighted reduction in one launch. Expert indices,
weights, intermediates, and outputs remain device-resident throughout this
two-launch expert MLP path.
`NvFp4Bf16Linear` consumes `ModelOpt` NVFP4 checkpoint tensors. It repacks E4M3
weight scales once, keeps packed E2M1 weights on the device, quantizes each BF16
activation block directly on CUDA, and dispatches native W4A4 Tensor Cores
through a persistent raw CUTLASS block-scaled FP4 plan. Global activation and
weight scales are read only while preparing the operation; execution has no
host synchronization or allocation. A checkpoint test compares the native
projection with an independent W4A16 reference on
`nvidia/Gemma-4-26B-A4B-NVFP4`.

`PagedKvBf16` implements the backend-neutral `KvBackendStorage` contract over a
single persistent CUDA page arena. Runtime-owned `BlockId` values select
physical pages; prefix ownership and session scheduling remain in `runtime`.
`PagedAttentionBf16` caches its device block table, updates it only when the
logical mapping changes, and applies decode GQA with online softmax directly
over fragmented pages. On tuned SM12 hardware, long contexts use 256-token
split-KV partitions with FP32 partial softmax state and a stable merge. Direct,
split, and merge kernels coexist in one graph and select work through device
predicates, without a host synchronization or threshold recapture.
`DecodeAttentionBf16` packs differently sized Q/K/V
weights once, issues one dense projection, and fuses Q/K normalization, V
normalization, and Q/K `RoPE` into one postprocess kernel. Paged writes,
sliding-window attention, and output projection remain allocation-free and
device-resident during execution.
`DecodeAttentionBf16::capture` transfers the prepared layer into a reusable
`CapturedDecodeAttentionBf16`. The graph owns its scratch and cache resources,
retains shared handles to fixed input/output allocations without copying device
memory, and rebinds exact capture-time QKV postprocess, K/V-store, and
paged-attention nodes for each token. Physical block mappings are explicit graph
invariants; acquiring or remapping a page returns a recapture-required error
instead of silently synchronizing or rebuilding the graph.
`RouterBf16` keeps the checkpoint-defined normalization, tensor-core BF16
projection, warp top-k softmax, and per-expert scaling in one prepared device
pipeline while retaining indices and weights on the device. `NvFp4ExpertBank`
uploads separate safetensors
experts directly into contiguous storage, and `SelectedNvFp4MoeBf16` executes
fused gate/up plus weighted down reduction without host routing.
`GroupedNvFp4MoeBf16` is the decode-oriented NVFP4 path. It uses router indices
directly in persistent indexed grouped CUTLASS W4A4 plans with one row per
selected expert. Decode gate/up quantization reads the shared BF16 activation and
computes its amax once while retaining independent checkpoint input scales and
packed outputs. `BucketedNvFp4MoeBf16` is the prefill path: one device kernel
builds expert counts, compact offsets, and bidirectional assignment maps, then
variable-row CUTLASS plans execute gate, up, and down with one group per expert.
Expert-bank scales are converted to CUTLASS layout once. Activation compaction,
paired gate/up quantization, gated activation, and router-weighted reduction stay
on the CUDA stream without host metadata reads or per-execution allocation.
Gate and up retain separate packed activations, weights, scales, and outputs but
share one paired variable-grouped CUTLASS scheduler; no expert bank is copied or
concatenated for this dispatch.
`DecodeMoeBlockBf16` combines these operations with the dense feed-forward
branch, all checkpoint norms, residuals, and the layer scalar. Compatible BF16
gate/up weights are packed once with a stream-ordered D2D transfer and shared by
all sessions; one dense plan emits adjacent gate/up rows consumed directly by a
packed gated-activation kernel.
`DecodeMoeBlockBf16::capture` produces a `CapturedDecodeMoeBlockBf16` whose
single graph replay includes attention, dense FFN, device routing, selected
W4A4 experts, residuals, and the output layer scalar. Consecutive tokens update
the same typed direct, split, and merge attention nodes used by the smaller
captured pipeline; all
other model operations retain fixed device addresses and plans.
`DecodeMoeBlockExecutor` is the prepared-layer lifecycle used while assembling
the model runner. The runner captures once during model warmup. Physical block
mappings update the fixed-capacity device table and typed graph arguments in
place, so switching sequences or allocating another page does not recapture the
model graph.
`DecodeMoeLayerTemplate` is the immutable model-owned boundary above those
executors. It retains one uploaded checkpoint weight set and expert-bank set,
then instantiates the model-owned plans, scratch, global K/V arena, and graph
without copying device memory for each request.
`CudaBackend::load_nvfp4_moe_layer_template` derives layer geometry from
`DecoderConfig`, resolves supported checkpoint prefixes, recognizes full-layer
`K=V`, uploads split NVFP4 experts, and returns that immutable template.
Standard partial and proportional `RoPE` are explicit geometry choices; the
latter keeps full-head pairing while rotating only the configured dimensions.
`CudaBackend::load_nvfp4_moe_model_template` adds BF16 token embeddings, every
discovered layer, final normalization, and the tied or independent output head.
`CudaMoeModelTemplate::instantiate` creates two fixed ping-pong activation
buffers and private layer executors. `CudaMoeModelSession::decode` keeps token
embedding, all layer replays, final normalization, and vocabulary logits on one
CUDA stream without per-token allocation or host readback. `sample` performs
greedy or bounded top-k/top-p selection into a stable device token buffer;
`decode_sampled` feeds that buffer directly into the next embedding lookup.
`prefill_from` uploads externally supplied prompt tokens through a reusable
pinned/device chunk, then executes embedding, projections, causal paged
attention, dense work, routing, and selected NVFP4 experts as a device-resident
batch. Prefill plans are cached lazily for each encountered chunk length and
reuse the decode executor's physical K/V pages. A small device kernel selects
only the final hidden row for final normalization and logits, so prefill never
materializes `[tokens, vocab]`. `CudaModelSessionConfig` makes chunk capacity
explicit.
Captured weights own shared GPU handles rather than references to a tensor set,
so a model or session does not become self-referential. `CudaTensorSet` and
`NvFp4ExpertBank` clones likewise share allocations and never duplicate or
re-upload checkpoint payloads.

CUDA source implementing these inference operations lives under
`cuda/kernels`; its typed plans live under `cuda/src/kernels`. Mircuda is used
only for source embedding, NVRTC compilation, typed symbol binding, streams,
memory, events, CUDA Graph mechanics, and raw CUTLASS dispatch. Model
mathematics and launch policy do not belong to the gateway crate.
`CudaConfig::nvrtc_include_paths` explicitly supplies CUDA toolkit or project
header roots. `CudaConfig::nvrtc_cache_directory` enables mircuda's
cross-process PTX cache. No library layer reads `CUDA_HOME`, cache-home
variables, or dotenv files implicitly.

Generation settings are resolved in this order:

1. explicit request or `GenerationOverrides` values;
2. `generation_config.json` values;
3. native defaults.

Model shape, attention, `RoPE`, `MoE`, normalization, and quantization come from
checkpoint configuration and required safetensors, not from CLI flags.

## Public API

### High-level facade

| API | Purpose |
|---|---|
| `Library` | Owns runtime configuration and the backend registry; loads models. |
| `ModelDescriptor` | Inspects configuration, tokenizer, tensors, and generation defaults without loading weights. |
| `Model` | Reusable loaded model handle; prepares prompts, creates sessions, and generates text. |
| `PreparedPrompt` | Rendered chat prompt plus native tokenization result. |
| `Session` | Independent K/V and recurrent state for manual prefill/decode. |
| `GenerationToken` | Incremental text tagged as semantic content or reasoning. |
| `GenerationOutput` | Separate final text and reasoning, token IDs, prompt count, metrics, and finish reason. |
| `ChatCompletionRequest` / `ChatMessage` | OpenAI-like request and message input types. |
| `GenerationOverrides` / `GenerationSettings` | Optional overrides and fully resolved generation policy. |
| `RuntimeConfig` | Explicit backend-neutral scheduler and K/V configuration. |
| `MetalConfig` | Explicit Metal batching, cache, fusion, and diagnostics policy. |
| `CudaBackend` / `CudaConfig` | Initialized native CUDA adapter and explicit device policy. |
| `CudaExecutionPlanner` / `CudaHardwareProfile` | Immutable model-level CUDA selector and model-independent device facts. |
| `CudaAttentionPolicy` / `AttentionPlanRequest` | Automatic or explicit direct/split-KV selection from hardware and attention geometry. |
| `CudaMemoryArchitecture` | Unified or discrete host/device memory topology used by CUDA planning. |
| `CudaPlanningPolicy` / `CudaNumericalPolicy` / `CudaKernelAdmission` | Explicit numerical and stability admission policy. |
| `DensePlanRequest` / `DensePlan` / `DenseRole` | Typed dense-operation selection key, decision, and semantic role. |
| `MoePlanRequest` / `MoePlan` / `MoeQuantization` | Typed routed-expert selection key, decision, and quantization layout. |
| `ExecutionPhase` / `PlanSource` | Decode/prefill phase and tuned/fallback decision provenance. |
| `CudaModelSessionConfig` | Explicit per-session CUDA prefill allocation policy. |
| `CudaTensorSet` / `TensorUploadBatch` | Device checkpoint storage and batched direct payload upload. |
| `Bf16Linear` | Cached allocation-free BF16 dense projection plan. |
| `Bf16VectorLinear` | Native bandwidth-oriented BF16 matrix-vector projection for an explicit one-row role. |
| `Bf16Projection` | Prepared matrix or vector implementation selected once by the CUDA planner. |
| `Bf16LinearPackWeights` | Model-owned row packing for differently sized BF16 projections executed by one plan. |
| `Bf16LinearPair` / `Bf16LinearPairWeights` | One-GEMM execution and model-owned packed weights for compatible BF16 projection pairs. |
| `Bf16Embedding` | Device-side BF16 token-row lookup without host materialization. |
| `DeviceSamplerBf16` | Greedy and bounded top-k/top-p BF16-logit sampling with a device-resident selected token. |
| `DeviceBatchSamplerBf16` | Independent per-row greedy or bounded sampling over packed logits without host readback. |
| `AffineQuantizedBf16Linear` | Allocation-free BF16-input affine Int4/Int8 decode projection. |
| `AffineQuantizedConfig` | Model-owned grouped affine quantization and matrix geometry. |
| `AffineQuantizedBf16Qmm` | Tiled scalar/WMMA Int4/Int8 prefill projection. |
| `NvFp4Config` / `NvFp4Tensors` | Geometry and `ModelOpt` tensor bundle for native NVIDIA FP4. |
| `NvFp4Bf16Linear` | Persistent native W4A4 Tensor Core projection with GPU activation quantization. |
| `RmsNormBf16` | Fixed-shape BF16 RMS normalization with FP32 reduction. |
| `RopeSpec` / `RopeBf16` | Standard half-split, optionally partial rotary embedding. |
| `PagedKvBf16` / `PagedAttentionBf16` | Runtime-owned physical BF16 pages and allocation-free decode GQA. |
| `BatchedPagedAttentionBf16` | One CUDA decode-attention launch over independent paged sequences. |
| `PagedDecodeBatch` | Shared device block tables and context lengths for one decode microbatch. |
| `DecodeAttentionConfig` / `DecodeAttentionWeights` | Config-derived geometry and checkpoint tensor bundle for decode attention. |
| `DecodeAttentionBf16` | Complete device-resident one-token attention pipeline. |
| `BatchedDecodeAttentionBf16` | Complete `RMSNorm`, QKV/RoPE, K/V store, attention, and output path for independent rows. |
| `CapturedDecodeAttentionBf16` | Reusable CUDA Graph attention pipeline with typed per-token rebinding. |
| `RouterSpec` / `RouterBf16` | Fused BF16 routed-expert projection, top-k softmax, and device selection. |
| `NvFp4ExpertBankConfig` / `NvFp4ExpertBank` | Contiguous direct-from-safetensors expert storage. |
| `SelectedNvFp4MoeBf16` | Device-selected NVFP4 gate/up and weighted down execution. |
| `GroupedNvFp4MoeBf16` | Device-indexed fixed-row CUTLASS W4A4 `MoE` for decode. |
| `BucketedNvFp4MoeBf16` | Device-bucketed variable-row CUTLASS W4A4 `MoE` for prefill. |
| `DecodeMoeBlockConfig` / `DecodeMoeBlockBf16` | Complete routed-MoE decode layer with residual policy. |
| `BatchedDecodeMoeBlockBf16` | Fixed-row grouped NVFP4 decode layer for one microbatch. |
| `BatchedDecodeMoeLayer` | Batched layer executor retaining immutable checkpoint weights. |
| `CapturedDecodeMoeBlockBf16` | Reusable complete routed-MoE CUDA Graph with typed K/V rebinding. |
| `PrefillAttentionBf16` / `PrefillMoeBlockBf16` | Fixed-chunk causal paged prefill sharing decode K/V state. |
| `DecodeMoeBlockExecutor` / `DecodeGraphAction` | Session-local direct, replay, and page-remap recapture lifecycle. |
| `DecodeMoeLayerTemplate` | Immutable uploaded layer weights and expert banks that instantiate no-copy session executors. |
| `NvFp4MoeLayerLoadConfig` | Explicit cache policy for config-discovered NVFP4 routed-MoE layer loading. |
| `CudaMoeModelTemplate` | Model-owned embedding, output tensors, and all immutable routed-MoE layer templates. |
| `CudaMoeModelSession` | Private activations, K/V state, chunked prompt input, graphs, logits, and a zero-readback sampled-token loop. |
| `CudaDecodeBatch` | Fixed-capacity full-model decode from embedding through every layer, final norm, logits, and per-row sampling. |
| `SelectedAffinePairBf16Linear` | One-launch gate/up projections for device-selected experts. |
| `GatedActivation` / `SelectedAffineGatedBf16Linear` | Metadata-driven GELU-tanh or `SiLU` selected-expert projection. |
| `SelectedAffineReduceBf16Linear` | Selected down projections with deterministic router-weighted reduction. |
| `CudaError` | Typed CUDA initialization and adapter error boundary. |
| `Error` / `Result` | Typed `thiserror` boundary for model and runtime failures. |

### Execution and diagnostics

| API | Purpose |
|---|---|
| `Engine` | Direct backend facade for advanced consumers. |
| `ProgressEvent`, `ProgressStage`, `ProgressUnit` | Weight-load, prefill, and per-token decode progress contract. |
| `BackendInfo` | Selected backend, device, and capability report. |
| `ModelHandle` | Stable loaded-model identity and backend name. |
| `PrefillOutput` / `DecodeOutput` | Incremental execution results. |
| `DecodeBatchRequest` / `DecodeBatchOutput` | Ordered, non-empty decode batch contract for one loaded model. |
| `SamplingLogits` | Selects device token, candidate, or full-logit output. |
| `CacheStats`, `KvCacheDType` | Cache state and storage-format controls. |
| `GenerationMetrics` | Serializable duration, throughput, token, and cache metrics. |

The public `foundation`, `models`, and `runtime` re-exports are currently an
advanced API used by Mirmir. They expose protocol responses, tokenizer and
layout inspection, scheduler contracts, sampler types, detailed K/V structures,
and trace payloads. They are less stable than the high-level facade and will be
grouped behind a deliberate advanced namespace before 1.0.

## Ownership and Concurrency

- `Library` and `Model` are cheap to clone; loaded model state is shared.
- Every `Model::session()` owns an independent logical block table and recurrent
  state. Physical K/V pages and execution plans are model-owned.
- CUDA prefill releases the shared runner after every bounded chunk. A weighted
  priority queue admits waiting decode work first, then guarantees one prefill
  chunk after `RuntimeConfig::scheduler.decode_priority_burst` decode quanta.
  FIFO order is preserved within both work classes. Concurrent decode callers
  enter a model-level admission window and are grouped into warmed binary
  buckets up to `max_batch_requests`. Scalar graph replay handles an unpaired
  final row. Scalar, prefill, and batch plans share the same physical page arena
  without copying K/V or model weights. Device-sampled scalar tokens retain
  their owning session; zero-copy scalar decode is used only while that owner
  remains current.
- Each loaded Metal model remains on a permanent owner thread because MLX stream
  registration is thread-local. Callers submit work through the backend instead
  of moving native handles between host threads.
- Concurrent Metal decode calls enter the shared admission window. Dense
  `SwiGLU`, routed hybrid-MoE, and hybrid linear-attention rows use one
  packed `[batch, 1, hidden]` graph for embedding, Q/K/V where applicable, MLP
  or routed experts, output projection, and sampling. Per-row positions, paged
  K/V, and full-attention cache state allow different context lengths in one
  batch. Hybrid linear attention packs every session's recurrent state into one
  compiled Gated Delta dispatch and commits independent state slices back on
  device. Host-logit sampling uses an ordered same-stream fallback.
- Token callbacks run synchronously on the generation caller. They should avoid
  blocking work or hand it to the application layer.
- Host reads and detailed logits are explicit slow paths. Prefer device sampling
  when the requested policy permits it.

## Errors

All public operations return `libmir::Result<T>`. `Error` uses `thiserror` and
converts model-discovery and runtime failures through `From`, so consumers can
use `?` without stringly typed remapping.

## Examples

- `cargo run --example inspect -- "$MODEL"` inspects a checkpoint without
  loading weights.
- `cargo run --example generate -- "$MODEL"` loads a model and streams text.
- `cargo run --example incremental -- "$MODEL"` drives `Session` prefill and
  decode manually.
- `cargo run --release --features cuda --example serve-throughput -- "$MODEL" 8 256 32`
  measures aggregate prefill and decode throughput for synchronized independent
  sessions without HTTP or model-load time.

## Documentation

Generate API documentation for the complete workspace:

```sh
make docs
make docs-open
```

The same commands expand to nightly `cargo doc --workspace --no-deps` with
rustdoc warnings denied. Crate-level documentation embeds this README, while
rustdoc lists every facade item and all advanced re-exported modules.

See [docs/roadmap.md](docs/roadmap.md) for the remaining public API, runtime,
cache, backend, model, and observability work.

## Validation

```sh
make examples
cargo clippy -p libmir --all-targets --no-default-features -- -D warnings
cargo clippy -p libmir --all-targets --no-default-features --features cuda -- -D warnings
cargo clippy -p libmir --all-targets --no-default-features --features metal -- -D warnings
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```
