# Libmir Agent Notes

## Project Goal

Libmir is a native Rust LLM inference library for local systems. It runs models without
PyTorch as a runtime dependency. PyTorch may be used only as an external
reference during validation, never as part of the product architecture.

The target shape is:

- Native Rust core runtime.
- Low-level Metal and CUDA backends.
- Efficient KV cache, prefix cache, scheduling, sampling, and model loading.

## Hard Rules

- No source file may exceed 250 LOC.
- Split modules only as `module/mod.rs` plus focused files such as
  `module/feature.rs`. Never flatten a child into a sibling such as
  `module_feature.rs`.
- Apply the same nesting to tests: use `module/tests.rs` or
  `module/tests/feature.rs`, never sibling files such as `module_tests.rs` or
  `feature_tests.rs` when they test an existing module.
- Avoid families of similarly named files in one directory.
- Use `thiserror` for typed errors and error mapping.
- Prefer `From`/`Into` error conversion with `?`; use `map_err` only as a
  last resort at unavoidable foreign/error-shape boundaries.
- Keep `anyhow` out of library APIs. It may be used only in binaries if needed.
- Keep server, CLI, and web layers out of the runtime core.
- Keep typed runtime configuration in `libmir::RuntimeConfig`. Environment,
  dotenv, CLI, and file parsing belong to consuming applications.
- Keep accelerator dependencies optional. The `metal` feature is the only path
  from the facade to `metal`/`mirtal`; the `cuda` feature is the only path to
  `cuda`/`mircuda`; the default feature set links neither backend.
- Do not add PyTorch, Transformers, or Python runtime dependencies.
- Prefer explicit model/backend manifests over dynamic Python-style model code.
- Keep unsafe code isolated behind small backend modules with clear invariants.
- Use the repository `rust-toolchain.toml`; nightly `rustfmt` and `clippy` are
  the source of truth.
- Run `cargo fmt --all -- --check` and validate the neutral, `metal`, and `cuda`
  feature sets before treating facade changes as complete. A full workspace
  `--all-features` check requires a host capable of building every backend.
- Every crate must keep `[lints] workspace = true` unless there is a documented
  reason to opt out.
- Imports are grouped and ordered by `rustfmt.toml`; do not hand-format `use`
  groups against formatter output.
- Keep domain states, stages, units, outcomes, and error codes typed throughout
  runtime and public contracts. Convert them to stable strings only at protocol,
  serialization, persistence, logging, or UI boundaries. Never infer state from
  display strings or duplicate enum-to-string mappings across consumers.

## Sliding KV Cache Invariants

- Sliding-attention layers must retain bulk prefill. Never reintroduce a
  one-token scheduler fallback after the configured attention window fills.
- A bulk write may temporarily retain a chronological
  `window - 1 + chunk_length` K/V view so every query in the chunk sees its
  correct causal window. The next scalar decode write compacts it to the
  bounded rotating ring.
- Build the bulk sliding causal mask from device arrays and pass it directly to
  SDPA. Cache updates and mask construction must not read tensor data back to
  Rust or synchronize the Metal stream.
- Preserve the explicit rotating write index across updates and snapshots;
  physical storage order must never be mistaken for temporal attention order.
- Paged-attention dependency arrays are rank-one `[1]` buffers. A scalar `[]`
  changes the generated Metal argument into a value and makes kernels that
  index the dependency fail to compile.

## Workspace Layout

- `core`: shared types, errors, model manifests, protocol structs. The Rust
  library crate is named `foundation` because `core` shadows Rust's standard
  `core` crate and breaks procedural macro expansions.
- `models`: local model layout inspection, tokenizer, chat prompt rendering.
- `runtime`: scheduler, backend traits, KV cache and session state.
- `libmir`: public model loading, prompt preparation, session, and generation API
  used by product binaries and embedders.
- `cuda`: CUDA backend adapter placeholder.
- `metal`: Apple Metal backend implemented through the public `mirtal` crate.

## Development Order

1. Lock down API contracts and error taxonomy.
2. Implement tokenizer/model manifest loading.
3. Implement KV block allocator, prefix hash, and continuous batching controls.
4. Stabilize the Metal backend for Apple Silicon.
5. Implement CUDA backend for one decoder-only family.
6. Add CPU dummy backend only for tests, not product direction.

## Style

- Domain code should be boring, explicit, and testable.
- Keep public structs small and serializable where they cross process/API
  boundaries.
- Avoid clever macros in core runtime code.
- Add comments only where invariants or hardware assumptions matter.

## Metal Runtime Configuration

- Libmir must never read runtime configuration from process environment or
  dotenv files. The `MIRMIR_*` names below are mappings owned by the Mirmir
  application; embedders set the corresponding `RuntimeConfig::metal` fields
  directly.

- Libmir contains no C/C++, CXX bridge, `build.rs`, direct `cc` dependency, or
  vendored Metal C++ headers. Native MLX binding code belongs exclusively to
  `mirtal`. `MLX_PREFIX` selects the MLX installation and defaults to
  `/opt/homebrew/opt/mlx`.
- Native MLX operations must receive an explicit stream handle. Do not add
  implicit global, device-default, or task-local stream fallbacks.

- `MIRMIR_METAL_DEVICE_TOKEN_PIPELINE=0` disables the default greedy device-token
  pipeline. The pipeline keeps the next token on MLX for the following
  decode step and is the preferred fast path when top-p and repetition penalty
  are inactive.
- Metal decode materializes the sampled token before building the following
  forward graph, then detaches the now-evaluated K/V roots without a second
  stream synchronization. Session release synchronizes before pages are
  recycled.
- Metal prefill scheduling is architecture-aware: routed models use physical
  waves of at most two rows and finish queued waves before streaming, while
  dense models retain completion-first interleaving. Keep this as a backend
  capability in the shared scheduler profile, not a model-name special case.
- Decode cohort admission uses exactly `decode_batch_wait_us`. Never multiply
  that window after a multi-row step or retain a hidden refill horizon; a
  latency/occupancy tradeoff must be explicit or backend-measured.
- The native MLX backend keeps an independent device-resident page-backed K/V
  state for each session, while dispatch remains serialized on one explicit GPU
  stream. Its longest-token-prefix LRU snapshots K/V and prompt logits as MLX
  array handles, so a hit does not copy tensors to Rust or recompute its cached
  prefix.
- `MIRMIR_METAL_PREFIX_CACHE_ENTRIES=<usize>` bounds device prefix snapshots;
  the default is `16`, while `0` disables the cache. Entries are keyed by model
  identity and the full token prefix, never by recyclable runtime block IDs.
- `MIRMIR_METAL_KV_RESERVE_TOKENS=<positive integer>` sets the minimum physical
  page arena reserved before prefill. The default is one 256-token allocation
  step; prompts larger than that reserve their complete token count. This keeps
  prefill growth out of the graph without changing `MIRMIR_METAL_PREFILL_STEP`.
- Physical paging is mandatory for every unbounded full-attention cache. The
  Rust cache owns persistent device K/V pages
  and a device-resident page table. K/V writes are lazy multi-output MLX graph
  primitives: their K/V page buffers are graph outputs, so the MLX page-backed
  view consumes them directly. Append and attention remain ordered on one
  explicit Metal stream without a scalar dependency or Rust-side synchronization.
  Prefix snapshots share physical pages through reference counts. Diverging
  sessions copy only a shared page, while an arena grows lazily only after its
  reserved physical pages are exhausted. Sliding layers retain their bounded
  ring and recurrent linear-attention layers retain their state.
- `MIRMIR_METAL_PAGED_ATTENTION_MIN_CONTEXT=<positive integer>` selects the
  physical-page activation threshold; the default is `128`. Below it Mirmir
  uses a temporary contiguous MLX cache before promoting it to physical pages.
  On the first update at the threshold, the Rust cache backfills head-major pages
  from live device K/V and releases the persistent contiguous cache. Identity
  page tables expose a zero-copy device view to MLX fast SDPA. Quantized K/V is
  not production-ready.
- Native paged SDPA is selected automatically from 8,192 cached tokens only for
  benchmarked shapes: head dimensions up to 256 and divisible by 32, with a GQA
  group factor from 5 through 32. It uses paged partial softmax plus a Metal
  reduction pass and preserves the long Qwen3.6 greedy digest. Other shapes,
  including Gemma4's 512-wide global heads, retain MLX SDPA over the same
  page-backed zero-copy view.
- Each native paged-attention cache owns persistent device partial, sum and
  maximum buffers plus cached dispatch/output specifications. Aliasing writes
  chain through the previous reduction output, so no scratch allocation or
  unordered overwrite occurs per decode token. Prefix snapshots always receive
  an independent scratch workspace even when they share immutable K/V pages.
- Keep paged-attention reduction on ordinary `MetalKernel::dispatch`. A typed
  prepared launch reduced host construction but regressed synchronized and
  full-model decode; do not restore it without a new controlled benchmark.
- Each physical page store lazily owns one typed prepared aliasing page-write
  plan. Decode updates its fixed constant buffer and launch geometry in place;
  do not recreate the pipeline, native input container, alias, stride or
  constant vectors per layer and token. Snapshots get an independent plan
  alongside their independent attention scratch.
- `MIRMIR_METAL_NATIVE_PAGED_SDPA=1` forces the native paged reader for compatible
  dimensions at any context. This is a diagnostic override; never broaden
  automatic selection without an alternating kernel benchmark and a real-model
  long-context digest regression.
- `MIRMIR_METAL_PREFILL_STEP=<positive integer>` limits the number of prompt
  tokens in a causal MLX prefill graph. The default is 2048 for hybrid linear
  MoE and 512 for other decoder archetypes; set it to `1` only when comparing
  against scalar-prefill behavior.
- `MIRMIR_METAL_PREFILL_EVAL_LAYERS=<positive integer>` materializes a causal
  prefill graph after each specified number of layers, without moving tensors
  out of MLX. It is an experimental graph-size control for benchmark tuning;
  unset means one graph for the full prefill chunk.
- `MIRMIR_METAL_PROFILE_LAYERS=1` synchronizes after every hybrid routed-MoE layer and emits
  `tracing` timings. It is for diagnosis only and must stay disabled in normal
  generation.
- `MIRMIR_METAL_PROFILE_COMPONENTS=1` synchronizes attention/KV and
  dense-plus-MoE feed-forward separately in every hybrid routed-MoE layer. It is solely a
  diagnostic profile and must stay disabled in normal generation.
- `MIRMIR_METAL_GRAPH_DUMP=<path>` exports the first lazy decode graph as DOT
  before evaluation for every decoder archetype. It is a one-shot diagnostic
  tool for graph boundaries and must stay unset in ordinary generation.
- `MIRMIR_METAL_FUSED_EXPERT_GATE_UP` controls hybrid routed-MoE expert gate/up fusion into
  one rank-three `gather_qmm` only for the direct `q_len=1` decoder path.
  Causal prefill continues to use sorted expert dispatch. Unset or `auto` is
  the default: Mirmir materializes ordinary fusions, samples the MLX Metal
  allocator, estimates the additional fused-expert arrays without evaluating
  them, and enables fusion only when `active + additional + reserve` fits the
  operating system's recommended working set. The reserve is the larger of
  2 GiB and 10% of that set. `=1` forces compatible fusion; `=0` disables it.
  The 26B reference-checkpoint fused path preserves the 128-token greedy sequence and
  improved a controlled benchmark from `62.55` to `64.20 tok/s`, while raising
  active Metal memory from `18,660,602,036` to `28,747,378,868` bytes. Any
  enabled expert fusion is reported directly by the executing Rust path.
- `MIRMIR_METAL_FUSED_SHARED_EXPERT_GATE_UP` applies the same auto/`=1`/`=0`
  policy to shared-expert routed SwiGLU models, including the native Qwen3.6
  hybrid linear stack. Routed expert fusion follows the automatic memory policy.
  `MIRMIR_METAL_FUSED_SHARED_DENSE_GATE_UP=1` additionally fuses the ordinary
  shared expert gate/up pair; it remains opt-in because its full-model benchmark
  did not show a sustained decode gain. Fused QMMs duplicate quantized arrays.
- The head-wide Qwen3.6 single-token recurrence kernel is enabled by default;
  `MIRMIR_METAL_FUSED_GATED_DELTA_DECODE=0` selects the general kernel for
  diagnosis. One 256-thread group handles a complete value head through eight
  SIMD groups. Its first thread computes the FP32 decay/update values directly
  from the projected gates, removing a dispatch and two temporary tensors. A
  two-step low-level regression requires identical output and complete FP32
  state, and the full 96/64 greedy digest matches the general path. The warm
  15-sample comparison measured `97.04 tok/s` versus `94.33 tok/s`; the decode
  graph contracts from 2,669 to 2,639 nodes and from 60 to 30 custom kernels.
- Keep the dedicated FP32 Gated Delta gate-preprocess Metal kernel for prefill
  and the diagnostic general recurrence. Single-token fused decode computes
  the same values inside its recurrence kernel. Reproducing `mlx_lm`
  preprocessing through shapeless
  `mx::compile` changed the expected digest and regressed decode to `34.25
  tok/s`; direct BF16 and register-emulated BF16 Metal variants produced the
  same reference digest but reached only `61.28` and `56.59 tok/s`. None of
  those diagnostic variants remain in runtime code.
- Gated Delta output normalization uses the same shapeless compiled precise
  SwiGLU graph as `mlx_lm`: gate and normalized values are promoted to FP32,
  SiLU and multiplication execute together, and only the result is cast back.
  This default path preserves the controlled digest while reducing the Qwen3.6
  decode graph from 2,819 to 2,669 nodes. Do not replace its three lazy Q/K/V
  slices with MLX `split`: that exact experiment regressed 96/64 decode from
  `97.28` to `79.77 tok/s`.
- Gated Delta Q/K normalization passes its fixed `1/sqrt(d)` and `1/d` scales
  as stride-zero one-dimensional weights directly to MLX RMSNorm. This removes
  60 post-normalization multiplies from the Qwen3.6 decode graph (`2,639` to
  `2,579` nodes), preserves the full greedy digest, and measured `90.02 tok/s`
  versus `89.49 tok/s` in the same thermally stabilized 15-sample series.
- Single-token Gated Delta fuses the exact MLX Q/K RMSNorm arithmetic into the
  recurrent Metal kernel. Preserve `metal::precise::rsqrt` and the two-stage
  input-dtype cast before applying the norm weight; changing either causes
  recurrent drift. The fusion removes 60 RMSNorm dispatches and 60 broadcasts
  from Qwen3.6 decode (`2,419` to `2,299` graph nodes), preserves the 128-token
  digest and measured `91.91`/`92.02 tok/s` versus `90.38`/`90.34 tok/s` in
  alternating runs. `MIRMIR_METAL_FUSED_GATED_DELTA_NORMALIZATION=0` is the
  diagnostic reference.
- Shared-expert MoE routing uses unscaled top-k when the model has no
  per-expert correction scale. Do not materialize an all-ones FP32 scale: its
  cast, gather, squeeze and multiply added 160 nodes across 40 Qwen3.6 layers.
  Removing them reduced decode from 2,579 to 2,419 nodes, preserved the full
  digest and measured `90.15 tok/s` versus `89.45 tok/s` in a matched run.
- The equivalent Homebrew `mlx_lm` Qwen3.6 single-token graph has 2,670 nodes
  versus Mirmir's 2,419. Both contain 391 ordinary quantized matmuls, but
  `mlx_lm` retains 120 routed `GatherQMM` operations while Mirmir's fused
  routed gate/up path uses 80. Disabling that fusion regressed a matched run
  from `98.70` to `96.22 tok/s`; keep the automatic memory-gated fusion.
- Do not wrap the shared-expert output sigmoid/multiply in `mx::compile`: it
  reduced the graph by 80 nodes but regressed decode from `98.45` to `90.03
  tok/s` and prefill from `665.96` to `612.98 tok/s`. A one-call native
  SharedExpertMoe ABI preserved the exact 2,419-node graph but measured only
  `98.94` versus `98.84`/`98.89 tok/s`; the duplicate implementation was
  removed. Resolving a thread-local stream once during loading likewise had no
  repeatable gain (`90.80`, then `90.21` versus a `90.21` baseline).
- `MIRMIR_METAL_FUSED_ATTENTION=1` and
  `MIRMIR_METAL_FUSED_DENSE_GATE_UP=1` opt dense SwiGLU decoders without affine
  projection biases into concatenated decode projections. They remain opt-in
  because Qwen3-8B showed no sustained throughput gain while the fused weights
  increase Metal memory. Bias-bearing checkpoints such as Bielik are rejected
  from this fusion automatically.
- Model trace records Metal wired and recommended limits plus active, cached,
  and peak MLX allocator bytes after model load. It also records the expert
  fusion policy decision. Use these values, rather than process RSS, when
  judging a Metal optimization's memory cost.
- The compatible hybrid routed-MoE affine router (`group_size=64`, `bits=8`,
  `top_k=8`) is a Rust graph compiled into a mirtal-owned typed handle. Native
  hybrid-MoE composition borrows that same cache. Set
  `MIRMIR_METAL_NATIVE_ROUTER=0` only to compare it with direct mirtal RMSNorm,
  QMM, and device top-k composition.
- Fused hybrid routed-MoE `q_len=1` projections are enabled by default: dense gate/up and
  compatible Q/K/V or K/V attention weights are merged and prewarmed. Set
  `MIRMIR_METAL_FUSED_DENSE_GATE_UP=0` or `MIRMIR_METAL_FUSED_ATTENTION=0` only
  for a lower-memory baseline; incompatible quantization is skipped safely.
- The native MLX loader raises Metal's wired allocation limit to the operating
  system's recommended working set before constructing the GPU model. This is
  process-wide and deliberately matches `mlx_lm` generation behavior.
- Metal execution policy is cloned into each `Library`, backend, and explicit
  stream. It is stable for that loaded model and may differ between independent
  library instances without process-global state.
- Device-side sampling must call `async_eval` on the selected token before Rust
  constructs and submits the following decode graph. This command boundary is
  what overlaps MLX execution with host graph construction; moving it after the
  model call serializes decode even though the token remains on the device.
- Paged KV allocation is always available. Native paged SDPA is a shape-based
  execution choice, not the definition of paged caching: identity page tables
  use MLX SDPA over a zero-copy page view when that kernel is faster. Supported
  fragmented/COW page tables must use native paged SDPA instead of gathering a
  contiguous K/V tensor. Keep this policy model-agnostic in `engine/kv/policy.rs`.
- `mirmir chat --bench` measures the production `libmir::Engine` chat prefill and
  decode path, not `LoadedModel` directly. It keeps model weights and Metal
  kernels warm but clears the device prefix cache before each run, preventing
  an exact-prefix hit from inflating prefill throughput. Defaults are one
  warmup and three measured samples; override them with `--bench-warmup` and
  `--bench-samples` (or their `MIRMIR_BENCH_*` environment variables). The
  report names the sampling path. Full-logit sampling is expected to be slower
  because it copies vocabulary logits to Rust every decode step.
## Mirtal Boundary

- The model-agnostic execution layer is provided by the public `mirtal` crate.
  Depend on its versioned crates.io release. Never add `mirtal-sys` as a direct
  dependency.
- New reusable MLX arrays, streams, graph operations, compiled graphs, memory
  controls, and Metal launch mechanics belong in `mirtal`. Model assembly,
  checkpoint interpretation, cache policy, and architecture selection remain
  in Mirmir.
- Declare Mirmir-specific fast kernels in Rust with `mirtal::metal_kernel!` and
  keep their MSL under `metal/kernels`. Complete direct-Metal libraries use
  `metal_library!`, then expose named functions with `MetalLibrary::export`.
  Kernels that mutate an existing MLX allocation use the generic
  `AliasingDispatch` contract; do not add a model- or cache-specific CXX API.
- `mirtal::Array` is the canonical array owner and `mirtal::Stream` is the only
  MLX stream. Mirmir wrappers contain safe Rust owners only; never reintroduce
  native pointer compatibility fields or direct `mirtal-sys` access.
- Mirmir owns model assembly, page allocation/refcounts, copy-on-write snapshot
  policy and cache promotion. Mirtal owns generic graph operations, compiled
  graphs, checked MSL import/export and execution mechanics.
- Metal exact-prefix hits may retain an unaligned terminal page and cached
  logits. A continuation from an unaligned terminal or checkpoint must restore
  only through the last complete K/V page and replay the partial page; never
  attach shared-arena continuation to page copy-on-write through MLX
  `slice_update`.
- A non-quantized Metal K/V view uses a slice for one increasing physical page
  run and one device `take` for multiple runs. Never rebuild fragmented shared
  arenas as per-run slices plus concatenation inside every attention layer.
- Metal packed-attention tuning retains a bounded causal runtime-discovery
  budget after startup. An unseen multi-token batch/context shape must compare
  row-wise and batched MLX SDPA once, persist the shape-keyed decision, and
  never fall back permanently merely because readiness did not cover it.
- An automatically resolved K/V block count is a shared capacity ceiling, not
  a per-session Metal residency request. Never copy it into
  `kv_reserve_tokens`; shared paged arenas grow from actual prompt demand.
- Affine routed Metal MLPs tune BF16-cast and native GatherQMM-output execution
  as complete shape/format candidates. Ordinary QMM retains its model dtype,
  and neither choice may be selected by a model or device-name rule.
- Affine quantization, dequantization, QMM/GatherQMM and generic graph operations
  are mirtal primitives. Sampling, routing, expert dispatch, embedding lookup,
  and RoPE scaling policy are Mirmir Rust compositions that must remain lazy and
  device-resident.
- Quantized embedding must gather packed weight/scale/bias rows before calling
  mirtal dequantize; never materialize a full vocabulary table.
- RoPE rotation is a mirtal primitive. Proportional and piecewise frequency
  construction belong to Mirmir because they interpret model configuration.
- Ordinary SDPA execution is owned by mirtal. Mirmir owns Q/K/V layout, model
  scale and mask selection, KV updates, physical paging, and the paged-attention
  policy; it must not call MLX fast SDPA directly. Pass learned tensor masks and
  per-head attention sinks as borrowed mirtal arrays without host conversion.
- Expert sorting, inverse restoration, and weighted reduction are owned by
  Mirmir and composed from mirtal operations without host reads.
- GeGLU, SwiGLU, precise FP32 SwiGLU, logit softcap, and the compatible affine
  router are Rust graph functions cached by mirtal at stream construction.
  Native hybrid-MoE and Gated Delta code borrow the same caches via non-owning
  handles that must not outlive `engine::Stream`; never duplicate or rebuild these
  graphs in decode.
