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
- Typed models must make invalid states unrepresentable. Do not replace related
  strings with parallel enums, booleans, and optional fields that can contradict
  one another. Wrapper enums must add domain semantics, not merely rename values.

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
  `mirtal`. `MLX_PREFIX` selects the MLX installation. Workmir configures
  `.native/mlx` using `sh scripts/mlx/setup.sh`; standalone builds fall back to
  `/opt/homebrew/opt/mlx` when no prefix is configured.
- Native MLX operations must receive an explicit stream handle. Do not add
  implicit global, device-default, or task-local stream fallbacks.
- Cohort-preserving prefill budget remainders remain a test-only candidate.
  The September 7 ABBA failed the performance gate despite passing semantic
  Qwen and synthetic clamped tests. Do not enable it by default from the first
  favorable pair; see Workmir's `2026-09-06-prefill-cohort-budget` report.
  Any candidate must retain scalar progress when the whole budget is tiny.
- Use the latest stable MLX as the tuning baseline (currently 0.32.2, pinned
  by Workmir). Cross-version greedy differences alone do not establish a quality
  regression or justify reverting the baseline. Validate semantic correctness
  on the new version and compare tuning candidates within that same version.
  The historical 0.32.0 C5/8k speedup was invalidated by host memory pressure;
  never reuse it as performance evidence. Version smoke tests must not hard-code
  the Homebrew minor version; benchmark metadata must record actual linked MLX.
  Reuse archived MLX-LM results instead of rerunning an unchanged reference.

- `MIRMIR_METAL_DEVICE_TOKEN_PIPELINE=0` disables the default greedy device-token
  pipeline. The pipeline keeps the next token on MLX for the following
  decode step and is the preferred fast path when top-p and repetition penalty
  are inactive.
- Metal decode schedules the sampled token before building the following
  forward graph, then evaluates that graph's logits, recurrent state and paged
  K/V arenas together as explicit roots before reading the token on the host.
  MLX therefore detaches each new state generation without invalidating a
  descendant graph. Session release synchronizes before pages are recycled.
- A ready decode group may mix full-logit and device-sampling requests. Keep
  compatible pending rows packed, execute other rows separately, and restore
  input order without adding an admission delay. Validate all session IDs and
  pending token IDs before advancing any row. A rejected token must not consume
  pending state. Returning from full logits to device sampling must recreate
  the pipeline so the session can rejoin packed decode on the following step.
- Persisted Metal tuning profiles identify the executable with a process-cached
  BLAKE3 hash, in addition to MLX version and GPU identity. This intentionally
  invalidates decisions after local builds without depending on Git or source
  files at deployment. Hashing failure disables persisted reuse; never admit
  an unidentified implementation. Profiles from different binaries are not
  interchangeable benchmark seeds.
- Metal prefill scheduling is architecture-aware: routed models finish queued,
  completion-balanced waves before streaming, while dense models retain
  completion-first interleaving. Derive routed wave width from the scheduler
  token budget and resident cache capacity; do not impose a model-name rule or
  a fixed row limit.
- Decode cohort admission uses exactly `decode_batch_wait_us`. Never multiply
  that window after a multi-row step or retain a hidden refill horizon; a
  latency/occupancy tradeoff must be explicit or backend-measured.
- The native MLX backend keeps an independent device-resident page-backed K/V
  state for each session, while dispatch remains serialized on one explicit GPU
  stream. Its longest-token-prefix LRU snapshots K/V and prompt logits as MLX
  array handles, so a hit does not copy tensors to Rust or recompute its cached
  prefix.
- Prefix byte accounting adds the union of actual retained recurrent allocations
  across every checkpoint and terminal snapshot to the existing K/V-plus-logits
  estimate. Use mirtal allocation ownership/identity, not logical row or tail
  sizes: views can retain a full parent buffer. Collect only materialized state;
  do not evaluate or synchronize merely to measure it. Keep recurrence out of
  the K/V-per-token estimator used by admission. Evicting one row does not free
  a shared parent while another cached row still retains it.
- An exact recurrent checkpoint without logits cannot rewind to compute its
  last token. Prefix lookup must select an earlier usable checkpoint or miss;
  preserve exact terminal hits and ordinary continuation at the current state.
- Multi-token GDN history compaction via `contiguous` remains test-only. The
  September 7 full-model ABBA reduced retained convolution memory by about 98%
  but failed the latency gate. Lower retention also changed chunk sizes through
  the existing pressure policy: disabled tuning does not freeze that schedule.
  Isolate identical chunk schedules before revisiting compaction; do not promote
  from memory savings or the favorable second pair alone. See Workmir's
  `2026-09-07-convolution-retention` report and archived candidate binaries.
- The subsequent uniform C5/128-token ABBA preserved all 640 tokens and actual
  stage schedules, but per-chunk history compaction still cost +17.4% prefill.
  Checkpoint-only compaction also remains test-only (`PrefixRetention`): it
  preserves the same memory saving with three preparations rather than 64,
  but its apparent speedup was invalidated by large drift in the zero-copy 2k
  control. Keep preparation before existing evaluation in scalar and packed
  paths; never add a synchronization at snapshot insertion. Do not promote
  either variant or another chunk policy from these unstable timing means.
  See Workmir's `2026-09-07-uniform-prefill` and `2026-09-07-checkpoint-retention`.
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
- Shared Qwen/clamped attention requests must preserve explicit masks and
  attention sinks. A Both-mode paged context may use its chronological K/V
  view for those capabilities; native-only placeholders must be rejected.
  Native paged kernels still do not support sinks.
- Qwen C5/8193 rejected the existing PagedBatched12 reader on a token mismatch.
  The exact-arithmetic batched two-pass experiment preserved all 640 tokens but
  measured only 1.00203x paired speedup, so it is test-only. Never specialize
  batched kernels on a changing used-page count: passing table stride through
  runtime metadata removed regular 16-token compilation stalls. Do not promote
  either candidate or repeat the same benchmark without a new hypothesis.
  See Workmir's September 6 Qwen attention report. Teacher forcing localized
  the Auto scalar/batch 8k divergence to MLX SDPA versus native paged; it did
  not qualify a global native-reader policy or establish semantic equivalence.
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
  or routed-expert decoders and 512 for dense softmax-only decoders; set it to
  `1` only when comparing against scalar-prefill behavior.
- `MIRMIR_METAL_PREFILL_EVAL_LAYERS=<positive integer>` materializes a causal
  prefill graph after each specified number of layers, without moving tensors
  out of MLX. It is an experimental graph-size control for benchmark tuning;
  unset means one graph for the full prefill chunk.
- `RUST_LOG=libmir::metal::prefill=debug` observes existing prefill boundaries
  with allocator memory and elapsed times. `Forward` contains nested `Segment`
  execution; never report all of it as CPU graph construction or sum nested
  timings twice. Diagnostics add no synchronization or tensor reads.
- The September 7 active-headroom scalar-budget candidate reduced evaluation
  count but failed a consistent end-to-end gate on MLX 0.32.2. Production keeps
  the original pressure policy; the patch, binaries and ABBA are archived in
  Workmir's `2026-09-07-prefill-diagnosis`. Do not promote from step counts or
  the favorable second pair alone.
- `MIRMIR_METAL_PROFILE_LAYERS=1` synchronizes after every hybrid routed-MoE layer and emits
  `tracing` timings. It is for diagnosis only and must stay disabled in normal
  generation.
- `MIRMIR_METAL_PROFILE_COMPONENTS=1` synchronizes attention/KV and
  dense-plus-MoE feed-forward separately in the shared packed layer loop,
  including clamped-routed decode. It is solely a
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
  Ordinary shared-expert gate/up fusion is an automatic complete-decode plan
  candidate. The tuner measures the whole device step on state snapshots and
  persists a decision per model and context bucket; it must not admit this path
  from the operator microbenchmark alone. Fused QMMs duplicate quantized arrays.
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
- Packed Gated Delta state may reuse its immutable full-batch parent only for
  the complete cohort in the original row order. Removed, reordered or mixed
  generations concatenate their actual row views. Keep the parent as an explicit
  evaluation/detachment root and preserve old snapshots; never turn this into an
  in-place recurrent-state write. The 128-step full-state regression covers
  cohort changes; real C5/2048 and C5/8193 gates require identical greedy tokens
  against forced row concatenation. See Workmir's September 6 C5 report.
- Packed GDN prefill remains a test-only candidate (`GatedDeltaPrefill`). On
  MLX 0.32.2, batching Qwen's MXFP4 alpha/beta and output projections changes
  their arithmetic and failed the identical-schedule greedy gate. Preserve
  these projections per row while packing QKV, z, convolution and recurrence.
  The corrected candidate matches the first-layer output/state bitwise and
  all 160 C5/128-token 2k/8k tokens, but its timing qualification was stopped
  after unchanged decode slowed 8.81x in the control. Do not promote from that
  incomplete pair or extrapolate to production chunks of 512. See Workmir's
  `2026-09-07-prefill-components` report and preserved failed binaries.
- The subsequent warmed single-process GDN replay matches output, recurrent
  state and convolution history bitwise at initialized offset 512 for both
  128/512 chunks and separate/shared-parent state. Its 512-token local gain
  is only 1.9–3.4%; 128-token timing fails the predeclared 15% spread control.
  This is one layer, not whole-model 512-token parity or throughput acceptance.
  Keep packed prefill test-only; do not repeat full-model BAAB from that small
  gain. See Workmir's `2026-09-07-gdn-paired` report.
- Packed Qwen and clamped prefill share explicit component profiling through
  `PrefillProfile`. Profiling settles prior graph roots before the first mixer
  and evaluates hidden, recurrent state and paged arenas after each component.
  These optional barriers change overlap: component times are diagnostic,
  not end-to-end throughput or GPU-only shader durations. Disabled profiling
  must not evaluate or synchronize; retain the clamped continuation regression.
- The September 7 actual MoE replay found gate/up + down at about 77% of
  barrier-profiled C5/512 time. SortedFused is neutral against GroupedFused;
  SortedGraph changes reduction arithmetic. Test-only MXFP4 bank gate/up
  fusion matches output bitwise but gives only 1.010x local 512 speedup and
  noisy 128 results. Do not enable it or repeat full-model series from that.
  Clamped sorted prefill also remains test-only; retain gathered biases,
  clamped activation, interleaving and graph reduction. See Workmir's
  `2026-09-07-moe-prefill` report. The missing native LHS gather index contract
  is a separate hypothesis, not an established optimization.
- Mirtal now exposes optional LHS/RHS gather indices for affine and MXFP4.
  Keep the Qwen indexed-input candidate test-only: MLX 0.32.2 disables its
  sorted-RHS shortcut with explicit LHS, and actual C5/128/512 MoE differs
  slightly from the materialized-input path. The bitwise gate skipped timing;
  no throughput improvement was qualified. Preserve RHS bias selection and
  do not replace default grouping or falsify the native sorted hint. See
  Workmir's `2026-09-07-gather-indices` report.
- The separate test-only indexed-MoE cost probe measures despite numerical
  differences and labels every record exploratory; it is not a quality gate.
  Its stable C5/128/512 ABBA found direct LHS 43%/93% slower locally. Reject
  this runtime candidate, retain mirtal's generic API, and do not repeat the
  unchanged probe or bypass MLX's grouped-kernel contract. The next projection
  experiment must target grouped compute itself, not merely remove `take`.
  See Workmir's `2026-09-07-indexed-moe-cost` report.
- Tile-aligned expert grouping remains test-only. Its bounded GPU padding
  preserves route inverse maps, gathered biases, and clamped/affine results;
  fused restoration may accept padded storage but must restore only real routes.
  Actual Qwen C5/128/512 MoE is bitwise equal and locally 1.166x/1.060x faster.
  Full C5/2k and 8k preserve 160 tokens and schedules, but whole-model timing
  failed the baseline stability control (25% prefill, 33% unchanged decode).
  Do not enable it or repeat that broad timing series. Expand short actual-layer
  and concentrated-routing coverage and account for extra allocator cache first.
  The 16-row layout targets MLX's non-NAX tile, not newer NAX tile sizes.
  Benchmark harnesses must declare enough manifest context and stop timing
  qualification as soon as their predeclared spread control fails. See
  Workmir's `2026-09-07-moe-aligned` report.
- Alignment budgets 4 and 8 were compared in actual Qwen layers 0/20/39 at
  C5/128/512, with actual and concentrated routing. All 24 comparisons match
  bitwise; 4 halves the extra grouped-input allocation (8 to 4 MiB). Neither
  qualifies for the tuner: hot-set routing regresses, and 4 gives under 3%
  improvement at actual layers 20/39 with 512. Two unstable cases were stopped,
  not retried. A fixed budget adds useless tiles when groups already align;
  layer number alone cannot predict routing on new inputs. Keep both test-only
  and target elimination of surplus tile work instead of another fixed budget
  or broad unchanged benchmark. See Workmir's `2026-09-07-moe-budgets` report.
- Compact GPU expert worklists and the custom BF16 MXFP4 tile projection remain
  test-only. After vectorized loads and bit-based FP4 decoding, September 7
  actual Qwen replay matches 12/12 outputs and gives 1.10–1.18x local C5/128,
  but only 1.01–1.05x C5/512. Two actual 512 cases miss the 3% latency gate;
  one hot-set 128 case is unstable. No input padding is allocated. The first
  scalar-load version was 20–29% slower at 512 and is archived too. Do not
  promote, change default chunks, or repeat broad benchmarks from these local
  gains. Profile the worklist and projections before changing the candidate.
  Normal replay continuation uses baseline kernels, not a candidate trajectory.
  See Workmir's `2026-09-07-moe-tiles` report.
- The follow-up compact BN64 kernel uses explicitly named SIMD accumulators.
  A generic matrix-array loop made BN64 6.3–7.5x slower and regressed BN32;
  `#pragma unroll` alone did not preserve performance. Keep the exact BN32
  shader and explicit BN64 MMA until a replacement is measured. BN64 passes
  all local gates (24/24 bitwise, 1.05–1.21x MoE, stable neutral hot routing)
  and whole-model C5/2049 plus C5/8193 token/schedule parity. The subsequent
  whole-model timing test stopped at 15.19% prefill spread after nine measured
  runs; all 1760 observed tokens still matched. Do not promote to the tuner or
  claim end-to-end/MLX-LM speedup. The cached stream-owned plan and shared
  `tiled_mlp` stay test-only; retain decode/short-tail fallback. Do not repeat
  unchanged long timing or infer compiler spilling without counters. See
  Workmir's `2026-09-07-tile-components` report and failed generic variant.
- The bounded-warmup BN64 follow-up reached readiness at round six (three
  recent samples per plan and phase within 5%). Its single measured series
  stopped after six runs: unchanged grouped decode spread 19.70%, although
  prefill spread was 2.67% baseline / 1.37% BN64. All 2880 tokens and schedules
  matched. Measured allocator active/cache snapshots were constant and equal
  between plans; this does not establish the cause of decode timing drift.
  Keep the six-round bound and independent 15% measured gate. Do not repeat
  the unchanged series, relax gates, compute incomplete aggregate speedups,
  or promote BN64. Diagnose decode host/GPU timing before another full-model
  qualification. See Workmir's `2026-09-07-settled-qwen` report.
- The bounded native decode trace (`2026-09-07-decode-boundaries`) covers
  99.11–99.30% of later C5/2049 decode wall time with target GPU execution
  envelopes. Rust graph construction stays at 39–42 ms; MLX roots submission
  grows from 682 to 748 ms and can itself wait for GPU work. Token reads cost
  about 0.22 ms. All 640 tokens/schedules match the prior reference. Keep the
  test-only boundary observer free of per-step barriers. Whole-model benchmark
  timing must settle the final pending graph once after its loop; its 3.5–4.8 ms
  tail does not explain the preceding 19.70% spread. Do not turn this benchmark
  barrier into a production barrier. GPU state labels changed but thermal state
  stayed Nominal; do not infer a cause or normalize timings from those labels.
  Further diagnosis should target GPU commands/kernels, not host orchestration.
- The decode-only Shader Timeline attempt (`2026-09-07-decode-kernels`) failed
  with GPU Service's unsupported selected-counter-profile warning and zero
  shader intervals. Do not repeat that configuration or infer shader durations
  from encoder wall times. A separate four-step component probe preserves160
  tokens and measures barrier-inclusive shares of44% FF/MoE,35% GDN,21% full
  attention; these are not pipelined runtime shares. At C5/top-k8 the40 routes
  select Unsorted; BN64's >1024-route prefill branch does not apply. Target
  actual unsorted MXFP4 expert GEMV projections for further decode investigation,
  including routing/reduction/shared-expert costs and real-model parity. Keep
  sparse profiling test-only and reset its stream switch after each call.
- Unsorted C5 decode MXFP4 bank fusion (`2026-09-07-decode-moe`) is test-only.
  Actual activations from steps8/24 and layers0/20/39 yield24 exact component
  replays and36 exact gate/up/whole-MoE comparisons across actual/hot routing.
  All12 timing cases stop at samples4–5 with >15% spread; do not compute
  incomplete averages, promote, or repeat the unchanged candidate. The local
  fused bank adds272.5–273 MiB while originals remain live.320 baseline trajectory
  tokens match; this does not qualify full candidate execution or GPT-OSS.
  Tiny component barriers dominate the decomposition, so stage shares are not
  GPU occupancy. Explain the systematic early timing jump before another local
  ranking; do not infer its cause or relax gates. Sparse probe selection remains
  typed and test-only and must be reset after each decode call.
- The submission-drift follow-up (`2026-09-07-submission-drift`) preserves640
  tokens and14 exact local replays. Native-only logging/profiler controls do not
  reproduce the jump. A factor run shows stalls before fused-bank allocation
  and after its drop, with an approximately16.5ms period. A later default trace
  correlates six slow measured samples after the first block with Ghostty or
  WindowServer GPU work; +/-1ms epoch uncertainty limits per-sample attribution.
  This is not a causal identification of every historical stall. Keep both
  probes test-only; do not rank fusion, filter outliers or loosen the15% gate.
  Next isolate elapsed-time periodicity from submission count with a fixed
  baseline sample-length experiment before choosing longer measurement windows.
  Do not stop user applications or repeat unchanged fusion/BN64/MLX-LM runs.
- The fixed sample-length diagnostic (`2026-09-07-submission-lengths`) records
  all944 samples and preserves160 tokens. Mirrored8/16/24-call blocks do not
  establish a coherent time or submission-count period. Inserting2ms sleeps
  is not a neutral control: the second spaced block slows by130.9% despite
  constant active/cache snapshots; continuous submissions recover near prior
  means. Do not use sleep for stabilization or repeat this probe. Next consider
  predeclared longer continuous measurement windows, with a separate baseline
  stability gate before ranking candidates; keep every outlier and the15% gate.
  Test-only sample lengths use NonZeroUsize and must not alter production decode.
- Prefill settlement synchronizes and detaches paged arena graphs, then reclaims
  the allocator cache under the existing memory-pressure thresholds. Preserve
  reusable allocations below those thresholds; do not restore unconditional
  clearing without a same-profile alternating benchmark.
- Shared-expert MoE routing uses unscaled top-k when the model has no
  per-expert correction scale. Do not materialize an all-ones FP32 scale: its
  cast, gather, squeeze and multiply added 160 nodes across 40 Qwen3.6 layers.
  Removing them reduced decode from 2,579 to 2,419 nodes, preserved the full
  digest and measured `90.15 tok/s` versus `89.45 tok/s` in a matched run.
- U32 and byte-packed U8 OCP MXFP4 projections use mirtal's native MLX
  `quantized_matmul` and `gather_qmm` primitives. U8 checkpoint weights are
  reinterpreted as U32 and reshaped lazily; do not dequantize them on the host.
  Apply output biases after QMM, gathering expert biases with the same indices
  before broadcasting them over gathered outputs. The custom MXFP4 kernels are
  test references, not runtime fallbacks. On Qwen3.6-35B-A3B MXFP4 the original
  U32 native change preserved the 128-token
  greedy digest and moved the controlled 128/64 benchmark from roughly
  `71`/`36` to `622.81`/`93.62` PP/TG tok/s; the same-machine `mlx_lm`
  reference measured `632.97`/`104.72` tok/s.
- The compiled single-token Gated Delta graph must admit ordinary U32 MXFP4
  projections, not only affine QMM projections. Qwen3.6 otherwise silently
  falls back to direct graph composition. The retained graph preserves the
  exact 128-token digest and improves geometric-mean 128–8192 decode by 2.3%
  in the final matrix; compiling the router, outer residual, or shared-output
  gate separately did not improve end-to-end throughput.
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
- Actual mixed shared-prefix exact-token gates are not fully qualified. The
  September 8 context probe passes GPT-OSS C2/2049 and mixed-prefix/2049, but
  fails GPT-OSS C5→C1 at offset10 and mixed-prefix/8193 at offset18; Qwen/2049
  fails the new mixed-prefix gate at offset59. Retaining inactive sessions
  reproduces the same first differences, so release itself is not necessary
  for these failures. Keep the strict gates and both controls; do not undo
  lifecycle fixes, force scalar execution or infer a specific numerical cause.
  Matched-state teacher forcing now localizes width-dependent attention output
  projection differences on identical inputs in both GPT-OSS and Qwen. Qwen's
  step-0 difference is now localized to the query/gate projection's gate
  slice; Q/K/V, RoPE and raw attention agree. The new f64 attention references
  sometimes favor scalar and sometimes packed. The fixed FP32 candidate is
  width-invariant but misses nearest BF16 in 2/324 new samples; do not promote
  it as an unconditional accuracy fix. See `2026-09-08-qwen-attention-width`.
  These one-step probes do not explain every accumulated failure.
  See Workmir's `2026-09-08-decode-width` report. Native MLX AddMM failed the
  BF16 cancellation precision hypothesis and its trial API was reverted; do
  not reintroduce it as a proven precision fix. Retain observer controls.
  The two passing GPT-OSS 7257-token Harmony retrieval cases are short semantic
  evidence, not batch quality or performance qualification. See Workmir's
  `2026-09-08-gpt-oss-context` report; do not repeat unchanged MLX-LM timing.
- Text prefill cancellation now reaches cache waits and scheduler queues.
  Metal yields between evaluated graphs, then retires selected rows before
  acknowledging cancellation; never let a waiter free cache before that ack.
  Keep original progress indices until the interrupted step returns, adjust
  cohort leases/counts together, and publish held completions when the last
  queued wave is cancelled. Wake cancellation even for queued/held requests.
  The quantum-only intermediate still took up to 7.78 s on GPT-OSS and was
  superseded; do not restore it as prompt cancellation. Actual Qwen/GPT-OSS
  public API tests stop at 1024/8193, preserve sibling/refill prefill+decode,
  and stop generate_cancellable before prompt completion without output.
  See Workmir's `2026-09-08-prefill-cancellation`. CUDA is compile-checked
  with retirement at scheduler boundaries; image prefill remains outside
  this text mechanism. No new HTTP quality or throughput gate was passed.
- Default HTTP reasoning differs from the no-thinking Qwen fixtures: multiple
  registry requests exhausted 1024 tokens without final content. Preserve that
  failed qualification, define reasoning-mode/budget behavior explicitly and
  do not silently disable reasoning globally to turn the test green.
- The Qwen registry quality gate passes shared prefixes of 2049 and 8193
  tokens with mixed full/device sampling, cancellation at step16 and refill
  at step32: 12/12 complete answers correct, all six mixed answers exactly
  match cold scalar controls, 88 non-EOS tokens each, zero resident arenas.
  See Workmir's `2026-09-08-qwen-prefix-quality`. This tests sequential prefix
  continuations and grouped decode, not packed prefill or production HTTP.
  Keep the existing strict synthetic failures visible; retrieval success
  does not qualify global numerical equivalence or throughput.
- Host diagnostic and example greedy fallbacks must choose the first maximum
  for finite tied logits, matching device argmax (including signed-zero ties).
  Iterator `max_by` selects the last equal element; do not use it without a
  deliberate tie rule. Metal diagnostics share `native::benchmark::argmax`.
- Preserve YaRN `truncate` through parsed and semantic model descriptions and
  both backends. Omitted fields default to true; official GPT-OSS specifies
  false and requires continuous correction bounds. Do not infer this from a
  model name or copy MLX-LM's unconditional floor/ceil behavior.
- Clamped expert activation must cast its clipping limit to the projection
  dtype. A strong FP32 limit promotes BF16 activation and makes the following
  MXFP4 projection fail. Keep strict MXFP4 validation and cover BF16 as well as
  FP32 fixtures. Actual official U8 GPT-OSS passes the reference, Harmony,
  pipeline and abandonment smokes in Workmir's `2026-09-08-gpt-oss-metal-smoke`;
  these do not qualify long-context quality, U8/U32 parity or performance.
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

## Request-scoped thinking templates

- `GenerationRequest::reasoning` selects `ReasoningMode` for text prompt and
  cache-checkpoint rendering. `ModelDefault` preserves the previous behavior;
  explicit modes require the Jinja `enable_thinking` input or a supported
  built-in template. Image requests reject explicit modes for now.
- Keep total completion limits inclusive of reasoning and final content. A
  disabled-template result does not qualify default-thinking quality. Workmir's
  `2026-09-08-qwen-reasoning-control` passes one Qwen HTTP disabled-mode case;
  GPT-OSS execution and concurrent HTTP are not qualified by that run.

## Optional cached refill admission

- `SchedulerConfig::cached_prefill_policy` defaults to `BackendDefault`.
  `InterleaveOneBlock` permits one short cached queue head during serialized
  backend decode, with slot/token/resident-capacity checks, one-row cohort and
  actual per-step replay cap. Preserve FIFO and existing cohort/handoff ownership.
- Workmir's `2026-09-09-cached-refill-admission` passes Qwen semantics and
  cancellation but fails default promotion: refill TTFT ~1 s→47 ms costs +22%
  survivor decode in ABBA (limit 15%). Keep this explicit opt-in tradeoff;
  do not globally enable routed interleaving. GPT-OSS new-mode execution is open.

## Decode timing attribution

- Metal structured decode timings split pre-worker wait from execution wall
  time at worker entry. Batch rows share a single completed operation interval;
  do not sum that shared duration across rows. Keep device_execution unavailable
  unless measured as device time, and preserve separate scheduler queue timing.
- Workmir's `2026-09-09-decode-refill-cost` verifies Qwen/GPT-OSS timing adapters,
  282 Metal tests and unchanged HTTP token sequences. C3 trace is fully packed,
  with ~16.5 ms execution and ~0.0066 ms worker wait. This does not identify
  a dominant kernel or prove a speedup; transitional C2 samples are not a matched
  throughput baseline. The cached-refill option remains opt-in after its prior
  survivor-latency gate failure.

## Mixed-position RoPE experiment

- Mirtal exposes device-offset RoPE for generated and explicit frequencies.
  Libmir's `RopeBatching::Offsets` integration remains test-only after the
  September 9 ragged C3 gate: 960 Qwen and 192 GPT-OSS token IDs match, but
  Qwen gains only 0.54% overall and regresses 0.25% in the first block. Do not
  promote or repeat unchanged. These synthetic-token tests are not chat quality.
- Batched rotation does not imply batched K/V concatenation: mixed-position
  view contexts may have different lengths. Preserve per-row attention, sinks,
  YaRN concentration and dtype. Keep the view/native unequal-length regression.
- The new C2/C3 component probe has stable same-width controls, but cross-width
  greedy trajectories differ. Barrier-inclusive attention +23%, MoE +10% and
  GDN +9% are scaling observations, not GPU shares or identical-work savings.
  See Workmir's `2026-09-09-decode-c3-components` for both failed promotion
  and the earlier candidate's invalid ragged-K/V concatenation error.

## K/V projection join experiment

- Ordinary MXFP4 and unclipped dense output joins remain test-only. Actual
  GPT-OSS K/V weights are dense BF16 with bias, not its routed MXFP4 banks.
  Preserve post-projection bias and output-major dense weight storage followed
  by the transposed view; reject clipping and incompatible bias/layout contracts.
- Workmir's `2026-09-09-mxfp4-kv-fusion` captures C3 step 8, layers 3/19 and passes
  all 6144 K/V output values per model bitwise. Observer controls preserve 192
  baseline IDs/model; this is not a whole-model candidate or chat-quality gate.
  Qwen layer 3 hot replay gains 32.36%, but layer 19 stops at 16.05% reference spread
  against 15%. Do not promote, average the incomplete case or retry unchanged.
  Added joined storage is 1.0625 MiB/layer Qwen and about 5.627 MiB GPT-OSS while
  originals remain live. No MLX-LM or full-model performance gate was run.

## K/V whole-decode qualification

- The ten-layer weight-set replay in Workmir's `2026-09-09-kv-working-set`
  passes locally (24.38% less K/V wall time, 30720 bitwise values), but the
  subsequent whole Qwen C3/2049 gate fails: 474.732→475.135ms/32 steps, a
  regressing second block and <3% gain, with both spreads below 4.3%.
  This does not invalidate the prior failed layer19 control. Do not rerun either
  unchanged or promote from local percentages; they are not full-decode shares.
- `KeyValueProjection::JoinedDecode` and `KeyValueJoin` remain test-only,
  restricted to packed noncausal sequence=1. Qwen/GPT-OSS uniform/ragged C3
  preserve 384 recorded IDs/model; expected 320/768 joined forward calls confirm
  use across all 10/24 attention layers. Keep scalar/prefill separate, preserve
  bias/sinks/YaRN and originals for A/B. Synthetic tokens do not qualify chat.
  Additional joined logical storage is 10.625 MiB Qwen, 135.046875 MiB GPT-OSS;
  allocator snapshots are not isolated peak-memory deltas. No MLX-LM run.

## Expert weight reuse structural gate

- Workmir's `2026-09-09-expert-weight-reuse` captures normal-prompt Qwen
  C3/C5 routes without decode host reads. All 1024 observed IDs match controls;
  1600 layer/step records include 51200 route occurrences. Diverse tasks allow
  only 5.32%/11.73% ideal repeated-weight elimination, below the declared 15%
  structural gate. Related LRU questions give 36.55%/53.98% but cannot qualify
  a general kernel. These are logical counts, not measured DRAM traffic or
  speedups. Keep this negative result; no kernel or MLX-LM timing was run.
- Pinned MLX sorted-RHS requires M==1, B>=16, right_sorted and B/E>=4 using the
  full expert-bank count. Qwen C3/C5 has B=24/40, E=256, so this gate is false.
  Explicit LHS also disables right_sorted. Ordinary qmv_wide is not selected
  by gathered dispatch; sorting or changing M alone does not add weight reuse.
- The test-only route observer supports shared and clamped models, retaining
  device indices and reading only after generation. Keep exact layer/row
  attribution and scope cleanup tests. Clamped fixture assertions cover the
  GPT-OSS architecture hook; no actual GPT-OSS route distribution is qualified.

## Four-lane GDN experiment

- Workmir's `2026-09-09-gdn-four-lane` ports the newer MLX-LM value-row SIMD
  layout, distinct from earlier packed-request GDN. Sequence and fused decode
  candidates preserve exact output/state, explicit-tree/native canary, f64
  reference, 128-step continuation and snapshots. Keep decode's original
  normalization casts, precise rsqrt and FP32 gates; do not drop their fusion.
- `GdnExecution` remains test-only. Actual Qwen C3/2049, C5/257 and normal
  questions C1/C3/C5 preserve 2496 recorded IDs across controls/candidates;
  short answers are capped at 64 tokens. Counters verify all30 GDN layers.
  Stable whole-model ABBA/BAAB rejects promotion: decode +0.87% time, prefill
  -2.57% time (both blocks positive but below the predefined3% mean gate).
  Do not retry unchanged, relax the gate, or extrapolate upstream M5 prefill
  speedups to M3 decode. No new MLX-LM or actual GPT-OSS benchmark was run.
- Candidate kernels require Dk128 and Dv divisible by8; sequence also requires
  FP32 state. Preserve native fallback, immutable state and compiled graph
  selection. GDN belongs in libmir, not mirtal; GPT-OSS has no GDN.

## K/V history assembly experiment

- Workmir's `2026-09-09-kv-history-assembly` confirms ten Qwen C3 joined K/V
  pairs allocate about 122 MiB of logical output per captured final decode step.
  This is not measured DRAM traffic. Native MLX SDPA requires compatible batch
  and head strides; a shared-arena view alone can trigger a contiguous copy.
  The 52.79% frozen resident-history local gain omits cache updates and is not
  whole-model performance. Per-session page gather cost remains unmeasured.
- `HistoryBatching::Rows` stays test-only, default Joined. It preserves native
  SDPA and joins small outputs for common-position, noncausal, single-token
  view contexts. Whole C3/2049 improves 3.57%; C2 improves 1.95% and passes its
  separate protection gate, not the generic 3% target. C5 aborts at 17.97%
  Joined spread against 15%; no incomplete mean, unchanged retry or promotion.
- Qwen/GPT-OSS uniform/ragged checks retain 384 recorded IDs/model; C2/C5
  short/long checks retain 896. GPT-OSS already uses row views with sinks.
  Qwen's hybrid batch caller forces native paging when all positions reach 8k;
  the lower cache policy alone does not describe that selection. Preserve
  the initial incorrect 8k counter-expectation failure in the archive. Corrected
  zero-selection checks prove fallback scope, not long-context candidate gains.
  These synthetic-token tests do not qualify chat quality. No MLX-LM rerun.
- The test observer retains only the last-step roots, evaluates and drains
  before session release, and replays without further page writes. Do not add
  these synchronization or allocation-inspection steps to normal decode.

## Direct page-to-batch gather experiment

- Workmir's `2026-09-09-kv-page-gather` proves that Qwen's per-session views
  already copy fragmented history before batch concatenation. All30 C3 views
  have a contiguous130-page prefill extent plus an out-of-run decode page;
  `take` allocates about122 MiB logical K/V output across layers. GPT-OSS full
  layers have36 gathered views (11.39 MiB), but no second batch-history join.
  Allocation identity and CPU chronological indexing are checked after drain;
  logical tensor sizes are not measured DRAM traffic.
- `HistoryBatching::Gathered` is test-only, default Joined. Direct gather into
  [B,H,T,D] retains native SDPA, cache writes, COW and device page tables. Select
  only common-position, noncausal sequence1 View mode,2–12 rows with existing
  unquantized pages. Keep native8k, ragged, prefill, scalar, unpaged and GPT-OSS
  sink-aware paths unchanged. Validate layouts and extents before launching;
  page-table contents are cache-owned, never read to the host for dispatch.
- The scalar-index kernel passes144 numerical cases and384 recorded IDs/model,
  but whole C3/2049 regresses12.01%. Preserve that binary/log, no unchanged retry.
  Grid-axis/four-value addressing passes147 numerical cases and another384 Qwen
  IDs; whole decode improves2.847%, below3%, despite both positive blocks and
  passing15% spreads. No promotion, rounding into a pass or repeat to rescue it.
  Absolute control times differ substantially across the two series; compare
  each candidate only with its own interleaved control. Drift cause is unknown.
- Specialize the gather only on stable head count/dimension and page size.
  Context length, page count/IDs and arena capacities must remain runtime data;
  do not reintroduce a new pipeline at each page boundary. Revised-kernel tests
  include partial vectors and a non-power-of-two page size. GPT-OSS dimensions
  pass shared kernel tests, but its actual decoder does not select this gather.

## Bounded decode page reservation experiment

- Workmir's `2026-09-09-decode-page-reservation` tests64 extra tokens instead
  of one page for packed-prefill ownership. PrefillRequest has no generation
  limit in that experiment; do not claim it propagated a public budget. Its
  fixed Tokens64 comparator remains test-only; the later GenerationBudget policy
  is described below. That trial kept scalar/vision behavior unchanged. Keep one
  immutable sequence plan for the actual cache target and every capacity check,
  including unowned tail pages and partial-prefix COW.
- Actual cold C3 Qwen/GPT-OSS pairs preserve192 recorded IDs/model and CPU
  chronological equality. All30/36 candidate full-layer views alias arenas,
  removing about122/11.39 MiB logical copied outputs. Qwen's second batch join
  remains. Planned extra ownership2.8125/3.375 MiB is not allocation or DRAM
  traffic; final arena shapes are unchanged, while global prefill active-memory
  snapshots differ by16 KiB/12 MiB. Do not claim zero memory cost.
- Capacity rejection before mutation, reserved-tail release, partial-prefix COW,
  cancellation/refill and full release pass; synthetic hybrid/clamped C5→C1→C5
  churn also passes with64-token horizon. Existing shared prefixes may remain
  fragmented; no relocation or guaranteed contiguous restoration is implemented.
- The fixed full-model series aborts after six runs on24.8366% OnePage prefill
  spread against15%. All576 completed IDs match, but neither prefill nor decode
  has an aggregate qualified result. Do not average the partial series, ignore
  the prefill protection gate, retry unchanged, or promote from copy counts.
  Runtime budget plumbing, broader memory/capacity behavior and performance
  remain open; no new MLX-LM reference run was made.
- Clippy also started during this rejected timing process; no phase timestamps
  prove it overlapped only warmup. Do not attribute the drift to it without
  evidence, but keep the confound explicit. Run future performance processes
  without concurrent builds, clippy or tests.

## Impossible prefill reservation

- Reject additional page requirements above absolute physical capacity before
  evicting any prefix in `reserve_prefill_pages`; use the typed `KvPageCapacity`
  error with the first full-attention layer. Eviction cannot admit such a request.
- Prefix lookup must not implicitly reserve a miss slot. Scalar and direct packed
  prefill restore states, check page capacity, then reserve prefix slots. Keep
  full-cache regressions for hybrid and clamped models, not only roomy-cache
  tests: earlier miss eviction bypassed the initial capacity guard.
- Workmir's `2026-09-09-impossible-reservation` preserves prefix hits and exact
  logits/tokens after rejected scalar/packed requests; six new regressions and
  all304 Metal tests pass. This is a production correctness fix, not a timing
  qualification of the test-only64-token reservation candidate.
- The absolute guard does not roll back pressure eviction for requests within
  capacity when active/leased pages prevent reclamation. Logical prefill cohorts
  retain their separate earlier lease/eviction policy across physical waves;
  do not reject their total future workload as though it were one physical wave.

## Generation-budget reservation

- Workmir's `2026-09-09-generation-reservation` adds `PrefillRequest::generation_tokens`
  from generation's existing logical reservation limit. Preserve the request through
  scalar facade dispatch. The count includes the first prediction; raw prefill has
  no declared limit. CUDA keeps its existing policy; vision is unchanged.
- `MetalCacheConfig::decode_reservation` offers OnePage (default) and GenerationBudget;
  the facade exports `MetalDecodeReservation`. Tokens64 remains test-only. Admit
  baseline pages before using spare capacity; optional tails must not trigger extra
  prefix eviction. Recheck pending plans and allow them to shrink before execution.
- Under pressure, return only unwritten reserved IDs from active sessions and weakly
  tracked unfinished batches, preserving history/remaining prompt plus the existing
  page allowance. Update future reservation targets too. Decode retries capacity
  after reclaiming, before any row advances. Do not add synchronization, relocate
  initialized pages or treat freed ownership as reduced arena allocation bytes.
- Exact prefix-hit completion must retire its pending reservation when its state
  moves into active sessions. Synthetic hybrid/clamped pressure, COW, refill and
  C5→C1→C5 checks pass; all311 Metal tests pass.
- The32-token actual-model pairs preserve192 IDs/model and remove observed copies
  from30/36 full-layer views. Qwen's second batch-history join remains. Shared-prefix
  contiguity is not guaranteed; logical copied bytes are not measured DRAM traffic.
- Whole Qwen timing aborts after6 runs on28.2322% control prefill spread. All576
  completed IDs match, but no aggregate performance result qualifies. This run had
  no concurrent assistant-launched build/test/clippy; the drift cause remains unknown.
  Keep OnePage default. Do not retry unchanged, average the partial series or claim
  an MLX-LM win. Diagnose prefill instability before another timing series.

## Prefill drift hardware diagnosis

- Workmir's `2026-09-09-prefill-drift` runs six OnePage C3/2049 samples and one
  recovery after20 seconds idle, with existing stage logging and external macmon.
  Warm prefill grows28.61% as active GPU frequency falls20.67% and temperature
  rises from about85 to98°C. Evaluate accounts for95.71% of the added latency;
  Forward and boundary allocator snapshots do not explain it. All672 IDs match.
- Idle partially restores frequency/performance while preserving the loaded model;
  it does not restore the earlier operating condition. Do not assume two warmups
  or a fixed20-second pause settles this workload. Choose sustained versus burst
  qualification explicitly and record hardware conditions before another A/B.
- The time×frequency product is only a post-hoc diagnostic. Never normalize away
  real variant-dependent frequency/power behavior or use it to rescue a failed gate.
  Telemetry is whole-machine, coarsely windowed; Evaluate is not GPU-only time.
  Global swap counters include loading and do not rule out other contributors.
  This capture cannot retroactively explain every historical outlier. No new
  candidate benchmark or production execution change was made in this diagnosis.


## Sustained baseline qualification

- Workmir's `2026-09-09-settled-reservation/cold-readiness-fix` contains the corrected
  external controller for the ignored `qualifies_sustained_qwen_reservation` driver.
  Low GPU active residency during cold warmup means not ready, not corrupt telemetry;
  never admit that row into the stable tail. The initial controller error is archived
  separately and must not be reported as a performance result.
- OnePage baseline passes the four-row warmup gate after57.99 seconds but the first
  independent validation row expands prefill spread to3.1850%, exceeding3%. All1152
  recorded IDs match. No candidate or A/B ran; native test success is not qualification.
  Preserve the anchored validation window; do not slide it or loosen gates after failure.
  Do not retry unchanged or rerun MLX-LM. A decode-only fixed-context experiment is a
  separate possible next hypothesis and cannot replace eventual whole-request validation.


## Resident-cohort reservation timing

- Workmir's `2026-09-09-resident-decode-reservation` prepares four independent C3/2049
  cohorts once and moves exclusive state ownership for each full decode sample.
  Do not replace this with SessionState snapshots: paged snapshots drop reserved
  tails and may introduce COW, changing the layout under test. Context advances
  equally per block; this is not fixed-position replay or a12-request scheduler test.
- The new budget256 workload passes isolated decode:578.933→488.216 ms/32 calls,
 15.6696% time reduction, blocks17.2449%/14.0402%; measured spreads9.3717%/4.0042%.
  All2376 decode IDs and positions match. In the captured conditioning step each
  OnePage cohort has30 copied full K/V views, each GenerationBudget cohort30 aliases.
  Keep the615.070 ms baseline sample; no filtering or frequency normalization.
- This does not rescue the earlier budget32 whole-request trials or qualify prefill,
  memory deltas, chat quality, GPT-OSS, C2/C5, or an MLX-LM win. OnePage stays default.
  Next assess budget256 preparation/memory and pressure/cache reuse, then protect
  prefill in a whole-request qualification. No unchanged retry or MLX-LM rerun.


## First-token COW after generation reservation

- Workmir's `2026-09-09-budget256-resources` exposes17/16-page failure with budget256
  and a33-token cold prefill. Terminal prefix insertion shares the partial last page
  after admission; greedy lookahead needs COW while optional tails own all pages.
- Keep scalar cold/exact-hit and batch first outputs routed through
  `native/prefill/reservation/output.rs`. For GenerationBudget device sampling, check
  the upcoming write; on typed KvPageCapacity reclaim the local state's unused tail
  and recheck before forward. It is not yet in the live-session map. No tensor reads,
  GPU barrier, decode-loop duplicate check or untyped error-string matching.
- Both synthetic architecture tests cover cold scalar/batch, exact33-token prefix,
  pressured extension and32-token trajectories. Clamped continuation legitimately
  restores aligned32 while Hybrid restores33. All315 Metal tests pass.
- Actual Qwen C3/2049 budget256 owns450 extra pages(14.0625 MiB); arena buffers grow
 180→200 MiB. Reclaim returns450 pages to the pool but does not shrink buffers;
  full session/prefix release drops all arenas. All30 recorded continuation IDs match.
  Cold/warm prefill timings are diagnostic only; preparation/whole-request latency
  remains unqualified. Keep OnePage default and archived MLX-LM/decode references.


## Complete native budget256 qualification

- Workmir's `2026-09-09-budget256-whole` runs C3/2049 with exactly256 predictions
  per row (first prefill output plus255 decode calls), preserving the device pipeline
  and final benchmark drain. It measures native fixed-length work, not HTTP/EOS.
- The single baseline-only attempt fails readiness within90s. Final four-sample
  spreads: prefill5.7324%, frequency6.3082% exceed3%; decode9.8916% is below15%.
  All7680 recorded IDs match. No validation/candidate/A/B ran. Keep OnePage default,
  no incomplete means, shifted window, relaxed gate, unchanged retry or MLX-LM rerun.
- The isolated resident decode gain does not qualify complete native generation.
  A separate application gap remains: mirmir configuration does not map public
  MetalDecodeReservation yet. Explicit opt-in plumbing is a next implementation
  task, not a default promotion or an explanation of this native timing failure.


## Reservation configuration boundary

- MetalDecodeReservation now serializes/deserializes production one_page and
  generation_budget values and provides Display. Keep test-only Tokens excluded
  from configuration deserialization. Its owning module is config/cache/mod.rs.
- Workmir's `2026-09-09-reservation-config` adds the macOS application setting
  runtime.metal_decode_reservation using this same public type. No parallel app
  enum or private backend dependency. OnePage remains the library default;
  application auto removes its override. This is plumbing, not a timing promotion.


## GenerationBudget application HTTP qualification

- Workmir's `2026-09-09-generation-budget-http` passes Qwen 6/6 and GPT-OSS 9/9
  complete answers through the current mirmir build with explicit GenerationBudget.
  Both cover C3 cancellation/refill, identical repeat IDs, logical prefix reuse and
  zero logical cache counters on unload. Qwen uses disabled reasoning; GPT-OSS
  uses default Harmony. Budget 256 is reserved but answers stop earlier.
- No product change or performance/default qualification follows from this run.
  Physical arena pressure, full-budget output and whole-request timing remain
  separate evidence domains; do not infer them from HTTP logical cache counts.


## Queue-aged prefill collection

- Prefill collection deadlines are anchored to PendingPrefill::enqueued, not
  collector entry. PrefillWindow keeps the latest-arrival quiet deadline capped
  by the oldest-arrival hard deadline, even when senders/priority order differ.
  Time spent behind accelerator work must not grant another gathering window.
  This is a collection bound, not a total queue-latency promise during GPU work.
- Preserve fresh-burst grouping, existing backend serialization and cached-refill
  opt-in behavior. Do not infer permission to interleave from an expired window.
- Workmir's `2026-09-10-prefill-admission-age` reproduces the redundant receive
  window and removes a 30–40 ms host-only wait for already-aged requests. Metal
  facade 103 tests, neutral 72 tests, final focused admission 22 tests, clippy,
  CUDA-feature compile and application build pass. Qwen 6/6 and GPT-OSS 9/9 HTTP
  answers, cancellation/refill/cache reuse, C3 cohorts and pre-change token parity
  all pass. This is not full-model throughput or a resolution of thermal drift.


## Cancellation during collection

- Generation-worker prefill/decode collectors process Command::Cancellation with
  the existing cancel_prefills maintenance while waiting. A notification must not
  wake recv_timeout only to be ignored until the gathering deadline. Stop when
  the relevant queue is empty; preserve survivors and their existing deadlines.
  Reuse cohort/session retirement and acknowledgement, never bypass ownership.
- Workmir's `2026-09-10-admission-cancellation` reproduces three worker failures
  and removes a 34–40 ms host-only wait for fresh cancelled requests. Metal facade
  106 tests, neutral 72 tests, clippy/CUDA compile/build and Qwen/GPT-OSS targeted
  HTTP collector disconnects pass. Both models recover and reuse cache with
  exact archived IDs. The probe's explicit 300 ms quiet window and ~115 ms coarse
  HTTP observations are diagnostic, not defaults or a full-model speedup.

## Persistent batch history experiment

- Workmir's `2026-09-10-persistent-history` implements test-only
  HistoryBatching::Persistent for common-position, noncausal sequence1 Qwen
  native views. Keep Joined default. GPT-OSS sink-aware row readers never select
  this path. This is incremental batch storage, not the rejected page gather,
  row SDPA or joined K/V projection candidates.
- Each cache owns its row index and an Arc to the cohort history generation.
  Take all handles before updating rows; require identical generation/order/width
  and one-token progress to reuse storage. Ordinary updates/reset invalidate;
  snapshots never inherit handles. Append only beyond all earlier view extents;
  allocate fresh on growth/churn. Keep the Metal library in the stream. Never
  introduce unsynchronized overwrites of positions visible to lazy readers.
- GPU tests cover bitwise FP32/FP16/BF16 SDPA, H2/D256 and H8/D64, allocation
  reuse/growth254–258, newest-first evaluation, churn and ragged fallback.
  321 backend tests and384 recorded IDs/model pass. Resident C3 records2178
  matching IDs; initial candidate block10 seeds/310 appends, later0/320.
- One resident GenerationBudget256 A/B trial fails in block5 at29.3757% baseline
  spread including conditioning anchor (15% limit), after passing3.8988% initial
  conditioning (5% limit). No aggregate speedup, unchanged retry, default
  promotion or MLX-LM rerun. Test process exit0 is not a performance pass;
  inspect resident.stop and absence of resident.gate.
- Duplicate batch buffers cost135 MiB logical capacity per C3 cohort at2304
  tokens across10 Qwen layers. Last-row ownership may retain a former group
  until the next decode/drop. Before production, qualify timing and cohort/context
  protection and integrate this memory into cache accounting/reclamation.

## Persistent history reclamation and prepared append

- Workmir's `2026-09-10-history-reclamation` extends test-only Persistent:
  KvCache::release_reservation_after clears its history handle; reclaiming all
  active session reservations releases cohort ownership. Continued decode seeds
  from authoritative per-session K/V. Do not retain diagnostic Allocation handles
  across reclamation; they themselves keep storage alive.
- Stream-owned Append caches one PreparedAliasing<4,2> per FP32/FP16/BF16 and
  rebinds geometry/constants. Validate dimensions, offset and uint address range
  before dispatch. Keep plans out of Send session/cache state. Plans must release
  tensor inputs after constructing lazy graphs, and later rebinds must not change
  pending graphs' parameters. No new GPU synchronization in execution/reclaim.
- 324 ordinary Metal tests pass; the isolated allocator regression runs separately.
  Actual Qwen C3/2049 GenerationBudget256:20 allocations/135 MiB before reclaim,
  zero after,20 after continuation; active MLX drops exactly135 MiB. GPT-OSS
  C3/129 retains no group history. Reclaim/continue plus uniform/ragged checks
  preserve768 recorded IDs/model. These are synthetic correctness checks.
- Host-only4096-dispatch construction drops4.131583→1.835709 ms (55.5689%),
  around0.561 microseconds saved/call, both blocks positive with15% spreads met.
  Output graphs are never evaluated in that timing: no GPU/model speedup claim,
  resident retry or MLX-LM rerun. Persistent stays test-only; per-history limits,
  global RAM-pressure integration and broader protection remain unqualified.

## Persistent history ownership budget and prefill pressure

- Workmir's `2026-09-10-history-budget` adds a test-only512 MiB budget per stream.
  Share one Arc lease per history storage across rows and append generations.
  Reserve before allocating, count old+new storage during growth, reject overflow,
  and use Joined SDPA on budget denial. Final History ownership releases the charge.
  The quota excludes arrays retained only by lazy readers, temporary graphs and
  allocator caches: never describe it as a hard total GPU-memory cap.
- Before scalar/packed prefill prefix handling, apply the existing >50% usable
  MLX pressure predicate. Suspend history admission and release active session
  histories before prefix RAM reclamation. Keep prefix/arena ownership intact.
  Resume admission only at a later low-pressure prefill check; decode must not
  immediately rebuild while suspended. No per-token MLX memory query or GPU sync.
- 328 Metal tests pass. Actual Qwen C3/2049 and GPT-OSS C3/129, GenerationBudget256:
  Joined and Persistent quotas0/40.5/512 MiB,32 decode calls before simulated
  pressure,32 suspended,32 after relief.1152 recorded IDs/model match. Qwen
  retains0/3/10 layers under those quotas, zero while suspended, and rebuilds;
  GPT retains none. Charges equal captured unique buffer bytes and end at0.
- Actual-model pressure observations are explicitly injected policy inputs, not
  measured OOM cycles. That harness has no prefixes; the separate hybrid/clamped
  regression verifies a nonempty prefix cache and preserved lookup. Keep allocator
  release measurements isolated from parallel tests. No timing, resident retry,
  MLX-LM rerun or default promotion is established by this resource qualification.


## Joined-only resident decode readiness

- Workmir's `2026-09-10-decode-readiness` is one new fixed Joined-only probe:
  four independent C3/2049 cohorts, GenerationBudget512,384 calls per cohort,
  four128-call warm-up windows then eight measured windows. No recurrent rewind.
  The combined external gate rejects11.2173% decode and39.1403% GPU-frequency
  spread (5% limits), despite99.9819% telemetry coverage and4644 matching IDs.
- Decode fails already within one matched-context round. Global swap/compression
  activity and other host work were observed; do not claim temperature alone
  explains it.328 ordinary Metal tests pass. No candidate/MLX-LM timing ran.
  Do not select only the favorable final round, relax gates or repeat unchanged
  warm-up loops. Require materially quieter/memory-stable conditions before
  qualification; Persistent remains test-only with C2/C3/C5/churn timing pending.
