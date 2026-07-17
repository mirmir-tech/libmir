# Libmir Roadmap

This document contains only forward-looking work. Implemented capabilities and
the current public contract belong in the README and API documentation. Kernel
experiments and performance measurements belong in reproducible benchmark
artifacts rather than this roadmap.

## Public API and 1.0 readiness

- Separate the stable `Library`, `Model`, `Session`, `Engine`, and generation
  facade from advanced runtime and model-building APIs.
- Add compatibility tests and complete examples for external consumers.
- Document semver expectations and migration requirements for facade changes.
- Define callback backpressure and propagate cancellation consistently through
  queued and batched execution.
- Make backend selection explicit when a build contains multiple executable
  backends.

## Correctness and model admission

- Retain independent checkpoint regressions for prompt rendering, tokenization,
  prefill, decode, sampling, and cache reuse.
- Expand CUDA comparison gates against an independent implementation across
  longer prompts, chunk boundaries, and sampled decoding.
- Replace remaining brand-specific fixtures with capability-based coverage for
  RoPE, normalization, activation, bias, sliding-window, and MoE variants.
- Admit additional text checkpoints through generic decoder capabilities and
  tensor schemas rather than model-name dispatch.
- Keep multimodal metadata discoverable; introduce vision or audio execution
  only through separate typed input contracts.

## Runtime and scheduling

- Add chunked prefill with bounded graph and memory growth.
- Complete bounded admission queues, deadlines, cancellation, and fairness
  metrics for concurrent generation.
- Add multi-model residency with an explicit memory-pressure and eviction
  policy.
- Tune decode buckets and prefill/decode graph selection using reproducible
  workloads rather than fixed assumptions about batch size.
- Expose stable queue-delay, TTFT, inter-token latency, cache-hit, active-memory,
  and request-completion metrics.

## Cache evolution

- Add a model-wide immutable prefix-block index with deterministic eviction.
- Test rollback semantics for cancellation and speculative decoding.
- Extend K/V storage beyond the BF16/E4M3 baseline only with explicit scale,
  quality, and long-context correctness gates.
- Consider CPU or disk cache tiers only after device eviction semantics are
  stable.

## Backend work

- Continue auditing Metal prefill and decode for host synchronization, tensor
  materialization, transient allocation, and long-context paged-attention
  regressions.
- Stabilize allocator, graph, kernel-selection, and cache diagnostics emitted by
  both accelerator backends.
- Extend CUDA hardware capability reporting and typed execution planning without
  introducing model-name or ambient-configuration decisions.
- Add checked offline tuning records keyed by hardware and operation geometry,
  with deterministic generic fallbacks.
- Keep experimental CUDA kernels outside automatic selection until independent
  numerical, memory-safety, and full-model performance gates pass.
- Implement additional quantized K/V formats without introducing Python,
  PyTorch, or host synchronization into the generation path.

## Future generation features

- Add speculative decoding after cache snapshot and rollback semantics pass on
  both Metal and CUDA.
- Add multimodal preprocessing and embeddings without coupling text-only model
  execution to multimodal dependencies.

## Observability and release gates

- Stabilize tracing event names, fields, and generation metrics before adding
  exporters.
- Keep OpenTelemetry and Grafana integration optional and owned by consuming
  applications.
- Require nightly rustfmt, strict clippy, tests, examples, and rustdoc before
  releases.
- Keep every Rust source file at or below 250 lines and every runtime path free
  of Python, PyTorch, and Transformers dependencies.
