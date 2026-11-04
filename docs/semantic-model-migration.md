# Semantic model migration

## Goal

Model execution is selected from a typed description of computation and physical weight
bindings. Repository names, `model_type`, `architectures`, and model-family enums must not select
backend code.

The optional checkpoint sidecar is `mir-model-spec.toml`. It contains a complete, versioned
`SemanticModelSpec`, especially semantics that Hugging Face configuration does not express. A
checkpoint without the sidecar is compiled from `config.json`, built-in import defaults,
and its tensor catalog. The precedence is: complete sidecar, explicit configuration,
built-in defaults, then generic tensor-based discovery.

## Built-in defaults and overrides

The checkpoint importer supplies Gemma 4 defaults for `gemma4`, `gemma4_text`, and
`gemma4_unified` text configurations, including nested `text_config`:

| Operation | Default |
|---|---|
| Q/K/V normalization | `query_key_value_rms` (V uses unweighted RMSNorm) |
| Attention scale | `1.0`, for both local and global heads |
| Embedding multiplier | `sqrt(hidden_size)` |

These preserve the computation used in our existing Gemma 4 26B Metal reference
and benchmark runs (`validation/references/dense-and-routed-gemma4-26b-a4b.toml`,
`benchmarks/gemma4-26b-a4b-it-8bit.md`). The CUDA 12B BF16 review reproduction
[confirmed the missing defaults](https://github.com/mirmir-tech/libmir/pull/12#discussion_r4468480847):
with hidden size 3840 the multiplier is 61.96773353931867, and the corrected
contract produces a normal greeting instead of repeated digits. This is provenance,
not a fresh full-checkpoint qualification of every release build.

Explicit `attention_scale` (or `attention_multiplier`), `embedding_multiplier`, and
`qk_normalization` in `config.json` override the import defaults. Export and edit
`mir-model-spec.toml` with `cargo run --example model-spec -- <checkpoint>` when
per-layer or complete semantic overrides are needed. A sidecar replaces the entire
contract; it is not a partial patch. Invalid values fail validation.

Model identity is used only at this import boundary to supply omitted semantics;
backends still select execution from the resulting typed operations. Unknown
identities retain generic discovery and may require an explicit spec. Renaming a
checkpoint directory cannot change defaults. Renaming a known `model_type` without
exporting its resolved spec can change omitted defaults.

Metal's compatibility runners reject unsupported normalization, attention-scale,
or embedding-multiplier overrides during admission and again before loading
weights. In particular, dense-and-routed execution requires Q/K/V RMSNorm, scale 1,
and sqrt(hidden_size) embeddings. CUDA consumes the declared values through its
semantic lowering. An override is never a promise that every backend supports it.

## Invariants

- `SemanticModelSpec` describes computation only. It does not contain checkpoint tensor names,
  quantization container details, or accelerator choices.
- `WeightBindingPlan` maps logical roles to physical tensors and storage formats. It does not
  choose model behavior.
- Metal and CUDA lower semantic operations according to backend capabilities. A fused kernel is
  selected by operation, storage, shape, and hardware predicates, never by model identity.
- Known import defaults fill omitted semantics; unsupported operations and invalid explicit
  semantics are errors. External specs can declare semantics absent from generic discovery.
- Ambiguous tensor layouts are rejected instead of resolved by rule order.
- `ModelFamily` is not part of the runtime or presentation API. `model_type` can select
  an import default profile; it does not select backend code.

## Phase 1: semantic contract and compatibility bridge

- Add the typed semantic model, attention, position, feed-forward, and routing structures.
- Add TOML serialization, parsing, schema-version validation, and `ModelLayout` discovery.
- Add `WeightBindingPlan` with dense, affine-quantized, and block-quantized storage descriptions.
- Compile existing `DecoderConfig` plus tensor evidence into `SemanticModelSpec`.
- Derive backend lowering from semantic operations.
- Remove `model_type` inference from decoder features and model-family sampling defaults.

Exit criteria:

- Official MXFP4 and MLX affine checkpoints produce equal semantic specs and unequal bindings.
- Explicit complete specs remain independent of `model_type` and `architectures`.
- Built-in defaults are tested separately from generic tensor discovery.
- Existing model tests continue to pass through the compatibility bridge.

## Phase 2: canonical tensor roles

- Replace readiness-only tensor schemas with required logical roles and shape constraints.
- Express separate/fused QKV, separate/interleaved gate-up, expert stacking, transposition, scales,
  and biases as binding transforms.
- Validate every bound tensor against dimensions derived from `SemanticModelSpec`.
- Resolve all candidate binding grammars and require exactly one complete solution.
- Attach quantization to individual bindings instead of `ModelManifest::quantization`.

Exit criteria:

- Native and converted checkpoints enter a shared loader after binding.
- Adding a new storage conversion does not add a decoder or model variant.

## Phase 3: operator lowering on Metal

- Introduce a shared decoder loop over lowered layer operations.
- Lower normalization, rotary position encoding, softmax attention, linear attention, dense FFN,
  routed FFN, and shared experts independently.
- Rename model-labelled kernels and modules after their operation and storage contracts.
- Move fusion selection into a Metal capability planner.
- Migrate existing GPT-OSS and hybrid runners one operation at a time, keeping numerical tests.

Exit criteria:

- Metal has no `DecoderModel::GptOss` or `DecoderModel::HybridLinearMoe` branch.
- Both GPT-OSS checkpoint layouts use one semantic decoder and differ only in weight bindings.

## Phase 4: operator lowering on CUDA

- Mirror the semantic lowering boundary used by Metal.
- Turn fixed geometry checks into kernel admission predicates.
- Provide composed fallbacks when a fused kernel rejects a valid geometry.
- Unify model sessions and K/V cache allocation around per-layer mixer specifications.

Exit criteria:

- CUDA has no model-specific execution/session enum variants.
- Unsupported cases report the exact operation, storage format, geometry, and missing capability.

## Phase 5: remove the compatibility taxonomy

- Delete `DecoderArchetype`, `AttentionFeature`, `FeedForwardFeature`, and all `uses_*_stack`
  helpers.
- Remove family-based chat fallbacks after tokenizer templates or explicit TOML protocol sections
  cover supported checkpoints.
- Keep family detection only if the application still needs it for presentation; otherwise remove
  it from manifests and telemetry too.

Exit criteria:

- Execution and session boundaries expose semantic operations rather than model-labelled variants.
- Chat fallback, manifests, traces, and inspection output contain no family-derived decisions.
- Once a complete spec is exported, `model_type`, `architectures`, and repository paths
  can change without changing that execution contract.

## Verification matrix

Every supported semantic configuration is tested against:

- config-derived and TOML-derived semantic specs;
- native and converted weight bindings;
- Metal and CUDA capability lowering;
- prefill, single-token decode, batched decode, and cache rollover;
- numerical reference logits and deterministic token digests;
- renamed or misleading `model_type`, `architectures`, and repository paths.

Synthetic config/sidecar equivalence and misleading-metadata coverage run in the regular test
suite. Real-checkpoint logits and token digests remain opt-in because they require the corresponding
checkpoint environment variables and accelerator hardware.
