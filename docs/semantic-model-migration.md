# Semantic model migration

## Goal

Model execution is selected from a typed description of computation and physical weight
bindings. Repository names, `model_type`, `architectures`, and model-family enums must not select
backend code.

The optional checkpoint sidecar is `mir-model-spec.toml`. It contains a complete, versioned
`SemanticModelSpec`, especially semantics that Hugging Face configuration does not express. A
checkpoint without the sidecar is compiled from `config.json` and its tensor catalog.

## Invariants

- `SemanticModelSpec` describes computation only. It does not contain checkpoint tensor names,
  quantization container details, or accelerator choices.
- `WeightBindingPlan` maps logical roles to physical tensors and storage formats. It does not
  choose model behavior.
- Metal and CUDA lower semantic operations according to backend capabilities. A fused kernel is
  selected by operation, storage, shape, and hardware predicates, never by model identity.
- Missing semantic information is an error unless a versioned TOML sidecar provides it.
- Ambiguous tensor layouts are rejected instead of resolved by rule order.
- `ModelFamily` is not part of the runtime or presentation API. Raw `model_type` and
  `architectures` values are retained only as checkpoint metadata.

## Phase 1: semantic contract and compatibility bridge

- Add the typed semantic model, attention, position, feed-forward, and routing structures.
- Add TOML serialization, parsing, schema-version validation, and `ModelLayout` discovery.
- Add `WeightBindingPlan` with dense, affine-quantized, and block-quantized storage descriptions.
- Compile existing `DecoderConfig` plus tensor evidence into `SemanticModelSpec`.
- Derive backend lowering from semantic operations.
- Remove `model_type` inference from decoder features and model-family sampling defaults.

Exit criteria:

- Official MXFP4 and MLX affine checkpoints produce equal semantic specs and unequal bindings.
- Changing `model_type` or `architectures` cannot change semantic discovery.
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
- `model_type`, `architectures`, and repository paths can change without changing the execution
  contract.

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
