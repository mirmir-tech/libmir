SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

.PHONY: help docs docs-open examples check-base check-metal check-cuda cuda-checkpoint \
	cuda-quality cuda-dense-gate cuda-profile

CUDA_NVFP4_MODEL ?= $(firstword $(wildcard $(HOME)/.cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/*))
CUDA_QUALITY_MODEL ?= $(firstword $(wildcard $(HOME)/.cache/mirmir/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*))
CUDA_QUALITY_MODES ?= throughput block-fp8-gate-up fp8-int4-gate-up
DENSE_FIXTURE_FAMILY ?=
DENSE_MODEL ?=
DENSE_REFERENCE ?=
DENSE_CUDA_POLICY ?= stable
HOST_OS := $(shell uname -s)
ifeq ($(HOST_OS),Darwin)
DOC_PACKAGES := --workspace --all-features
else
DOC_PACKAGES := -p libmir -p libmir-cuda -p libmir-core -p libmir-models \
	-p libmir-runtime --features libmir/cuda
endif

help: ## Show documentation targets.
	@awk 'BEGIN {FS = ":.*## "; printf "Libmir commands:\n\n"} \
		/^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

docs: ## Build rustdoc for the complete public workspace API.
	@RUSTDOCFLAGS="-D warnings" cargo doc $(DOC_PACKAGES) --no-deps

docs-open: ## Build and open the libmir API documentation.
	@RUSTDOCFLAGS="-D warnings" cargo doc $(DOC_PACKAGES) --no-deps --open

examples: ## Type-check every public libmir example.
	@cargo check -p libmir --examples --features metal

check-base: ## Validate libmir without an accelerator backend.
	@cargo clippy -p libmir --all-targets --no-default-features -- -D warnings
	@cargo test -p libmir --no-default-features

check-metal: ## Validate the complete Metal facade.
	@cargo clippy -p libmir --all-targets --no-default-features \
		--features metal -- -D warnings
	@cargo test -p libmir --no-default-features --features metal

check-cuda: ## Validate the CUDA facade without building Metal.
	@cargo clippy -p libmir-cuda -p libmir --all-targets --no-default-features \
		--features libmir/cuda -- -D warnings
	@cargo test -p libmir-cuda -p libmir --no-default-features --features libmir/cuda

cuda-checkpoint: ## Validate native NVFP4 against the local NVIDIA checkpoint.
	@test -d "$(CUDA_NVFP4_MODEL)" || { printf 'NVFP4 checkpoint not found.\n' >&2; exit 2; }
	@LIBMIR_CUDA_NVFP4_MODEL="$(CUDA_NVFP4_MODEL)" \
		cargo test -p libmir-cuda checkpoint_

cuda-quality: ## Gate experimental CUDA dense modes against stable BF16 logits.
	@test -d "$(CUDA_QUALITY_MODEL)" || { printf 'CUDA quality checkpoint not found.\n' >&2; exit 2; }
	@for mode in $(CUDA_QUALITY_MODES); do \
		LIBMIR_CUDA_GATE_THROUGHPUT_QUALITY="$$mode" \
		LIBMIR_CUDA_QUALITY_MODEL="$(CUDA_QUALITY_MODEL)" \
		cargo test --release -p libmir-cuda checkpoint_throughput_quality_report \
			-- --nocapture || exit $$?; \
	done

cuda-dense-gate: ## Run one configured CUDA dense-checkpoint V2-V4 matrix row.
	@test -n "$(DENSE_FIXTURE_FAMILY)" || { printf 'DENSE_FIXTURE_FAMILY is required.\n' >&2; exit 2; }
	@test -d "$(DENSE_MODEL)" || { printf 'DENSE_MODEL is not a checkpoint directory.\n' >&2; exit 2; }
	@test -f "$(DENSE_REFERENCE)" || { printf 'DENSE_REFERENCE is not a file.\n' >&2; exit 2; }
	@case "$(DENSE_FIXTURE_FAMILY)" in \
		dense) model_env=MIRMIR_DENSE_MODEL; reference_env=MIRMIR_DENSE_REFERENCE ;; \
		dense_and_routed) model_env=MIRMIR_DENSE_AND_ROUTED_MODEL; \
			reference_env=MIRMIR_DENSE_AND_ROUTED_REFERENCE ;; \
		shared_routed) model_env=MIRMIR_SHARED_ROUTED_DENSE_MODEL; \
			reference_env=MIRMIR_SHARED_ROUTED_DENSE_REFERENCE ;; \
		clamped_routed) model_env=MIRMIR_CLAMPED_ROUTED_DENSE_MODEL; \
			reference_env=MIRMIR_CLAMPED_ROUTED_DENSE_REFERENCE ;; \
		*) printf 'Unknown DENSE_FIXTURE_FAMILY: %s\n' "$(DENSE_FIXTURE_FAMILY)" >&2; exit 2 ;; \
	esac; \
	env MIRMIR_DENSE_FIXTURE_FAMILY="$(DENSE_FIXTURE_FAMILY)" \
		MIRMIR_DENSE_CUDA_POLICY="$(DENSE_CUDA_POLICY)" \
		"$$model_env=$(DENSE_MODEL)" "$$reference_env=$(DENSE_REFERENCE)" \
		cargo test --release --features cuda --test dense_checkpoints \
			validates_dense_checkpoint_matrix_v2_to_v4 -- --ignored --nocapture

cuda-profile: ## Profile libmir-owned CUDA inference kernels.
	@cargo run --release -p libmir-cuda --example affine-gemv
	@cargo run --release -p libmir-cuda --example affine-qmm
	@cargo run --release -p libmir-cuda --example selected-affine-pair
	@cargo run --release -p libmir-cuda --example selected-moe
	@cargo run --release -p libmir-cuda --example selected-dense-moe
	@cargo run --release -p libmir-cuda --example output-head
