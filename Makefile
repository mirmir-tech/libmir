SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

.PHONY: help docs docs-open examples check-base check-metal check-cuda cuda-checkpoint cuda-profile

CUDA_NVFP4_MODEL ?= $(firstword $(wildcard $(HOME)/.cache/huggingface/hub/models--nvidia--Gemma-4-26B-A4B-NVFP4/snapshots/*))
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

cuda-profile: ## Profile libmir-owned CUDA inference kernels.
	@cargo run --release -p libmir-cuda --example affine-gemv
	@cargo run --release -p libmir-cuda --example affine-qmm
	@cargo run --release -p libmir-cuda --example selected-affine-pair
	@cargo run --release -p libmir-cuda --example selected-moe
