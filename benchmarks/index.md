# Benchmarks

These reports track our work to match and outperform established inference
engines: vLLM on CUDA and MLX-LM on Apple Silicon.

## Methodology

All systems are exercised through their OpenAI-compatible API with
[`llama-benchy`](https://github.com/eugr/llama-benchy)
`0.4.1.dev1+ge9be34457` at revision
[`e9be344578cec17745066b220798b80a0d2686d3`](https://github.com/eugr/llama-benchy/commit/e9be344578cec17745066b220798b80a0d2686d3).

| Setting | Value |
|:---|:---|
| Prompt processing (PP) | 2,048 tokens |
| Text generation (TG) | Exactly 128 tokens |
| Concurrency | 1 / 2 / 5 / 10 |
| Qwen3-4B context depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 |
| Qwen3.6-35B-A3B context depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 |
| Qwen3.8-27B context depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 |
| Gemma 4 26B-A4B context depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 |
| GPT-OSS-20B context depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 / 65,535 / 100,000 |
| Cache | Prefix caching enabled |
| Latency | API mode |
| Repetitions | 1 warm-up + 3 measured runs per cell; Gemma mirmir uses 2 warm-ups |
| Aggregation | Geometric mean over common positive cells |

Only cells shared by mirmir and the reference implementation are included in
the headline comparisons. Higher PP and TG are better; lower TTFT is better.
The model pages below record the exact checkpoint, hardware, reference version,
matrix coverage, and per-depth results.

## CUDA

| Model | Device | PP tok/s ↑<br>mirmir / vLLM | TG tok/s ↑<br>mirmir / vLLM | TTFT ms ↓<br>mirmir / vLLM |
|:---|:---|---:|---:|---:|
| [Qwen3-4B BF16](qwen3-4b.md) | NVIDIA GX10 (GB10) | 4,538.6 / 4,782.4<br>94.9% · 6/36 wins | 34.14 / 32.84<br>103.9% · 26/36 wins | 2,424 / 2,015<br>1.203× · 1/36 wins |
| [Qwen3.6-35B-A3B NVFP4](qwen3.6-35b-a3b-nvfp4.md) | NVIDIA GX10 (GB10) | 4,681.9 / 4,562.6<br>102.6% · 19/36 wins | 75.60 / 79.25<br>95.4% · 7/36 wins | 2,159 / 2,305<br>0.937× · 22/36 wins |
| [Qwen3.8-27B BF16](qwen3.8-27b.md) | NVIDIA GX10 (GB10) | 1,249.1 / 1,067.3<br>117.0% · 36/36 wins | 9.02 / 8.99<br>100.3% · 20/36 wins | 8,001 / 10,218<br>0.783× · 36/36 wins |
| [GPT-OSS-20B BF16](gpt-oss-20b-bf16.md) | NVIDIA GX10 (GB10) | 2,315.1 / 1,159.8<br>199.6% · 34/52 wins | 36.64 / 19.49<br>188.0% · 49/52 wins | 8,285 / 13,525<br>0.613× · 27/52 wins |

## Metal

mirmir 0.6.2 and MLX-LM 0.32.2, both on MLX 0.32.4. MLX-LM runs out of GPU
memory (`Insufficient Memory`, confirmed by a second run) in the deepest
high-concurrency cells, which mirmir completes on the same machine; those
cells are counted in the last column and left out of the ratios.

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM | TTFT ms ↓<br>mirmir / MLX-LM | MLX-LM out of memory |
|:---|:---|---:|---:|---:|---:|
| [Qwen3.6-35B-A3B 4-bit](qwen3.6-35b-a3b-4bit.md) | Apple M5 Pro<br>64 GiB | 2,164.3 / 1,617.8<br>133.8% · 25/34 wins | 144.26 / 105.44<br>136.8% · 33/34 wins | 5,810 / 7,526<br>0.772× · 22/34 wins | 2 of 36 |
| [Gemma 4 26B-A4B IT 4-bit](gemma4-26b-a4b-it-4bit.md) | Apple M5 Pro<br>64 GiB | 1,972.6 / 1,595.4<br>123.6% · 23/32 wins | 105.27 / 87.63<br>120.1% · 32/32 wins | 5,880 / 7,040<br>0.835× · 22/32 wins | 4 of 36 |
| [Qwen3.8-27B 4-bit](qwen3.8-27b-4bit.md) | Apple M5 Pro<br>64 GiB | 383.1 / 351.7<br>108.9% · 3/26 wins | 27.50 / 26.66<br>103.2% · 20/26 wins | 22,907 / 24,902<br>0.920× · 5/26 wins | 8 of 34 |
| [Qwen3-4B BF16](qwen3-4b.md) | Apple M5 Pro<br>64 GiB | 1,662.2 / 1,738.0<br>95.6% · 13/28 wins | 53.44 / 54.67<br>97.7% · 10/28 wins | 6,164 / 5,846<br>1.054× · 14/28 wins | 8 of 36 |

## Single-request Metal diagnostics (M3 Max, earlier releases)

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM |
|:---|:---|---:|---:|
| [Qwen3.6-35B-A3B MXFP4](qwen3.6-35b-a3b-nvfp4.md) | Apple M3 Max<br>40-core GPU · 64 GiB | 1,085.09 / 1,069.22<br>101.5% · 2/4 wins | 109.19 / 100.21<br>109.0% · 4/4 wins |

## Partial Metal reference diagnostics (M3 Max, earlier releases)

These rows are excluded from the canonical table because the reference process
did not finish. They retain only common positive cells and document the exact
coverage and failure on the linked model page.

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM | TTFT ms ↓<br>mirmir / MLX-LM |
|:---|:---|---:|---:|---:|
| [Gemma 4 26B-A4B IT 8-bit](gemma4-26b-a4b-it-8bit.md) | Apple M3 Max<br>40-core GPU · 64 GiB | 569.0 / 362.2<br>157.1% · 25/26 wins | 40.04 / 35.14<br>113.9% · 21/26 wins | 18,017 / 28,815<br>0.625× · 25/26 wins |
