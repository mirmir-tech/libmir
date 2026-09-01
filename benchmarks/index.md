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
| [Qwen3-4B BF16](qwen3-4b.md) | NVIDIA GX10 (GB10) | 4,556.7 / 4,720.7<br>96.5% · 8/36 wins | 58.70 / 34.07<br>172.3% · 33/36 wins | 2,953 / 2,191<br>1.347× · 0/36 wins |
| [Qwen3.6-35B-A3B NVFP4](qwen3.6-35b-a3b-nvfp4.md) | NVIDIA GX10 (GB10) | 2,939.3 / 2,399.0<br>122.5% · 21/36 wins | 41.60 / 54.10<br>76.9% · 8/36 wins | 3,379 / 3,899<br>0.867× · 21/36 wins |
| [GPT-OSS-20B BF16](gpt-oss-20b-bf16.md) | NVIDIA GX10 (GB10) | 2,827.2 / 2,254.3<br>125.4% · 43/52 wins | 27.74 / 25.53<br>108.7% · 26/52 wins | 6,088 / 7,422<br>0.820× · 44/52 wins |

## Metal

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM | TTFT ms ↓<br>mirmir / MLX-LM |
|:---|:---|---:|---:|---:|
| [Qwen3-4B BF16](qwen3-4b.md) | Apple M3 Max<br>40-core GPU · 64 GiB | 545.0 / 593.4<br>91.8% · 5/26 wins | 22.10 / 27.34<br>80.8% · 6/26 wins | 19,149 / 17,405<br>1.100× · 5/26 wins |

## Single-request Metal diagnostics

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM |
|:---|:---|---:|---:|
| [Qwen3.6-35B-A3B MXFP4](qwen3.6-35b-a3b-nvfp4.md) | Apple M3 Max<br>40-core GPU · 64 GiB | 1,085.09 / 1,069.22<br>101.5% · 2/4 wins | 109.19 / 100.21<br>109.0% · 4/4 wins |

## Partial Metal reference diagnostics

These rows are excluded from the canonical table because the reference process
did not finish. They retain only common positive cells and document the exact
coverage and failure on the linked model page.

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM | TTFT ms ↓<br>mirmir / MLX-LM |
|:---|:---|---:|---:|---:|
| [Gemma 4 26B-A4B IT 8-bit](gemma4-26b-a4b-it-8bit.md) | Apple M3 Max<br>40-core GPU · 64 GiB | 569.0 / 362.2<br>157.1% · 25/26 wins | 40.04 / 35.14<br>113.9% · 21/26 wins | 18,017 / 28,815<br>0.625× · 25/26 wins |
