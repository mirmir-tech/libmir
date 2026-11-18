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
| GPT-OSS-20B context depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 / 65,535 / 100,000 |
| Cache | Prefix caching enabled |
| Latency | API mode |
| Repetitions | 1 warm-up + 3 measured runs per cell |
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
