# Benchmarks

These reports track our work to match and outperform established inference
engines: vLLM and llama.cpp on CUDA, and MLX-LM and llama.cpp on Apple Silicon.

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

## CUDA — GGUF

mirmir serving llama.cpp's GGUF files against `llama-server` on the same
files, llama.cpp `b1b770d`, mirmir with BF16 K/V and llama.cpp with its
default F16. Depth 0 and 4,096 at concurrency 1 / 2 / 10 (Qwen3-8B Q4_K_M:
five depths, four concurrencies). The first eight models are the Metal GGUF
matrix; the [page](gguf-cuda.md) records the methodology, perplexity parity,
the cells mirmir did not complete and every cell.

| Model | Device | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|:---|:---|---:|---:|---:|
| [Qwen3-30B-A3B Q4_K_M](gguf-cuda.md#qwen3-30b-a3b-q4_k_m) | NVIDIA GX10 (GB10) | 4,182.5 / 2,734.8<br>152.9% · 6/6 wins | 118.36 / 94.81<br>124.8% · 6/6 wins | 1,340 / 1,887<br>0.710× · 6/6 wins |
| [Qwen3-30B-A3B Q8_0](gguf-cuda.md#qwen3-30b-a3b-q8_0) | NVIDIA GX10 (GB10) | 3,918.4 / 2,058.9<br>190.3% · 6/6 wins | 80.38 / 65.48<br>122.8% · 6/6 wins | 1,427 / 2,498<br>0.571× · 6/6 wins |
| [GPT-OSS-20B MXFP4](gguf-cuda.md#gpt-oss-20b-mxfp4) | NVIDIA GX10 (GB10) | 5,752.8 / 3,693.3<br>155.8% · 6/6 wins | 135.04 / 112.04<br>120.5% · 5/6 wins | 1,003 / 1,471<br>0.682× · 6/6 wins |
| [Gemma 3 12B IT Q4_K_M](gguf-cuda.md#gemma-3-12b-it-q4_k_m) | NVIDIA GX10 (GB10) | 2,595.5 / 1,427.3<br>181.9% · 6/6 wins | 48.48 / 40.98<br>118.3% · 6/6 wins | 1,890 / 3,443<br>0.549× · 6/6 wins |
| [Qwen3-8B Q4_K_M](gguf-cuda.md#qwen3-8b-q4_k_m) | NVIDIA GX10 (GB10) | 2,782.0 / 1,119.4<br>248.5% · 20/20 wins | 61.61 / 40.34<br>152.7% · 17/20 wins | 2,319 / 5,069<br>0.458× · 20/20 wins |
| [Qwen3-8B Q8_0](gguf-cuda.md#qwen3-8b-q8_0) | NVIDIA GX10 (GB10) | 3,638.4 / 2,194.9<br>165.8% · 6/6 wins | 56.82 / 47.79<br>118.9% · 5/6 wins | 1,556 / 2,402<br>0.648× · 6/6 wins |
| [Llama 3.1 8B Instruct Q4_K_M](gguf-cuda.md#llama-31-8b-instruct-q4_k_m) | NVIDIA GX10 (GB10) | 3,880.0 / 2,662.6<br>145.7% · 6/6 wins | 75.93 / 65.58<br>115.8% · 6/6 wins | 1,256 / 1,793<br>0.701× · 6/6 wins |
| [Mistral 7B Instruct v0.3 Q4_K_M](gguf-cuda.md#mistral-7b-instruct-v03-q4_k_m) | NVIDIA GX10 (GB10) | 4,059.8 / 2,644.3<br>153.5% · 6/6 wins | 79.78 / 68.90<br>115.8% · 6/6 wins | 1,202 / 1,829<br>0.657× · 6/6 wins |
| [Qwen3.6-35B-A3B UD-Q4_K_M](gguf-cuda.md#qwen36-35b-a3b-ud-q4_k_m) | NVIDIA GX10 (GB10) | 3,460.8 / 2,164.4<br>159.9% · 5/6 wins | 108.02 / 84.60<br>127.7% · 6/6 wins | 1,423 / 2,288<br>0.622× · 5/6 wins |
| [Gemma 4 26B-A4B IT UD-Q4_K_M](gguf-cuda.md#gemma-4-26b-a4b-it-ud-q4_k_m) | NVIDIA GX10 (GB10) | 3,927.3 / 2,051.9<br>191.4% · 6/6 wins | 66.28 / 77.13<br>85.9% · 2/6 wins | 1,483 / 2,652<br>0.559× · 6/6 wins |
| [GLM-4.5-Air Q2_K](gguf-cuda.md#glm-45-air-q2_k) | NVIDIA GX10 (GB10) | 1,229.2 / 758.0<br>162.2% · 2/2 wins | 38.53 / 37.09<br>103.9% · 2/2 wins | 2,357 / 3,858<br>0.611× · 2/2 wins |
| [Llama 4 Scout 17B-16E Q2_K](gguf-cuda.md#llama-4-scout-17b-16e-q2_k) | NVIDIA GX10 (GB10) | 195.0 / 629.8<br>31.0% · 0/5 wins | 29.54 / 32.43<br>91.1% · 2/5 wins | 20,312 / 6,288<br>3.230× · 0/5 wins |
| [Nemotron Nano 9B v2 Q4_K_M](gguf-cuda.md#nemotron-nano-9b-v2-q4_k_m) | NVIDIA GX10 (GB10) | 2,151.7 / 1,649.2<br>130.5% · 5/6 wins | 61.19 / 49.73<br>123.0% · 4/6 wins | 2,282 / 2,998<br>0.761× · 5/6 wins |
| [Mistral Small 4 119B UD-Q4_K_M](gguf-cuda.md#mistral-small-4-119b-ud-q4_k_m) | NVIDIA GX10 (GB10) | 1,962.7 / 627.2<br>313.0% · 6/6 wins | 54.93 / 32.52<br>168.9% · 6/6 wins | 2,169 / 6,240<br>0.348× · 6/6 wins |

## Metal

mirmir 0.7 candidate (libmir `891a79b`, 2026-11-23) and MLX-LM 0.32.2, both on
MLX 0.32.4. MLX-LM runs out of GPU memory (`Insufficient Memory`, confirmed by
a second run) in the deepest high-concurrency cells, which mirmir completes on
the same machine; those cells are counted in the last column and left out of
the ratios.

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM | TTFT ms ↓<br>mirmir / MLX-LM | MLX-LM out of memory |
|:---|:---|---:|---:|---:|---:|
| [Qwen3.6-35B-A3B 4-bit](qwen3.6-35b-a3b-4bit.md) | Apple M5 Pro<br>64 GiB | 2,140.0 / 1,617.8<br>132.3% · 23/34 wins | 143.40 / 105.44<br>136.0% · 33/34 wins | 5,876 / 7,526<br>0.781× · 21/34 wins | 2 of 36 |
| [Gemma 4 26B-A4B IT 4-bit](gemma4-26b-a4b-it-4bit.md) | Apple M5 Pro<br>64 GiB | 1,903.3 / 1,595.4<br>119.3% · 21/32 wins | 104.18 / 87.63<br>118.9% · 30/32 wins | 6,075 / 7,040<br>0.863× · 19/32 wins | 4 of 36 |
| [Qwen3.8-27B 4-bit](qwen3.8-27b-4bit.md) | Apple M5 Pro<br>64 GiB | 349.2 / 351.7<br>99.3% · 3/26 wins | 26.15 / 26.66<br>98.1% · 8/26 wins | 25,157 / 24,902<br>1.010× · 3/26 wins | 10 of 36 |
| [Qwen3-4B BF16](qwen3-4b.md) | Apple M5 Pro<br>64 GiB | 1,725.1 / 1,738.0<br>99.3% · 13/28 wins | 54.76 / 54.67<br>100.2% · 11/28 wins | 5,947 / 5,846<br>1.017× · 13/28 wins | 8 of 36 |

## Metal — GGUF

mirmir serving llama.cpp's GGUF files against `llama-server` on the same
files, llama.cpp `b1b770d90`, BF16 K/V on both. Depth 0 and 4,096 at
concurrency 1 / 2 / 10 (Qwen3-8B Q4_K_M: five depths, four concurrencies).
Measured with the 0.7 candidate (libmir `891a79b`, 2026-11-23); the
[page](gguf-metal.md) records the methodology, perplexity parity and every
cell.

| Model | Device | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|:---|:---|---:|---:|---:|
| [Qwen3-30B-A3B Q4_K_M](gguf-metal.md#qwen3-30b-a3b-q4_k_m) | Apple M5 Pro<br>64 GiB | 1,784.0 / 1,196.2<br>149.1% · 6/6 wins | 94.83 / 71.09<br>133.4% · 5/6 wins | 3,135 / 4,283<br>0.732× · 6/6 wins |
| [Qwen3-30B-A3B Q8_0](gguf-metal.md#qwen3-30b-a3b-q8_0) | Apple M5 Pro<br>64 GiB | 1,647.7 / 1,104.5<br>149.2% · 6/6 wins | 78.83 / 53.47<br>147.4% · 6/6 wins | 3,349 / 4,639<br>0.722× · 6/6 wins |
| [GPT-OSS-20B MXFP4](gguf-metal.md#gpt-oss-20b-mxfp4) | Apple M5 Pro<br>64 GiB | 2,458.7 / 1,590.8<br>154.6% · 6/6 wins | 112.69 / 96.28<br>117.0% · 2/6 wins | 2,331 / 3,433<br>0.679× · 6/6 wins |
| [Gemma 3 12B IT Q4_K_M](gguf-metal.md#gemma-3-12b-it-q4_k_m) | Apple M5 Pro<br>64 GiB | 868.9 / 640.4<br>135.7% · 6/6 wins | 48.89 / 33.27<br>146.9% · 5/6 wins | 6,338 / 7,544<br>0.840× · 5/6 wins |
| [Qwen3-8B Q4_K_M](gguf-metal.md#qwen3-8b-q4_k_m) | Apple M5 Pro<br>64 GiB | 926.4 / 639.1<br>145.0% · 17/17 wins | 60.92 / 42.27<br>144.1% · 15/17 wins | 5,945 / 7,993<br>0.744× · 17/17 wins |
| [Qwen3-8B Q8_0](gguf-metal.md#qwen3-8b-q8_0) | Apple M5 Pro<br>64 GiB | 1,208.8 / 1,058.3<br>114.2% · 5/6 wins | 58.04 / 43.94<br>132.1% · 5/6 wins | 4,641 / 4,884<br>0.950× · 3/6 wins |
| [Llama 3.1 8B Instruct Q4_K_M](gguf-metal.md#llama-31-8b-instruct-q4_k_m) | Apple M5 Pro<br>64 GiB | 1,369.4 / 1,038.6<br>131.9% · 6/6 wins | 79.78 / 53.70<br>148.6% · 4/6 wins | 4,097 / 4,601<br>0.890× · 4/6 wins |
| [Mistral 7B Instruct v0.3 Q4_K_M](gguf-metal.md#mistral-7b-instruct-v03-q4_k_m) | Apple M5 Pro<br>64 GiB | 1,397.2 / 1,047.9<br>133.3% · 6/6 wins | 82.79 / 56.41<br>146.8% · 4/6 wins | 3,991 / 4,545<br>0.878× · 4/6 wins |

## Single-request Metal diagnostics (M3 Max, earlier releases)

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM |
|:---|:---|---:|---:|
| Qwen3.6-35B-A3B MXFP4 | Apple M3 Max<br>40-core GPU · 64 GiB | 1,085.09 / 1,069.22<br>101.5% · 2/4 wins | 109.19 / 100.21<br>109.0% · 4/4 wins |

## Partial Metal reference diagnostics (M3 Max, earlier releases)

These rows are excluded from the canonical table because the reference process
did not finish; they retain only common positive cells.

| Model | Device | PP tok/s ↑<br>mirmir / MLX-LM | TG tok/s ↑<br>mirmir / MLX-LM | TTFT ms ↓<br>mirmir / MLX-LM |
|:---|:---|---:|---:|---:|
| Gemma 4 26B-A4B IT 8-bit | Apple M3 Max<br>40-core GPU · 64 GiB | 569.0 / 362.2<br>157.1% · 25/26 wins | 40.04 / 35.14<br>113.9% · 21/26 wins | 18,017 / 28,815<br>0.625× · 25/26 wins |
