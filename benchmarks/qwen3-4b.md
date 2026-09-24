# Qwen3-4B BF16

## Model

| Property | Value |
|:---|:---|
| Hugging Face | `Qwen/Qwen3-4B` |
| Snapshot | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Parameters | 4B |
| Architecture | Dense decoder-only Transformer |
| Checkpoint | BF16 SafeTensors |
| Weight size | 7.49 GiB |
| Layers | 36 |
| Hidden size | 2,560 |
| SwiGLU size | 9,728 |
| Q heads | 32 |
| KV heads | 8 |
| Head dimension | 128 |
| Context | 40,960 |

## Matrix

| Property | Value |
|:---|:---|
| Harness | `llama-benchy` `0.4.1.dev1+ge9be34457` |
| Harness revision | `e9be344578cec17745066b220798b80a0d2686d3` |
| PP | 2,048 |
| TG | 128 |
| Concurrency | 1 / 2 / 5 / 10 |
| Depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 |
| Cache | Prefix |
| Latency | API |
| Runs | 1 warm-up + 3 measured |

## CUDA — NVIDIA GX10 (GB10)

| Property | Value |
|:---|:---|
| Machine | NVIDIA GX10 (GB10) |
| Driver | 580.173.02 |
| mirmir | 0.3.1 (2026-09-24) |
| Reference | vLLM 0.29.0 |
| K/V cache | BF16 on both engines; mirmir blocks of 16 tokens sized from the memory estimate (the graph runtime keeps its pages) |
| Token budget | mirmir 1,024 / vLLM 2,096 |
| mirmir cells | 36 |
| Reference cells | 36 |

| Metric | mirmir | vLLM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 4,538.6 | 4,782.4 | 94.9% | 6/36 |
| TG tok/s | 34.14 | 32.84 | 103.9% | 26/36 |
| TTFT ms | 2,424 | 2,015 | 1.203× | 1/36 |

| Depth | Phase | PP mirmir | PP vLLM | PP % | TG mirmir | TG vLLM | TG % | TTFT mirmir ms | TTFT vLLM ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 7,881.7 | 8,234.5 | 95.7% | 66.26 | 62.07 | 106.7% | 695 | 561 | 1.239× |
| 4,096 | load | 7,273.8 | 7,697.6 | 94.5% | 53.83 | 50.66 | 106.3% | 1,422 | 1,195 | 1.189× |
| 4,096 | reuse | 5,426.4 | 6,209.3 | 87.4% | 53.01 | 50.03 | 105.9% | 1,007 | 735 | 1.369× |
| 8,192 | load | 6,464.1 | 6,629.2 | 97.5% | 38.20 | 36.96 | 103.4% | 2,974 | 2,729 | 1.090× |
| 8,192 | reuse | 4,306.1 | 4,918.5 | 87.5% | 44.25 | 42.22 | 104.8% | 1,267 | 924 | 1.372× |
| 16,384 | load | 4,980.2 | 5,071.7 | 98.2% | 23.01 | 23.03 | 99.9% | 7,299 | 6,949 | 1.050× |
| 16,384 | reuse | 3,048.7 | 3,311.8 | 92.1% | 33.44 | 32.10 | 104.1% | 1,776 | 1,335 | 1.331× |
| 32,768 | load | 3,360.2 | 3,220.1 | 104.4% | 11.48 | 11.32 | 101.5% | 20,962 | 20,847 | 1.006× |
| 32,768 | reuse | 1,849.5 | 1,885.4 | 98.1% | 22.30 | 21.65 | 103.0% | 2,839 | 2,274 | 1.248× |

Both engines serve ten sequences with prefix caching enabled; every one of the
486 measured responses per engine contains exactly 128 tokens. The graph
runtime now completes prefill rows in arrival order and runs decode rows in
the same packed forward as prefill chunks (AD-031), as vLLM does: first
tokens arrive request by request instead of for the whole cohort at once, so
TTFT fell from 1.70× to 1.20× of vLLM and prompt processing rose from 86% to
95%. Decode now overlaps other requests' prefill, which lowers the aggregate
decode rate at concurrency (the table published earlier the same day showed
TG at 133% of vLLM with every request waiting for the slowest prefill);
estimated request completion time is shorter in every measured cell. mirmir
matches vLLM on load rows (95–104% PP) and trails on reuse prompt processing
(87–98%). A CUDA prefill wave holds every admitted row (AD-030).

## Metal — Apple M3 Max, 40-core GPU, 64 GiB

| Property | Value |
|:---|:---|
| Machine | Apple M3 Max |
| GPU | 40 cores |
| Memory | 64 GiB unified |
| Reference | MLX-LM 0.31.3 |
| mirmir cells | 36 |
| Reference cells | 26 |

| Metric | mirmir | MLX-LM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 545.0 | 593.4 | 91.8% | 5/26 |
| TG tok/s | 22.10 | 27.34 | 80.8% | 6/26 |
| TTFT ms | 19,149 | 17,405 | 1.100× | 5/26 |

| Depth | Phase | Cells | PP mirmir | PP MLX | PP % | TG mirmir | TG MLX | TG % | TTFT mirmir ms | TTFT MLX ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 4 | 964.0 | 934.8 | 103.1% | 47.26 | 43.21 | 109.4% | 6,748 | 6,931 | 0.974× |
| 4,096 | load | 4 | 714.8 | 721.4 | 99.1% | 27.21 | 31.23 | 87.1% | 18,179 | 17,948 | 1.013× |
| 4,096 | reuse | 4 | 479.6 | 571.1 | 84.0% | 23.98 | 29.10 | 82.4% | 13,849 | 11,318 | 1.224× |
| 8,192 | load | 4 | 621.4 | 669.9 | 92.8% | 20.76 | 26.09 | 79.6% | 41,738 | 37,756 | 1.105× |
| 8,192 | reuse | 4 | 403.2 | 482.8 | 83.5% | 18.44 | 26.67 | 69.2% | 16,102 | 13,347 | 1.206× |
| 16,384 | load | 3 | 507.2 | 530.8 | 95.5% | 13.46 | 18.46 | 73.0% | 69,636 | 66,510 | 1.047× |
| 16,384 | reuse | 3 | 283.7 | 328.6 | 86.3% | 12.37 | 18.68 | 66.2% | 15,573 | 13,436 | 1.159× |



