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
| Driver | 580.159.03 |
| Reference | vLLM 0.25.1 |
| mirmir cells | 36 |
| Reference cells | 36 |

| Metric | mirmir | vLLM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 4,556.7 | 4,720.7 | 96.5% | 8/36 |
| TG tok/s | 58.70 | 34.07 | 172.3% | 33/36 |
| TTFT ms | 2,953 | 2,191 | 1.347× | 0/36 |

| Depth | Phase | PP mirmir | PP vLLM | PP % | TG mirmir | TG vLLM | TG % | TTFT mirmir ms | TTFT vLLM ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 6,759.8 | 7,885.1 | 85.7% | 72.48 | 61.51 | 117.8% | 1,001 | 621 | 1.611× |
| 4,096 | load | 7,124.6 | 7,295.6 | 97.7% | 69.07 | 51.61 | 133.8% | 1,788 | 1,343 | 1.331× |
| 4,096 | reuse | 5,254.6 | 5,716.5 | 91.9% | 68.27 | 51.16 | 133.4% | 1,199 | 886 | 1.354× |
| 8,192 | load | 6,582.1 | 6,471.9 | 101.7% | 62.53 | 37.65 | 166.1% | 3,751 | 2,873 | 1.306× |
| 8,192 | reuse | 4,067.3 | 4,690.1 | 86.7% | 65.74 | 43.90 | 149.7% | 1,539 | 1,110 | 1.386× |
| 16,384 | load | 5,333.1 | 5,057.0 | 105.5% | 51.12 | 23.68 | 215.9% | 9,048 | 7,081 | 1.278× |
| 16,384 | reuse | 3,126.2 | 3,247.0 | 96.3% | 62.42 | 33.13 | 188.4% | 2,022 | 1,484 | 1.362× |
| 32,768 | load | 3,835.0 | 3,494.4 | 109.8% | 35.61 | 12.53 | 284.2% | 24,821 | 20,018 | 1.240× |
| 32,768 | reuse | 1,955.0 | 2,032.7 | 96.2% | 51.86 | 23.44 | 221.2% | 3,035 | 2,352 | 1.291× |

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
