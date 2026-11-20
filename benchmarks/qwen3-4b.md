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
| mirmir | 0.4.0 (2026-09-24) |
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

## Metal — Apple M5 Pro, 64 GiB

| Property | Value |
|:---|:---|
| Machine | Apple M5 Pro |
| Memory | 64 GiB unified |
| macOS | 27.2 |
| mirmir | 0.6.2 (2026-11-15), MLX 0.32.4 |
| Reference | MLX-LM 0.32.2, MLX 0.32.4 |
| mirmir cells | 36 |
| Reference cells | 28 |

| Metric | mirmir | MLX-LM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 1,662.2 | 1,738.0 | 95.6% | 13/28 |
| TG tok/s | 53.44 | 54.67 | 97.7% | 10/28 |
| TTFT ms | 6,164 | 5,846 | 1.054× | 14/28 |

| Depth | Phase | C | PP mirmir | PP MLX | PP % | TG mirmir | TG MLX | TG % | TTFT mirmir ms | TTFT MLX ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 1 | 2,849.3 | 2,694.8 | 105.7% | 31.69 | 31.74 | 99.8% | 753 | 772 | 0.975× |
| 0 | plain | 2 | 2,792.5 | 2,814.8 | 99.2% | 59.83 | 60.02 | 99.7% | 1,473 | 1,461 | 1.008× |
| 0 | plain | 5 | 2,756.8 | 2,715.3 | 101.5% | 128.09 | 132.91 | 96.4% | 3,703 | 3,792 | 0.977× |
| 0 | plain | 10 | 2,423.8 | 2,448.3 | 99.0% | 202.39 | 208.54 | 97.1% | 8,375 | 8,210 | 1.020× |
| 4,096 | load | 1 | 2,763.6 | 2,595.6 | 106.5% | 29.97 | 30.26 | 99.0% | 1,521 | 1,592 | 0.955× |
| 4,096 | load | 2 | 2,435.8 | 2,331.5 | 104.5% | 55.07 | 55.63 | 99.0% | 3,377 | 3,532 | 0.956× |
| 4,096 | load | 5 | 2,344.8 | 2,356.5 | 99.5% | 107.64 | 103.42 | 104.1% | 8,676 | 8,744 | 0.992× |
| 4,096 | load | 10 | 2,439.5 | 2,260.2 | 107.9% | 157.08 | 163.39 | 96.1% | 16,736 | 17,727 | 0.944× |
| 4,096 | reuse | 1 | 1,748.9 | 2,004.5 | 87.2% | 28.79 | 29.09 | 99.0% | 1,208 | 1,031 | 1.172× |
| 4,096 | reuse | 2 | 1,602.1 | 1,843.7 | 86.9% | 51.79 | 52.83 | 98.0% | 2,558 | 2,222 | 1.151× |
| 4,096 | reuse | 5 | 1,643.2 | 1,911.9 | 85.9% | 101.07 | 103.26 | 97.9% | 6,236 | 5,362 | 1.163× |
| 4,096 | reuse | 10 | 1,604.4 | 1,918.8 | 83.6% | 129.25 | 139.33 | 92.8% | 12,466 | 10,336 | 1.206× |
| 8,192 | load | 1 | 2,390.2 | 2,270.7 | 105.3% | 28.01 | 27.92 | 100.3% | 3,474 | 3,630 | 0.957× |
| 8,192 | load | 2 | 2,078.5 | 1,958.9 | 106.1% | 48.73 | 49.93 | 97.6% | 7,896 | 8,379 | 0.942× |
| 8,192 | load | 5 | 2,128.0 | 2,064.3 | 103.1% | 82.36 | 91.30 | 90.2% | 19,164 | 19,896 | 0.963× |
| 8,192 | load | 10 | 2,181.3 | 1,875.5 | 116.3% | 90.85 | 82.39 | 110.3% | 37,461 | 39,646 | 0.945× |
| 8,192 | reuse | 1 | 1,264.2 | 1,541.7 | 82.0% | 27.05 | 26.99 | 100.2% | 1,668 | 1,341 | 1.244× |
| 8,192 | reuse | 2 | 1,185.2 | 1,424.0 | 83.2% | 45.68 | 47.30 | 96.6% | 3,456 | 2,876 | 1.202× |
| 8,192 | reuse | 5 | 1,256.0 | 1,465.4 | 85.7% | 82.20 | 83.20 | 98.8% | 8,153 | 6,994 | 1.166× |
| 8,192 | reuse | 10 | 1,203.1 | 1,455.6 | 82.7% | 80.40 | 100.66 | 79.9% | 16,560 | 13,569 | 1.220× |
| 16,384 | load | 1 | 1,780.4 | 1,605.8 | 110.9% | 24.96 | 24.35 | 102.5% | 9,273 | 10,249 | 0.905× |
| 16,384 | load | 2 | 1,712.2 | 1,628.4 | 105.1% | 41.02 | 40.60 | 101.0% | 19,162 | 20,147 | 0.951× |
| 16,384 | load | 5 | 1,759.0 | 1,606.7 | 109.5% | 48.33 | 63.79 | 75.8% | 46,434 | 51,036 | 0.910× |
| 16,384 | load | 10 | 1,421.7 | OOM | – | 32.00 | OOM | – | 85,111 | OOM | – |
| 16,384 | reuse | 1 | 778.2 | 999.2 | 77.9% | 24.15 | 23.50 | 102.8% | 2,688 | 2,060 | 1.305× |
| 16,384 | reuse | 2 | 816.8 | 991.4 | 82.4% | 39.40 | 39.01 | 101.0% | 5,015 | 4,132 | 1.214× |
| 16,384 | reuse | 5 | 873.6 | 997.7 | 87.6% | 56.16 | 59.47 | 94.4% | 11,722 | 10,263 | 1.142× |
| 16,384 | reuse | 10 | 284.2 | OOM | – | 22.37 | OOM | – | 32,525 | OOM | – |
| 32,768 | load | 1 | 1,299.0 | 1,104.5 | 117.6% | 20.39 | 19.04 | 107.1% | 25,289 | 29,693 | 0.852× |
| 32,768 | load | 2 | 1,183.1 | OOM | – | 30.26 | OOM | – | 55,441 | OOM | – |
| 32,768 | load | 5 | 1,173.4 | OOM | – | 12.78 | OOM | – | 105,869 | OOM | – |
| 32,768 | load | 10 | 1,121.6 | OOM | – | 6.13 | OOM | – | 178,105 | OOM | – |
| 32,768 | reuse | 1 | 495.0 | 645.8 | 76.6% | 19.94 | 18.59 | 107.3% | 4,195 | 3,182 | 1.318× |
| 32,768 | reuse | 2 | 503.9 | OOM | – | 27.02 | OOM | – | 8,129 | OOM | – |
| 32,768 | reuse | 5 | 121.7 | OOM | – | 10.79 | OOM | – | 43,481 | OOM | – |
| 32,768 | reuse | 10 | 74.5 | OOM | – | 5.01 | OOM | – | 132,707 | OOM | – |

OOM marks cells where MLX-LM's server stopped with `[METAL] Command buffer
execution failed: Insufficient Memory` (`generation thread died`); a second
run on 2026-11-16 failed the same way. mirmir completed them on the same
machine. They are left out of the ratios above, which cover only cells both
engines finished (8 of 36 cells have no MLX-LM result).
