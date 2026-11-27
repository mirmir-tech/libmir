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
| macOS | 27.2.1 |
| mirmir | 0.7 candidate (libmir `891a79b`, 2026-11-23), MLX 0.32.4 |
| Reference | MLX-LM 0.32.2, MLX 0.32.4 |
| mirmir cells | 36 |
| Reference cells | 28 |

| Metric | mirmir | MLX-LM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 1,725.1 | 1,738.0 | 99.3% | 13/28 |
| TG tok/s | 54.76 | 54.67 | 100.2% | 11/28 |
| TTFT ms | 5,947 | 5,846 | 1.017× | 13/28 |

| Depth | Phase | C | PP mirmir | PP MLX | PP % | TG mirmir | TG MLX | TG % | TTFT mirmir ms | TTFT MLX ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 1 | 2,682.5 | 2,694.8 | 99.5% | 31.65 | 31.74 | 99.7% | 800 | 772 | 1.036× |
| 0 | plain | 2 | 2,683.2 | 2,814.8 | 95.3% | 59.66 | 60.02 | 99.4% | 1,533 | 1,461 | 1.049× |
| 0 | plain | 5 | 2,555.6 | 2,715.3 | 94.1% | 129.11 | 132.91 | 97.1% | 4,023 | 3,792 | 1.061× |
| 0 | plain | 10 | 2,582.8 | 2,448.3 | 105.5% | 185.85 | 208.54 | 89.1% | 7,305 | 8,210 | 0.890× |
| 4,096 | load | 1 | 2,816.3 | 2,595.6 | 108.5% | 29.86 | 30.26 | 98.7% | 1,494 | 1,592 | 0.938× |
| 4,096 | load | 2 | 2,705.4 | 2,331.5 | 116.0% | 54.83 | 55.63 | 98.6% | 3,038 | 3,532 | 0.860× |
| 4,096 | load | 5 | 2,451.5 | 2,356.5 | 104.0% | 113.21 | 103.42 | 109.5% | 8,384 | 8,744 | 0.959× |
| 4,096 | load | 10 | 2,327.1 | 2,260.2 | 103.0% | 171.00 | 163.39 | 104.7% | 17,662 | 17,727 | 0.996× |
| 4,096 | reuse | 1 | 1,965.8 | 2,004.5 | 98.1% | 28.74 | 29.09 | 98.8% | 1,081 | 1,031 | 1.048× |
| 4,096 | reuse | 2 | 1,816.2 | 1,843.7 | 98.5% | 51.76 | 52.83 | 98.0% | 2,266 | 2,222 | 1.020× |
| 4,096 | reuse | 5 | 1,626.0 | 1,911.9 | 85.0% | 100.52 | 103.26 | 97.3% | 6,299 | 5,362 | 1.175× |
| 4,096 | reuse | 10 | 1,550.9 | 1,918.8 | 80.8% | 136.93 | 139.33 | 98.3% | 13,063 | 10,336 | 1.264× |
| 8,192 | load | 1 | 2,550.7 | 2,270.7 | 112.3% | 28.07 | 27.92 | 100.5% | 3,255 | 3,630 | 0.897× |
| 8,192 | load | 2 | 2,380.8 | 1,958.9 | 121.5% | 49.75 | 49.93 | 99.6% | 6,896 | 8,379 | 0.823× |
| 8,192 | load | 5 | 2,048.6 | 2,064.3 | 99.2% | 89.53 | 91.30 | 98.1% | 20,028 | 19,896 | 1.007× |
| 8,192 | load | 10 | 2,048.2 | 1,875.5 | 109.2% | 99.98 | 82.39 | 121.3% | 39,646 | 39,646 | 1.000× |
| 8,192 | reuse | 1 | 1,532.9 | 1,541.7 | 99.4% | 27.06 | 26.99 | 100.3% | 1,383 | 1,341 | 1.031× |
| 8,192 | reuse | 2 | 1,462.2 | 1,424.0 | 102.7% | 47.14 | 47.30 | 99.7% | 2,827 | 2,876 | 0.983× |
| 8,192 | reuse | 5 | 1,217.7 | 1,465.4 | 83.1% | 80.55 | 83.20 | 96.8% | 8,419 | 6,994 | 1.204× |
| 8,192 | reuse | 10 | 1,165.1 | 1,455.6 | 80.0% | 94.52 | 100.66 | 93.9% | 17,578 | 13,569 | 1.295× |
| 16,384 | load | 1 | 1,836.1 | 1,605.8 | 114.3% | 24.99 | 24.35 | 102.6% | 8,973 | 10,249 | 0.876× |
| 16,384 | load | 2 | 1,749.5 | 1,628.4 | 107.4% | 40.98 | 40.60 | 100.9% | 18,772 | 20,147 | 0.932× |
| 16,384 | load | 5 | 1,676.6 | 1,606.7 | 104.4% | 59.51 | 63.79 | 93.3% | 48,817 | 51,036 | 0.957× |
| 16,384 | load | 10 | 1,364.8 | OOM | – | 22.99 | OOM | – | 86,093 | OOM | – |
| 16,384 | reuse | 1 | 960.1 | 999.2 | 96.1% | 24.20 | 23.50 | 103.0% | 2,193 | 2,060 | 1.065× |
| 16,384 | reuse | 2 | 856.5 | 991.4 | 86.4% | 39.32 | 39.01 | 100.8% | 4,826 | 4,132 | 1.168× |
| 16,384 | reuse | 5 | 828.1 | 997.7 | 83.0% | 56.48 | 59.47 | 95.0% | 12,387 | 10,263 | 1.207× |
| 16,384 | reuse | 10 | 274.6 | OOM | – | 23.07 | OOM | – | 37,146 | OOM | – |
| 32,768 | load | 1 | 1,298.7 | 1,104.5 | 117.6% | 20.38 | 19.04 | 107.0% | 25,293 | 29,693 | 0.852× |
| 32,768 | load | 2 | 1,258.3 | OOM | – | 30.27 | OOM | – | 52,112 | OOM | – |
| 32,768 | load | 5 | 1,133.0 | OOM | – | 13.50 | OOM | – | 112,310 | OOM | – |
| 32,768 | load | 10 | 1,043.7 | OOM | – | 5.79 | OOM | – | 189,158 | OOM | – |
| 32,768 | reuse | 1 | 595.7 | 645.8 | 92.2% | 19.95 | 18.59 | 107.3% | 3,500 | 3,182 | 1.100× |
| 32,768 | reuse | 2 | 490.5 | OOM | – | 27.06 | OOM | – | 8,376 | OOM | – |
| 32,768 | reuse | 5 | 123.4 | OOM | – | 11.94 | OOM | – | 48,393 | OOM | – |
| 32,768 | reuse | 10 | 78.3 | OOM | – | 5.36 | OOM | – | 133,699 | OOM | – |

OOM marks cells where MLX-LM's server stopped with `[METAL] Command buffer
execution failed: Insufficient Memory` (`generation thread died`); a second
run on 2026-11-16 failed the same way. mirmir completed them on the same
machine. They are left out of the ratios above, which cover only cells both
engines finished (8 of 36 cells have no MLX-LM result).
