# Qwen3.8-27B 4-bit

## Model

| Property | Value |
|:---|:---|
| Hugging Face | `mlx-community/Qwen3.8-27B-4bit` |
| Snapshot | `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` |
| Parameters | 27B dense |
| Architecture | Hybrid linear/full-attention dense VLM |
| Checkpoint | MLX 4-bit SafeTensors |

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

## Metal — Apple M5 Pro, 64 GiB

| Property | Value |
|:---|:---|
| Machine | Apple M5 Pro |
| Memory | 64 GiB unified |
| macOS | 27.2 |
| mirmir | 0.6.2 (2026-11-15), MLX 0.32.4 |
| Reference | MLX-LM 0.32.2, MLX 0.32.4 |
| mirmir cells | 34 |
| Reference cells | 26 |

| Metric | mirmir | MLX-LM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 383.1 | 351.7 | 108.9% | 3/26 |
| TG tok/s | 27.50 | 26.66 | 103.2% | 20/26 |
| TTFT ms | 22,907 | 24,902 | 0.920× | 5/26 |

| Depth | Phase | C | PP mirmir | PP MLX | PP % | TG mirmir | TG MLX | TG % | TTFT mirmir ms | TTFT MLX ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 1 | 426.4 | 440.8 | 96.7% | 17.71 | 16.29 | 108.7% | 4,940 | 4,785 | 1.032× |
| 0 | plain | 2 | 389.4 | 396.7 | 98.2% | 32.11 | 31.80 | 101.0% | 10,791 | 10,590 | 1.019× |
| 0 | plain | 5 | 414.7 | 434.9 | 95.4% | 57.32 | 57.58 | 99.5% | 25,316 | 24,144 | 1.049× |
| 0 | plain | 10 | 418.0 | 431.3 | 96.9% | 59.59 | 57.47 | 103.7% | 50,239 | 47,822 | 1.051× |
| 4,096 | load | 1 | 402.8 | 409.9 | 98.3% | 17.45 | 16.50 | 105.8% | 10,307 | 10,149 | 1.016× |
| 4,096 | load | 2 | 418.7 | 424.2 | 98.7% | 31.83 | 30.44 | 104.6% | 19,819 | 19,566 | 1.013× |
| 4,096 | load | 5 | 417.3 | 435.3 | 95.9% | 53.99 | 53.29 | 101.3% | 49,725 | 47,664 | 1.043× |
| 4,096 | load | 10 | 416.2 | OOM | – | 57.24 | OOM | – | 99,710 | OOM | – |
| 4,096 | reuse | 1 | 382.6 | 406.6 | 94.1% | 17.31 | 17.07 | 101.4% | 5,357 | 5,059 | 1.059× |
| 4,096 | reuse | 2 | 397.9 | 423.5 | 94.0% | 31.19 | 29.65 | 105.2% | 10,295 | 9,670 | 1.065× |
| 4,096 | reuse | 5 | 396.8 | 194.7 | 203.8% | 52.94 | 50.54 | 104.7% | 25,805 | 52,232 | 0.494× |
| 4,096 | reuse | 10 | 393.8 | OOM | – | 55.21 | OOM | – | 52,009 | OOM | – |
| 8,192 | load | 1 | 410.7 | 412.4 | 99.6% | 17.17 | 16.28 | 105.5% | 20,082 | 20,020 | 1.003× |
| 8,192 | load | 2 | 417.3 | 421.1 | 99.1% | 30.53 | 29.07 | 105.0% | 39,522 | 39,159 | 1.009× |
| 8,192 | load | 5 | 411.7 | 424.7 | 96.9% | 50.08 | 50.03 | 100.1% | 100,146 | 97,081 | 1.032× |
| 8,192 | load | 10 | 413.1 | OOM | – | 51.74 | OOM | – | 199,600 | OOM | – |
| 8,192 | reuse | 1 | 382.1 | 400.6 | 95.4% | 17.06 | 16.90 | 100.9% | 5,362 | 5,136 | 1.044× |
| 8,192 | reuse | 2 | 381.9 | 406.7 | 93.9% | 29.52 | 29.76 | 99.2% | 10,727 | 10,069 | 1.065× |
| 8,192 | reuse | 5 | 377.8 | 121.4 | 311.2% | 49.54 | 34.84 | 142.2% | 27,102 | 79,978 | 0.339× |
| 8,192 | reuse | 10 | 375.5 | OOM | – | 50.89 | OOM | – | 54,540 | OOM | – |
| 16,384 | load | 1 | 399.9 | 408.3 | 97.9% | 16.43 | 16.21 | 101.4% | 41,104 | 40,282 | 1.020× |
| 16,384 | load | 2 | 409.6 | 413.1 | 99.2% | 28.58 | 28.56 | 100.1% | 80,258 | 79,581 | 1.009× |
| 16,384 | load | 5 | 401.5 | 404.8 | 99.2% | 29.53 | 45.45 | 65.0% | 188,226 | 203,053 | 0.927× |
| 16,384 | load | 10 | 399.7 | OOM | – | 26.82 | OOM | – | 411,304 | OOM | – |
| 16,384 | reuse | 1 | 342.5 | 375.5 | 91.2% | 16.50 | 15.88 | 103.9% | 5,979 | 5,474 | 1.092× |
| 16,384 | reuse | 2 | 340.4 | 375.9 | 90.6% | 28.27 | 28.32 | 99.8% | 12,033 | 10,895 | 1.104× |
| 16,384 | reuse | 5 | 347.1 | 92.3 | 376.1% | 36.95 | 18.07 | 204.5% | 29,505 | 94,011 | 0.314× |
| 16,384 | reuse | 10 | 322.0 | OOM | – | 38.67 | OOM | – | 63,624 | OOM | – |
| 32,768 | load | 1 | 372.4 | 379.8 | 98.1% | 15.51 | 15.24 | 101.8% | 88,135 | 86,455 | 1.019× |
| 32,768 | load | 2 | 373.9 | 375.3 | 99.6% | 17.55 | 24.80 | 70.8% | 160,850 | 174,929 | 0.920× |
| 32,768 | load | 10 | 366.6 | OOM | – | 33.59 | OOM | – | 895,199 | OOM | – |
| 32,768 | reuse | 1 | 294.1 | 320.7 | 91.7% | 15.33 | 15.28 | 100.3% | 6,966 | 6,407 | 1.087× |
| 32,768 | reuse | 2 | 283.2 | 321.8 | 88.0% | 24.43 | 24.66 | 99.1% | 14,466 | 12,727 | 1.137× |
| 32,768 | reuse | 10 | 32.8 | OOM | – | 8.72 | OOM | – | 501,073 | OOM | – |

OOM marks cells where MLX-LM's server stopped with `[METAL] Command buffer
execution failed: Insufficient Memory` (`generation thread died`); a second
run on 2026-11-16 failed the same way. mirmir completed them on the same
machine. They are left out of the ratios above, which cover only cells both
engines finished (8 of 34 cells have no MLX-LM result). mirmir has no result for depth 32,768 at
concurrency 5: all 30 responses arrived without errors, but 10 did not contain
exactly 128 tokens, so the harness discards the group.
